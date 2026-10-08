from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from common.data import (
    load_yaml,
    read_jsonl,
    repo_path,
    prompt_messages_from_preference,
    preference_responses,
    encode_prompt_response,
    pad_batch,
)
from common.data import write_jsonl
from common.logging_utils import set_seed
from common.metrics import sampled_kl
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode, clear_gpu
from common.generation import batch_generate, response_token_logprobs
from task1_dpo.dpo import dpo_loss
from task1_dpo.train import get_batch_logps, make_collate


def score_responses_with_rm(rm_model, rm_tokenizer, prompts, responses, batch_size=4, max_length=1280):
    device = next(rm_model.parameters()).device
    scores = []
    for i in range(0, len(prompts), batch_size):
        chunk_p = prompts[i : i + batch_size]
        chunk_r = responses[i : i + batch_size]
        formatted = []
        for p, r in zip(chunk_p, chunk_r):
            # Format prompt and assistant response
            msgs = list(p) + [{"role": "assistant", "content": r}]
            formatted.append(rm_tokenizer.apply_chat_template(msgs, tokenize=False))
        enc = rm_tokenizer(formatted, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            out = rm_model(**enc)
            batch_scores = out.logits.squeeze(-1).float().tolist()
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            scores.extend(batch_scores)
    return scores


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["dpo_standard_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def generation_eval(cfg, model, tokenizer, rm_model, rm_tokenizer, prompts, batch_size=4):
    """Sample responses on fixed held-out prompts; RM score, sampled-response KL, length."""
    set_seed(int(cfg["seed"]))
    responses, lengths, kls, seq_kls = [], [], [], []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        gen = batch_generate(
            model,
            tokenizer,
            chunk,
            max_prompt_length=int(cfg.get("max_prompt_length", 512)),
            max_new_tokens=int(cfg.get("max_generation_tokens", 256)),
            temperature=float(cfg["generation"]["temperature"]),
            top_p=float(cfg["generation"]["top_p"]),
            do_sample=True,
        )
        with torch.no_grad():
            pol_logp, _ = response_token_logprobs(model, gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"])
            with reference_mode(model):
                ref_logp, _ = response_token_logprobs(model, gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"])
        m = gen["response_mask"]
        for j in range(len(chunk)):
            kls.append(sampled_kl(pol_logp[j], ref_logp[j], m[j]).item())
            seq_kls.append(float(((pol_logp[j] - ref_logp[j]) * m[j]).sum().item()))
        responses.extend(gen["responses"])
        lengths.extend(gen["response_lengths"])
    rm_scores = score_responses_with_rm(rm_model, rm_tokenizer, prompts, responses)
    return responses, lengths, kls, seq_kls, rm_scores


def evaluate_dpo(config_path: str, adapter: str, name: str = "standard", max_eval_examples: int | None = None, beta: float | None = None, num_gen_prompts: int = 100):
    bundle = load_evaluation_bundle(config_path, adapter)
    cfg = bundle["cfg"]
    rows = bundle["rows"]
    if max_eval_examples is not None:
        rows = rows[:max_eval_examples]

    tokenizer = bundle["tokenizer"]
    model = bundle["policy"]
    rm_model, rm_tokenizer = bundle["reward"]
    beta = float(cfg.get("beta", 0.1) if beta is None else beta)

    loader = DataLoader(
        rows,
        batch_size=int(cfg.get("eval_batch_size", 4)),
        shuffle=False,
        collate_fn=make_collate(tokenizer, int(cfg.get("max_sequence_length", 1024))),
    )

    p_ch, p_rj, r_ch, r_rj = [], [], [], []
    print(f"Evaluating DPO model '{name}' (beta={beta}) on {len(rows)} held-out pairs...")
    with torch.no_grad():
        for chosen_batch, rejected_batch in loader:
            p_ch.append(get_batch_logps(model, chosen_batch).float().cpu())
            p_rj.append(get_batch_logps(model, rejected_batch).float().cpu())
            with reference_mode(model):
                r_ch.append(get_batch_logps(model, chosen_batch).float().cpu())
                r_rj.append(get_batch_logps(model, rejected_batch).float().cpu())
    p_ch, p_rj, r_ch, r_rj = (torch.cat(x) for x in (p_ch, p_rj, r_ch, r_rj))
    loss, _ = dpo_loss(p_ch, p_rj, r_ch, r_rj, beta)
    margins = (p_ch - p_rj) - (r_ch - r_rj)

    # Sample generations on the first held-out prompts for reward scoring, KL & length stats
    gen_rows = bundle["rows"][:num_gen_prompts]
    prompts_to_generate = [prompt_messages_from_preference(r) for r in gen_rows]
    responses, lengths, kls, seq_kls, rm_scores = generation_eval(cfg, model, tokenizer, rm_model, rm_tokenizer, prompts_to_generate)

    results = {
        "model_name": name,
        "adapter": adapter,
        "beta": beta,
        "num_pairs": len(rows),
        "dpo_loss": round(float(loss.item()), 4),
        "preference_accuracy": round(float((margins > 0).float().mean().item()), 4),
        "mean_dpo_margin": round(float(margins.mean().item()), 4),
        "mean_policy_margin": round(float((p_ch - p_rj).mean().item()), 4),
        "mean_ref_margin": round(float((r_ch - r_rj).mean().item()), 4),
        "num_generation_prompts": len(prompts_to_generate),
        # released estimator: response-token logp(policy) - logp(ref) on the policy's own samples, token-averaged
        "kl_divergence": round(float(np.mean(kls)), 5),
        "kl_sequence_sum_mean": round(float(np.mean(seq_kls)), 4),
        "reward_model_score_mean": round(float(np.mean(rm_scores)), 4),
        "reward_model_score_std": round(float(np.std(rm_scores)), 4),
        "response_length_mean": round(float(np.mean(lengths)), 2),
        "response_length_std": round(float(np.std(lengths)), 2),
        "response_length_iqr": round(float(np.percentile(lengths, 75) - np.percentile(lengths, 25)), 2),
    }

    print("\n--- Evaluation Results ---")
    for k, v in results.items():
        print(f"{k}: {v}")

    out_file = repo_path(f"results/dpo_{name}_eval.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    write_jsonl(f"results/samples/dpo_{name}_eval_samples.jsonl", [
        {"prompt_id": r.get("prompt_id"), "prompt": p[-1]["content"], "response": resp, "rm_score": round(sc, 4),
         "kl": round(k, 5), "response_tokens": n}
        for r, p, resp, sc, k, n in zip(gen_rows, prompts_to_generate, responses, rm_scores, kls, lengths)
    ])
    write_jsonl(f"results/samples/dpo_{name}_pair_margins.jsonl", [
        {"prompt_id": r.get("prompt_id"), "dpo_margin": round(float(m), 4)} for r, m in zip(rows, margins.tolist())
    ])
    print(f"Saved results to {out_file}")

    del model, rm_model, bundle
    clear_gpu()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--beta", type=float, help="beta the adapter was trained with (defaults to config beta)")
    args = ap.parse_args()
    evaluate_dpo(args.config, args.adapter, args.name, args.max_examples, args.beta)


if __name__ == "__main__":
    main()
