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
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from common.generation import batch_generate
from task1_dpo.dpo import dpo_loss
from task1_dpo.train import get_batch_logps, make_collate


def score_responses_with_rm(rm_model, rm_tokenizer, prompts, responses, batch_size=4):
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
        enc = rm_tokenizer(formatted, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            out = rm_model(**enc)
            batch_scores = out.logits.squeeze(-1).tolist()
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


def evaluate_dpo(config_path: str, adapter: str, name: str = "standard", max_eval_examples: int | None = None):
    bundle = load_evaluation_bundle(config_path, adapter)
    cfg = bundle["cfg"]
    rows = bundle["rows"]
    if max_eval_examples is not None:
        rows = rows[:max_eval_examples]

    tokenizer = bundle["tokenizer"]
    model = bundle["policy"]
    rm_model, rm_tokenizer = bundle["reward"]
    beta = float(cfg.get("beta", 0.1))

    loader = DataLoader(
        rows,
        batch_size=int(cfg.get("eval_batch_size", 4)),
        shuffle=False,
        collate_fn=make_collate(tokenizer, int(cfg.get("max_sequence_length", 1024))),
    )

    losses = []
    accuracies = []
    policy_margins = []
    ref_margins = []
    kl_diffs = []

    print(f"Evaluating DPO model '{name}' on {len(rows)} held-out pairs...")
    with torch.no_grad():
        for chosen_batch, rejected_batch in loader:
            p_chosen = get_batch_logps(model, chosen_batch)
            p_rejected = get_batch_logps(model, rejected_batch)

            with reference_mode(model):
                r_chosen = get_batch_logps(model, chosen_batch)
                r_rejected = get_batch_logps(model, rejected_batch)

            loss, diags = dpo_loss(p_chosen, p_rejected, r_chosen, r_rejected, beta)
            losses.append(loss.item())
            accuracies.append(diags["preference_accuracy"].item())
            policy_margins.extend((p_chosen - p_rejected).tolist())
            ref_margins.extend((r_chosen - r_rejected).tolist())

            # Token-level or sequence KL difference proxy: (log pi_theta - log pi_ref) on chosen
            kl_chosen = (p_chosen - r_chosen)
            kl_diffs.extend(kl_chosen.tolist())

    # Sample generations on held-out prompts for reward scoring & length stats
    prompts_to_generate = [prompt_messages_from_preference(r) for r in rows[:min(len(rows), 100)]]
    gen_results = batch_generate(
        model,
        tokenizer,
        prompts_to_generate,
        max_prompt_length=256,
        max_new_tokens=int(cfg.get("max_new_tokens", 256)),
        temperature=float(cfg.get("generation", {}).get("temperature", 0.7)),
        top_p=float(cfg.get("generation", {}).get("top_p", 0.9)),
        do_sample=True,
    )

    responses = gen_results["responses"]
    lengths = gen_results["response_lengths"]
    rm_scores = score_responses_with_rm(rm_model, rm_tokenizer, prompts_to_generate, responses)

    results = {
        "model_name": name,
        "beta": beta,
        "num_pairs": len(rows),
        "dpo_loss": round(float(np.mean(losses)), 4),
        "preference_accuracy": round(float(np.mean(accuracies)), 4),
        "mean_policy_margin": round(float(np.mean(policy_margins)), 4),
        "mean_ref_margin": round(float(np.mean(ref_margins)), 4),
        "mean_kl_divergence": round(float(np.mean(kl_diffs)), 4),
        "reward_model_score_mean": round(float(np.mean(rm_scores)), 4),
        "reward_model_score_std": round(float(np.std(rm_scores)), 4),
        "response_length_mean": round(float(np.mean(lengths)), 2),
        "response_length_std": round(float(np.std(lengths)), 2),
    }

    print("\n--- Evaluation Results ---")
    for k, v in results.items():
        print(f"{k}: {v}")

    out_file = repo_path(f"results/dpo_{name}_eval.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_file}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--max-examples", type=int, default=None)
    args = ap.parse_args()
    evaluate_dpo(args.config, args.adapter, args.name, args.max_examples)


if __name__ == "__main__":
    main()
