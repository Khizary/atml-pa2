from __future__ import annotations

import argparse
import json
import numpy as np
import torch

from common.data import load_yaml, read_jsonl, prompt_messages, repo_path, write_jsonl
from common.logging_utils import set_seed
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode, clear_gpu
from common.generation import batch_generate, response_token_logprobs
from common.metrics import sampled_kl, logits_entropy
from task2_ppo.continue_train import score_rollouts


def adapter_ok(adapter) -> bool:
    p = repo_path(adapter)
    return p.exists() and (p / "adapter_config.json").exists()


def evaluate_rl_policy(cfg, adapter, max_new_tokens: int, max_examples: int | None = 100, batch_size: int = 4):
    """Held-out eval shared by ppo/grpo: same prompts, same seed, same sampling for every condition."""
    rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])
    if max_examples is not None:
        rows = rows[:max_examples]
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    rm_model, rm_tok = load_reward_model(cfg)
    set_seed(int(cfg["seed"]))

    scores, kls, ents, lengths, truncs, records = [], [], [], [], [], []
    for i in range(0, len(rows), batch_size):
        chunk_rows = rows[i : i + batch_size]
        chunk_msgs = [prompt_messages(r) for r in chunk_rows]
        gen = batch_generate(
            policy,
            tokenizer,
            chunk_msgs,
            max_prompt_length=int(cfg.get("max_prompt_length", 256)),
            max_new_tokens=max_new_tokens,
            temperature=float(cfg["generation"]["temperature"]),
            top_p=float(cfg["generation"]["top_p"]),
            do_sample=True,
        )
        chunk_scores = score_rollouts(rm_model, rm_tok, chunk_msgs, gen["responses"], int(cfg.get("reward_max_length", 1280))).float().tolist()
        with torch.no_grad():
            pol_logp, logits = response_token_logprobs(policy, gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"])
            ent_rows = [logits_entropy(logits[j : j + 1], gen["response_mask"][j : j + 1]).item() for j in range(len(chunk_rows))]
            del logits
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(policy, gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"])
            kl_rows = [sampled_kl(pol_logp[j], ref_logp[j], gen["response_mask"][j]).item() for j in range(len(chunk_rows))]

        for j, r in enumerate(chunk_rows):
            records.append({
                "prompt_id": r.get("prompt_id", r.get("source_index")),
                "prompt": chunk_msgs[j][-1]["content"],
                "response": gen["responses"][j],
                "reward": round(chunk_scores[j], 4),
                "kl": round(kl_rows[j], 5),
                "entropy": round(ent_rows[j], 4),
                "response_tokens": gen["response_lengths"][j],
                "truncated": gen["truncated"][j],
            })
        scores.extend(chunk_scores)
        kls.extend(kl_rows)
        ents.extend(ent_rows)
        lengths.extend(gen["response_lengths"])
        truncs.extend(gen["truncated"])

    del policy, rm_model
    clear_gpu()

    results = {
        "adapter": str(adapter),
        "eval_examples": len(rows),
        "max_new_tokens": max_new_tokens,
        "mean_reward": round(float(np.mean(scores)), 4),
        "std_reward": round(float(np.std(scores)), 4),
        # per-response token-averaged sampled kl (policy - ref logp on own samples), then averaged over prompts
        "kl_divergence": round(float(np.mean(kls)), 5),
        "entropy": round(float(np.mean(ents)), 4),
        "response_length_mean": round(float(np.mean(lengths)), 2),
        "response_length_std": round(float(np.std(lengths)), 2),
        "response_length_iqr": round(float(np.percentile(lengths, 75) - np.percentile(lengths, 25)), 2),
        "truncation_rate": round(float(np.mean(truncs)), 4),
    }
    return results, records


def evaluate_ppo(config_path: str, adapter: str, name: str = "standard", max_examples: int | None = 100):
    cfg = load_yaml(config_path)
    print(f"Evaluating PPO policy '{name}' ({adapter}) on held-out prompts...")
    results, records = evaluate_rl_policy(cfg, adapter, int(cfg.get("eval_max_response_length", 768)), max_examples)
    results = {"policy_name": name, **results}

    print("\n--- PPO Evaluation Results ---")
    for k, v in results.items():
        print(f"{k}: {v}")

    out_file = repo_path(f"results/ppo_{name}_eval.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    write_jsonl(f"results/samples/ppo_{name}_eval_samples.jsonl", records)
    print(f"Saved evaluation to {out_file}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--max-examples", type=int, default=100)
    args = ap.parse_args()
    if not adapter_ok(args.adapter):
        raise SystemExit(f"Adapter {args.adapter} missing adapter_config.json, train it first")
    evaluate_ppo(args.config, args.adapter, args.name, args.max_examples)


if __name__ == "__main__":
    main()
