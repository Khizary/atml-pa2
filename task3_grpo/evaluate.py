from __future__ import annotations

import argparse
import json
import numpy as np
import torch

from common.data import load_yaml, read_jsonl, prompt_messages, repo_path
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from common.generation import batch_generate, response_token_logprobs
from common.metrics import sampled_kl
from task2_ppo.continue_train import score_rollouts


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def evaluate_grpo(config_path: str, adapter: str, name: str = "standard", max_examples: int | None = 100):
    bundle = load_evaluation_bundle(config_path, adapter)
    cfg = bundle["cfg"]
    rows = bundle["rows"]
    if max_examples is not None:
        rows = rows[:max_examples]

    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]
    rm_model, rm_tok = bundle["reward"]

    print(f"Evaluating GRPO policy '{name}' on {len(rows)} held-out prompts...")
    prompt_msgs = [prompt_messages(r) for r in rows]

    scores, kl_divs, lengths = [], [], []
    batch_size = int(cfg.get("eval_batch_size", 4))
    for i in range(0, len(prompt_msgs), batch_size):
        chunk_msgs = prompt_msgs[i : i + batch_size]
        gen = batch_generate(
            policy,
            tokenizer,
            chunk_msgs,
            max_prompt_length=int(cfg.get("max_prompt_length", 256)),
            max_new_tokens=int(cfg.get("max_completion_length", 512)),
            temperature=float(cfg.get("generation", {}).get("temperature", 0.7)),
            top_p=float(cfg.get("generation", {}).get("top_p", 0.9)),
            do_sample=True,
        )
        chunk_scores = score_rollouts(rm_model, rm_tok, chunk_msgs, gen["responses"]).tolist()
        scores.extend(chunk_scores if isinstance(chunk_scores, list) else [chunk_scores])

        with torch.no_grad():
            pol_logp, _ = response_token_logprobs(policy, gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"])
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(policy, gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"])
            kl_val = sampled_kl(pol_logp, ref_logp, gen["response_mask"]).item()
            kl_divs.append(kl_val)
        lengths.extend(gen["response_lengths"])

    results = {
        "policy_name": name,
        "eval_examples": len(rows),
        "mean_reward": round(float(np.mean(scores)), 4),
        "std_reward": round(float(np.std(scores)), 4),
        "kl_divergence": round(float(np.mean(kl_divs)), 4) if kl_divs else 0.0,
        "response_length_mean": round(float(np.mean(lengths)), 2),
        "response_length_std": round(float(np.std(lengths)), 2),
    }

    print("\n--- GRPO Evaluation Results ---")
    for k, v in results.items():
        print(f"{k}: {v}")

    out_file = repo_path(f"results/grpo_{name}_eval.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved evaluation to {out_file}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--max-examples", type=int, default=100)
    args = ap.parse_args()
    evaluate_grpo(args.config, args.adapter, args.name, args.max_examples)


if __name__ == "__main__":
    main()
