from __future__ import annotations

import argparse
import json

from common.data import load_yaml, repo_path, write_jsonl
from task2_ppo.evaluate import adapter_ok, evaluate_rl_policy


def evaluate_grpo(config_path: str, adapter: str, name: str = "standard", max_examples: int | None = 100):
    cfg = load_yaml(config_path)
    print(f"Evaluating GRPO policy '{name}' ({adapter}) on held-out prompts...")
    results, records = evaluate_rl_policy(cfg, adapter, int(cfg.get("max_completion_length", 512)), max_examples)
    results = {"policy_name": name, **results}

    print("\n--- GRPO Evaluation Results ---")
    for k, v in results.items():
        print(f"{k}: {v}")

    out_file = repo_path(f"results/grpo_{name}_eval.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    write_jsonl(f"results/samples/grpo_{name}_eval_samples.jsonl", records)
    print(f"Saved evaluation to {out_file}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--max-examples", type=int, default=100)
    args = ap.parse_args()
    if not adapter_ok(args.adapter):
        raise SystemExit(f"Adapter {args.adapter} missing adapter_config.json, train it first")
    evaluate_grpo(args.config, args.adapter, args.name, args.max_examples)


if __name__ == "__main__":
    main()
