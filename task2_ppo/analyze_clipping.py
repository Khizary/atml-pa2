from __future__ import annotations

import argparse
import json
import torch
import numpy as np

from common.data import load_yaml, repo_path
from task2_ppo.continue_train import run_ppo


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def analyze_cached_batch(rows, epsilons):
    results = {}
    for eps in epsilons:
        clipped_fractions = []
        for r in rows:
            old_lp = torch.as_tensor(r["old_logprobs"])
            ref_lp = torch.as_tensor(r["ref_logprobs"])
            # approximate ratio between old and ref or synthetic ratio
            ratio = torch.exp(old_lp - ref_lp)
            affected = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
            clipped_fractions.append(affected.mean().item())
        results[str(eps)] = {
            "cached_clip_fraction": round(float(np.mean(clipped_fractions)), 4),
            "cached_affected_token_rate": round(float(np.mean(clipped_fractions)), 4),
        }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    rows = load_cached_rollouts(cfg["cached_rollouts"])
    epsilons = cfg.get("clip_values", [0.05, 0.20, 0.50])
    fork_updates = int(cfg.get("fork_updates", 8))

    print(f"Loaded {len(rows)} cached PPO rollouts. Analyzing epsilons={epsilons}...")
    cached_metrics = analyze_cached_batch(rows, epsilons)

    fork_results = {}
    if not args.skip_train:
        for eps in epsilons:
            run_name = f"clip_{eps}".replace(".", "_")
            out_dir = f"outputs/task2_ppo/{run_name}"
            print(f"\n=== Running PPO short fork for epsilon={eps} ===")
            run_ppo(
                config_path=args.config,
                output=out_dir,
                updates=fork_updates,
                clip_epsilon=eps,
                run_name=run_name,
            )
            # Load logged continuation
            res_path = repo_path(f"results/ppo_{run_name}_continuation.json")
            if res_path.exists():
                with open(res_path) as f:
                    fork_results[str(eps)] = json.load(f)

    final_results = {
        "cached_rollout_analysis": cached_metrics,
        "short_fork_results": fork_results,
    }

    out_file = repo_path("results/ppo_clipping_study.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nPPO clipping study complete. Results saved to {out_file}")


if __name__ == "__main__":
    main()
