from __future__ import annotations

import argparse
import json
import numpy as np

from common.data import load_yaml, repo_path
from common.metrics import safe_corr
from task3_grpo.continue_train import run_grpo
from task3_grpo.evaluate import evaluate_grpo


def length_conditioned_stats(traj):
    """How gradient weight is split between short and long completions.

    Uses every unmasked completion with a non-zero advantage. Short/long = below/above the
    median length of those completions. Weight of completion k = |A_k| * T_k / norm.
    """
    lens, weights, advs = [], [], []
    for t in traj:
        for L, w, a, m in zip(t["completion_lengths"], t["completion_grad_weight"], t["completion_advantages"], t["completion_masked"]):
            if m or abs(a) < 1e-8:
                continue
            lens.append(L)
            weights.append(w)
            advs.append(abs(a))
    if not lens:
        return {}
    lens, weights, advs = np.array(lens, float), np.array(weights, float), np.array(advs, float)
    med = float(np.median(lens))
    short, long_ = lens <= med, lens > med
    per_token = weights / np.maximum(lens, 1.0)
    return {
        "num_completions": int(len(lens)),
        "median_length": med,
        "short_share_of_grad_weight": round(float(weights[short].sum() / weights.sum()), 4),
        "long_share_of_grad_weight": round(float(weights[long_].sum() / weights.sum()), 4),
        "mean_per_token_weight_short": round(float(per_token[short].mean()), 6) if short.any() else None,
        "mean_per_token_weight_long": round(float(per_token[long_].mean()), 6) if long_.any() else None,
        "corr_length_vs_seq_grad_weight": round(safe_corr(lens, weights), 4),
        "mean_grad_norm": round(float(np.mean([t["grad_norm"] for t in traj])), 5),
        "mean_train_length": round(float(np.mean([t["response_length"] for t in traj])), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--max-examples", type=int, default=100)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    fork_updates = int(cfg.get("fork_updates", 8))

    print(f"Running normalization comparison (canonical GRPO vs Dr. GRPO) for {fork_updates} updates...")
    results = {}
    for name, loss_type in [("canonical", "grpo"), ("dr_grpo", "dr_grpo")]:
        print(f"\n=== Running {name} (loss_type={loss_type}) ===")
        out_dir = f"outputs/task3_grpo/{name}"
        run_grpo(config_path=args.config, output=out_dir, updates=fork_updates, loss_type=loss_type, run_name=name)
        with open(repo_path(f"results/grpo_{name}_continuation.json")) as f:
            train = json.load(f)
        held_out = evaluate_grpo(args.config, out_dir, name, args.max_examples)
        results[name] = {
            "held_out": held_out,
            "length_conditioned": length_conditioned_stats(train["trajectory"]),
            "train": train,
        }

    out_file = repo_path("results/grpo_normalization_study.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nNormalization comparison complete. Results saved to {out_file}")


if __name__ == "__main__":
    main()
