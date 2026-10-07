from __future__ import annotations

import argparse
import json
from collections import defaultdict
import numpy as np

from common.data import load_yaml, read_jsonl, repo_path


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int):
    """Partition 8 completions per prompt into 8/k groups of size k."""
    groups = []
    for pid, completions in by_prompt.items():
        # completions has length >= 8
        avail = completions[:8]
        num_subgroups = 8 // k
        for i in range(num_subgroups):
            sub = avail[i * k : (i + 1) * k]
            groups.append({
                "prompt_id": pid,
                "completions": sub,
                "rewards": [float(c["reward"]) for c in sub],
            })
    return groups


def analyze_groups(groups, prompt_difficulty_map=None):
    stds = []
    informative_flags = []
    adv_variances = []

    diff_breakdown = defaultdict(lambda: {"stds": [], "informative": []})

    for g in groups:
        rews = np.array(g["rewards"], dtype=float)
        std_val = float(np.std(rews))
        stds.append(std_val)
        is_info = float(std_val > 1e-5)
        informative_flags.append(is_info)

        if std_val > 1e-5:
            adv = (rews - np.mean(rews)) / (std_val + 1e-6)
            adv_variances.append(float(np.var(adv)))
        else:
            adv_variances.append(0.0)

        if prompt_difficulty_map:
            diff = prompt_difficulty_map.get(g["prompt_id"], "unknown")
            diff_breakdown[diff]["stds"].append(std_val)
            diff_breakdown[diff]["informative"].append(is_info)

    summary = {
        "num_groups": len(groups),
        "mean_within_group_std": round(float(np.mean(stds)), 4),
        "informative_group_rate": round(float(np.mean(informative_flags)), 4),
        "uninformative_group_rate": round(1.0 - float(np.mean(informative_flags)), 4),
        "relative_signal_variance": round(float(np.mean(adv_variances)), 4),
    }

    if prompt_difficulty_map:
        summary["difficulty_bins"] = {}
        for diff, stats in diff_breakdown.items():
            summary["difficulty_bins"][diff] = {
                "count": len(stats["stds"]),
                "mean_within_group_std": round(float(np.mean(stats["stds"])), 4),
                "informative_group_rate": round(float(np.mean(stats["informative"])), 4),
            }

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    by_prompt = load_k8_cache(cfg["group_cache"])
    group_sizes = cfg.get("group_sizes", [2, 4, 8])
    print(f"Loaded {len(by_prompt)} cached prompts. Analyzing K in {group_sizes}...")

    # Calculate overall prompt difficulty based on full 8-sample mean reward
    prompt_difficulty_map = {}
    mean_prompt_rewards = []
    for pid, compl in by_prompt.items():
        m_rew = float(np.mean([float(c["reward"]) for c in compl[:8]]))
        mean_prompt_rewards.append(m_rew)

    median_diff = float(np.median(mean_prompt_rewards)) if mean_prompt_rewards else 0.5
    for pid, compl in by_prompt.items():
        m_rew = float(np.mean([float(c["reward"]) for c in compl[:8]]))
        prompt_difficulty_map[pid] = "easy" if m_rew >= median_diff else "hard"

    results = {}
    for k in group_sizes:
        print(f"\nEvaluating K = {k}...")
        groups = regroup_equal_generation_budget(by_prompt, k)
        stats = analyze_groups(groups, prompt_difficulty_map)
        results[f"K_{k}"] = stats
        print(f"K={k}: informative_rate={stats['informative_group_rate']:.3f}, within_std={stats['mean_within_group_std']:.3f}")

    out_file = repo_path("results/grpo_group_size.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nGroup size analysis complete. Saved to {out_file}")


if __name__ == "__main__":
    main()
