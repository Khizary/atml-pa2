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


def _group_stats(groups, tol=1e-6):
    stds, info, centered, standardized = [], [], [], []
    for g in groups:
        rews = np.array(g["rewards"], dtype=float)
        sd = float(np.std(rews))
        stds.append(sd)
        info.append(float(sd > tol))
        centered.extend((rews - rews.mean()).tolist())
        # same clamp as group_relative_advantages: zero-std groups give all-zero advantages
        standardized.extend(((rews - rews.mean()) / max(sd, tol)).tolist())
    return {
        "num_groups": len(groups),
        "num_completions": len(centered),
        "informative_group_rate": round(float(np.mean(info)), 4),
        "uninformative_group_rate": round(1.0 - float(np.mean(info)), 4),
        "mean_within_group_std": round(float(np.mean(stds)), 4),
        # variance of the raw relative signal r_k - mean_group (what the advantage is built from)
        "centered_reward_variance": round(float(np.var(centered)), 4),
        # variance of the standardized advantage pooled over every completion in the budget
        "standardized_advantage_variance": round(float(np.var(standardized)), 4),
    }


def analyze_groups(groups, prompt_difficulty_map=None):
    summary = _group_stats(groups)
    if prompt_difficulty_map:
        summary["difficulty_bins"] = {}
        for diff in sorted(set(prompt_difficulty_map.values())):
            sub = [g for g in groups if prompt_difficulty_map.get(g["prompt_id"]) == diff]
            if sub:
                summary["difficulty_bins"][diff] = _group_stats(sub)
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

    results = {
        "binning_rule": "prompt mean reward over all 8 cached completions; easy if >= median across prompts, hard otherwise",
        "median_prompt_reward": round(median_diff, 4),
        "num_prompts": len(by_prompt),
        "bin_sizes": {b: sum(1 for v in prompt_difficulty_map.values() if v == b) for b in ["easy", "hard"]},
    }
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
