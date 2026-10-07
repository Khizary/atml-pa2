from __future__ import annotations

import argparse
import json
from collections import defaultdict
import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def compare_pair_verifier(gold, resp_a, resp_b):
    ra = exact_reward(resp_a, gold)
    rb = exact_reward(resp_b, gold)
    if ra > rb:
        return "A"
    elif rb > ra:
        return "B"
    else:
        return "TIE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print(f"Loaded {len(groups)} diagnostic problems with all 5 variants.")

    judge = PairwiseAIJudge(cfg, cache_path="cached/ai_judge_diagnostics.json")

    # We evaluate pairs: (better_variant, worse_variant)
    contrast_pairs = [
        ("clean_correct", "corrupt_reasoning_correct_final", "reasoning_corruption"),
        ("clean_correct", "good_reasoning_wrong_final", "outcome_error"),
        ("clean_correct", "persuasive_filler_correct", "persuasive_filler"),
        ("clean_correct", "gold_distractor_wrong_final", "distractor_error"),
    ]

    analysis = {
        "verifier": defaultdict(lambda: {"better": 0, "tie": 0, "worse": 0}),
        "ai_judge": defaultdict(lambda: {"better": 0, "tie": 0, "worse": 0}),
    }

    for pid, variants in groups.items():
        prob = variants["clean_correct"]["question"]
        gold = variants["clean_correct"]["gold_final"]

        for better_var, worse_var, pair_name in contrast_pairs:
            resp_better = variants[better_var]["response"]
            resp_worse = variants[worse_var]["response"]

            # Verifier comparison
            v_pref = compare_pair_verifier(gold, resp_better, resp_worse)
            if v_pref == "A":
                analysis["verifier"][pair_name]["better"] += 1
            elif v_pref == "TIE":
                analysis["verifier"][pair_name]["tie"] += 1
            else:
                analysis["verifier"][pair_name]["worse"] += 1

            # AI judge comparison
            j_pref = judge.compare(prob, resp_better, resp_worse)
            if j_pref == "A":
                analysis["ai_judge"][pair_name]["better"] += 1
            elif j_pref == "TIE":
                analysis["ai_judge"][pair_name]["tie"] += 1
            else:
                analysis["ai_judge"][pair_name]["worse"] += 1

    total_probs = len(groups)
    results = {}
    for mechanism in ["verifier", "ai_judge"]:
        results[mechanism] = {}
        for pair_name, counts in analysis[mechanism].items():
            results[mechanism][pair_name] = {
                "better_rate": round(counts["better"] / total_probs, 4),
                "tie_rate": round(counts["tie"] / total_probs, 4),
                "wrong_preference_rate": round(counts["worse"] / total_probs, 4),
            }

        # Specific Sensitivity metrics
        results[mechanism]["S_reason"] = results[mechanism]["reasoning_corruption"]["better_rate"]
        results[mechanism]["S_outcome"] = results[mechanism]["outcome_error"]["better_rate"]

    print("\n=== Perturbation Diagnostics Summary ===")
    print("Exact Verifier:")
    print(f"  S_reason: {results['verifier']['S_reason']:.3f} | S_outcome: {results['verifier']['S_outcome']:.3f}")
    print("AI Pairwise Judge:")
    print(f"  S_reason: {results['ai_judge']['S_reason']:.3f} | S_outcome: {results['ai_judge']['S_outcome']:.3f}")

    out_file = repo_path("results/feedback_diagnostics.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved diagnostic results to {out_file}")


if __name__ == "__main__":
    main()
