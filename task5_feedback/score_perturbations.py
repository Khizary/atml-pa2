from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
import numpy as np

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
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

    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)
    judge = PairwiseAIJudge(cfg, cache_path=outdir / "judge_cache_diagnostics.json")

    # (diagnostically better, worse, name). clean_correct is the better side of every pair.
    contrast_pairs = [
        ("clean_correct", "corrupt_reasoning_correct_final", "reasoning_corruption"),
        ("clean_correct", "good_reasoning_wrong_final", "outcome_error"),
        ("clean_correct", "persuasive_filler_correct", "persuasive_filler"),
        ("clean_correct", "gold_distractor_wrong_final", "gold_distractor"),
    ]
    counts = {m: defaultdict(Counter) for m in ["verifier", "ai_judge"]}
    pair_records = []
    variant_scores = {m: defaultdict(list) for m in ["verifier", "ai_judge_group_winrate"]}

    for pid, variants in groups.items():
        prob = str(variants["clean_correct"]["question"])
        gold = str(variants["clean_correct"]["gold_final"])
        for better_var, worse_var, pair_name in contrast_pairs:
            resp_better = variants[better_var]["response"]
            resp_worse = variants[worse_var]["response"]
            prefs = {
                "verifier": compare_pair_verifier(gold, resp_better, resp_worse),
                "ai_judge": judge.compare(prob, resp_better, resp_worse),
            }
            for mech, pref in prefs.items():
                counts[mech][pair_name][{"A": "better", "B": "worse", "TIE": "tie"}[pref]] += 1
            pair_records.append({"problem_id": pid, "pair": pair_name, **{f"{m}_pref": v for m, v in prefs.items()}})

        # rewards each mechanism would hand out if all five variants were one sampled group
        names = sorted(EXPECTED_VARIANTS)
        group = judge.group_rewards(prob, [variants[v]["response"] for v in names])
        for v, g in zip(names, group):
            variant_scores["ai_judge_group_winrate"][v].append(g)
            variant_scores["verifier"][v].append(exact_reward(variants[v]["response"], gold))

    total_probs = len(groups)
    results = {"num_problems": total_probs}
    for mech in ["verifier", "ai_judge"]:
        results[mech] = {}
        for _, _, pair_name in contrast_pairs:
            c = counts[mech][pair_name]
            results[mech][pair_name] = {
                "better_rate": round(c["better"] / total_probs, 4),
                "tie_rate": round(c["tie"] / total_probs, 4),
                "wrong_preference_rate": round(c["worse"] / total_probs, 4),
            }
        # S_reason: final answer held correct, reasoning degraded. S_outcome: reasoning held, final changed.
        results[mech]["S_reason"] = results[mech]["reasoning_corruption"]["better_rate"]
        results[mech]["S_outcome"] = results[mech]["outcome_error"]["better_rate"]
    results["mean_reward_per_variant"] = {
        m: {v: round(float(np.mean(x)), 4) for v, x in d.items()} for m, d in variant_scores.items()
    }

    print("\n=== Perturbation Diagnostics Summary ===")
    for mech in ["verifier", "ai_judge"]:
        print(f"{mech}: S_reason={results[mech]['S_reason']:.3f} S_outcome={results[mech]['S_outcome']:.3f}")
        for _, _, pair_name in contrast_pairs:
            print(f"   {pair_name}: {results[mech][pair_name]}")
    print("mean reward per variant:", json.dumps(results["mean_reward_per_variant"], indent=1))

    out_file = outdir / "diagnostics.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    write_jsonl(outdir / "diagnostics_pairs.jsonl", pair_records)
    print(f"\nSaved diagnostic results to {out_file}")


if __name__ == "__main__":
    main()
