from __future__ import annotations

import argparse
import json
import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.models import load_policy, load_tokenizer
from common.generation import batch_generate
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def evaluate_dataset(config_path: str, dataset: str, max_examples: int | None = None):
    cfg, rows, tokenizer = load_math_evaluation(config_path, dataset)
    if max_examples is not None:
        rows = rows[:max_examples]

    print(f"\n=== Evaluating Math on {dataset.upper()} ({len(rows)} problems) ===")
    policies = ["sft", "rlvr", "rlaif"]
    policy_responses = {}

    for pol in policies:
        print(f"Generating completions for policy '{pol}'...")
        model = load_frozen_policy(cfg, pol)
        prompts = [[{"role": "user", "content": str(r.get("question", r.get("prompt", "")))}] for r in rows]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=int(cfg.get("math_max_new_tokens", 512)),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        policy_responses[pol] = gen["responses"]
        del model

    judge = PairwiseAIJudge(cfg, cache_path=f"cached/ai_judge_{dataset}.json")

    results = {"dataset": dataset, "num_problems": len(rows), "policies": {}}

    for pol in policies:
        resps = policy_responses[pol]
        exact_accs = []
        format_compliances = []
        lengths = []
        for r, resp in zip(rows, resps):
            gold = str(r.get("gold_final", ""))
            exact_accs.append(exact_reward(resp, gold))
            format_compliances.append(float(extract_designated_final(resp) is not None))
            lengths.append(len(resp.split()))

        results["policies"][pol] = {
            "exact_accuracy": round(float(np.mean(exact_accs)), 4),
            "format_compliance_rate": round(float(np.mean(format_compliances)), 4),
            "mean_response_words": round(float(np.mean(lengths)), 2),
        }
        print(f"[{pol.upper()}] Exact Acc: {results['policies'][pol]['exact_accuracy']:.4f} | Format: {results['policies'][pol]['format_compliance_rate']:.4f} | Words: {results['policies'][pol]['mean_response_words']:.1f}")

    # Pairwise comparisons against SFT
    pairwise = {}
    for pol in ["rlvr", "rlaif"]:
        wins, ties, losses = 0, 0, 0
        for i, r in enumerate(rows):
            prob = str(r.get("question", r.get("prompt", "")))
            cand_pol = policy_responses[pol][i]
            cand_sft = policy_responses["sft"][i]
            pref = judge.compare(prob, cand_pol, cand_sft)
            if pref == "A":
                wins += 1
            elif pref == "B":
                losses += 1
            else:
                ties += 1

        total = max(1, len(rows))
        win_rate = (wins + 0.5 * ties) / total
        pairwise[f"{pol}_vs_sft"] = {
            "win_rate": round(float(win_rate), 4),
            "wins": wins,
            "ties": ties,
            "losses": losses,
        }
        print(f"[{pol.upper()} vs SFT] Win Rate: {win_rate:.4f} (W:{wins}, T:{ties}, L:{losses})")

    results["pairwise"] = pairwise

    out_file = repo_path(f"results/feedback_{dataset}_eval.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved evaluation to {out_file}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--max-examples", type=int, default=None)
    args = ap.parse_args()
    evaluate_dataset(args.config, args.dataset, args.max_examples)


if __name__ == "__main__":
    main()
