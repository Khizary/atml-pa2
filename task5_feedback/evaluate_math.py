from __future__ import annotations

import argparse
import json
from collections import Counter
import numpy as np

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.models import load_policy, load_tokenizer, clear_gpu
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


def failure_type(resp, gold, truncated):
    pred = extract_designated_final(resp)
    if pred is None:
        return "truncated_no_final" if truncated else "no_final_format"
    return "correct" if exact_reward(resp, gold) == 1.0 else "wrong_final"


def evaluate_dataset(config_path: str, dataset: str, max_examples: int | None = None):
    cfg, rows, tokenizer = load_math_evaluation(config_path, dataset)
    if max_examples is not None:
        rows = rows[:max_examples]
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Evaluating Math on {dataset.upper()} ({len(rows)} problems) ===")
    policies = ["sft", "rlvr", "rlaif"]
    gens = {}
    max_new = int(cfg.get("math_max_new_tokens", 512))
    batch_size = int(cfg.get("math_eval_batch_size", 16))
    # the course prompt (with the `#### <number>` instruction) is in `messages`
    prompts = [prompt_messages(r) for r in rows]

    for pol in policies:
        print(f"Generating completions for policy '{pol}'...")
        model = load_frozen_policy(cfg, pol)
        out = {"responses": [], "lengths": [], "truncated": []}
        for start in range(0, len(prompts), batch_size):
            gen = batch_generate(
                model,
                tokenizer,
                prompts[start : start + batch_size],
                max_prompt_length=512,
                max_new_tokens=max_new,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
            )
            out["responses"].extend(gen["responses"])
            out["lengths"].extend(gen["response_lengths"])
            out["truncated"].extend(gen["truncated"])
        gens[pol] = out
        del model
        clear_gpu()

    results = {"dataset": dataset, "num_problems": len(rows), "max_new_tokens": max_new, "decoding": "greedy", "policies": {}}
    for pol in policies:
        g = gens[pol]
        golds = [str(r["gold_final"]) for r in rows]
        exact = [exact_reward(resp, gold) for resp, gold in zip(g["responses"], golds)]
        fmt = [float(extract_designated_final(resp) is not None) for resp in g["responses"]]
        ftypes = [failure_type(resp, gold, t) for resp, gold, t in zip(g["responses"], golds, g["truncated"])]
        results["policies"][pol] = {
            "exact_accuracy": round(float(np.mean(exact)), 4),
            "format_compliance_rate": round(float(np.mean(fmt)), 4),
            "response_tokens_mean": round(float(np.mean(g["lengths"])), 2),
            "response_tokens_std": round(float(np.std(g["lengths"])), 2),
            "truncation_rate": round(float(np.mean(g["truncated"])), 4),
            "failure_types": dict(Counter(ftypes)),
        }
        g["exact"], g["ftypes"] = exact, ftypes
        print(f"[{pol.upper()}] Exact Acc: {np.mean(exact):.4f} | Format: {np.mean(fmt):.4f} | Tokens: {np.mean(g['lengths']):.1f}")

    judge = PairwiseAIJudge(cfg, cache_path=outdir / f"judge_cache_{dataset}.json")

    pairwise = {}
    for pol in ["rlvr", "rlaif"]:
        prefs, counts = [], Counter()
        agree = dec_n = dec_agree = 0
        for i, r in enumerate(rows):
            pref = judge.compare(str(r["question"]), gens[pol]["responses"][i], gens["sft"]["responses"][i])
            prefs.append(pref)
            # verifier preference: the exactly-correct side wins, tie if both or neither are correct
            dv = gens[pol]["exact"][i] - gens["sft"]["exact"][i]
            v_pref = "A" if dv > 0 else ("B" if dv < 0 else "TIE")
            counts[f"verifier_{v_pref}|judge_{pref}"] += 1
            agree += v_pref == pref
            if v_pref != "TIE":
                dec_n += 1
                dec_agree += v_pref == pref
        n = max(1, len(rows))
        wins, losses, ties = prefs.count("A"), prefs.count("B"), prefs.count("TIE")
        pairwise[f"{pol}_vs_sft"] = {
            "win_rate": round((wins + 0.5 * ties) / n, 4),
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "tie_rate": round(ties / n, 4),
            "verifier_judge_agreement": round(agree / n, 4),
            "verifier_decisive_pairs": dec_n,
            "judge_agrees_on_verifier_decisive_pairs": round(dec_agree / dec_n, 4) if dec_n else None,
            "verifier_vs_judge_counts": dict(counts),
        }
        gens[pol]["judge_vs_sft"] = prefs
        print(f"[{pol.upper()} vs SFT] Win Rate: {pairwise[f'{pol}_vs_sft']['win_rate']:.4f} (W:{wins}, T:{ties}, L:{losses}) verifier-judge agreement={agree / n:.3f}")
    results["pairwise"] = pairwise

    for pol in policies:
        recs = []
        for i, r in enumerate(rows):
            recs.append({
                "source_index": r.get("source_index"),
                "gold_final": r["gold_final"],
                "response": gens[pol]["responses"][i],
                "pred_final": extract_designated_final(gens[pol]["responses"][i]),
                "exact": gens[pol]["exact"][i],
                "failure_type": gens[pol]["ftypes"][i],
                "response_tokens": gens[pol]["lengths"][i],
                "judge_vs_sft": gens[pol].get("judge_vs_sft", [None] * len(rows))[i],
            })
        write_jsonl(outdir / f"{dataset}_{pol}_responses.jsonl", recs)

    out_file = outdir / f"{dataset}_eval.json"
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
