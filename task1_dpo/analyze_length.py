from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from common.data import (
    load_yaml,
    read_jsonl,
    repo_path,
    prompt_messages,
    preference_responses,
    encode_prompt_response,
    pad_batch,
)
from common.models import load_policy, load_tokenizer, reference_mode
from common.metrics import word_limit_compliance, word_count, parse_word_limit
from common.generation import batch_generate
from task1_dpo.train import run_training, get_batch_logps, make_collate


def eval_stratified_subset(model, tokenizer, rows, beta=0.1):
    loader = DataLoader(
        rows,
        batch_size=4,
        shuffle=False,
        collate_fn=make_collate(tokenizer, max_length=1024),
    )
    accs, policy_margins, dpo_margins = [], [], []
    with torch.no_grad():
        for chosen_batch, rejected_batch in loader:
            p_chosen = get_batch_logps(model, chosen_batch)
            p_rejected = get_batch_logps(model, rejected_batch)

            with reference_mode(model):
                r_chosen = get_batch_logps(model, chosen_batch)
                r_rejected = get_batch_logps(model, rejected_batch)

            margin = (p_chosen - p_rejected) - (r_chosen - r_rejected)
            acc = (margin > 0).float()
            accs.extend(acc.tolist())
            dpo_margins.extend(margin.float().tolist())
            policy_margins.extend((p_chosen - p_rejected).tolist())
    return {
        "count": len(rows),
        "preference_accuracy": round(float(np.mean(accs)), 4) if accs else 0.0,
        "mean_policy_margin": round(float(np.mean(policy_margins)), 4) if policy_margins else 0.0,
        "mean_dpo_margin": round(float(np.mean(dpo_margins)), 4) if dpo_margins else 0.0,
    }


def evaluate_word_limits(model, tokenizer, prompt_rows, max_new_tokens=256):
    """Greedy responses on the common word-limit prompts; compliance + generated length."""
    prompts = [prompt_messages(r) for r in prompt_rows]
    gen = batch_generate(
        model,
        tokenizer,
        prompts,
        max_prompt_length=256,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        do_sample=False,
    )
    records, compliances, words, toks = [], [], [], []
    for r, msgs, response, n_tok, trunc in zip(prompt_rows, prompts, gen["responses"], gen["response_lengths"], gen["truncated"]):
        p_text = msgs[-1]["content"]
        limit = parse_word_limit(p_text)
        comp = word_limit_compliance(p_text, response)
        if comp is None:
            raise ValueError(f"No word limit parsed from prompt {r.get('prompt_id')}: {p_text}")
        wc = word_count(response)
        compliances.append(comp)
        words.append(wc)
        toks.append(n_tok)
        records.append({
            "prompt_id": r.get("prompt_id"),
            "prompt": p_text,
            "word_limit": limit,
            "response": response,
            "word_count": wc,
            "response_tokens": n_tok,
            "truncated": trunc,
            "compliant": bool(comp),
        })
    return {
        "num_prompts": len(records),
        "word_limit_compliance_rate": round(float(np.mean(compliances)), 4),
        "mean_word_count": round(float(np.mean(words)), 2),
        "std_word_count": round(float(np.std(words)), 2),
        "mean_response_tokens": round(float(np.mean(toks)), 2),
        "std_response_tokens": round(float(np.std(toks)), 2),
        "mean_words_over_limit": round(float(np.mean([max(0, rec["word_count"] - rec["word_limit"]) for rec in records])), 2),
        "examples": records,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    balanced_train_path = cfg["paths"]["dpo_length_train"]
    stratified_eval_path = cfg["paths"]["dpo_length_eval"]
    word_limit_path = cfg["paths"]["word_limit_prompts"]

    balanced_out = "outputs/task1_dpo/length_balanced"
    standard_out = "outputs/task1_dpo/standard"

    if not args.skip_train:
        print("\n=== Training Length-Balanced DPO Model ===")
        run_training(
            config_path=args.config,
            run_name="length_balanced",
            dataset_path=balanced_train_path,
            output_path=balanced_out,
            beta=float(cfg.get("beta", 0.1)),
        )

    # Load models
    tokenizer = load_tokenizer(cfg["base_model"])
    model_std = load_policy(cfg, adapter_path=standard_out, trainable=False)
    model_bal = load_policy(cfg, adapter_path=balanced_out, trainable=False)

    strat_rows = read_jsonl(stratified_eval_path)
    # Filter into 3 strata based on row metadata or length
    # strata are supplied by the course file, don't recompute them
    strata = {name: [r for r in strat_rows if r["length_stratum"] == name] for name in ["preferred_longer", "length_matched", "rejected_longer"]}
    missing = len(strat_rows) - sum(len(v) for v in strata.values())
    if missing:
        raise ValueError(f"{missing} stratified rows have an unknown length_stratum")

    results = {
        "stratified_evaluation": {
            "standard_model": {},
            "length_balanced_model": {},
        },
        "word_limit_compliance": {},
    }

    print("\n=== Evaluating Stratified Subsets ===")
    for s_name, rows in strata.items():
        if rows:
            print(f"Evaluating {s_name} ({len(rows)} rows)...")
            results["stratified_evaluation"]["standard_model"][s_name] = eval_stratified_subset(model_std, tokenizer, rows)
            results["stratified_evaluation"]["length_balanced_model"][s_name] = eval_stratified_subset(model_bal, tokenizer, rows)
            results["stratified_evaluation"].setdefault("dataset_length_stats", {})[s_name] = {
                "mean_chosen_tokens": round(float(np.mean([float(r["chosen_tokens"]) for r in rows])), 2),
                "mean_rejected_tokens": round(float(np.mean([float(r["rejected_tokens"]) for r in rows])), 2),
            }

    print("\n=== Evaluating Word-Limit Compliance ===")
    word_rows = read_jsonl(word_limit_path)
    max_new = int(cfg.get("max_generation_tokens", 256))
    results["word_limit_compliance"]["standard_model"] = evaluate_word_limits(model_std, tokenizer, word_rows, max_new)
    results["word_limit_compliance"]["length_balanced_model"] = evaluate_word_limits(model_bal, tokenizer, word_rows, max_new)
    for k, v in results["word_limit_compliance"].items():
        print(f"[{k}] compliance={v['word_limit_compliance_rate']:.2f} words={v['mean_word_count']:.1f} tokens={v['mean_response_tokens']:.1f}")

    out_file = repo_path("results/dpo_length_analysis.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nLength confounding analysis complete. Saved to {out_file}")


if __name__ == "__main__":
    main()
