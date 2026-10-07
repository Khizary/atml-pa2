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
from common.metrics import word_limit_compliance, word_count
from common.generation import batch_generate
from task1_dpo.train import run_training, get_batch_logps, make_collate


def eval_stratified_subset(model, tokenizer, rows, beta=0.1):
    loader = DataLoader(
        rows,
        batch_size=4,
        shuffle=False,
        collate_fn=make_collate(tokenizer, max_length=1024),
    )
    accs, policy_margins = [], []
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
            policy_margins.extend((p_chosen - p_rejected).tolist())
    return {
        "count": len(rows),
        "preference_accuracy": round(float(np.mean(accs)), 4) if accs else 0.0,
        "mean_policy_margin": round(float(np.mean(policy_margins)), 4) if policy_margins else 0.0,
    }


def evaluate_word_limits(model, tokenizer, prompt_rows):
    prompts = [prompt_messages(r) for r in prompt_rows]
    gen = batch_generate(
        model,
        tokenizer,
        prompts,
        max_prompt_length=256,
        max_new_tokens=128,
        temperature=0.0,
        top_p=1.0,
        do_sample=False,
    )
    records = []
    compliances = []
    lengths = []
    for r, response in zip(prompt_rows, gen["responses"]):
        p_text = r.get("prompt", r.get("question", ""))
        comp = word_limit_compliance(p_text, response)
        wc = word_count(response)
        compliances.append(comp if comp is not None else 1.0)
        lengths.append(wc)
        records.append({
            "prompt": p_text,
            "response": response,
            "word_count": wc,
            "compliant": bool(comp),
        })
    return {
        "word_limit_compliance_rate": round(float(np.mean(compliances)), 4),
        "mean_word_count": round(float(np.mean(lengths)), 2),
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
    strata = {
        "preferred_longer": [r for r in strat_rows if r.get("stratum") == "preferred_longer" or (len(str(r.get("chosen",""))) > len(str(r.get("rejected",""))) * 1.2)],
        "matched_length": [r for r in strat_rows if r.get("stratum") == "matched_length" or (0.8 <= len(str(r.get("chosen",""))) / max(1, len(str(r.get("rejected","")))) <= 1.2)],
        "rejected_longer": [r for r in strat_rows if r.get("stratum") == "rejected_longer" or (len(str(r.get("rejected",""))) > len(str(r.get("chosen",""))) * 1.2)],
    }

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

    print("\n=== Evaluating Word-Limit Compliance ===")
    word_rows = read_jsonl(word_limit_path)
    results["word_limit_compliance"]["standard_model"] = evaluate_word_limits(model_std, tokenizer, word_rows)
    results["word_limit_compliance"]["length_balanced_model"] = evaluate_word_limits(model_bal, tokenizer, word_rows)

    out_file = repo_path("results/dpo_length_analysis.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nLength confounding analysis complete. Saved to {out_file}")


if __name__ == "__main__":
    main()
