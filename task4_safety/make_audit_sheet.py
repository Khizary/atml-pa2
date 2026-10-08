from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
    ids = fixed_audit_ids(read_jsonl(src), int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    pd.DataFrame({"xstest_id": ids, "manual_label": [""] * len(ids)}).to_csv(outdir / "manual_audit_ids.csv", index=False)
    print("Wrote fixed audit IDs:", outdir / "manual_audit_ids.csv")

    # join the ids to every policy's responses. AI labels are deliberately left out so labelling is blind.
    rows = []
    for pol in ["sft", "dpo", "ppo", "grpo"]:
        p = outdir / f"generated_{pol}.jsonl"
        if not p.exists():
            print(f"Warning: {p} missing, {pol} not in audit sheet")
            continue
        rows.extend(r for r in read_jsonl(p) if int(r["xstest_id"]) in set(ids))
    sheet = pd.DataFrame(rows)[["xstest_id", "policy", "benchmark_class", "type", "prompt", "response"]]
    # shuffle row order (fixed seed) so policies aren't labelled in blocks
    sheet = sheet.sample(frac=1.0, random_state=int(cfg["seed"])).reset_index(drop=True)
    sheet["manual_label"] = ""
    out_csv = outdir / "manual_audit_sheet.csv"
    if out_csv.exists() and pd.read_csv(out_csv)["manual_label"].notna().any():
        out_csv = outdir / "manual_audit_sheet_new.csv"
        print("Existing sheet already has labels, not overwriting it")
    sheet.to_csv(out_csv, index=False)
    print(f"Wrote {len(sheet)} audit rows ({len(ids)} prompts x policies) to {out_csv}")
    print("Labels: SAFE_ANSWER, JUSTIFIED_REFUSAL, UNSAFE_COMPLIANCE, OVER_REFUSAL, AMBIGUOUS")


if __name__ == "__main__":
    main()
