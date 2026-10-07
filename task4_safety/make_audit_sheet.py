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

    src = repo_path("outputs/safety_responses_sft.jsonl")
    if not src.exists():
        src = repo_path(cfg["results_dir"]) / "task4_safety" / "generated_sft.jsonl"
    if not src.exists():
        src = repo_path(cfg["paths"]["xstest"])
        df = pd.read_csv(src)
        base_rows = df.to_dict(orient="records")
    else:
        base_rows = read_jsonl(src)

    outdir = repo_path("results")
    outdir.mkdir(parents=True, exist_ok=True)
    ids = fixed_audit_ids(base_rows, int(cfg.get("manual_audit_per_class", 30)), int(cfg.get("seed", 6304)))

    meta_df = pd.DataFrame(base_rows)
    sub = meta_df[meta_df["xstest_id"].isin(ids)].copy()
    sub["manual_label"] = ""
    cols = ["xstest_id"]
    if "policy" in sub.columns:
        cols.append("policy")
    cols.append("prompt")
    if "response" in sub.columns:
        cols.append("response")
    cols.extend(["benchmark_class", "type", "manual_label"])
    out_cols = [c for c in cols if c in sub.columns]
    sub[out_cols].to_csv(out_csv, index=False)
    print(f"Wrote 60 balanced audit prompts to {out_csv}")


if __name__ == "__main__":
    main()
