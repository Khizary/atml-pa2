from __future__ import annotations

import argparse
import json
from common.data import load_yaml, repo_path


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"

    gsm = load(outdir / "gsm_eval.json")
    trans = load(outdir / "transfer_eval.json")
    summary = {
        "in_domain_gsm8k": gsm,
        "out_of_domain_transfer": trans,
        "diagnostics": load(outdir / "diagnostics.json"),
        "transfer_drop": {},
    }

    if gsm and trans:
        for pol in ["sft", "rlvr", "rlaif"]:
            g, t = gsm["policies"][pol], trans["policies"][pol]
            row = {
                "gsm_acc": g["exact_accuracy"],
                "transfer_acc": t["exact_accuracy"],
                "accuracy_drop": round(g["exact_accuracy"] - t["exact_accuracy"], 4),
                "gsm_format": g["format_compliance_rate"],
                "transfer_format": t["format_compliance_rate"],
                "gsm_tokens": g["response_tokens_mean"],
                "transfer_tokens": t["response_tokens_mean"],
                "transfer_failure_types": t["failure_types"],
            }
            key = f"{pol}_vs_sft"
            if key in gsm.get("pairwise", {}) and key in trans.get("pairwise", {}):
                row["gsm_win_rate_vs_sft"] = gsm["pairwise"][key]["win_rate"]
                row["transfer_win_rate_vs_sft"] = trans["pairwise"][key]["win_rate"]
                row["win_rate_drop"] = round(row["gsm_win_rate_vs_sft"] - row["transfer_win_rate_vs_sft"], 4)
            summary["transfer_drop"][pol] = row

    out_file = outdir / "feedback_synthesis.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nFeedback synthesis complete. Saved to {out_file}")
    for pol, d in summary["transfer_drop"].items():
        print(f"  {pol.upper()}: acc {d['gsm_acc']:.3f} -> {d['transfer_acc']:.3f} (drop {d['accuracy_drop']:.3f})")


if __name__ == "__main__":
    main()
