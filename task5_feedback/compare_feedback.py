from __future__ import annotations

import argparse
import json
from common.data import load_yaml, repo_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    gsm_file = repo_path("results/feedback_gsm_eval.json")
    trans_file = repo_path("results/feedback_transfer_eval.json")
    diag_file = repo_path("results/feedback_diagnostics.json")

    summary = {
        "in_domain_gsm8k": {},
        "out_of_domain_transfer": {},
        "diagnostics": {},
        "transfer_drop": {},
    }

    if gsm_file.exists():
        with open(gsm_file) as f:
            summary["in_domain_gsm8k"] = json.load(f)

    if trans_file.exists():
        with open(trans_file) as f:
            summary["out_of_domain_transfer"] = json.load(f)

    if diag_file.exists():
        with open(diag_file) as f:
            summary["diagnostics"] = json.load(f)

    # Compute transfer drop if both are present
    if "policies" in summary["in_domain_gsm8k"] and "policies" in summary["out_of_domain_transfer"]:
        for pol in ["sft", "rlvr", "rlaif"]:
            gsm_acc = summary["in_domain_gsm8k"]["policies"].get(pol, {}).get("exact_accuracy", 0.0)
            trans_acc = summary["out_of_domain_transfer"]["policies"].get(pol, {}).get("exact_accuracy", 0.0)
            drop = gsm_acc - trans_acc
            summary["transfer_drop"][pol] = {
                "gsm_acc": gsm_acc,
                "transfer_acc": trans_acc,
                "accuracy_drop": round(drop, 4),
            }

    out_file = repo_path("results/feedback_synthesis.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nFeedback synthesis complete. Saved to {out_file}")
    if summary["transfer_drop"]:
        print("Transfer performance drop (GSM8K -> SVAMP):")
        for pol, d in summary["transfer_drop"].items():
            print(f"  {pol.upper()}: {d['gsm_acc']:.3f} -> {d['transfer_acc']:.3f} (drop: {d['accuracy_drop']:.3f})")


if __name__ == "__main__":
    main()
