from __future__ import annotations

import argparse
import json
from common.data import load_yaml, repo_path
from task1_dpo.train import run_training
from task1_dpo.evaluate import evaluate_dpo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    betas = cfg.get("betas", [0.03, 0.10, 0.30])
    max_examples = int(cfg.get("short_ablation_examples", 300))

    print(f"Running DPO beta ablation for betas={betas} with {max_examples} examples each...")
    summary = {}

    for b in betas:
        run_name = f"beta_{b}".replace(".", "_")
        out_dir = f"outputs/task1_dpo/{run_name}"
        print(f"\n=================== Training beta = {b} ===================")
        run_training(
            config_path=args.config,
            run_name=run_name,
            output_path=out_dir,
            beta=b,
            max_examples=max_examples,
        )

        print(f"\n=================== Evaluating beta = {b} ===================")
        res = evaluate_dpo(
            config_path=args.config,
            adapter=out_dir,
            name=run_name,
            max_eval_examples=min(max_examples, 100),
        )
        summary[str(b)] = res

    out_file = repo_path("results/dpo_beta_ablation.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll beta ablations complete. Summary saved to {out_file}")


if __name__ == "__main__":
    main()
