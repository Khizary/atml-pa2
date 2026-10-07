from __future__ import annotations

import argparse
import json
from common.data import load_yaml, repo_path
from task3_grpo.continue_train import run_grpo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    fork_updates = int(cfg.get("fork_updates", 8))

    print(f"Running normalization comparison (canonical GRPO vs Dr. GRPO) for {fork_updates} updates...")

    print("\n=== Running Canonical GRPO (1/T_k) ===")
    run_grpo(
        config_path=args.config,
        output="outputs/task3_grpo/canonical",
        updates=fork_updates,
        loss_type="grpo",
        run_name="canonical",
    )

    print("\n=== Running Dr. GRPO (1/T_max) ===")
    run_grpo(
        config_path=args.config,
        output="outputs/task3_grpo/dr_grpo",
        updates=fork_updates,
        loss_type="dr_grpo",
        run_name="dr_grpo",
    )

    # Collate results
    results = {}
    for name in ["canonical", "dr_grpo"]:
        res_file = repo_path(f"results/grpo_{name}_continuation.json")
        if res_file.exists():
            with open(res_file) as f:
                results[name] = json.load(f)

    out_file = repo_path("results/grpo_normalization_study.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nNormalization comparison complete. Results saved to {out_file}")


if __name__ == "__main__":
    main()
