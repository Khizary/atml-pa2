from __future__ import annotations

import argparse
import json
from common.data import load_yaml, repo_path
from task2_ppo.continue_train import run_ppo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    kl_values = cfg.get("kl_values", [0.0, 0.10, 0.20])
    fork_updates = int(cfg.get("fork_updates", 8))

    print(f"Running PPO KL ablation for kl_values={kl_values} with {fork_updates} updates each...")
    summary = {}

    for b in kl_values:
        run_name = f"kl_{b}".replace(".", "_")
        out_dir = f"outputs/task2_ppo/{run_name}"
        print(f"\n=== Running PPO short fork for kl_beta={b} ===")
        run_ppo(
            config_path=args.config,
            output=out_dir,
            updates=fork_updates,
            kl_beta=b,
            run_name=run_name,
        )

        res_path = repo_path(f"results/ppo_{run_name}_continuation.json")
        if res_path.exists():
            with open(res_path) as f:
                summary[str(b)] = json.load(f)

    out_file = repo_path("results/ppo_kl_ablation.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPPO KL pressure study complete. Results saved to {out_file}")


if __name__ == "__main__":
    main()
