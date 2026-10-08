from __future__ import annotations

import argparse
import json
import numpy as np

from common.data import load_yaml, repo_path, write_jsonl
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import evaluate_rl_policy


def trajectory_stats(traj):
    """Simple stability numbers for a short fork trajectory."""
    def col(k):
        return np.array([t[k] for t in traj if t.get(k) == t.get(k)], dtype=float)  # drops nans
    pol, gn, kl, clip, rew = col("policy_loss"), col("policy_grad_norm"), col("kl"), col("clip_fraction"), col("reward")
    return {
        "policy_loss_std": round(float(pol.std()), 5) if len(pol) else None,
        "policy_grad_norm_mean": round(float(gn.mean()), 4) if len(gn) else None,
        "policy_grad_norm_max": round(float(gn.max()), 4) if len(gn) else None,
        "kl_final": round(float(kl[-1]), 5) if len(kl) else None,
        "kl_max_abs": round(float(np.abs(kl).max()), 5) if len(kl) else None,
        "clip_fraction_mean": round(float(clip.mean()), 4) if len(clip) else None,
        "train_reward_mean": round(float(rew.mean()), 4) if len(rew) else None,
    }


def fork_and_eval(config_path, cfg, run_name, max_examples=100, **kw):
    out_dir = f"outputs/task2_ppo/{run_name}"
    run_ppo(config_path=config_path, output=out_dir, updates=int(cfg.get("fork_updates", 8)), run_name=run_name, **kw)
    with open(repo_path(f"results/ppo_{run_name}_continuation.json")) as f:
        train = json.load(f)
    held_out, records = evaluate_rl_policy(cfg, out_dir, int(cfg.get("eval_max_response_length", 768)), max_examples)
    write_jsonl(f"results/samples/ppo_{run_name}_eval_samples.jsonl", records)
    print(f"[{run_name}] held-out: {held_out}")
    return {"held_out": held_out, "stability": trajectory_stats(train["trajectory"]), "train": train}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--max-examples", type=int, default=100)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    kl_values = cfg.get("kl_values", [0.0, 0.10, 0.20])
    print(f"Running PPO KL ablation for kl_values={kl_values} with {cfg.get('fork_updates', 8)} updates each...")
    summary = {}
    for b in kl_values:
        run_name = f"kl_{b}".replace(".", "_")
        print(f"\n=== Running PPO short fork for kl_beta={b} ===")
        summary[str(b)] = fork_and_eval(args.config, cfg, run_name, args.max_examples, kl_beta=b)

    out_file = repo_path("results/ppo_kl_ablation.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPPO KL pressure study complete. Results saved to {out_file}")


if __name__ == "__main__":
    main()
