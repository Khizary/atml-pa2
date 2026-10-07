from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import set_seed
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    trainable_parameters,
    reference_mode,
)
from common.generation import batch_generate, response_token_logprobs
from task2_ppo.continue_train import score_rollouts
from task3_grpo.grpo import (
    group_relative_advantages,
    grpo_policy_loss,
    mask_truncated_sequences,
)


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo", run_name: str = "standard"):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    num_updates = int(updates if updates is not None else cfg.get("updates", 20))
    out = repo_path(output or cfg.get("output", f"outputs/task3_grpo/{run_name}"))
    out.mkdir(parents=True, exist_ok=True)

    policy = bundle["policy"]
    tokenizer = bundle["tokenizer"]
    rm_model = bundle["reward_model"]
    rm_tok = bundle["reward_tokenizer"]
    prompts = bundle["prompt_rows"]
    optimizer = bundle["optimizer"]

    eps = float(cfg.get("clip_epsilon", 0.20))
    beta = float(cfg.get("kl_beta", 0.10))
    K = int(cfg.get("num_generations", 4))
    max_compl_len = int(cfg.get("max_completion_length", 512))
    mask_trunc = bool(cfg.get("mask_truncated_completions", True))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    history = []
    print(f"Starting GRPO continuation: run={run_name}, loss_type={loss_type}, updates={num_updates}, K={K}")

    for u in range(num_updates):
        p_row = prompts[u % len(prompts)]
        p_msgs = prompt_messages(p_row)
        batch_prompts = [p_msgs] * K

        # Generate K completions
        gen = batch_generate(
            policy,
            tokenizer,
            batch_prompts,
            max_prompt_length=int(cfg.get("max_prompt_length", 256)),
            max_new_tokens=max_compl_len,
            temperature=float(cfg.get("generation", {}).get("temperature", 0.7)),
            top_p=float(cfg.get("generation", {}).get("top_p", 0.9)),
            do_sample=True,
        )

        seqs = gen["sequences"]
        attn = gen["attention_mask"]
        p_width = gen["prompt_width"]
        resp_ids = gen["response_ids"]
        rmask = gen["response_mask"]

        # Reward scoring
        rewards = score_rollouts(rm_model, rm_tok, batch_prompts, gen["responses"])

        # Informativeness check
        rew_std = float(rewards.std(unbiased=False).item())
        is_uninformative = float(rew_std < 1e-5)

        # Truncation mask
        token_mask = mask_truncated_sequences(rmask, gen["truncated"]) if mask_trunc else rmask

        # Group advantages (all K belong to group 0)
        group_ids = torch.zeros(K, dtype=torch.long, device=rewards.device)
        seq_adv = group_relative_advantages(rewards, group_ids)

        # Old and ref logprobs
        with torch.no_grad():
            old_logp, _ = response_token_logprobs(policy, seqs, attn, p_width, resp_ids)
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(policy, seqs, attn, p_width, resp_ids)

        # Optimization step
        optimizer.zero_grad()
        new_logp, _ = response_token_logprobs(policy, seqs, attn, p_width, resp_ids)
        loss, diags = grpo_policy_loss(
            new_logp,
            old_logp,
            seq_adv,
            token_mask,
            ref_logp,
            eps=eps,
            beta=beta,
            loss_type=loss_type,
            max_completion_length=max_compl_len,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_norm=float(cfg.get("max_grad_norm", 1.0)))
        optimizer.step()

        record = {
            "update": u + 1,
            "reward_mean": round(float(rewards.mean().item()), 4),
            "reward_std": round(rew_std, 4),
            "uninformative": is_uninformative,
            "policy_loss": round(diags["policy_term"].item(), 4),
            "kl": round(diags["sampled_kl"].item(), 4),
            "clip_fraction": round(diags["clip_fraction"].item(), 4),
            "entropy": round(diags["sample_entropy"].item(), 4),
            "response_length": round(float(rmask.sum(-1).mean().item()), 2),
        }
        history.append(record)
        print(f"[Update {u+1:02d}/{num_updates:02d}] rew={record['reward_mean']:.3f} (std={record['reward_std']:.3f}) kl={record['kl']:.3f} clip={record['clip_fraction']:.3f} len={record['response_length']:.1f}")

    wall_clock = time.time() - start_time
    peak_vram = (torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0

    print(f"\nGRPO Continuation complete in {wall_clock:.1f}s. Peak VRAM: {peak_vram:.2f} GB")

    policy.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    res_file = repo_path(f"results/grpo_{run_name}_continuation.json")
    res_file.parent.mkdir(parents=True, exist_ok=True)
    with open(res_file, "w") as f:
        json.dump({
            "run_name": run_name,
            "loss_type": loss_type,
            "updates": num_updates,
            "wall_clock_sec": round(wall_clock, 2),
            "peak_vram_gb": round(peak_vram, 3),
            "trajectory": history,
        }, f, indent=2)
    print(f"Saved results to {res_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()
