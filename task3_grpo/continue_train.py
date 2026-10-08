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
    clear_gpu,
    disable_dropout,
)
from common.generation import batch_generate, response_token_logprobs
from common.metrics import sampled_kl, logits_entropy
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
    disable_dropout(policy)
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
        rewards = score_rollouts(rm_model, rm_tok, batch_prompts, gen["responses"], int(cfg.get("reward_max_length", 1280))).float().to(seqs.device)

        # Informativeness check
        rew_std = float(rewards.std(unbiased=False).item())
        is_uninformative = float(rew_std < 1e-5)

        # Truncation mask
        token_mask = mask_truncated_sequences(rmask, gen["truncated"]) if mask_trunc else rmask

        # Group advantages (all K belong to group 0)
        group_ids = torch.zeros(K, dtype=torch.long, device=seqs.device)
        seq_adv = group_relative_advantages(rewards, group_ids).to(seqs.device)

        # Old and ref logprobs
        with torch.no_grad():
            old_logp, _ = response_token_logprobs(policy, seqs, attn, p_width, resp_ids)
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(policy, seqs, attn, p_width, resp_ids)

        # Optimization step
        optimizer.zero_grad()
        new_logp, new_logits = response_token_logprobs(policy, seqs, attn, p_width, resp_ids)
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
        gnorm = float(torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_norm=float(cfg.get("max_grad_norm", 1.0))))
        optimizer.step()

        with torch.no_grad():
            ent = logits_entropy(new_logits.detach(), rmask).item()
            kl_sampled = sampled_kl(old_logp, ref_logp, rmask).item()
        del new_logits

        # per-completion pieces for the normalization study: each completion's total gradient
        # weight is |A_k| * T_k / norm, with norm = T_k (canonical) or T_max (dr grpo)
        lens = token_mask.sum(-1)
        norm = lens.clamp_min(1.0) if loss_type == "grpo" else torch.full_like(lens, float(max_compl_len))
        seq_weight = (seq_adv.abs() * lens / norm).tolist()

        record = {
            "update": u + 1,
            "prompt_id": p_row.get("prompt_id", p_row.get("source_index")),
            "reward_mean": round(float(rewards.mean().item()), 4),
            "reward_std": round(rew_std, 4),
            "uninformative": is_uninformative,
            "policy_loss": round(diags["policy_term"].item(), 5),
            "loss_total": round(loss.item(), 5),
            "grad_norm": round(gnorm, 5),
            "kl": round(kl_sampled, 5),
            "kl_k3_loss_term": round(diags["sampled_kl"].item(), 6),
            "clip_fraction": round(diags["clip_fraction"].item(), 4),
            "entropy": round(ent, 4),
            "response_length": round(float(rmask.sum(-1).mean().item()), 2),
            "truncated_fraction": round(float(np.mean(gen["truncated"])), 4),
            "completion_lengths": [int(x) for x in rmask.sum(-1).tolist()],
            "completion_rewards": [round(float(x), 4) for x in rewards.tolist()],
            "completion_advantages": [round(float(x), 4) for x in seq_adv.tolist()],
            "completion_masked": [bool(x) for x in gen["truncated"]] if mask_trunc else [False] * K,
            "completion_grad_weight": [round(float(x), 5) for x in seq_weight],
        }
        history.append(record)
        print(f"[Update {u+1:02d}/{num_updates:02d}] rew={record['reward_mean']:.3f} (std={record['reward_std']:.3f}) kl={record['kl']:.5f} gnorm={record['grad_norm']:.4f} ent={record['entropy']:.3f} clip={record['clip_fraction']:.3f} len={record['response_length']:.1f}")

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
            "num_generations": K,
            "clip_epsilon": eps,
            "kl_beta": beta,
            "max_completion_length": max_compl_len,
            "mask_truncated_completions": mask_trunc,
            "wall_clock_sec": round(wall_clock, 2),
            "peak_vram_gb": round(peak_vram, 3),
            "mean_within_group_reward_std": round(float(np.mean([h["reward_std"] for h in history])), 4),
            "uninformative_group_fraction": round(float(np.mean([h["uninformative"] for h in history])), 4),
            "trajectory": history,
        }, f, indent=2)
    print(f"Saved results to {res_file}")

    del bundle, policy, rm_model, optimizer
    clear_gpu()
    return str(out)


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
