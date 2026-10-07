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
    load_value_model,
    trainable_parameters,
    value_parameter_groups,
    reference_mode,
    token_values,
)
from common.generation import batch_generate, response_token_logprobs
from common.metrics import sampled_kl, sample_entropy, masked_mean
from task2_ppo.ppo import (
    compute_gae,
    shaped_rewards,
    ppo_policy_loss,
    value_mse_loss,
    normalize_advantages,
)


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "lora_head"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def score_rollouts(rm_model, rm_tokenizer, prompts, responses):
    device = next(rm_model.parameters()).device
    formatted = []
    for p, r in zip(prompts, responses):
        msgs = list(p) + [{"role": "assistant", "content": r}]
        formatted.append(rm_tokenizer.apply_chat_template(msgs, tokenize=False))
    enc = rm_tokenizer(formatted, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = rm_model(**enc)
        scores = out.logits.squeeze(-1)
        if scores.ndim == 0:
            scores = scores.unsqueeze(0)
    return scores


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    num_updates = int(updates if updates is not None else cfg.get("updates", 20))
    eps = float(clip_epsilon if clip_epsilon is not None else cfg.get("clip_epsilon", 0.2))
    beta_kl = float(kl_beta if kl_beta is not None else cfg.get("kl_beta", 0.1))

    out = repo_path(output or cfg.get("output", f"outputs/task2_ppo/{run_name}"))
    out.mkdir(parents=True, exist_ok=True)

    policy = bundle["policy"]
    value_model = bundle["value_model"]
    tokenizer = bundle["tokenizer"]
    rm_model = bundle["reward_model"]
    rm_tok = bundle["reward_tokenizer"]
    prompts = bundle["prompt_rows"]
    pol_opt = bundle["policy_optimizer"]
    val_opt = bundle["value_optimizer"]

    ppo_epochs = int(cfg.get("ppo_epochs", 2))
    gamma = float(cfg.get("gamma", 1.0))
    lam = float(cfg.get("gae_lambda", 0.95))
    prompts_per_update = int(cfg.get("prompts_per_update", 1))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    history = []
    print(f"Starting PPO continuation: run={run_name}, updates={num_updates}, eps={eps}, beta_kl={beta_kl}")

    for u in range(num_updates):
        # 1. Sample prompts
        start_idx = (u * prompts_per_update) % len(prompts)
        end_idx = start_idx + prompts_per_update
        if end_idx <= len(prompts):
            batch_slice = prompts[start_idx:end_idx]
        else:
            batch_slice = prompts[start_idx:] + prompts[: end_idx % len(prompts)]
        prompt_msgs = [prompt_messages(r) for r in batch_slice]

        # 2. Rollout
        gen = batch_generate(
            policy,
            tokenizer,
            prompt_msgs,
            max_prompt_length=int(cfg.get("max_prompt_length", 256)),
            max_new_tokens=int(cfg.get("max_response_length", 512)),
            temperature=float(cfg.get("generation", {}).get("temperature", 0.7)),
            top_p=float(cfg.get("generation", {}).get("top_p", 0.9)),
            do_sample=True,
        )

        seqs = gen["sequences"]
        attn = gen["attention_mask"]
        p_width = gen["prompt_width"]
        resp_ids = gen["response_ids"]
        rmask = gen["response_mask"]

        # 3. Reward scoring
        task_rew = score_rollouts(rm_model, rm_tok, prompt_msgs, gen["responses"]).to(seqs.device)
        # Apply missing EOS penalty if applicable
        if cfg.get("missing_eos_penalty"):
            for i, term in enumerate(gen["terminated_with_eos"]):
                if not term:
                    task_rew[i] -= float(cfg["missing_eos_penalty"])

        # 4. Old & Ref Logprobs
        with torch.no_grad():
            old_logp, _ = response_token_logprobs(policy, seqs, attn, p_width, resp_ids)
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(policy, seqs, attn, p_width, resp_ids)

        # 5. KL Shaping
        rewards = shaped_rewards(task_rew, old_logp, ref_logp, rmask, beta_kl)

        # 6. Values & GAE
        with torch.no_grad():
            full_vals = token_values(value_model, seqs, attn)
            vals = full_vals[:, p_width - 1 : -1][:, : resp_ids.shape[1]]
            advs, rets = compute_gae(rewards, vals, rmask, gamma, lam)
            norm_advs = normalize_advantages(advs, rmask)

        # 7. Update loop
        step_pol_loss = 0.0
        step_val_loss = 0.0
        step_clip_frac = 0.0

        for _ in range(ppo_epochs):
            # Policy update
            pol_opt.zero_grad()
            new_logp, _ = response_token_logprobs(policy, seqs, attn, p_width, resp_ids)
            p_loss, _, c_frac = ppo_policy_loss(new_logp, old_logp, norm_advs, rmask, eps=eps)
            p_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_norm=float(cfg.get("max_grad_norm", 1.0)))
            pol_opt.step()

            # Value update
            val_opt.zero_grad()
            new_full_vals = token_values(value_model, seqs, attn)
            new_vals = new_full_vals[:, p_width - 1 : -1][:, : resp_ids.shape[1]]
            v_loss = value_mse_loss(new_vals, rets, rmask)
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters(value_model), max_norm=float(cfg.get("max_grad_norm", 1.0)))
            val_opt.step()

            step_pol_loss += p_loss.item() / ppo_epochs
            step_val_loss += v_loss.item() / ppo_epochs
            step_clip_frac += c_frac.item() / ppo_epochs

        # Compute diagnostics
        with torch.no_grad():
            cur_kl = sampled_kl(new_logp, ref_logp, rmask).item()
            cur_ent = sample_entropy(new_logp, rmask).item()
            mean_len = float(rmask.sum(-1).mean().item())
            mean_rew = float(task_rew.mean().item())

        record = {
            "update": u + 1,
            "reward": round(mean_rew, 4),
            "kl": round(cur_kl, 4),
            "entropy": round(cur_ent, 4),
            "policy_loss": round(step_pol_loss, 4),
            "value_loss": round(step_val_loss, 4),
            "clip_fraction": round(step_clip_frac, 4),
            "response_length": round(mean_len, 2),
        }
        history.append(record)
        print(f"[Update {u+1:02d}/{num_updates:02d}] rew={record['reward']:.3f} kl={record['kl']:.3f} pol_loss={record['policy_loss']:.3f} val_loss={record['value_loss']:.3f} clip_frac={record['clip_fraction']:.3f} len={record['response_length']:.1f}")

    wall_clock = time.time() - start_time
    peak_vram = (torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0

    print(f"\nPPO Continuation complete in {wall_clock:.1f}s. Peak VRAM: {peak_vram:.2f} GB")

    policy.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    res_file = repo_path(f"results/ppo_{run_name}_continuation.json")
    res_file.parent.mkdir(parents=True, exist_ok=True)
    with open(res_file, "w") as f:
        json.dump({
            "run_name": run_name,
            "updates": num_updates,
            "clip_epsilon": eps,
            "kl_beta": beta_kl,
            "wall_clock_sec": round(wall_clock, 2),
            "peak_vram_gb": round(peak_vram, 3),
            "trajectory": history,
        }, f, indent=2)
    print(f"Saved results to {res_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
