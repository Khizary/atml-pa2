from __future__ import annotations

import argparse
import json
import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import set_seed
from common.models import load_policy, load_tokenizer, trainable_parameters, clear_gpu, disable_dropout
from common.generation import response_token_logprobs
from task2_ppo.ppo import compute_gae, shaped_rewards, normalize_advantages, ppo_policy_loss
from task2_ppo.ablate_kl import fork_and_eval


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"prompt_id", "response", "old_logprobs", "ref_logprobs", "values", "effective_terminal_reward"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def build_cached_batch(cfg, tokenizer, rows, beta_kl):
    """Rebuild prompt+response token ids and the fixed GAE advantages for every cached rollout."""
    prompt_rows = {}
    for key in ["rl_prompt_train", "rl_prompt_eval"]:
        for r in read_jsonl(cfg["paths"][key]):
            prompt_rows[str(r["prompt_id"])] = r

    items, skipped = [], 0
    for r in rows:
        old = torch.as_tensor(r["old_logprobs"]).float()
        if str(r["prompt_id"]) not in prompt_rows:
            skipped += 1
            continue
        msgs = prompt_messages(prompt_rows[str(r["prompt_id"])])
        p_ids = list(tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
        r_ids = tokenizer(r["response"], add_special_tokens=False)["input_ids"]
        # cached logprobs may or may not include the eos position
        if len(r_ids) + 1 == len(old):
            r_ids = r_ids + [tokenizer.eos_token_id]
        if len(r_ids) != len(old):
            # retokenized text doesn't line up with the cached logprobs, can't use it
            skipped += 1
            continue
        mask = torch.ones(1, len(r_ids))
        rew = shaped_rewards(
            torch.tensor([float(r["effective_terminal_reward"])]),
            old[None], torch.as_tensor(r["ref_logprobs"]).float()[None], mask, beta_kl,
        )
        adv, _ = compute_gae(rew, torch.as_tensor(r["values"]).float()[None], mask,
                             float(cfg.get("gamma", 1.0)), float(cfg.get("gae_lambda", 0.95)))
        items.append({"p_ids": p_ids, "r_ids": r_ids, "old": old, "adv": adv[0]})

    if not items:
        return items, skipped
    # whiten advantages over the whole cached batch (same as the continuation loop does per batch)
    all_adv = torch.cat([it["adv"] for it in items])
    mask_all = torch.ones_like(all_adv)
    normed = normalize_advantages(all_adv[None], mask_all[None])[0]
    pos = 0
    for it in items:
        n = len(it["adv"])
        it["adv"] = normed[pos : pos + n]
        pos += n
    return items, skipped


def cached_pass(policy, items, eps, device, optimizer=None):
    """One pass over the cached batch. Returns token-level ratio diagnostics; steps the optimizer if given."""
    n_tok = sum(len(it["r_ids"]) for it in items)
    ratios, advs = [], []
    surr_clip, surr_unclip = 0.0, 0.0
    if optimizer is not None:
        optimizer.zero_grad()
    for it in items:
        seq = torch.tensor([it["p_ids"] + it["r_ids"]], device=device)
        attn = torch.ones_like(seq)
        resp = torch.tensor([it["r_ids"]], device=device)
        old = it["old"].to(device)[None]
        adv = it["adv"].to(device)[None]
        mask = torch.ones_like(old)
        with torch.set_grad_enabled(optimizer is not None):
            new_logp, _ = response_token_logprobs(policy, seq, attn, len(it["p_ids"]), resp)
            loss, ratio, _ = ppo_policy_loss(new_logp, old, adv, mask, eps=eps)
            if optimizer is not None:
                # token-weighted mean over the whole batch
                (loss * len(it["r_ids"]) / n_tok).backward()
        ratio = ratio.float()
        ratios.append(ratio[0].cpu())
        advs.append(adv[0].cpu())
        surr_unclip += float((ratio * adv).sum())
        surr_clip += float(torch.minimum(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv).sum())
    gnorm = None
    if optimizer is not None:
        gnorm = float(torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), 1.0))
        optimizer.step()

    rho, a = torch.cat(ratios), torch.cat(advs)
    outside = ((rho < 1 - eps) | (rho > 1 + eps)).float()
    # tokens where the min() actually picks the clipped branch, i.e. their gradient is cut off
    affected = (((a > 0) & (rho > 1 + eps)) | ((a < 0) & (rho < 1 - eps))).float()
    return {
        "clip_fraction": round(float(outside.mean()), 5),
        "affected_token_fraction": round(float(affected.mean()), 5),
        "clipped_surrogate": round(surr_clip / n_tok, 6),
        "unclipped_surrogate": round(surr_unclip / n_tok, 6),
        "ratio_mean": round(float(rho.mean()), 5),
        "ratio_std": round(float(rho.std()), 5),
        "ratio_max_abs_log": round(float(rho.log().abs().max()), 5),
        "grad_norm": gnorm,
    }


def analyze_cached_batch(cfg, rows, epsilons):
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
    disable_dropout(policy)
    device = next(policy.parameters()).device
    items, skipped = build_cached_batch(cfg, tokenizer, rows, float(cfg.get("kl_beta", 0.1)))
    print(f"Cached batch: {len(items)} usable rollouts ({skipped} skipped), {sum(len(i['r_ids']) for i in items)} response tokens")
    if not items:
        raise RuntimeError("no cached rollout could be re-tokenized to match its cached logprobs")

    init_state = {n: p.detach().clone() for n, p in policy.named_parameters() if p.requires_grad}
    ppo_epochs = int(cfg.get("ppo_epochs", 2))
    results = {"num_rollouts": len(items), "skipped_rollouts": skipped, "ppo_epochs": ppo_epochs}
    for eps in epsilons:
        # every epsilon starts from the identical midpoint weights + fresh optimizer
        with torch.no_grad():
            for n, p in policy.named_parameters():
                if n in init_state:
                    p.copy_(init_state[n])
        set_seed(int(cfg["seed"]))
        opt = AdamW(trainable_parameters(policy), lr=float(cfg["policy_learning_rate"]))
        passes = []
        for ep in range(ppo_epochs):
            policy.train()
            passes.append({"epoch": ep, **cached_pass(policy, items, eps, device, opt)})
        policy.eval()
        after = cached_pass(policy, items, eps, device, None)
        results[str(eps)] = {
            # epoch 0 measures the cached old policy vs current midpoint before any step
            "per_epoch": passes,
            "after_update": after,
            "cached_clip_fraction": passes[-1]["clip_fraction"],
            "cached_affected_token_fraction": passes[-1]["affected_token_fraction"],
        }
        print(f"eps={eps}: {results[str(eps)]['per_epoch']} | after: {after}")

    del policy
    clear_gpu()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--max-examples", type=int, default=100)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    rows = load_cached_rollouts(cfg["cached_rollouts"])
    epsilons = cfg.get("clip_values", [0.05, 0.20, 0.50])
    print(f"Loaded {len(rows)} cached PPO rollouts. Analyzing epsilons={epsilons}...")
    try:
        cached_metrics = analyze_cached_batch(cfg, rows, epsilons)
    except Exception as exc:
        # don't lose the forks because of the cached study
        import traceback
        traceback.print_exc()
        cached_metrics = {"error": repr(exc)}
        clear_gpu()

    fork_results = {}
    if not args.skip_train:
        for eps in epsilons:
            run_name = f"clip_{eps}".replace(".", "_")
            print(f"\n=== Running PPO short fork for epsilon={eps} ===")
            fork_results[str(eps)] = fork_and_eval(args.config, cfg, run_name, args.max_examples, clip_epsilon=eps)

    final_results = {
        "cached_rollout_analysis": cached_metrics,
        "short_fork_results": fork_results,
    }

    out_file = repo_path("results/ppo_clipping_study.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nPPO clipping study complete. Results saved to {out_file}")


if __name__ == "__main__":
    main()
