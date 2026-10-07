from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.logging_utils import set_seed
from common.models import load_policy, load_tokenizer, trainable_parameters, reference_mode
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def get_batch_logps(model, batch):
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    response_mask = batch["response_mask"][:, 1:].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[:, :-1, :]
    logp = torch.log_softmax(logits.float(), dim=-1)
    target_ids = input_ids[:, 1:].unsqueeze(-1)
    token_logps = torch.gather(logp, -1, target_ids).squeeze(-1)
    return (token_logps * response_mask).sum(-1)


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    model = bundle["model"]
    loader = bundle["loader"]
    optimizer = bundle["optimizer"]
    dpo_beta = bundle["beta"]

    output = repo_path(output_path or cfg.get("output", f"outputs/task1_dpo/{run_name}"))
    output.mkdir(parents=True, exist_ok=True)

    grad_accum = int(cfg.get("gradient_accumulation_steps", 1))
    epochs = int(cfg.get("epochs", 1))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    metrics_log = []
    step = 0
    total_loss = 0.0

    print(f"Starting DPO training: run={run_name}, beta={dpo_beta}, examples={len(bundle['rows'])}, epochs={epochs}")
    optimizer.zero_grad()

    for epoch in range(epochs):
        for i, (chosen_batch, rejected_batch) in enumerate(loader):
            step += 1
            # Forward policy
            model.train()
            policy_chosen_logp = get_batch_logps(model, chosen_batch)
            policy_rejected_logp = get_batch_logps(model, rejected_batch)

            # Forward reference policy (using disable_adapter)
            with torch.no_grad():
                with reference_mode(model):
                    ref_chosen_logp = get_batch_logps(model, chosen_batch)
                    ref_rejected_logp = get_batch_logps(model, rejected_batch)

            # Compute loss
            loss, diags = dpo_loss(
                policy_chosen_logp,
                policy_rejected_logp,
                ref_chosen_logp,
                ref_rejected_logp,
                dpo_beta,
            )

            loss_scaled = loss / grad_accum
            loss_scaled.backward()
            total_loss += loss.item()

            if step % grad_accum == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % int(cfg.get("log_every", 10)) == 0 or (i + 1) == len(loader):
                log_entry = {
                    "step": step,
                    "epoch": epoch,
                    "loss": round(loss.item(), 4),
                    "preference_accuracy": round(diags["preference_accuracy"].item(), 4),
                    "policy_margin": round(diags["policy_margin_mean"].item(), 4),
                    "logit_mean": round(diags["logit_mean"].item(), 4),
                }
                metrics_log.append(log_entry)
                print(f"[Step {step:04d}] loss={log_entry['loss']:.4f} acc={log_entry['preference_accuracy']:.4f} margin={log_entry['policy_margin']:.4f}")

    wall_clock = time.time() - start_time
    peak_vram_gb = (torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0

    print(f"Training finished in {wall_clock:.1f}s. Peak VRAM: {peak_vram_gb:.2f} GB")

    # Save model adapter
    model.save_pretrained(str(output))
    bundle["tokenizer"].save_pretrained(str(output))

    # Save metrics
    res_path = repo_path("results")
    res_path.mkdir(parents=True, exist_ok=True)
    with open(res_path / f"dpo_{run_name}_train.json", "w") as f:
        json.dump({
            "run_name": run_name,
            "beta": dpo_beta,
            "steps": step,
            "wall_clock_sec": round(wall_clock, 2),
            "peak_vram_gb": round(peak_vram_gb, 3),
            "logs": metrics_log,
        }, f, indent=2)

    print(f"Saved checkpoint to {output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
