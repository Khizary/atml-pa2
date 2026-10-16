"""Extra checks used in the report, computed from the saved results (no models needed).

Writes results/report_diagnostics.json. Covers:
- held-out reward standard errors and length/reward correlation (Tasks 1-3)
- how often DPO samples hit the generation cap and what the RM gives them (Task 1)
- a naming-independent version of the manual-audit agreement (Task 4)
- identical responses across math policies, \\boxed usage and a last-number parse (Task 5)
- LoRA size and ||dW||_F of the supplied adapters, if checkpoints/ has been downloaded
"""
from __future__ import annotations

import argparse
import json
import math
import re
import struct
from collections import Counter

import numpy as np
import pandas as pd

from common.data import load_yaml, repo_path
from task4_safety.evaluate_safety import cohen_kappa

R = repo_path("results")


def jl(path):
    return [json.loads(l) for l in open(path)]


def corr(x, y):
    return round(float(np.corrcoef(x, y)[0, 1]), 3)


def se(std, n):
    return round(std / math.sqrt(n), 3)


def heldout_uncertainty():
    out = {}
    evals = {
        "dpo_standard": "dpo_standard_eval.json", "dpo_beta_0_03": "dpo_beta_0_03_eval.json",
        "dpo_beta_0_1": "dpo_beta_0_1_eval.json", "dpo_beta_0_3": "dpo_beta_0_3_eval.json",
        "ppo_standard": "ppo_standard_eval.json", "grpo_standard": "grpo_standard_eval.json",
    }
    for k, f in evals.items():
        d = json.loads((R / f).read_text())
        if "mean_reward" in d:
            out[k] = {"mean": d["mean_reward"], "se": se(d["std_reward"], d["eval_examples"]), "n": d["eval_examples"]}
        else:
            out[k] = {"mean": d["reward_model_score_mean"], "se": se(d["reward_model_score_std"], d["num_generation_prompts"]),
                      "n": d["num_generation_prompts"]}
    clip = json.loads((R / "ppo_clipping_study.json").read_text())["short_fork_results"]
    for e, v in clip.items():
        h = v["held_out"]
        out[f"ppo_clip_{e}"] = {"mean": h["mean_reward"], "se": se(h["std_reward"], h["eval_examples"]), "n": h["eval_examples"]}
    kl = json.loads((R / "ppo_kl_ablation.json").read_text())
    for b, v in kl.items():
        h = v["held_out"]
        out[f"ppo_kl_{b}"] = {"mean": h["mean_reward"], "se": se(h["std_reward"], h["eval_examples"]), "n": h["eval_examples"]}
    nm = json.loads((R / "grpo_normalization_study.json").read_text())
    for k, v in nm.items():
        h = v["held_out"]
        out[f"grpo_{k}"] = {"mean": h["mean_reward"], "se": se(h["std_reward"], h["eval_examples"]), "n": h["eval_examples"]}
    return out


def paired_differences():
    """Same 100 held-out prompts for every policy, so compare them prompt by prompt."""
    def scores(name):
        rows = jl(R / "samples" / f"{name}_eval_samples.jsonl")
        key = "rm_score" if "rm_score" in rows[0] else "reward"
        return {r["prompt_id"]: r[key] for r in rows}

    pairs = [("dpo_beta_0_1", "dpo_beta_0_03"), ("dpo_beta_0_3", "dpo_beta_0_1"), ("dpo_standard", "dpo_beta_0_1"),
             ("ppo_clip_0_05", "ppo_clip_0_2"), ("ppo_kl_0_0", "ppo_kl_0_1"), ("ppo_kl_0_2", "ppo_kl_0_1"),
             ("grpo_dr_grpo", "grpo_canonical"), ("grpo_standard", "grpo_canonical")]
    out = {}
    for a, b in pairs:
        sa, sb = scores(a), scores(b)
        ids = sorted(set(sa) & set(sb))
        d = np.array([sa[i] - sb[i] for i in ids])
        out[f"{a}_minus_{b}"] = {"n": len(ids), "mean_diff": round(float(d.mean()), 3),
                                 "paired_se": round(float(d.std(ddof=1) / math.sqrt(len(d))), 3)}
    return out


def rl_training_checks():
    ppo = json.loads((R / "ppo_standard_continuation.json").read_text())["trajectory"]
    grpo = json.loads((R / "grpo_standard_continuation.json").read_text())["trajectory"]
    kl_tok = float(np.mean([t["kl"] for t in ppo]))
    length = float(np.mean([t["response_length"] for t in ppo]))
    comp = [(l, m) for t in grpo for l, m in zip(t["completion_lengths"], t["completion_masked"])]
    return {
        "ppo": {
            "mean_token_kl": round(kl_tok, 6), "mean_response_length": round(length, 1),
            "kl_penalty_per_response_at_beta_0.1": round(0.1 * kl_tok * length, 5),
            "mean_abs_task_reward": round(float(np.mean([abs(t["reward_with_eos_penalty"]) for t in ppo])), 3),
            "corr_value_mean_vs_reward": corr([t["value_mean"] for t in ppo], [t["reward_with_eos_penalty"] for t in ppo]),
            "value_grad_norm_preclip_min_max": [round(min(t["value_grad_norm"] for t in ppo), 1), round(max(t["value_grad_norm"] for t in ppo), 1)],
            "explained_variance_min_max": [round(min(t["value_explained_variance"] for t in ppo), 2), round(max(t["value_explained_variance"] for t in ppo), 2)],
            "updates_ending_with_eos": int(sum(t["terminated_with_eos"] for t in ppo)),
        },
        "grpo": {
            "completions": len(comp), "masked_truncated": sum(m for _, m in comp),
            "zero_grad_updates": [t["update"] for t in grpo if t["grad_norm"] == 0],
            "within_group_std_min_max": [min(t["reward_std"] for t in grpo), max(t["reward_std"] for t in grpo)],
        },
    }


def length_vs_reward():
    out = {}
    for name in ["dpo_standard", "ppo_standard", "grpo_standard"]:
        rows = jl(R / "samples" / f"{name}_eval_samples.jsonl")
        key = "rm_score" if "rm_score" in rows[0] else "reward"
        lens = [r["response_tokens"] for r in rows]
        rew = [r[key] for r in rows]
        cap = max(lens)
        hit = [r[key] for r in rows if r["response_tokens"] >= cap]
        rest = [r[key] for r in rows if r["response_tokens"] < cap]
        out[name] = {"corr_length_reward": corr(lens, rew), "generation_cap_observed": cap, "num_hit_cap": len(hit),
                     "mean_reward_hit_cap": round(float(np.mean(hit)), 3) if hit else None,
                     "mean_reward_below_cap": round(float(np.mean(rest)), 3)}
    return out


def audit_coarse(cfg):
    coarse = {"SAFE_ANSWER": "ANSWER", "OVER_REFUSAL": "REFUSAL", "JUSTIFIED_REFUSAL": "REFUSAL",
              "UNSAFE_COMPLIANCE": "UNSAFE", "AMBIGUOUS": "AMBIGUOUS"}
    d = repo_path(cfg["results_dir"]) / "task4_safety"
    ai = {}
    for pol in ["sft", "dpo", "ppo", "grpo"]:
        for r in jl(d / f"judged_{pol}.jsonl"):
            ai[(pol, int(r["xstest_id"]))] = r["label"]
    df = pd.read_csv(d / "manual_audit_sheet.csv")
    rows = [(r.policy, r.benchmark_class, r.manual_label.strip().upper(), ai[(r.policy, int(r.xstest_id))]) for r in df.itertuples()]

    def summ(sub):
        m = [coarse[x[2]] for x in sub]
        a = [coarse[x[3]] for x in sub]
        conf = Counter(f"{x}|{y}" for x, y in zip(m, a))
        return {"n": len(sub), "agreement": round(float(np.mean([x == y for x, y in zip(m, a)])), 4),
                "cohen_kappa": round(cohen_kappa(m, a), 4), "confusion_manual|ai": dict(conf)}

    out = {"overall": summ(rows)}
    for c in ["SAFE", "UNSAFE"]:
        out[c] = summ([x for x in rows if x[1] == c])
    for p in ["sft", "dpo", "ppo", "grpo"]:
        out[p] = summ([x for x in rows if x[0] == p])
    # labels whose name contradicts the prompt class under the judge's own definitions
    bad_manual = Counter(f"{c}|{m}" for _, c, m, _ in rows
                         if (c == "SAFE" and m in ("JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE"))
                         or (c == "UNSAFE" and m in ("SAFE_ANSWER", "OVER_REFUSAL")))
    out["manual_labels_inconsistent_with_class"] = dict(bad_manual)
    full = Counter()
    for pol in ["sft", "dpo", "ppo", "grpo"]:
        for r in jl(d / f"judged_{pol}.jsonl"):
            c = r["benchmark_class"]
            if (c == "SAFE" and r["label"] in ("JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE")) or (c == "UNSAFE" and r["label"] in ("SAFE_ANSWER", "OVER_REFUSAL")):
                full[f"{c}|{r['label']}"] += 1
    out["judge_labels_inconsistent_with_class_all_1800"] = dict(full)
    return out


NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def last_number(text):
    nums = NUM.findall(text.replace("**", "").replace("$", ""))
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


def math_checks():
    d = R / "task5_feedback"
    out = {}
    for ds in ["gsm", "transfer"]:
        rows = {p: jl(d / f"{ds}_{p}_responses.jsonl") for p in ["sft", "rlvr", "rlaif"]}
        res = {}
        for p, rs in rows.items():
            ft = Counter(r["failure_type"] for r in rs)
            nf = [r for r in rs if r["failure_type"] == "no_final_format"]
            gold = [float(str(r["gold_final"]).replace(",", "")) for r in rs]
            ln = [last_number(r["response"]) for r in rs]
            res[p] = {
                "exact_accuracy": round(ft["correct"] / len(rs), 4),
                "accuracy_given_format": round(ft["correct"] / max(1, ft["correct"] + ft["wrong_final"]), 4),
                "no_final_format": len(nf),
                "no_final_format_with_boxed": sum("\\boxed" in r["response"] for r in nf),
                "last_number_accuracy": round(float(np.mean([x is not None and abs(x - g) < 1e-6 for x, g in zip(ln, gold)])), 4),
            }
        for p in ["rlvr", "rlaif"]:
            same = [b["judge_vs_sft"] for a, b in zip(rows["sft"], rows[p]) if a["response"] == b["response"]]
            diff = [b["judge_vs_sft"] for a, b in zip(rows["sft"], rows[p]) if a["response"] != b["response"]]
            res[p]["identical_to_sft"] = len(same)
            # judge verdicts (A = this policy wins, B = SFT wins) on byte-identical vs differing pairs
            res[p]["judge_on_identical_pairs"] = dict(Counter(same))
            res[p]["judge_on_differing_pairs"] = dict(Counter(diff))
        out[ds] = res
    return out


def adapter_norms(ckpt_dir):
    out = {}
    for name in ["ppo_midpoint_policy", "grpo_midpoint_policy", "rlvr_policy", "rlaif_policy"]:
        f = ckpt_dir / name / "adapter_model.safetensors"
        if not f.exists():
            continue
        raw = f.read_bytes()
        n = struct.unpack("<Q", raw[:8])[0]
        hdr = json.loads(raw[8:8 + n])
        acfg = json.loads((ckpt_dir / name / "adapter_config.json").read_text())
        scale = acfg["lora_alpha"] / acfg["r"]
        t, total = {}, 0
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            s, e = v["data_offsets"]
            dt = {"F32": np.float32, "F16": np.float16}[v["dtype"]]
            t[k] = np.frombuffer(raw[8 + n + s:8 + n + e], dtype=dt).reshape(v["shape"]).astype(np.float64)
            total += int(np.prod(v["shape"]))
        norms = [float(np.linalg.norm(scale * t[k.replace("lora_A", "lora_B")] @ t[k])) for k in t if "lora_A" in k]
        out[name] = {"trainable_params": total, "modules": len(norms),
                     "delta_w_fro_mean": round(float(np.mean(norms)), 5), "delta_w_fro_max": round(float(np.max(norms)), 5)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--checkpoints", default="checkpoints")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out = {
        "heldout_reward_uncertainty": heldout_uncertainty(),
        "paired_reward_differences": paired_differences(),
        "rl_training_checks": rl_training_checks(),
        "length_vs_reward": length_vs_reward(),
        "task4_audit_answer_vs_refusal": audit_coarse(cfg),
        "task5_math_checks": math_checks(),
        "supplied_adapters": adapter_norms(repo_path(args.checkpoints)),
    }
    (R / "report_diagnostics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
