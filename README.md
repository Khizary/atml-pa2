# ATML PA2 - LLM Post-Training

Implementation and results for ATML PA2: DPO, PPO, GRPO / Dr. GRPO, XSTest safety calibration, and RLVR vs RLAIF, all on Qwen2.5-1.5B-Instruct with LoRA adapters. It's built on the course starter repository; the training loops, ablations, evaluation and analysis code are ours.

## Setup

```bash
git clone https://github.com/Khizary/atml-pa2.git
cd atml-pa2
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
python -m scripts.check_environment
```

`download_assets` fetches the fixed course data, cached rollouts and midpoint checkpoints from
https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets at the pinned revision
`0b350481fb03f5525a35bcdec4131bd4fe487f98`. Base, reward and judge models are pulled from Hugging Face at runtime.

Reference environment: Transformers 4.57.1, TRL 0.27.2, PEFT 0.17.1, Tokenizers 0.22.1. Our results were produced on a single Kaggle T4 (16 GB, fp16). The full pipeline takes about 8 hours there.

## Reproducing the results

Run from the repo root, in this order. Every config fixes the seed (6304), and each fork starts from the same supplied midpoint checkpoint.

```bash
# Task 1 - DPO
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.ablate_beta --config configs/dpo.yaml
python -m task1_dpo.analyze_length --config configs/dpo.yaml

# Task 2 - PPO
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml
python -m task2_ppo.ablate_kl --config configs/ppo.yaml

# Task 3 - GRPO
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml
python -m task3_grpo.compare_normalization --config configs/grpo.yaml

# Task 4 - Safety calibration (needs the Task 1-3 adapters)
python -m task4_safety.generate_responses --config configs/feedback.yaml
python -m task4_safety.judge_responses --config configs/feedback.yaml
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml
# fill the manual_label column of results/task4_safety/manual_audit_sheet.csv, then:
python -m task4_safety.evaluate_safety --config configs/feedback.yaml

# Task 5 - RLVR vs RLAIF
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml

# Extra checks used in the report (standard errors, audit agreement on merged labels,
# math format checks, size of the supplied adapters); reads results/ and checkpoints/ only
python -m scripts.report_diagnostics --config configs/feedback.yaml
```

Trained adapters go to `outputs/` (not committed). Everything reported is in `results/`:

| Path | Contents |
|---|---|
| `results/dpo_*.json` | DPO training logs, eval (pair accuracy, margins, sampled KL, RM score, length), beta ablation, length-stratified analysis |
| `results/ppo_*.json` | PPO continuation trajectories, held-out eval, clipping study (cached-batch + forks), KL ablation |
| `results/grpo_*.json` | GRPO continuation trajectories, eval, group-size study, canonical vs Dr. GRPO comparison |
| `results/samples/` | generated eval samples per policy/fork, for qualitative examples |
| `results/task4_safety/` | XSTest generations, judge labels, manual audit sheet (with our labels), `safety_calibration.json` |
| `results/task5_feedback/` | GSM8K and SVAMP-transfer responses, pairwise judge results, perturbation diagnostics, comparison summary |
| `results/report_diagnostics.json` | paired reward differences and standard errors, length/reward correlations, audit agreement on merged labels, math format checks, supplied-adapter sizes |

## Implementation notes

- **Starter defects fixed:** DPO margin sign (`task1_dpo`), PPO clipped surrogate now takes `min` rather than `max` (`task2_ppo/ppo.py`), and GRPO rewards are standardized per group rather than over the batch (`task3_grpo`).
- **Reference policy:** the base model, obtained by disabling the LoRA adapter (`common.models.reference_mode`).
- **KL:** response-token `log pi - log pi_ref` on the policy's own samples, averaged over tokens.
- **PPO critic:** the trainable critic parameters (LoRA + score head) are kept in fp32, because Adam on fp16 weights produced NaNs on the T4. Value estimates come from the decoder hidden states passed through the head (`common.models.token_values`).
- **Dropout:** disabled during RL updates, so old and new log-probs come from the same deterministic forward pass.
- **Task 4:** the supplied judge prompt, loader and parser are used unchanged. Rates follow the judge spec, i.e. P(label | prompt class).

## Attribution

- Starter code, data, cached rollouts, checkpoints, the XSTest safety judge, and the Task 5 verifier and pairwise judge come from the course release (ATML PA2 starter repository and the Hugging Face asset repo above).
- Models: Qwen2.5-1.5B-Instruct (policy), `yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback` (reward model), and Qwen2.5-3B-Instruct (judge), all from Hugging Face.
- Datasets: XSTest (Röttger et al., 2024), GSM8K, and the SVAMP challenge set, as packaged by the course.
- LLM coding assistants were used while writing and debugging the code, as the course policy allows for code. All experimental design choices, result interpretation and the report are our own.
