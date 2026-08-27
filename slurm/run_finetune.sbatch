#!/bin/bash
# Encoder baseline / QLoRA fine-tuning Slurm job. Mirrors run_model.sbatch.
#
# Not meant to be `sbatch`'d directly. WHICH selects encoder vs qlora, MODEL
# is only used (and required) for qlora, one model per job for parallelism
# across GPUs. Use submit_finetune.sh, which fills these in via
# `sbatch --export=WHICH=...,MODEL=... ...`.
#
# Encoder is a single shared baseline, not per-model: submit it exactly
# once. Each qlora job writes to its own checkpoint dir and results path,
# so concurrent qlora jobs for different models don't collide; don't
# submit two jobs for the same target (a given MODEL, or encoder)
# concurrently, since the results CSV write isn't safe for concurrent
# writers.
#
# Not resumable the way the main grid is: run_finetune.py doesn't wire up
# Trainer's resume_from_checkpoint, so a time-limit kill restarts training
# from scratch. The time limit below is set generously for that reason.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=/path/to/sexism-study/logs/%x-%j.out
#SBATCH --error=/path/to/sexism-study/logs/%x-%j.err

set -euo pipefail

if [ -z "${WHICH:-}" ]; then
  echo "WHICH not set. Submit via submit_finetune.sh, not sbatch run_finetune.sbatch directly." >&2
  exit 1
fi
if [ "$WHICH" = "qlora" ] && [ -z "${MODEL:-}" ]; then
  echo "MODEL not set; required for WHICH=qlora." >&2
  exit 1
fi

REPO_ROOT=/path/to/sexism-study/repo
cd "$REPO_ROOT"
source /path/to/sexism-study/venv/bin/activate
export HF_HOME=/path/to/sexism-study/hf_cache

# HF_TOKEN for the two gated models (llama3.1, gemma2; see models.py's
# GATED_MODELS).
if [ -f "$HOME/.hf_token" ]; then
  export HF_TOKEN="$(cat "$HOME/.hf_token")"
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ "$WHICH" = "encoder" ]; then
  python scripts/run_finetune.py \
    --which encoder \
    --data-dir /path/to/sexism-study/data \
    --results-path /path/to/sexism-study/results/finetune_encoder.csv
else
  python scripts/run_finetune.py \
    --which qlora \
    --only-models "$MODEL" \
    --data-dir /path/to/sexism-study/data \
    --results-path "/path/to/sexism-study/results/finetune_qlora_${MODEL}.csv"
fi
