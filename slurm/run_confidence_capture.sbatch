#!/bin/bash
# Confidence-capture run: the 7 non-CoT variants x all 6 models, full EDOS
# test split. See scripts/run_confidence_capture.py's module docstring.
#
# One job for all 6 models, not one per model: a dry run put the whole
# thing at well under an hour, so per-model queueing overhead isn't worth
# it here. Resubmittable via skip-if-output-exists per (model, variant)
# unit (see the script's own docstring).
#
# --exclude=gorina8: its GPU index 3 fails fast on allocation. gorina9 is
# not excluded even though its GPU7 has the same issue, since excluding
# both nodes leaves nothing for the job to land on; gorina9 has 7 other
# working GPUs, so this accepts a partial risk of landing on the bad one
# and resubmitting if it does.
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=gorina8
#SBATCH --output=/path/to/sexism-study/logs/%x-%j.out
#SBATCH --error=/path/to/sexism-study/logs/%x-%j.err

set -euo pipefail

REPO_ROOT=/path/to/sexism-study/repo
cd "$REPO_ROOT"
source /path/to/sexism-study/venv/bin/activate
export HF_HOME=/path/to/sexism-study/hf_cache

# HF_TOKEN for the two gated models (llama3.1, gemma2).
if [ -f "$HOME/.hf_token" ]; then
  export HF_TOKEN="$(cat "$HOME/.hf_token")"
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/run_confidence_capture.py \
  --models mistral,phi3,llama3.1,qwen2.5,gemma2,qwen3 \
  --data-dir /path/to/sexism-study/data \
  --output-dir /path/to/sexism-study/checkpoints/confidence
