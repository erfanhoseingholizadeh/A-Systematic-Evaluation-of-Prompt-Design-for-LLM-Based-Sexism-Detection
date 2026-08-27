#!/bin/bash
# Generic per-model Slurm job.
#
# Not meant to be `sbatch`'d directly. MODEL, GRES, and TIME differ per
# model (qwen3 needs the 80GB card; the other 5 use generic gpu:1). Use
# submit_grid.sh, which fills these in per model via
# `sbatch --export=MODEL=... --gres=... ...`.
#
# Resubmittable: RunCheckpoint is manifest-based (configs/cluster.yaml,
# cluster_setup.py), so re-running this same job after a 24h MaxTime kill
# picks up from the last completed unit automatically.
#
# RunCheckpoint is not safe for two processes writing the same
# checkpoint-dir concurrently, so each model here gets its own
# --checkpoint-dir; don't point two concurrently-running jobs at the same
# one.
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00
#SBATCH --output=/path/to/sexism-study/logs/%x-%j.out
#SBATCH --error=/path/to/sexism-study/logs/%x-%j.err

set -euo pipefail

if [ -z "${MODEL:-}" ]; then
  echo "MODEL not set. Submit via submit_grid.sh, not sbatch run_model.sbatch directly." >&2
  exit 1
fi

REPO_ROOT=/path/to/sexism-study/repo
cd "$REPO_ROOT"
source /path/to/sexism-study/venv/bin/activate
export HF_HOME=/path/to/sexism-study/hf_cache

# HF_TOKEN for the two gated models (llama3.1, gemma2; see models.py's
# GATED_MODELS). Never hardcoded here. Drop a token at ~/.hf_token
# (chmod 600) before submitting those two jobs; the other 4 models load
# fine without this file present.
if [ -f "$HOME/.hf_token" ]; then
  export HF_TOKEN="$(cat "$HOME/.hf_token")"
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/run_on_cluster.py \
  --only-models "$MODEL" \
  --data-dir /path/to/sexism-study/data \
  --checkpoint-dir "/path/to/sexism-study/checkpoints/${MODEL}" \
  --results-path "/path/to/sexism-study/results/${MODEL}.csv"
