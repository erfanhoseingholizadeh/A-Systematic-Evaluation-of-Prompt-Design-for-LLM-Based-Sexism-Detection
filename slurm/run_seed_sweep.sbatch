#!/bin/bash
# Generic per-model few-shot-demonstration-seed sweep Slurm job. Mirrors
# run_extra_sweep.sbatch, but scoped to `--which seed` only and writing to
# its own checkpoint/results namespace, so it doesn't touch the order/size
# sweep results already produced by run_extra_sweep.sbatch's default.
#
# Not meant to be `sbatch`'d directly. MODEL differs per model. Use
# submit_seed_sweep.sh, which fills it in via `sbatch --export=MODEL=... ...`.
#
# Must run after the main grid has produced results/results.csv -- the
# sweep targets each model's best variant from the main grid.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=/path/to/sexism-study/logs/%x-%j.out
#SBATCH --error=/path/to/sexism-study/logs/%x-%j.err

set -euo pipefail

if [ -z "${MODEL:-}" ]; then
  echo "MODEL not set. Submit via submit_seed_sweep.sh, not sbatch run_seed_sweep.sbatch directly." >&2
  exit 1
fi

REPO_ROOT=/path/to/sexism-study/repo
cd "$REPO_ROOT"
source /path/to/sexism-study/venv/bin/activate
export HF_HOME=/path/to/sexism-study/hf_cache

if [ -f "$HOME/.hf_token" ]; then
  export HF_TOKEN="$(cat "$HOME/.hf_token")"
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/run_extra_sweep_on_cluster.py \
  --which seed \
  --only-models "$MODEL" \
  --data-dir /path/to/sexism-study/data \
  --checkpoint-dir "/path/to/sexism-study/checkpoints/seed_sweep/${MODEL}" \
  --results-path /path/to/sexism-study/results/results.csv \
  --sweep-results-path "/path/to/sexism-study/results/seed_sweep_${MODEL}.csv"
