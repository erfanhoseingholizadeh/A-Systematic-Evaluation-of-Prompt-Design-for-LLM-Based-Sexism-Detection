#!/bin/bash
# Generic per-model sensitivity-sweep Slurm job. Mirrors run_model.sbatch,
# but for scripts/run_extra_sweep_on_cluster.py instead of run_on_cluster.py.
#
# Not meant to be `sbatch`'d directly. MODEL differs per model. Use
# submit_extra_sweep.sh, which fills it in via `sbatch --export=MODEL=... ...`.
#
# Must run after the main grid (run_model.sbatch/submit_grid.sh) has
# produced every model's results CSV and those have been merged into
# results/results.csv -- the sweep targets each model's best variant from
# the main grid, which doesn't exist before that.
#
# Own --checkpoint-dir per model: RunCheckpoint isn't safe for two
# processes writing the same directory concurrently.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=/path/to/sexism-study/logs/%x-%j.out
#SBATCH --error=/path/to/sexism-study/logs/%x-%j.err

set -euo pipefail

if [ -z "${MODEL:-}" ]; then
  echo "MODEL not set. Submit via submit_extra_sweep.sh, not sbatch run_extra_sweep.sbatch directly." >&2
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
  --only-models "$MODEL" \
  --data-dir /path/to/sexism-study/data \
  --checkpoint-dir "/path/to/sexism-study/checkpoints/extra_sweep/${MODEL}" \
  --results-path /path/to/sexism-study/results/results.csv \
  --sweep-results-path "/path/to/sexism-study/results/extra_sweep_${MODEL}.csv"
