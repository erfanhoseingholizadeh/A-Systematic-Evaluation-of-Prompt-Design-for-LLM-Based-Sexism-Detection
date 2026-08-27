#!/bin/bash
# Single-(model, variant) job for the cross-hardware determinism check:
# reruns one model's known-best variant on a specific node/GPU type, so
# the same unit can be compared across different hardware. Not meant to
# be `sbatch`'d directly; use submit_crosshw_check.sh, which sets
# MODEL/VARIANT/TAG and pins a specific node via --nodelist/--gres.
#
# Writes to a dedicated crosshw_<TAG> checkpoint/results namespace, never
# the main grid's, so it can't collide with the production run.
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
#SBATCH --output=/path/to/sexism-study/logs/%x-%j.out
#SBATCH --error=/path/to/sexism-study/logs/%x-%j.err

set -euo pipefail

if [ -z "${MODEL:-}" ] || [ -z "${VARIANT:-}" ] || [ -z "${TAG:-}" ]; then
  echo "MODEL, VARIANT, and TAG must be set. Submit via submit_crosshw_check.sh, not sbatch directly." >&2
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

python scripts/run_on_cluster.py \
  --only-models "$MODEL" \
  --only-variants "$VARIANT" \
  --data-dir /path/to/sexism-study/data \
  --checkpoint-dir "/path/to/sexism-study/checkpoints/crosshw_${TAG}/${MODEL}" \
  --results-path "/path/to/sexism-study/results/crosshw_${TAG}_${MODEL}.csv"
