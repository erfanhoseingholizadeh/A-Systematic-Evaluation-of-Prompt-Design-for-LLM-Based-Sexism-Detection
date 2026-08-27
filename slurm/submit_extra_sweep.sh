#!/bin/bash
# Submit all 6 models' sensitivity-sweep Slurm jobs, one job per model
# (mirrors submit_grid.sh).
#
# Run from the Slurm submit host, inside the repo:
#   cd /path/to/sexism-study/repo/slurm && ./submit_extra_sweep.sh
#
# Requires results/results.csv (the main grid's per-variant results,
# merged across all 6 models) to already exist -- not enforced here, just
# the intended precondition; run_extra_sweep_on_cluster.py fails fast with
# a clear message if it's missing.
#
# qwen3 pinned to the 80GB A100 like submit_grid.sh: its batch_size=128
# (configs/cluster.yaml) needs the larger card.
set -euo pipefail

mkdir -p /path/to/sexism-study/logs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sbatch --job-name=sexism-sweep-qwen3 --gres=gpu:nvidia_a100_80gb_pcie:1 \
  --export=MODEL=qwen3 "$SCRIPT_DIR/run_extra_sweep.sbatch"

for model in mistral phi3 llama3.1 qwen2.5 gemma2; do
  sbatch --job-name="sexism-sweep-${model}" --gres=gpu:1 \
    --export="MODEL=${model}" "$SCRIPT_DIR/run_extra_sweep.sbatch"
done

echo "Submitted all 6 sensitivity-sweep jobs. Check with: squeue -u \$USER"
