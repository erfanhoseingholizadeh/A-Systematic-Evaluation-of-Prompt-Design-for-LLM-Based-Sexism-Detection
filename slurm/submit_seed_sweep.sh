#!/bin/bash
# Submit the few-shot-demonstration-seed sensitivity sweep, one job per
# model (mirrors submit_extra_sweep.sh). Reruns each model's best
# main-grid variant with several alternate seeds for which demonstrations
# are drawn (holding count fixed at the grid's default of 2 per class),
# as opposed to the order/size sweeps, which vary arrangement and count
# but never which specific examples are chosen.
#
# Run from the Slurm submit host, inside the repo:
#   cd /path/to/sexism-study/repo/slurm && ./submit_seed_sweep.sh
#
# All six models use few-shot in their best main-grid variant, so unlike
# the order sweep (which skips models whose best variant isn't order-
# sensitive), no model is skipped here.
#
# qwen3 pinned to the 80GB A100 like submit_grid.sh/submit_extra_sweep.sh.
set -euo pipefail

mkdir -p /path/to/sexism-study/logs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sbatch --job-name=sexism-seedsweep-qwen3 --gres=gpu:nvidia_a100_80gb_pcie:1 \
  --export=MODEL=qwen3 "$SCRIPT_DIR/run_seed_sweep.sbatch"

for model in mistral phi3 llama3.1 qwen2.5 gemma2; do
  sbatch --job-name="sexism-seedsweep-${model}" --gres=gpu:1 \
    --export="MODEL=${model}" "$SCRIPT_DIR/run_seed_sweep.sbatch"
done

echo "Submitted all 6 seed-sweep jobs. Check with: squeue -u \$USER"
