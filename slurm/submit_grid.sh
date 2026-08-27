#!/bin/bash
# Submit all 6 models' Slurm jobs.
#
# Run from the Slurm submit host, inside the repo:
#   cd /path/to/sexism-study/repo/slurm && ./submit_grid.sh
#
# All 6 are submitted at once. The cluster's own QoS caps concurrent GPUs
# per user, so the rest simply queue until one frees up; no manual
# "N at a time" orchestration needed here.
#
# qwen3 is pinned to the 80GB A100: its batch_size=128 (configs/cluster.yaml)
# uses too much VRAM for the 40GB cards. The other 5 request a generic
# gpu:1 (any free GPU) at the default batch_size=32, confirmed safe
# (no OOM) for all five.
#
# Each gpu:1 request is bound to one distinct physical GPU by the
# scheduler's own resource-tracking config, so none of these jobs can
# collide on the same device even when several land on the same node.
set -euo pipefail

mkdir -p /path/to/sexism-study/logs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sbatch --job-name=sexism-qwen3 --gres=gpu:nvidia_a100_80gb_pcie:1 \
  --export=MODEL=qwen3 "$SCRIPT_DIR/run_model.sbatch"

for model in mistral phi3 llama3.1 qwen2.5 gemma2; do
  sbatch --job-name="sexism-${model}" --gres=gpu:1 \
    --export="MODEL=${model}" "$SCRIPT_DIR/run_model.sbatch"
done

echo "Submitted all 6. Check with: squeue -u \$USER"
