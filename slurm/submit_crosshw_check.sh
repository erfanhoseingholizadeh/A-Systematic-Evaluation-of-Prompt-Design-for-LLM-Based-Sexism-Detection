#!/bin/bash
# Cross-hardware determinism check: reruns each model's known-best
# main-grid variant on genuinely different hardware, so the "does bit-
# identical output hold across GPUs" question in the paper's Limitations
# can be tested directly rather than left as an untested hedge.
#
# mistral/phi3/llama3.1/qwen2.5/gemma2: gorina8 (A100-PCIE-40GB) vs
# gorina9 (A100-80GB-PCIE) -- two different GPU models, two different
# nodes.
# qwen3: gorina9 A100-80GB-PCIE (its normal grid card) vs gorina9
# H100-NVL -- two different GPU architectures on the same node, since
# qwen3's batch=128 footprint doesn't fit gorina8's 40GB cards.
#
# Run from the Slurm submit host, inside the repo:
#   cd /path/to/sexism-study/repo/slurm && ./submit_crosshw_check.sh
set -euo pipefail

mkdir -p /path/to/sexism-study/logs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for pair in mistral:V7 phi3:V9 llama3.1:V6 qwen2.5:V6 gemma2:V7; do
  model="${pair%%:*}"
  variant="${pair##*:}"

  sbatch --job-name="crosshw-g8-${model}" --nodelist=gorina8 --gres=gpu:1 \
    --export=MODEL="${model}",VARIANT="${variant}",TAG=gorina8 \
    "$SCRIPT_DIR/run_crosshw_check.sbatch"

  sbatch --job-name="crosshw-g9-${model}" --nodelist=gorina9 --gres=gpu:nvidia_a100_80gb_pcie:1 \
    --export=MODEL="${model}",VARIANT="${variant}",TAG=gorina9 \
    "$SCRIPT_DIR/run_crosshw_check.sbatch"
done

sbatch --job-name=crosshw-a100-qwen3 --nodelist=gorina9 --gres=gpu:nvidia_a100_80gb_pcie:1 \
  --export=MODEL=qwen3,VARIANT=V8,TAG=gorina9a100 \
  "$SCRIPT_DIR/run_crosshw_check.sbatch"

sbatch --job-name=crosshw-h100-qwen3 --nodelist=gorina9 --gres=gpu:nvidia_h100_nvl:1 \
  --export=MODEL=qwen3,VARIANT=V8,TAG=gorina9h100 \
  "$SCRIPT_DIR/run_crosshw_check.sbatch"

echo "Submitted 12 cross-hardware check jobs. Check with: squeue -u \$USER"
