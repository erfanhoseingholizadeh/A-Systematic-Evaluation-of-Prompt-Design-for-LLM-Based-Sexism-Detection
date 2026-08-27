#!/bin/bash
# Submit the encoder baseline (once) and QLoRA fine-tuning (one job per
# model in configs/finetune.yaml's qlora.model_keys).
#
# Run from the Slurm submit host, inside the repo:
#   cd /path/to/sexism-study/repo/slurm && ./submit_finetune.sh
#
# Each qlora job gets a generic gpu:1 request; Slurm queues whatever
# doesn't fit immediately, same as submit_grid.sh.
set -euo pipefail

mkdir -p /path/to/sexism-study/logs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sbatch --job-name=sexism-finetune-encoder --gres=gpu:1 \
  --export=WHICH=encoder "$SCRIPT_DIR/run_finetune.sbatch"

for model in mistral phi3 llama3.1 gemma2 qwen2.5 qwen3; do
  sbatch --job-name="sexism-qlora-${model}" --gres=gpu:1 \
    --export="WHICH=qlora,MODEL=${model}" "$SCRIPT_DIR/run_finetune.sbatch"
done

echo "Submitted encoder baseline + 6 qlora jobs. Check with: squeue -u \$USER"
