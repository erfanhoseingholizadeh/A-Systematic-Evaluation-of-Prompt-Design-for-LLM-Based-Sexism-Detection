# Sexism Prompting

A study of prompt design for sexism detection with instruction-tuned LLMs: eighteen prompt variants (role framing, context, aspect decomposition, chain-of-thought, few-shot) evaluated zero- and few-shot across six models, checked against a classical TF-IDF baseline and two fine-tuned baselines on the same official EDOS test split. The short version of the finding: careful prompt design moves the numbers a lot within its own space, but for five of the six models it never catches up to a plain bag-of-words logistic regression classifier.

This is a real codebase, not a notebook. Everything under `src/sexism_prompting/` is tested and importable, every run is seeded and checkpointed, and the evaluation grid is resumable if it gets killed partway through.

## What's here

Six models (Mistral-7B, Phi-3-mini, Llama-3.1-8B, Qwen2.5-7B-Instruct, Gemma-2-9B-it, Qwen3-8B) are run through the same 18-variant prompt grid on the full official EDOS test split, loaded one at a time so a single GPU is enough. Chain-of-thought variants get a real token budget instead of a two-token cap, and the model produces its own `ANSWER: YES` / `ANSWER: NO` marker after reasoning rather than being forced into an instant answer.

Alongside the prompting grid: a QLoRA fine-tune and a supervised RoBERTa encoder for each model as fine-tuned baselines, and a CPU-only TF-IDF + logistic regression classifier as the classical baseline. Two sensitivity sweeps rerun each model's own best variant under an alternate prompt-component order and alternate few-shot counts, and a third sweep checks sensitivity to which specific few-shot examples get drawn. First-token logits give a calibrated P(sexist) score for the non-CoT variants, which feeds both an AUC comparison against the classical baseline and a confidence-based abstention policy.

Every run seeds everything that could introduce randomness (few-shot sampling, prompt component order, data splits, training) from one place, and every `(model, variant)` unit is checkpointed as it finishes, so a killed job resumes instead of starting over. Statistical testing (bootstrap CIs, McNemar vs. the baseline variant, Holm correction) runs post hoc from the saved predictions, no GPU needed.

## Layout

```
src/sexism_prompting/
  data.py, edos_full.py     shared download/label utilities; edos_full.py is the sole dataset source
  prompts.py                the R/A/C/T/F prompt component builder
  inference.py               batched generation, separate configs for CoT vs. direct
  metrics.py                 YES/NO parsing, accuracy/precision/recall/F1
  models.py, cluster_setup.py   model registry + loaders (local and Slurm-cluster)
  evaluate.py, checkpoint.py     the evaluation loop and its resumable manifest
  analysis.py                bootstrap CIs, McNemar tests, failure-policy comparisons
  finetune.py, classical.py   the encoder/QLoRA and TF-IDF baselines
  confidence.py               per-example P(sexist) from first-token logits
  sensitivity_sweeps.py       the order/size/seed sweeps
  sft.py, io_utils.py, seeding.py   SFT example construction, results I/O, the one set_seed()
configs/    models.yaml, prompt_variants.yaml, finetune.yaml, cluster.yaml
scripts/    CLI entry points, local or cluster
slurm/      Slurm job templates and submission scripts
tests/      pure-Python unit tests, no GPU required
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

That installs and tests everything except the actual model-loading and inference code, which needs a CUDA GPU:

```bash
pip install -e ".[gpu]"
```

The two gated models (Llama-3.1, Gemma-2) need an HF token with access granted, exported as `HF_TOKEN`.

Direct dependencies in `pyproject.toml` are pinned to exact versions. For a byte-identical environment, including the full transitive dependency closure, use the lock files instead: `requirements-lock.txt` for local development (no GPU needed) or `requirements-gpu-lock.txt` for the GPU stack (needs a compatible CUDA setup).

## Running it

```bash
python scripts/download_data.py                  # full EDOS release, SHA-256 verified
python scripts/run_experiment.py                  # full 6-model x 18-variant grid
python scripts/run_experiment.py --only-models mistral --only-variants V1,V6 --subsample 50  # quick pilot
python scripts/run_finetune.py                     # encoder + QLoRA baselines, needs the gpu extra
python scripts/run_classical_baseline.py           # TF-IDF + logistic regression, CPU-only
python scripts/run_extra_sweep.py                  # order + few-shot-size sensitivity sweeps, run after run_experiment.py
python scripts/analyze_results.py --checkpoint-dir <ckpt dir>   # bootstrap CIs + McNemar, no GPU
```

Every entry point takes a `seed` (default 42) and calls `seeding.set_seed` before anything else. Decoding is greedy, which removes sampling randomness but not kernel-level nondeterminism: `seeding.set_seed`'s `full_determinism` option isn't enabled by default, so exact byte-for-byte reproduction across different hardware isn't guaranteed, only that the decoding strategy itself introduces no extra randomness on top of the seed.

The paper-scale runs were done on a Slurm GPU cluster. `slurm/run_model.sbatch` + `slurm/submit_grid.sh` submit the full grid, `slurm/run_finetune.sbatch` + `slurm/submit_finetune.sh` submit the fine-tuned baselines, both going through `scripts/run_on_cluster.py`, which batches real GPU generation instead of the one-at-a-time default path. These scripts have a placeholder workspace path (`/path/to/sexism-study`) that needs editing for your own cluster before submitting anything.

## Authors

Erfan Hoseingholizadeh, Shakiba Sadat Mirbagheri, Mina Farmanbar.

## License

Apache 2.0. See `LICENSE`.
