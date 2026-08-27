#!/usr/bin/env python3
"""Run the model x variant grid on the UiS TekNat Slurm cluster.

Same resumable-checkpoint grid loop as run_experiment.py, but loads models
via cluster_setup.load_one_for_cluster instead of models.load_one: real
GPU batching (batch_size passed to the HF pipeline constructor, not just
used for caller-side list chunking) and bf16 instead of 4-bit quantization,
both verified on real gorina8/gorina9 hardware (see cluster_setup.py's
docstring for the measurements). models.py itself is untouched.

Intended to run inside an `srun`/`sbatch` allocation on the `gpu` partition
(see slurm/run_model.sbatch + slurm/submit_grid.sh), not on a login node
(ssh1-4) or the Slurm submit host (gorina11) directly.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml  # noqa: E402

from sexism_prompting.checkpoint import RunCheckpoint  # noqa: E402
from sexism_prompting.edos_full import download_edos_full, get_split, load_edos_full  # noqa: E402
from sexism_prompting.evaluate import evaluate_all_variants_sequential, load_prompt_variants  # noqa: E402
from sexism_prompting.io_utils import pivot_metric, save_results  # noqa: E402
from sexism_prompting.seeding import set_seed  # noqa: E402

# cluster_setup imports torch/transformers at module level (same precedent
# as models.py itself), deferred to inside main(), after argparse has
# already had a chance to handle --help, so this script stays runnable in a
# torch-less sandbox up to that point. Mirrors evaluate.py's own lazy
# `from .models import load_one, unload_one` for the identical reason (see
# tests/test_cli_entrypoints.py's module docstring for why this matters:
# pyflakes can't catch a broken import chain that only a real run exercises).


def make_should_continue(margin_minutes: float):
    """Stop starting new units once fewer than ``margin_minutes`` remain
    before Slurm kills the job. Without this, a unit still generating when
    MaxTime hits gets SIGKILLed mid-unit with no checkpoint, so its
    progress is lost again on every resubmission.

    Reads SLURM_JOB_END_TIME (a Unix timestamp Slurm sets for any job
    submitted with --time), so the budget always matches whatever time
    limit the job actually got instead of a second hardcoded value that can
    drift out of sync with slurm/run_model.sbatch. Returns None (no budget
    enforced) outside a Slurm allocation with a time limit, e.g. local
    testing, or a bare `srun` with no --time.

    ``margin_minutes`` must be at least as large as the slowest single unit
    this job could hit, since the check happens only *before* starting a
    unit, never mid-unit: the default (150 min) covers the slowest unit
    measured so far (qwen3 CoT, ~109 min at 4000 rows / 1.64s/example) with
    real margin to spare for the final checkpoint write and process exit.
    """
    end_time_str = os.environ.get("SLURM_JOB_END_TIME")
    if not end_time_str:
        return None
    end_time = float(end_time_str)
    margin_seconds = margin_minutes * 60

    def should_continue() -> bool:
        return (end_time - time.time()) > margin_seconds

    return should_continue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--checkpoint-dir", default="./checkpoints/experiment")
    parser.add_argument("--results-path", default="./results/results.csv")
    parser.add_argument("--models-config", default=None)
    parser.add_argument("--variants-config", default=None)
    parser.add_argument("--cluster-config", default=None, help="batch_size/quantize settings, see configs/cluster.yaml")
    parser.add_argument("--subsample", type=int, default=None, help="evaluate only the first N test rows (for a quick pilot)")
    parser.add_argument("--only-models", default=None, help="comma-separated model keys to restrict to, e.g. 'qwen3' (one Slurm job per model)")
    parser.add_argument("--only-variants", default=None, help="comma-separated variant ids to restrict to, e.g. 'V1,V6' (for a quick pilot)")
    parser.add_argument(
        "--time-margin-minutes", type=float, default=150.0,
        help="stop starting new units once fewer than this many minutes remain before Slurm's time limit "
             "(via SLURM_JOB_END_TIME); no effect outside a time-limited Slurm allocation",
    )
    args = parser.parse_args()

    from sexism_prompting.cluster_setup import detect_gpu_name, make_cluster_load_fn, unload_one

    repo_root = Path(args.repo_root)
    models_config = args.models_config or repo_root / "configs" / "models.yaml"
    variants_config = args.variants_config or repo_root / "configs" / "prompt_variants.yaml"
    cluster_config = args.cluster_config or repo_root / "configs" / "cluster.yaml"

    with open(models_config) as f:
        models_cfg = yaml.safe_load(f)
    with open(cluster_config) as f:
        cluster_cfg = yaml.safe_load(f)
    set_seed(models_cfg.get("seed", 42))

    gpu_name = detect_gpu_name()
    print(f"GPU: {gpu_name}")

    edos_full_dir = Path(args.data_dir) / "edos_full"
    download_edos_full(edos_full_dir)
    full_df = load_edos_full(edos_full_dir)
    test_df = get_split(full_df, "test")
    demos_df = get_split(full_df, "train")
    if args.subsample:
        test_df = test_df.iloc[: args.subsample].reset_index(drop=True)
        print(f"Subsampled test set to {len(test_df)} rows for a quick pilot.")

    variants, fewshot_n, fewshot_seed = load_prompt_variants(variants_config)
    model_keys = models_cfg["models"]
    if args.only_models:
        model_keys = [k.strip() for k in args.only_models.split(",")]
        unknown = [k for k in model_keys if k not in models_cfg["models"]]
        if unknown:
            # Fail fast rather than silently running an empty grid: an empty
            # grid_df still reaches save_results(..., append=False) below,
            # which would overwrite a good results.csv with zero rows. A
            # typo in --only-models (e.g. a Slurm job template mismatch)
            # would otherwise destroy real results with no error at all.
            raise SystemExit(
                f"--only-models: unknown model key(s) {unknown}; valid keys are {models_cfg['models']}"
            )
    if args.only_variants:
        wanted = {v.strip() for v in args.only_variants.split(",")}
        variants = [v for v in variants if v["id"] in wanted]

    checkpoint = RunCheckpoint(args.checkpoint_dir)

    cluster_batch_size = cluster_cfg.get("batch_size", 32)
    cluster_batch_size_overrides = cluster_cfg.get("batch_size_overrides", {})
    load_fn = make_cluster_load_fn(
        default_batch_size=cluster_batch_size,
        batch_size_overrides=cluster_batch_size_overrides,
        quantize=cluster_cfg.get("quantize", False),
    )

    def on_unit_done(unit_id: str, metrics: dict) -> None:
        print(f"[{unit_id}] " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items() if isinstance(v, float)))

    grid_df = evaluate_all_variants_sequential(
        model_keys,
        test_df,
        demos_df,
        variants,
        checkpoint,
        fewshot_num_per_class=fewshot_n,
        fewshot_seed=fewshot_seed,
        batch_size=cluster_batch_size,
        batch_size_overrides=cluster_batch_size_overrides,
        gen_kwargs=models_cfg.get("generation"),
        cot_gen_kwargs=models_cfg.get("generation_cot"),
        on_unit_done=on_unit_done,
        load_fn=load_fn,
        unload_fn=unload_one,
        should_continue=make_should_continue(args.time_margin_minutes),
        failure_threshold=cluster_cfg.get("failure_threshold", 0.5),
        failure_threshold_overrides=cluster_cfg.get("failure_threshold_overrides", {}),
        cot_gen_kwargs_overrides=models_cfg.get("generation_cot_overrides", {}),
    )

    save_results(grid_df, args.results_path, append=False)
    print(f"\nSaved {len(grid_df)} rows to {args.results_path}")
    print(pivot_metric(grid_df, "f1").to_string())


if __name__ == "__main__":
    main()
