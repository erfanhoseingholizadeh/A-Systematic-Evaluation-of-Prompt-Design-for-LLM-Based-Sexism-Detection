#!/usr/bin/env python3
"""Prompt-component-order, few-shot-size, and few-shot-demonstration-seed
sensitivity sweeps, on the UiS TekNat Slurm cluster.

Same sweep logic as run_extra_sweep.py, but loads models via
cluster_setup.make_cluster_load_fn instead of models.load_one, mirroring
run_on_cluster.py's reasoning for the main grid: models.load_one's
device_map="auto" silently falls back to full CPU placement on a bad GPU
index (e.g. gorina8's index 3) instead of raising, which would turn a sweep
unit into a multi-hour hang instead of a fast, visible failure. See
cluster_setup.py's docstring for the full mechanism.

Must be run AFTER scripts/run_on_cluster.py has produced real main-grid
results; there is no meaningful "best variant" before that.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from sexism_prompting.edos_full import download_edos_full, get_split, load_edos_full  # noqa: E402
from sexism_prompting.evaluate import load_prompt_variants  # noqa: E402
from sexism_prompting.io_utils import save_results  # noqa: E402
from sexism_prompting.seeding import set_seed  # noqa: E402
from sexism_prompting.sensitivity_sweeps import (  # noqa: E402
    DEFAULT_FEWSHOT_SEEDS,
    DEFAULT_FEWSHOT_SIZES,
    plan_order_sweep,
    plan_seed_sweep,
    plan_size_sweep,
    run_order_sweep_unit,
    run_seed_sweep_unit,
    run_size_sweep_unit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--checkpoint-dir", default="./checkpoints/extra_sweep")
    parser.add_argument(
        "--results-path", default="./results/results.csv",
        help="main grid's results CSV, used to find each model's best variant",
    )
    parser.add_argument("--sweep-results-path", default="./results/extra_sweep_results.csv")
    parser.add_argument("--models-config", default=None)
    parser.add_argument("--variants-config", default=None)
    parser.add_argument("--cluster-config", default=None, help="batch_size/quantize settings, see configs/cluster.yaml")
    parser.add_argument(
        "--only-models", default=None,
        help="comma-separated model keys to restrict the sweep to, e.g. 'qwen3' (one Slurm job per model, "
             "mirroring run_on_cluster.py's --only-models) -- keeps any single job's worst-case runtime "
             "bounded to one model's best variant instead of all 6 run sequentially",
    )
    parser.add_argument(
        "--which", choices=["order", "size", "seed", "both", "all"], default="both",
        help="'both' (default, unchanged meaning) = order+size, for backward compatibility with existing "
             "invocations; 'all' = order+size+seed; 'seed' runs only the demonstration-selection sweep",
    )
    parser.add_argument("--n-orders", type=int, default=5, help="distinct component orderings to try per model")
    parser.add_argument(
        "--fewshot-sizes", default=",".join(str(s) for s in DEFAULT_FEWSHOT_SIZES),
        help="comma-separated fewshot_num_per_class values to try per model, e.g. '1,3,4'",
    )
    parser.add_argument(
        "--fewshot-seeds", default=",".join(str(s) for s in DEFAULT_FEWSHOT_SEEDS),
        help="comma-separated fewshot_seed values to try per model (which specific demonstrations are drawn, "
             "not how many), e.g. '1,7,123'",
    )
    parser.add_argument("--subsample", type=int, default=None, help="evaluate only the first N test rows (for a quick pilot)")
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

    if not Path(args.results_path).exists():
        raise SystemExit(
            f"No main-grid results found at {args.results_path}. Run scripts/run_on_cluster.py first. "
            "Sensitivity sweeps target each model's best variant, which doesn't exist until the main grid does."
        )
    results_df = pd.read_csv(args.results_path)
    if args.only_models:
        wanted = [k.strip() for k in args.only_models.split(",")]
        unknown = [k for k in wanted if k not in set(results_df["model"])]
        if unknown:
            raise SystemExit(
                f"--only-models: unknown model key(s) {unknown}; models present in {args.results_path} "
                f"are {sorted(set(results_df['model']))}"
            )
        results_df = results_df[results_df["model"].isin(wanted)].reset_index(drop=True)

    edos_full_dir = Path(args.data_dir) / "edos_full"
    download_edos_full(edos_full_dir)
    full_df = load_edos_full(edos_full_dir)
    test_df = get_split(full_df, "test")
    demos_df = get_split(full_df, "train")
    if args.subsample:
        test_df = test_df.iloc[: args.subsample].reset_index(drop=True)
        print(f"Subsampled test set to {len(test_df)} rows for a quick pilot.")

    variants, fewshot_n, fewshot_seed = load_prompt_variants(variants_config)
    variants_by_id = {v["id"]: v for v in variants}

    cluster_batch_size = cluster_cfg.get("batch_size", 32)
    cluster_batch_size_overrides = cluster_cfg.get("batch_size_overrides", {})
    load_fn = make_cluster_load_fn(
        default_batch_size=cluster_batch_size,
        batch_size_overrides=cluster_batch_size_overrides,
        quantize=cluster_cfg.get("quantize", False),
    )

    evaluate_kwargs = dict(
        batch_size=cluster_batch_size,
        batch_size_overrides=cluster_batch_size_overrides,
        gen_kwargs=models_cfg.get("generation"),
        cot_gen_kwargs=models_cfg.get("generation_cot"),
        cot_gen_kwargs_overrides=models_cfg.get("generation_cot_overrides", {}),
        failure_threshold=cluster_cfg.get("failure_threshold", 0.5),
        failure_threshold_overrides=cluster_cfg.get("failure_threshold_overrides", {}),
        load_fn=load_fn,
        unload_fn=unload_one,
    )

    all_rows = []

    if args.which in ("order", "both", "all"):
        plan, skipped = plan_order_sweep(results_df, variants_by_id, n_orders=args.n_orders)
        if skipped:
            print(
                f"Order sweep: skipping {skipped}; their best variant has fewer than 2 active "
                "shufflable components, so every ordering renders an identical prompt."
            )
        print(f"Order sweep: {len(plan)} units planned across {len({u['model_key'] for u in plan})} models.")
        for unit in plan:
            print(f"  order sweep: {unit['model_key']} / {unit['variant_id']} / order {unit['order_index']}")
            df = run_order_sweep_unit(
                unit, test_df, demos_df, variants_by_id, args.checkpoint_dir,
                fewshot_num_per_class=fewshot_n, fewshot_seed=fewshot_seed, **evaluate_kwargs,
            )
            if not df.empty:
                df["sweep"] = "order"
                df["order_index"] = unit["order_index"]
                all_rows.append(df)

    if args.which in ("size", "both", "all"):
        sizes = [int(s) for s in args.fewshot_sizes.split(",") if s.strip()]
        plan, skipped = plan_size_sweep(results_df, variants_by_id, sizes=sizes)
        if skipped:
            print(f"Size sweep: skipping {skipped}; their best variant doesn't use few-shot demonstrations.")
        print(f"Size sweep: {len(plan)} units planned across {len({u['model_key'] for u in plan})} models.")
        for unit in plan:
            print(f"  size sweep: {unit['model_key']} / {unit['variant_id']} / size {unit['fewshot_num_per_class']}")
            df = run_size_sweep_unit(
                unit, test_df, demos_df, variants_by_id, args.checkpoint_dir,
                fewshot_seed=fewshot_seed, **evaluate_kwargs,
            )
            if not df.empty:
                df["sweep"] = "size"
                df["fewshot_num_per_class"] = unit["fewshot_num_per_class"]
                all_rows.append(df)

    if args.which in ("seed", "all"):
        seeds = [int(s) for s in args.fewshot_seeds.split(",") if s.strip()]
        plan, skipped = plan_seed_sweep(results_df, variants_by_id, seeds=seeds)
        if skipped:
            print(f"Seed sweep: skipping {skipped}; their best variant doesn't use few-shot demonstrations.")
        print(f"Seed sweep: {len(plan)} units planned across {len({u['model_key'] for u in plan})} models.")
        for unit in plan:
            print(f"  seed sweep: {unit['model_key']} / {unit['variant_id']} / seed {unit['fewshot_seed']}")
            df = run_seed_sweep_unit(
                unit, test_df, demos_df, variants_by_id, args.checkpoint_dir,
                fewshot_num_per_class=fewshot_n, **evaluate_kwargs,
            )
            if not df.empty:
                df["sweep"] = "seed"
                df["fewshot_seed"] = unit["fewshot_seed"]
                all_rows.append(df)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        save_results(combined, args.sweep_results_path, append=False)
        print(f"\nSaved {len(combined)} rows to {args.sweep_results_path}")
    else:
        print("\nNo sweep units ran (nothing planned, or every candidate was skipped).")


if __name__ == "__main__":
    main()
