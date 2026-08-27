#!/usr/bin/env python3
"""Prompt-component-order, few-shot-size, and few-shot-demonstration-seed
sensitivity sweeps, run locally.

Additive and isolated from the main grid's checkpoint. Reads the main
grid's own results CSV to find each model's best variant (see
sensitivity_sweeps.py), then sweeps that one variant along one axis at a
time: component order or few-shot demonstration count. Must be run AFTER
scripts/run_experiment.py (or scripts/run_on_cluster.py) has produced real
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

    repo_root = Path(args.repo_root)
    models_config = args.models_config or repo_root / "configs" / "models.yaml"
    variants_config = args.variants_config or repo_root / "configs" / "prompt_variants.yaml"

    with open(models_config) as f:
        models_cfg = yaml.safe_load(f)
    set_seed(models_cfg.get("seed", 42))

    if not Path(args.results_path).exists():
        raise SystemExit(
            f"No main-grid results found at {args.results_path}. Run scripts/run_experiment.py first. "
            "Sensitivity sweeps target each model's best variant, which doesn't exist until the main grid does."
        )
    results_df = pd.read_csv(args.results_path)

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

    evaluate_kwargs = dict(
        batch_size=models_cfg.get("batch_size", 32),
        batch_size_overrides=models_cfg.get("batch_size_overrides", {}),
        gen_kwargs=models_cfg.get("generation"),
        cot_gen_kwargs=models_cfg.get("generation_cot"),
        cot_gen_kwargs_overrides=models_cfg.get("generation_cot_overrides", {}),
        failure_threshold=models_cfg.get("failure_threshold", 0.5),
        failure_threshold_overrides=models_cfg.get("failure_threshold_overrides", {}),
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
