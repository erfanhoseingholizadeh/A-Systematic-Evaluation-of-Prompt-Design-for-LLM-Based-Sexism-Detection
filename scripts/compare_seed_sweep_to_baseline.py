#!/usr/bin/env python3
"""Compare each model's demonstration-seed sweep against its own grid-default
(fewshot_seed=42) F1 on the same best variant.

No GPU: reads the already-completed ``results/seed_sweep_<model>.csv`` files
(3 alternate seeds per model, from ``scripts/run_extra_sweep.py --which
seed``) and the main grid's ``results/results.csv`` for each model's
best-variant baseline (via ``sensitivity_sweeps.best_variant_per_model``, the
same selection the seed sweep itself targets -- never a hardcoded guess).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402

from sexism_prompting.sensitivity_sweeps import best_variant_per_model  # noqa: E402

GRID_DEFAULT_SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-path", default="./results/results.csv")
    parser.add_argument("--seed-sweep-dir", default="./results")
    parser.add_argument("--out", default="./results/seed_sweep_vs_baseline.csv")
    args = parser.parse_args()

    grid_df = pd.read_csv(args.results_path)
    best_variant = best_variant_per_model(grid_df)

    rows = []
    for model, variant in sorted(best_variant.items()):
        baseline_row = grid_df[(grid_df["model"] == model) & (grid_df["variant_id"] == variant)]
        if baseline_row.empty:
            print(f"SKIP {model}: no grid-default row for {variant} in {args.results_path}")
            continue
        baseline_f1 = float(baseline_row["f1"].iloc[0])

        sweep_path = Path(args.seed_sweep_dir) / f"seed_sweep_{model}.csv"
        if not sweep_path.exists():
            print(f"SKIP {model}: {sweep_path} not found")
            continue
        sweep_df = pd.read_csv(sweep_path)

        for _, sweep_row in sweep_df.sort_values("fewshot_seed").iterrows():
            rows.append(
                {
                    "model": model,
                    "variant": variant,
                    "fewshot_seed": int(sweep_row["fewshot_seed"]),
                    "seed_f1": float(sweep_row["f1"]),
                    "seed_fail_ratio": float(sweep_row["fail_ratio"]),
                    "seed_n_fail": int(sweep_row["n_fail"]),
                    "seed_n": int(sweep_row["n"]),
                    "grid_default_seed": GRID_DEFAULT_SEED,
                    "grid_default_f1": baseline_f1,
                    "grid_default_fail_ratio": float(baseline_row["fail_ratio"].iloc[0]),
                    "delta_f1": float(sweep_row["f1"]) - baseline_f1,
                }
            )

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}\n")
    print(out_df.to_string(index=False))

    print("\nPer-model spread (delta_f1 = seed_f1 - grid_default_f1):")
    summary = out_df.groupby(["model", "variant", "grid_default_f1"])["delta_f1"].agg(
        ["min", "max", "mean"]
    )
    summary["range"] = summary["max"] - summary["min"]
    print(summary.sort_values("range", ascending=False).to_string())


if __name__ == "__main__":
    main()
