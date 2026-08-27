#!/usr/bin/env python3
"""Aggregate scripts/run_confidence_capture.py's 42 per-unit JSON files into
a paper-ready summary.

No GPU, no new generation: reads the already-completed checkpoint dir,
recomputes the agreement-rate diagnostic correctly (see
analysis.aggregate_confidence_capture's docstring for why the live-printed
value under-counts), and rolls each model up to its F1-best covered
(non-CoT) variant. Cross-references configs/models.yaml's model list and
paper/latex/data/baseline_comparison.csv's true overall-best-by-F1 variant
(computed across all 18 variants, including chain-of-thought) so a model
whose true best variant isn't covered by this run -- currently only
qwen3, whose best (V8) is chain-of-thought -- is flagged explicitly rather
than silently reported as if the covered-set selection were the real best.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402

from sexism_prompting.analysis import (  # noqa: E402
    aggregate_confidence_capture,
    summarize_confidence_capture_by_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default="./checkpoints/confidence")
    parser.add_argument("--classical-results-path", default="./results/finetune_classical.csv")
    parser.add_argument("--baseline-comparison-path", default="./paper/latex/data/baseline_comparison.csv")
    parser.add_argument("--units-out", default="./results/confidence_capture_units.csv")
    parser.add_argument("--by-model-out", default="./results/confidence_capture_by_model.csv")
    args = parser.parse_args()

    classical_df = pd.read_csv(args.classical_results_path)
    classical_auc = float(classical_df.iloc[-1]["auc"])

    units_df = aggregate_confidence_capture(args.checkpoint_dir)
    if units_df.empty:
        print(f"No unit JSON files found under {args.checkpoint_dir}; nothing to aggregate.")
        return

    by_model = summarize_confidence_capture_by_model(units_df, classical_auc=classical_auc)

    true_best = pd.read_csv(args.baseline_comparison_path)[["model", "best_prompt_variant", "best_prompt_f1"]]
    by_model = by_model.merge(true_best, on="model", how="left")
    by_model["covers_true_best_variant"] = by_model["best_covered_variant"] == by_model["best_prompt_variant"]

    for out_path, df in [(args.units_out, units_df), (args.by_model_out, by_model)]:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Wrote {len(df)} rows to {out_path}")

    print(f"\nClassical baseline AUC: {classical_auc:.4f}\n")
    print(by_model.to_string(index=False))

    uncovered = by_model[~by_model["covers_true_best_variant"]]
    if not uncovered.empty:
        print(
            "\nCAVEAT: the following model(s)' true overall-best-by-F1 variant is NOT in this "
            "run's non-CoT coverage, so the AUC reported above is a stand-in, not the model's "
            "actual best variant's AUC:"
        )
        for _, row in uncovered.iterrows():
            print(
                f"  {row['model']}: true best is {row['best_prompt_variant']} "
                f"(f1={row['best_prompt_f1']}, chain-of-thought, not AUC-scorable by this method); "
                f"reporting {row['best_covered_variant']} instead (f1={row['f1']:.3f}, auc={row['auc']:.4f})"
            )


if __name__ == "__main__":
    main()
