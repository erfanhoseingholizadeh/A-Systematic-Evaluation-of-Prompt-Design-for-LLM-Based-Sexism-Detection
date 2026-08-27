#!/usr/bin/env python3
"""Statistical analysis of a Stage 1 checkpoint's saved predictions (no GPU).

Produces, from raw per-example responses already in the checkpoint dir:
- analysis_metrics.csv: per (model, variant) unit, headline metrics,
  failures-excluded metrics + coverage, and bootstrap CIs;
- mcnemar_vs_baseline.csv: exact McNemar tests of every variant against the
  baseline variant per model, Holm-corrected.

Works on any checkpoint directory produced by the grid, local or synced from
the cluster.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sexism_prompting.analysis import analyze_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default="./checkpoints/experiment",
                        help="dir containing manifest.json + predictions/")
    parser.add_argument("--out-dir", default="./results/analysis")
    parser.add_argument("--baseline-variant", default="V1")
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Every predictions file already embeds its own labels (see
    # checkpoint.RunCheckpoint's aligned {responses, rewire_ids, labels}
    # format), so a unit without embedded labels is simply skipped below
    # rather than risk a positional mismatch against the wrong dataset.
    fallback_labels = None

    metrics_df, mcnemar_df = analyze_checkpoint(
        args.checkpoint_dir,
        fallback_labels=fallback_labels,
        baseline_variant=args.baseline_variant,
        n_bootstrap=args.bootstrap_iters,
        seed=args.seed,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "analysis_metrics.csv"
    mcnemar_path = out_dir / "mcnemar_vs_baseline.csv"
    metrics_df.to_csv(metrics_path, index=False)
    mcnemar_df.to_csv(mcnemar_path, index=False)
    print(f"Wrote {len(metrics_df)} unit rows to {metrics_path}")
    print(f"Wrote {len(mcnemar_df)} comparison rows to {mcnemar_path}")

    if not metrics_df.empty:
        print("\nTop units by headline F1 (with 95% bootstrap CI):")
        top = metrics_df.sort_values("f1", ascending=False).head(10)
        for _, row in top.iterrows():
            print(f"  {row['unit_id']:<24} f1={row['f1']:.3f} [{row['f1_ci_low']:.3f}, {row['f1_ci_high']:.3f}] "
                  f"coverage={row['excl_coverage']:.2f}")
        flagged = metrics_df[metrics_df["fail_ratio"] > 0]
        if not flagged.empty:
            print("\nUnits with nonzero fail_ratio (headline vs failures-excluded F1):")
            for _, row in flagged.sort_values("fail_ratio", ascending=False).iterrows():
                print(f"  {row['unit_id']:<24} fail_ratio={row['fail_ratio']:.3f} "
                      f"f1={row['f1']:.3f} vs excl_f1={row['excl_f1']:.3f}")

    if not mcnemar_df.empty:
        significant = mcnemar_df[mcnemar_df["p_holm"] < 0.05]
        print(f"\nMcNemar (accuracy) vs {args.baseline_variant}: {len(significant)}/{len(mcnemar_df)} "
              f"comparisons significant at Holm-adjusted p<0.05.")
        for _, row in significant.sort_values("p_holm").iterrows():
            better = row["variant_id"] if row["variant_only_correct"] > row["baseline_only_correct"] else row["baseline"]
            print(f"  {row['model']:<10} {row['variant_id']:>4} vs {row['baseline']}: "
                  f"p_holm={row['p_holm']:.4f} (better: {better})")

        f1_significant = mcnemar_df[mcnemar_df["f1_diff_p_holm"] < 0.05]
        print(f"\nPaired bootstrap F1-difference vs {args.baseline_variant}: "
              f"{len(f1_significant)}/{len(mcnemar_df)} comparisons significant at Holm-adjusted p<0.05 "
              f"(independent test/family from McNemar above).")
        for _, row in f1_significant.sort_values("f1_diff_p_holm").iterrows():
            print(f"  {row['model']:<10} {row['variant_id']:>4} vs {row['baseline']}: "
                  f"f1_diff={row['f1_diff']:+.4f} [{row['f1_diff_ci_low']:+.4f}, {row['f1_diff_ci_high']:+.4f}] "
                  f"p_holm={row['f1_diff_p_holm']:.4f}")


if __name__ == "__main__":
    main()
