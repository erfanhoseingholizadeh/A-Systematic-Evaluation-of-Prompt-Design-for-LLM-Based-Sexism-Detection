#!/usr/bin/env python3
"""Break down a checkpoint's already-collected predictions by EDOS Task-B
sexism subtype (false positives on genuinely non-sexist rows, false
negatives per subtype).

No GPU, no new generation: reuses predictions + rewire_ids already sitting
in the checkpoint dir plus the full EDOS release's category labels (joined
locally by rewire_id, see edos_full.join_categories).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sexism_prompting.analysis import error_breakdown_by_category  # noqa: E402
from sexism_prompting.edos_full import download_edos_full, get_split, load_edos_full  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default="./checkpoints/experiment")
    parser.add_argument("--data-dir", default="./data", help="EDOS data location")
    parser.add_argument("--edos-full-dir", default=None, help="defaults to <data-dir>/edos_full")
    parser.add_argument("--out", default="./results/analysis/error_by_subtype.csv")
    args = parser.parse_args()

    full_dir = args.edos_full_dir or str(Path(args.data_dir) / "edos_full")
    download_edos_full(full_dir)
    full_df = load_edos_full(full_dir)
    test_df = get_split(full_df, "test")

    breakdown = error_breakdown_by_category(args.checkpoint_dir, test_df, full_df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    breakdown.to_csv(out_path, index=False)
    print(f"Wrote {len(breakdown)} (model, variant, category) rows to {out_path}")

    # "none" (genuinely non-sexist rows) never has a false-negative rate by
    # construction -- fn_rate only applies to the 4 real sexism subtypes,
    # "none" only ever contributes fp_rate (see error_breakdown_by_category's
    # docstring). Its (model, "none") group is therefore all-NaN on fn_rate,
    # which idxmax raises on (pandas >= 2.x) rather than silently skips.
    fn_rows = breakdown.dropna(subset=["fn_rate"])
    if not fn_rows.empty:
        print("\nWorst false-negative rate per (model, category) across all variants:")
        worst_fn = fn_rows.loc[fn_rows.groupby(["model", "category"])["fn_rate"].idxmax()]
        for _, row in worst_fn.sort_values("fn_rate", ascending=False).iterrows():
            print(f"  {row['model']:<10} {row['category']:<40} variant={row['variant_id']:>4} "
                  f"fn_rate={row['fn_rate']:.3f} (n={row['n']})")


if __name__ == "__main__":
    main()
