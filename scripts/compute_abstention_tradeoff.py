#!/usr/bin/env python3
"""Confidence-based abstention tradeoff for each model's F1-best covered
(non-CoT) confidence-capture variant.

No GPU, no new generation: reads the already-completed
scripts/run_confidence_capture.py checkpoint dir (the same 42 per-unit JSON
files scripts/aggregate_confidence_capture.py reads) and
results/confidence_capture_by_model.csv (that script's own output, for each
model's best_covered_variant selection), then sweeps
metrics.compute_metrics_with_abstention across a fixed set of abstention
band widths for exactly those 6 (model, variant) pairs -- the same cells
already reported in paper/latex Table IX -- so the tradeoff is directly
comparable to that table, not a different selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402

from sexism_prompting.metrics import compute_metrics_with_abstention  # noqa: E402

BANDS = (0.0, 0.1, 0.2, 0.3, 0.4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default="./checkpoints/confidence")
    parser.add_argument("--by-model-path", default="./results/confidence_capture_by_model.csv")
    parser.add_argument("--out", default="./results/abstention_tradeoff.csv")
    parser.add_argument("--bands", type=float, nargs="+", default=list(BANDS))
    args = parser.parse_args()

    by_model = pd.read_csv(args.by_model_path)
    checkpoint_dir = Path(args.checkpoint_dir)

    rows = []
    for _, row in by_model.iterrows():
        model, variant = row["model"], row["best_covered_variant"]
        unit_path = checkpoint_dir / f"{variant}__{model}.json"
        if not unit_path.exists():
            print(f"SKIP {model}/{variant}: {unit_path} not found")
            continue
        with open(unit_path) as f:
            d = json.load(f)

        for band in args.bands:
            m = compute_metrics_with_abstention(d["responses"], d["y_true"], d["confidences"], band=band)
            rows.append({"model": model, "variant": variant, "band": band, **m})

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}\n")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
