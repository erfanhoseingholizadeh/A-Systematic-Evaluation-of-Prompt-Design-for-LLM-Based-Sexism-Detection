#!/usr/bin/env python3
"""Extend the refusal-vs-truncation check (originally done ad hoc for
llama3.1's chain-of-thought variants, see Sec. V-C / Table III) to
other models' smaller elevated-fail_ratio cells.

No GPU, no new generation: reads each unit's already-saved predictions
file (synced from checkpoints/<model>/predictions/ on the cluster) and
classifies its failed rows by regex.

Usage:
    python scripts/check_cot_refusal_vs_truncation.py \
        --unit gemma2:V8 --unit qwen3:V8 --unit qwen3:V12 --unit qwen3:V14 \
        --unit mistral:V8 --unit phi3:V12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402

from sexism_prompting.analysis import classify_cot_failure_causes  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unit", action="append", required=True,
        help="model:variant pair, e.g. gemma2:V8. Repeatable.",
    )
    parser.add_argument("--checkpoints-root", default="./checkpoints")
    parser.add_argument("--out", default="./results/analysis/cot_refusal_vs_truncation.csv")
    args = parser.parse_args()

    by_model: dict[str, list[str]] = {}
    for spec in args.unit:
        model, variant = spec.split(":", 1)
        by_model.setdefault(model, []).append(variant)

    frames = []
    for model, variants in by_model.items():
        unit_ids = [f"{v}__{model}" for v in variants]
        df = classify_cot_failure_causes(Path(args.checkpoints_root) / model, unit_ids)
        df.insert(0, "model", model)
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"Wrote {len(result)} rows to {out_path}\n")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
