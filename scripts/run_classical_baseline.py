#!/usr/bin/env python3
"""Train and evaluate the classical TF-IDF + logistic-regression baseline.

CPU-only, no GPU/torch needed. Unlike run_finetune.py's encoder/QLoRA
baselines, this runs anywhere the base dependencies are installed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402

from sexism_prompting.classical import evaluate_classical_baseline, train_classical_baseline  # noqa: E402
from sexism_prompting.edos_full import download_edos_full, get_split, load_edos_full  # noqa: E402
from sexism_prompting.io_utils import save_results  # noqa: E402
from sexism_prompting.seeding import set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-path", default="./results/finetune_results.csv")
    args = parser.parse_args()

    set_seed(args.seed)

    edos_full_dir = Path(args.data_dir) / "edos_full"
    download_edos_full(edos_full_dir)
    full_df = load_edos_full(edos_full_dir)
    train_df = get_split(full_df, "train")
    test_df = get_split(full_df, "test")

    pipeline = train_classical_baseline(train_df, seed=args.seed)
    metrics = evaluate_classical_baseline(pipeline, test_df)
    print(f"[classical:tfidf+logreg] {metrics}")

    row = pd.DataFrame([{"baseline": "classical", "model": "tfidf+logreg", **metrics}])
    save_results(row, args.results_path, append=True)
    print(f"\nSaved 1 row to {args.results_path}")


if __name__ == "__main__":
    main()
