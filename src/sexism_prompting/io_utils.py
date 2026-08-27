"""Saving and reshaping evaluation results as local CSV files, plus small
file-integrity helpers shared by the data-download and checkpoint modules."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def sha256_of_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Hex SHA-256 of a file's contents (streamed, so large files are fine)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def save_results(results_df: pd.DataFrame, output_path: str | Path, append: bool = True) -> Path:
    """Save a tidy results DataFrame to CSV, appending to an existing file by default."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if append and output_path.exists():
        existing = pd.read_csv(output_path)
        combined = pd.concat([existing, results_df], ignore_index=True)
    else:
        combined = results_df

    combined.to_csv(output_path, index=False)
    return output_path


def pivot_metric(results_df: pd.DataFrame, metric: str = "f1") -> pd.DataFrame:
    """Wide table (rows = variant_id, columns = model) for one metric."""
    if results_df.empty:
        return pd.DataFrame()
    return results_df.pivot(index="variant_id", columns="model", values=metric)
