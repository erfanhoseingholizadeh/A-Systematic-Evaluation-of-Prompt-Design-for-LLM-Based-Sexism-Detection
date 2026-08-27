"""The full official EDOS release (Kirk et al., SemEval-2023 Task 10): ~20k
labeled examples with the official train/dev/test split and a 4-category
sexism taxonomy for the sexist subset. The only dataset source this project
uses (see ``data.py``'s module docstring).

``get_split`` selects the official train/dev/test partition; ``join_categories``
attaches sexism-subtype labels (by shared ``rewire_id``) to any subset of
these rows, e.g. the test split's own predictions, for subtype-level error
analysis.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import _download_file, normalize_labels, verify_sha256

# Commit b1f1fbc of rewire-online/edos (branch: main as of 2026-07-06),
# pinned for the same reason as the SHA-256 verification below: an upstream
# force-push must not silently change what this study evaluates.
SOURCE_URL = (
    "https://raw.githubusercontent.com/rewire-online/edos/"
    "b1f1fbcca9d42de0395432ada8fcfb5c35f68c77/data/edos_labelled_aggregated.csv"
)
FILENAME = "edos_labelled_aggregated.csv"
EXPECTED_SHA256 = "607bdbb766e1803a28f0eb8fc21c14b6b9dfa20039f726ba6767438584b47f9b"

# The 4 top-level sexism categories used for label_category on the sexist
# rows; label_category == "none" for the non-sexist rows.
CATEGORIES = [
    "1. threats, plans to harm and incitement",
    "2. derogation",
    "3. animosity",
    "4. prejudiced discussions",
]


def download_edos_full(
    data_dir: Path | str = "./data",
    force: bool = False,
    source_url: str = SOURCE_URL,
    expected_sha256: str | None = None,
) -> Path:
    """Download the full aggregated EDOS CSV if not already present, verifying
    its SHA-256 (pre-existing files included). ``expected_sha256`` defaults to
    ``EXPECTED_SHA256`` when using the pinned ``SOURCE_URL``; pass ``""`` to
    skip verification for a custom source."""
    if expected_sha256 is None:
        expected_sha256 = EXPECTED_SHA256 if source_url == SOURCE_URL else ""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / FILENAME
    if force or not dest.exists():
        _download_file(source_url, dest)
    if expected_sha256:
        verify_sha256(dest, expected_sha256)
    return dest


def load_edos_full(data_dir: Path | str = "./data") -> pd.DataFrame:
    """Load the full EDOS dataset (columns: rewire_id, text, label_sexist,
    label_category, label_vector, split, plus normalized binary ``label``)."""
    df = pd.read_csv(Path(data_dir) / FILENAME)
    return normalize_labels(df)


def get_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Filter to one of the official splits: 'train', 'dev', or 'test'."""
    if split not in ("train", "dev", "test"):
        raise ValueError(f"split must be one of train/dev/test, got {split!r}")
    return df[df["split"] == split].reset_index(drop=True)


def join_categories(examples_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``label_category``/``label_vector`` to ``examples_df`` by joining
    on ``rewire_id`` against the full release (``examples_df`` is typically a
    subset of ``full_df``, e.g. its own test split).

    ``examples_df`` is often ``get_split(full_df, "test")`` itself (see
    ``analysis.error_breakdown_by_category``'s real call path), which already
    carries these two columns since ``get_split`` only filters rows, never
    drops columns: join() would otherwise raise ``ValueError: columns
    overlap``. Dropping them first (a no-op when absent) means the join
    always wins with the freshly-looked-up values, which are identical to
    whatever was already there for any genuine subset of ``full_df``.
    """
    if "rewire_id" not in examples_df.columns:
        raise ValueError("examples_df has no 'rewire_id' column to join on")
    lookup = full_df.set_index("rewire_id")[["label_category", "label_vector"]]
    examples_df = examples_df.drop(columns=["label_category", "label_vector"], errors="ignore")
    return examples_df.join(lookup, on="rewire_id")
