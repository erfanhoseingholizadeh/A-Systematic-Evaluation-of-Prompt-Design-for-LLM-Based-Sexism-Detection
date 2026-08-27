"""Shared download/label utilities used across this project's data loaders.

The only dataset source for this project is the official EDOS release (see
``edos_full.py``); there is no course-derived mini-set loader here. A file
matching `a2_test.csv`/`demonstrations.csv` (mirrored from a University of
Bologna course repository, not the official EDOS release) is an
undocumented, artificially-balanced 50/50 sample with no real EDOS sampling
methodology behind it, and must never be used for anything in this project.

See ``edos_full.py`` for the full ~20k-example official release with train/
dev/test splits and sexism-category labels.
"""

from __future__ import annotations

import time
import shutil
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

from .io_utils import sha256_of_file

LABEL_MAP = {"sexist": 1, "not sexist": 0}


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary ``label`` column derived from ``label_sexist``, if missing."""
    if "label" in df.columns:
        return df
    df = df.copy()
    mapped = df["label_sexist"].map(LABEL_MAP)
    if mapped.isna().any():
        bad = sorted(df.loc[mapped.isna(), "label_sexist"].astype(str).unique())
        raise ValueError(f"unexpected label_sexist values {bad}; expected one of {sorted(LABEL_MAP)}")
    df["label"] = mapped.astype(int)
    return df


def _download_file(url: str, dest: Path, timeout_seconds: int = 60, retries: int = 3) -> None:
    """Download ``url`` to ``dest`` atomically (tmp file + rename), with a
    timeout and retries. A killed or hung session must not leave a torn file
    that a later session then trusts because it "already exists"."""
    last_error: Optional[Exception] = None
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout_seconds) as response, open(tmp, "wb") as f:
                shutil.copyfileobj(response, f)
            tmp.replace(dest)
            return
        except Exception as e:  # noqa: BLE001
            last_error = e
            tmp.unlink(missing_ok=True)
            if attempt < retries:
                print(f"Download attempt {attempt}/{retries} for {url} failed ({e!r}); retrying.")
                time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download {url} after {retries} attempts: {last_error!r}")


def verify_sha256(path: Path, expected: str) -> None:
    """Raise (with a how-to-fix message) unless ``path`` hashes to ``expected``."""
    actual = sha256_of_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{path} sha256 mismatch: expected {expected[:12]}…, got {actual[:12]}…. "
            f"Either the local copy is corrupt (delete it to re-download) or upstream changed "
            f"(re-pin the URL commit + EXPECTED_SHA256 deliberately if so)."
        )


