#!/usr/bin/env python3
"""Download the official EDOS release (the only dataset source this project uses)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sexism_prompting.edos_full import download_edos_full  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--force", action="store_true", help="re-download even if the file already exists")
    args = parser.parse_args()

    full_path = download_edos_full(Path(args.data_dir) / "edos_full", force=args.force)
    print(f"Full EDOS release: {full_path}")


if __name__ == "__main__":
    main()
