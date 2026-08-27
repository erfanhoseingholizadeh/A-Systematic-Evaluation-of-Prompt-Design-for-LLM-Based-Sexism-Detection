"""Subprocess smoke tests for every CLI entry point in scripts/.

Pytest/pyflakes alone miss a real class of bug: a lazy import (inside a
function body or after an early guard, rather than at module top level)
referencing a name that no longer exists in the target module. Pyflakes only
checks names within the file it's scanning, not whether an imported name
actually exists where it's imported from, so that class of breakage is
invisible to static analysis and only surfaces when the code path runs.
These tests run every entry point (as a subprocess, so a crash can't take
the test process down with it) far enough to exercise its real import
chain, without needing a GPU.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# scripts/*.py: real argparse CLIs, runnable locally. `--help` parses args
# and imports the whole module without needing a GPU, a network connection,
# or real data on disk.
SCRIPTS_WITH_HELP = [
    "scripts/download_data.py",
    "scripts/error_analysis_by_subtype.py",
    "scripts/run_experiment.py",
    "scripts/run_on_cluster.py",
    "scripts/analyze_results.py",
    "scripts/run_finetune.py",
    "scripts/run_extra_sweep.py",
    "scripts/run_extra_sweep_on_cluster.py",
    "scripts/run_classical_baseline.py",
]

def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("script", SCRIPTS_WITH_HELP)
def test_script_help_runs_cleanly(script: str) -> None:
    result = _run([script, "--help"])
    assert result.returncode == 0, f"{script} --help failed:\n{result.stdout}\n{result.stderr}"
    assert "Traceback" not in result.stderr, f"{script} --help raised:\n{result.stderr}"
    assert "usage:" in result.stdout.lower()
