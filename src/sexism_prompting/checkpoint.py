"""Resumable checkpointing for long, multi-hour runs.

``RunCheckpoint`` keeps a manifest of completed (model, variant) units plus
their raw predictions and metrics, persisted to a directory. A restarted run
skips anything already in the manifest instead of redoing it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

MANIFEST_FILENAME = "manifest.json"
RUN_INFO_FILENAME = "run_info.json"


class RunCheckpoint:
    """Tracks which (model, variant) units are done and persists their output."""

    def __init__(self, checkpoint_dir: str | Path):
        self.dir = Path(checkpoint_dir)
        self.predictions_dir = self.dir / "predictions"
        self.failures_dir = self.dir / "failures"
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.predictions_dir.mkdir(parents=True, exist_ok=True)
            self.failures_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            # A read-only checkpoint dir (opened purely to inspect another
            # run's manifest) may have no write access, and a subdir this
            # class never wrote to (typically failures/) won't exist yet.
            # Fine to ignore: reads only touch the already-loaded manifest
            # or files that must already exist. A real write attempt
            # (mark_done, save_failure_evidence) still raises its own clear
            # OSError.
            pass
        self.manifest_path = self.dir / MANIFEST_FILENAME
        self._manifest: Dict[str, Dict] = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Dict]:
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {}

    def _save_manifest(self) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self._manifest, f, indent=2, sort_keys=True)
        tmp.replace(self.manifest_path)  # atomic on POSIX: avoids a torn manifest file

    @staticmethod
    def unit_id(variant_id: str, model_key: str) -> str:
        return f"{variant_id}__{model_key}"

    def is_done(self, unit_id: str, config_fingerprint: Optional[str] = None) -> bool:
        """True if this unit is checkpointed AND, when checked, its config still matches.

        ``config_fingerprint``, when given, is compared against the one
        stored when the unit was marked done. A mismatch means the prompt,
        generation settings, data, or model has changed since, treated as
        NOT done, so a stale result is never silently reused. Keying by
        `variant_id__model_key` alone can't detect that kind of drift.
        Entries with no stored fingerprint are grandfathered in as trusted,
        so this never retroactively invalidates already-completed work.
        """
        entry = self._manifest.get(unit_id)
        if entry is None:
            return False
        if config_fingerprint is None:
            return True
        stored = entry.get("config_fingerprint")
        if stored is None:
            return True
        return stored == config_fingerprint

    def completed_units(self) -> List[str]:
        return sorted(self._manifest)

    def metrics_for(self, unit_id: str) -> Optional[Dict]:
        return self._manifest.get(unit_id, {}).get("metrics")

    def mark_done(
        self,
        unit_id: str,
        metrics: Dict,
        predictions: Optional[List[str]] = None,
        rewire_ids: Optional[Sequence[str]] = None,
        labels: Optional[Sequence[int]] = None,
        config_fingerprint: Optional[str] = None,
    ) -> None:
        """Record a completed unit's metrics and (optionally) its raw predictions.

        When ``rewire_ids``/``labels`` are given, the predictions file stores a
        dict aligning every response with its example id and gold label, so
        later statistical analysis never has to re-download a byte-identical
        test set just to know which row each response belongs to. Without
        them, the legacy bare-list shape is kept (existing files from earlier
        sessions stay valid either way; see ``load_predictions_normalized``).

        ``config_fingerprint``, when given, is stored alongside the metrics
        so a later ``is_done`` call can detect a stale result; see its
        docstring. Omitted entirely (not just ``null``) when not given, so
        legacy manifest entries and fresh ones written without a fingerprint
        look identical on disk.
        """
        if predictions is not None:
            if rewire_ids is not None or labels is not None:
                payload: object = {
                    "responses": list(predictions),
                    "rewire_ids": list(rewire_ids) if rewire_ids is not None else None,
                    "labels": [int(label) for label in labels] if labels is not None else None,
                }
            else:
                payload = list(predictions)
            with open(self.predictions_dir / f"{unit_id}.json", "w") as f:
                json.dump(payload, f)
        entry = {
            "metrics": metrics,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if config_fingerprint is not None:
            entry["config_fingerprint"] = config_fingerprint
        self._manifest[unit_id] = entry
        self._save_manifest()

    def save_failure_evidence(
        self,
        unit_id: str,
        responses: Sequence[str],
        reason: str,
        rewire_ids: Optional[Sequence[str]] = None,
        labels: Optional[Sequence[int]] = None,
        config_fingerprint: Optional[str] = None,
    ) -> None:
        """Persist a failed unit's raw responses for diagnosis, WITHOUT marking it done.

        A unit whose fail_ratio crosses ``failure_threshold`` is deliberately
        never checkpointed (see ``evaluate.py``), so it retries in a future
        session; this method still persists its raw generation text so the
        failure can be inspected directly instead of needing one-off
        instrumentation and a fresh retry. It writes to a separate
        ``failures/`` directory that ``is_done`` never looks at, so it has
        zero effect on resume/skip behavior.
        """
        self.failures_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "reason": reason,
            "responses": list(responses),
            "rewire_ids": list(rewire_ids) if rewire_ids is not None else None,
            "labels": [int(label) for label in labels] if labels is not None else None,
            "config_fingerprint": config_fingerprint,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with open(self.failures_dir / f"{unit_id}.json", "w") as f:
            json.dump(payload, f)

    def load_predictions(self, unit_id: str) -> List[str]:
        with open(self.predictions_dir / f"{unit_id}.json") as f:
            return json.load(f)

    def load_predictions_normalized(self, unit_id: str) -> Dict:
        """Predictions in a uniform dict shape, whether the file is the legacy
        bare list or the current aligned dict.

        Returns {"responses": [...], "rewire_ids": [...] or None, "labels": [...] or None}.
        """
        raw = self.load_predictions(unit_id)
        if isinstance(raw, dict):
            return {
                "responses": raw.get("responses", []),
                "rewire_ids": raw.get("rewire_ids"),
                "labels": raw.get("labels"),
            }
        return {"responses": raw, "rewire_ids": None, "labels": None}

    def append_run_info(self, info: Dict) -> None:
        """Append one session's environment/config snapshot to ``run_info.json``.

        The file holds a list (one entry per session touching this checkpoint
        dir), so a multi-session run records every GPU/library/config context
        its units were produced under: the traceability the results need for
        the paper's reproducibility claims.
        """
        path = self.dir / RUN_INFO_FILENAME
        entries: List[Dict] = []
        if path.exists():
            try:
                with open(path) as f:
                    entries = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"Warning: could not read existing {path.name} ({e!r}); starting a fresh list.")
        entries.append(info)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(entries, f, indent=2)
        tmp.replace(path)

