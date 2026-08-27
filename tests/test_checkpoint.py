import json
import os
import stat

from sexism_prompting.checkpoint import RunCheckpoint


def test_unit_id_format():
    assert RunCheckpoint.unit_id("V1", "mistral") == "V1__mistral"


def test_mark_done_and_is_done(tmp_path):
    cp = RunCheckpoint(tmp_path / "ckpt")
    uid = RunCheckpoint.unit_id("V1", "mistral")
    assert not cp.is_done(uid)

    cp.mark_done(uid, {"f1": 0.75}, predictions=["YES", "NO"])
    assert cp.is_done(uid)
    assert cp.metrics_for(uid) == {"f1": 0.75}
    assert cp.load_predictions(uid) == ["YES", "NO"]


def test_manifest_persists_across_instances(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    cp1 = RunCheckpoint(ckpt_dir)
    uid = RunCheckpoint.unit_id("V1", "mistral")
    cp1.mark_done(uid, {"f1": 0.75})

    # Simulates a resumed/restarted run re-reading the same directory.
    cp2 = RunCheckpoint(ckpt_dir)
    assert cp2.is_done(uid)
    assert cp2.completed_units() == [uid]


def test_completed_units_sorted(tmp_path):
    cp = RunCheckpoint(tmp_path / "ckpt")
    cp.mark_done("V2__mistral", {"f1": 0.1})
    cp.mark_done("V1__mistral", {"f1": 0.2})
    assert cp.completed_units() == ["V1__mistral", "V2__mistral"]


def test_construction_survives_a_read_only_mount_missing_failures_dir(tmp_path):
    """A read-only checkpoint directory with no failures/ subdir (e.g. one
    read cross-run by the sensitivity-sweep code, never written to)
    constructs without raising PermissionError from failures_dir.mkdir()
    before any manifest read can happen."""
    ckpt_dir = tmp_path / "ckpt"
    (ckpt_dir / "predictions").mkdir(parents=True)
    (ckpt_dir / "manifest.json").write_text(json.dumps({"V1__mistral": {"metrics": {"f1": 0.5}}}))
    # No failures/: exactly what a real read-only mounted checkpoint looks like.

    original_modes = {}
    for dirpath, dirnames, filenames in os.walk(ckpt_dir):
        original_modes[dirpath] = os.stat(dirpath).st_mode
        os.chmod(dirpath, stat.S_IRUSR | stat.S_IXUSR)  # read + traverse only
    try:
        cp = RunCheckpoint(ckpt_dir)
        assert cp.completed_units() == ["V1__mistral"]
        assert cp.metrics_for("V1__mistral") == {"f1": 0.5}
        assert not (ckpt_dir / "failures").exists()  # never created: no write access
    finally:
        for dirpath, mode in original_modes.items():
            os.chmod(dirpath, mode)


def test_mark_done_with_alignment_metadata(tmp_path):
    cp = RunCheckpoint(tmp_path / "ckpt")
    cp.mark_done("V1__m", {"f1": 1.0}, predictions=["YES", "NO"], rewire_ids=["id1", "id2"], labels=[1, 0])

    norm = cp.load_predictions_normalized("V1__m")
    assert norm == {"responses": ["YES", "NO"], "rewire_ids": ["id1", "id2"], "labels": [1, 0]}

    manifest = json.loads((tmp_path / "ckpt" / "manifest.json").read_text())
    assert manifest["V1__m"]["metrics"] == {"f1": 1.0}
    assert "completed_at_utc" in manifest["V1__m"]


def test_load_predictions_normalized_handles_legacy_bare_list(tmp_path):
    """Prediction files from sessions before the alignment change are bare
    lists; they must stay loadable alongside the new dict shape."""
    cp = RunCheckpoint(tmp_path / "ckpt")
    cp.mark_done("V1__m", {"f1": 1.0}, predictions=["YES"])
    assert cp.load_predictions("V1__m") == ["YES"]
    assert cp.load_predictions_normalized("V1__m") == {"responses": ["YES"], "rewire_ids": None, "labels": None}


def test_is_done_detects_stale_config_fingerprint(tmp_path):
    """A unit marked done under one config must be treated as pending again
    once the prompt/generation/data/model config changes. Keying by
    variant_id__model_key alone can't detect that kind of change, so
    without the fingerprint check a stale result would be silently reused
    forever."""
    cp = RunCheckpoint(tmp_path / "ckpt")
    uid = RunCheckpoint.unit_id("V1", "mistral")
    cp.mark_done(uid, {"f1": 0.75}, config_fingerprint="fp-old")

    assert cp.is_done(uid, "fp-old") is True
    assert cp.is_done(uid, "fp-new") is False  # config changed since: stale, must retry
    assert cp.is_done(uid) is True  # no fingerprint check requested: unaffected, backward compatible


def test_is_done_grandfathers_in_legacy_entries_without_a_stored_fingerprint(tmp_path):
    """Entries with no stored config fingerprint at all (written before
    fingerprinting was tracked) must not be retroactively invalidated."""
    cp = RunCheckpoint(tmp_path / "ckpt")
    uid = RunCheckpoint.unit_id("V1", "mistral")
    cp.mark_done(uid, {"f1": 0.75})  # no config_fingerprint: a legacy-shaped entry

    assert cp.is_done(uid, "fp-anything") is True  # trusted, not nuked


def test_save_failure_evidence_persists_without_marking_done(tmp_path):
    cp = RunCheckpoint(tmp_path / "ckpt")
    uid = RunCheckpoint.unit_id("V13", "llama3.1")

    cp.save_failure_evidence(
        uid, ["hmm???", "not sure", "YES"], reason="fail_ratio=0.667 >= threshold=0.5",
        rewire_ids=["r1", "r2", "r3"], labels=[1, 0, 1],
    )

    assert not cp.is_done(uid)  # must not gate resume/skip
    saved = json.loads((tmp_path / "ckpt" / "failures" / f"{uid}.json").read_text())
    assert saved["responses"] == ["hmm???", "not sure", "YES"]
    assert saved["reason"] == "fail_ratio=0.667 >= threshold=0.5"
    assert saved["rewire_ids"] == ["r1", "r2", "r3"]
    assert saved["labels"] == [1, 0, 1]


def test_append_run_info_accumulates(tmp_path):
    cp = RunCheckpoint(tmp_path / "ckpt")
    cp.append_run_info({"session": 1})
    cp.append_run_info({"session": 2})
    entries = json.loads((tmp_path / "ckpt" / "run_info.json").read_text())
    assert [e["session"] for e in entries] == [1, 2]


