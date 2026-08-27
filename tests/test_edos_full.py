import pandas as pd
import pytest

from sexism_prompting import edos_full


def _make_full_df():
    return pd.DataFrame(
        {
            "rewire_id": ["a", "b", "c", "d"],
            "text": ["t1", "t2", "t3", "t4"],
            "label_sexist": ["sexist", "not sexist", "sexist", "not sexist"],
            "label_category": ["2. derogation", "none", "3. animosity", "none"],
            "label_vector": ["2.1", "none", "3.1", "none"],
            "split": ["train", "train", "dev", "test"],
        }
    )


def test_load_edos_full_normalizes_labels(tmp_path, monkeypatch):
    _make_full_df().to_csv(tmp_path / edos_full.FILENAME, index=False)
    df = edos_full.load_edos_full(tmp_path)
    assert list(df["label"]) == [1, 0, 1, 0]


def test_get_split_filters_correctly():
    df = _make_full_df()
    train = edos_full.get_split(df, "train")
    assert list(train["rewire_id"]) == ["a", "b"]
    dev = edos_full.get_split(df, "dev")
    assert list(dev["rewire_id"]) == ["c"]


def test_get_split_rejects_invalid_split_name():
    with pytest.raises(ValueError):
        edos_full.get_split(_make_full_df(), "validation")


def test_join_categories_attaches_subtype_labels():
    full_df = _make_full_df()
    examples_df = pd.DataFrame({"rewire_id": ["a", "d"], "text": ["t1", "t4"], "label": [1, 0]})
    joined = edos_full.join_categories(examples_df, full_df)
    assert joined.loc[joined["rewire_id"] == "a", "label_category"].iloc[0] == "2. derogation"
    assert joined.loc[joined["rewire_id"] == "d", "label_category"].iloc[0] == "none"


def test_join_categories_handles_examples_df_that_already_has_the_columns():
    # Mirrors the real call path (analysis.error_breakdown_by_category):
    # examples_df is get_split(full_df, ...), which already carries
    # label_category/label_vector since get_split only filters rows;
    # join_categories must handle that column overlap instead of raising
    # "ValueError: columns overlap but no suffix specified".
    full_df = _make_full_df()
    examples_df = edos_full.get_split(full_df, "train")
    joined = edos_full.join_categories(examples_df, full_df)
    assert joined.loc[joined["rewire_id"] == "a", "label_category"].iloc[0] == "2. derogation"
    assert joined.loc[joined["rewire_id"] == "b", "label_category"].iloc[0] == "none"


def test_join_categories_requires_rewire_id():
    with pytest.raises(ValueError):
        edos_full.join_categories(pd.DataFrame({"text": ["x"]}), _make_full_df())


def test_download_edos_full_skips_existing(tmp_path, monkeypatch):
    calls = []

    def _download(url, dest, **kwargs):
        calls.append(url)
        _make_full_df().to_csv(dest, index=False)

    monkeypatch.setattr(edos_full, "_download_file", _download)
    edos_full.download_edos_full(tmp_path, expected_sha256="")
    edos_full.download_edos_full(tmp_path, expected_sha256="")
    assert len(calls) == 1


def test_download_edos_full_verifies_checksum(tmp_path, monkeypatch):
    import hashlib

    monkeypatch.setattr(edos_full, "_download_file", lambda url, dest, **kwargs: dest.write_text("abc"))
    good = hashlib.sha256(b"abc").hexdigest()

    edos_full.download_edos_full(tmp_path, expected_sha256=good)  # matches

    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        edos_full.download_edos_full(tmp_path, force=True, expected_sha256="0" * 64)
