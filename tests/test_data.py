import pandas as pd

from sexism_prompting import data


def test_normalize_labels_adds_binary_column():
    df = pd.DataFrame({"text": ["a", "b"], "label_sexist": ["sexist", "not sexist"]})
    out = data.normalize_labels(df)
    assert list(out["label"]) == [1, 0]


def test_normalize_labels_noop_if_already_present():
    df = pd.DataFrame({"text": ["a"], "label_sexist": ["sexist"], "label": [0]})
    out = data.normalize_labels(df)
    assert list(out["label"]) == [0]  # untouched, not recomputed from label_sexist


def test_normalize_labels_rejects_unexpected_values():
    import pytest

    df = pd.DataFrame({"text": ["a"], "label_sexist": ["maybe sexist"]})
    with pytest.raises(ValueError, match="maybe sexist"):
        data.normalize_labels(df)


def test_download_file_retries_then_succeeds(tmp_path, monkeypatch):
    import io

    attempts = {"n": 0}

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(url, timeout):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise OSError("flaky network")
        return _Response(b"payload")

    monkeypatch.setattr(data.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(data.time, "sleep", lambda seconds: None)

    dest = tmp_path / "f.csv"
    data._download_file("http://example.test/f.csv", dest)
    assert dest.read_bytes() == b"payload"
    assert attempts["n"] == 3
    assert not dest.with_suffix(dest.suffix + ".part").exists()  # tmp cleaned up


def test_download_file_gives_up_after_retries(tmp_path, monkeypatch):
    import pytest

    def fake_urlopen(url, timeout):
        raise OSError("network down")

    monkeypatch.setattr(data.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(data.time, "sleep", lambda seconds: None)

    dest = tmp_path / "f.csv"
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        data._download_file("http://example.test/f.csv", dest)
    assert not dest.exists()  # no torn destination file left behind


def test_verify_sha256_raises_on_mismatch(tmp_path):
    import pytest

    path = tmp_path / "f.csv"
    path.write_text("content-a")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        data.verify_sha256(path, "0" * 64)
