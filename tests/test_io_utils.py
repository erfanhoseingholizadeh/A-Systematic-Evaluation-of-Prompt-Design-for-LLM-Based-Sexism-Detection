import pandas as pd

from sexism_prompting.io_utils import pivot_metric, save_results


def test_save_results_creates_new_file(tmp_path):
    path = tmp_path / "results.csv"
    df = pd.DataFrame([{"variant_id": "V1", "model": "mistral", "f1": 0.7}])
    save_results(df, path, append=False)
    assert path.exists()
    assert len(pd.read_csv(path)) == 1


def test_save_results_appends_by_default(tmp_path):
    path = tmp_path / "results.csv"
    save_results(pd.DataFrame([{"variant_id": "V1", "model": "mistral", "f1": 0.7}]), path)
    save_results(pd.DataFrame([{"variant_id": "V2", "model": "mistral", "f1": 0.8}]), path)
    combined = pd.read_csv(path)
    assert len(combined) == 2
    assert sorted(combined["variant_id"]) == ["V1", "V2"]


def test_save_results_overwrites_when_append_false(tmp_path):
    path = tmp_path / "results.csv"
    save_results(pd.DataFrame([{"variant_id": "V1", "model": "mistral", "f1": 0.7}]), path)
    save_results(pd.DataFrame([{"variant_id": "V2", "model": "mistral", "f1": 0.8}]), path, append=False)
    combined = pd.read_csv(path)
    assert len(combined) == 1
    assert combined["variant_id"].iloc[0] == "V2"


def test_pivot_metric_wide_table():
    df = pd.DataFrame(
        [
            {"variant_id": "V1", "model": "mistral", "f1": 0.7},
            {"variant_id": "V1", "model": "phi3", "f1": 0.6},
            {"variant_id": "V2", "model": "mistral", "f1": 0.75},
            {"variant_id": "V2", "model": "phi3", "f1": 0.65},
        ]
    )
    wide = pivot_metric(df, "f1")
    assert wide.loc["V1", "mistral"] == 0.7
    assert wide.loc["V2", "phi3"] == 0.65
