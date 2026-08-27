import pandas as pd

from sexism_prompting.sft import balance_classes, build_masked_example


def test_prompt_masked_completion_labeled():
    input_ids, labels = build_masked_example([1, 2, 3], [7, 8], max_length=10)
    assert input_ids == [1, 2, 3, 7, 8]
    assert labels == [-100, -100, -100, 7, 8]


def test_boundary_is_exact():
    """The whole point of assembling ids instead of re-tokenizing strings:
    the last prompt token is masked and the first completion token is not,
    no off-by-one regardless of how a tokenizer would merge the boundary."""
    prompt, completion = [5, 6, 7], [9]
    input_ids, labels = build_masked_example(prompt, completion, max_length=512)
    assert labels[len(prompt) - 1] == -100
    assert labels[len(prompt)] == 9


def test_left_truncation_keeps_prompt_tail_and_completion():
    input_ids, labels = build_masked_example(list(range(100)), [7, 8], max_length=10)
    assert len(input_ids) == 10
    assert input_ids[-2:] == [7, 8]
    assert input_ids[:8] == list(range(92, 100))  # the tail (answer cue) survives
    assert labels == [-100] * 8 + [7, 8]


def test_never_all_masked():
    """Even with a zero prompt budget the completion keeps its labels:
    an all -100 example would make the batch loss NaN."""
    input_ids, labels = build_masked_example(list(range(100)), [7], max_length=1)
    assert input_ids == [7]
    assert labels == [7]


def _imbalanced_df(n_majority: int = 76, n_minority: int = 24) -> pd.DataFrame:
    rows = [{"text": f"maj-{i}", "label": 0} for i in range(n_majority)]
    rows += [{"text": f"min-{i}", "label": 1} for i in range(n_minority)]
    return pd.DataFrame(rows)


def test_balance_classes_equalizes_counts():
    balanced = balance_classes(_imbalanced_df(), seed=42)
    counts = balanced["label"].value_counts()
    assert counts[0] == counts[1] == 76


def test_balance_classes_keeps_all_majority_rows():
    df = _imbalanced_df()
    balanced = balance_classes(df, seed=42)
    majority_texts = set(df[df["label"] == 0]["text"])
    kept_majority_texts = set(balanced[balanced["label"] == 0]["text"])
    assert majority_texts == kept_majority_texts


def test_balance_classes_oversamples_minority_with_repeats():
    balanced = balance_classes(_imbalanced_df(), seed=42)
    minority_texts = balanced[balanced["label"] == 1]["text"]
    assert minority_texts.nunique() == 24
    assert len(minority_texts) == 76


def test_balance_classes_deterministic_given_seed():
    df = _imbalanced_df()
    a = balance_classes(df, seed=7)
    b = balance_classes(df, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_balance_classes_noop_shape_when_already_balanced():
    df = _imbalanced_df(n_majority=50, n_minority=50)
    balanced = balance_classes(df, seed=42)
    assert len(balanced) == 100
    assert set(balanced["text"]) == set(df["text"])
