import math

from sexism_prompting.metrics import (
    compute_auc,
    compute_metrics,
    compute_metrics_excluding_failures,
    compute_metrics_from_predictions,
    compute_metrics_with_abstention,
    is_failure,
    process_response,
)


def test_process_response_direct_yes_no():
    assert process_response("YES") == 1
    assert process_response("NO") == 0
    assert process_response(" no") == 0
    assert process_response("yes please") == 1


def test_process_response_finds_last_answer_marker():
    """The CoT case: a long reasoning trace, decision only at the very end."""
    response = (
        "Let's think step by step. The text uses derogatory language about women, "
        "which suggests hostility. However it could also be read as neutral.\n"
        "ANSWER: YES"
    )
    assert process_response(response) == 1


def test_process_response_prefers_last_of_multiple_answer_markers():
    response = "ANSWER: NO\nActually, reconsidering...\nANSWER: YES"
    assert process_response(response) == 1


def test_process_response_no_marker_falls_back_to_yes_no_search():
    assert process_response("I think this is YES actually") == 1
    assert process_response("no sexist content here") == 0


def test_process_response_unparseable_defaults_to_zero():
    assert process_response("") == 0
    assert process_response("I cannot answer that.") == 0
    assert process_response(None) == 0


def test_is_failure():
    assert is_failure("YES") is False
    assert is_failure("no") is False
    assert is_failure("I decline to answer") is True
    assert is_failure(None) is True


def test_hedge_mentioning_both_words_is_a_failure_not_a_confident_prediction():
    """is_failure must flag a hedge mentioning both words ("I can't tell if
    this is YES or NO without more context") as a failure. Scanning the
    full raw text for either word appearing anywhere would miss this, and
    process_response would silently pick a label based on which word
    happens to come first, turning a non-answer into a confident,
    wrong-by-construction prediction."""
    hedge = "I can't tell if this is YES or NO without more context."
    assert is_failure(hedge) is True
    assert process_response(hedge) == 0  # ambiguous defaults to 0, not word-order-based


def test_clean_answer_marker_still_wins_even_if_reasoning_mentions_both_words():
    """A real decision after real CoT reasoning must not be treated as
    ambiguous just because the reasoning trace discusses both labels before
    concluding; only responses with NO clean leading marker are ambiguous."""
    response = (
        "This could be read as either YES (sexist) or NO (not sexist) depending on "
        "context, but the derogatory language tips it.\nANSWER: YES"
    )
    assert is_failure(response) is False
    assert process_response(response) == 1


def test_compute_metrics_from_predictions_perfect():
    metrics = compute_metrics_from_predictions([1, 0, 1, 0], [1, 0, 1, 0])
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_compute_metrics_from_predictions_all_wrong():
    metrics = compute_metrics_from_predictions([0, 1, 0, 1], [1, 0, 1, 0])
    assert metrics["accuracy"] == 0.0
    assert metrics["f1"] == 0.0


def test_compute_metrics_empty_is_nan_not_crash():
    import math

    metrics = compute_metrics_from_predictions([], [])
    assert math.isnan(metrics["accuracy"])


def test_compute_metrics_tracks_fail_ratio():
    responses = ["YES", "NO", "I cannot decide", "YES"]
    y_true = [1, 0, 0, 1]
    metrics = compute_metrics(responses, y_true)
    assert metrics["n"] == 4
    assert metrics["n_fail"] == 1
    assert metrics["fail_ratio"] == 0.25
    assert metrics["accuracy"] == 1.0  # the failure defaults to 0, matching y_true=0 here


def test_compute_metrics_handles_long_cot_responses():
    responses = [
        "Reasoning: this text objectifies women.\nANSWER: YES",
        "Reasoning: this is a neutral statement.\nANSWER: NO",
    ]
    metrics = compute_metrics(responses, [1, 0])
    assert metrics["accuracy"] == 1.0
    assert metrics["fail_ratio"] == 0.0


def test_compute_metrics_excluding_failures_drops_unparseable_rows():
    responses = ["YES", "NO", "I cannot decide", "YES"]
    y_true = [1, 0, 1, 0]
    m = compute_metrics_excluding_failures(responses, y_true)
    assert m["n"] == 4
    assert m["n_answered"] == 3
    assert m["coverage"] == 0.75
    # Kept rows: (YES,1)=tp, (NO,0)=tn, (YES,0)=fp -> precision 0.5, recall 1.0
    assert m["precision"] == 0.5
    assert m["recall"] == 1.0
    # Contrast with the headline policy, where the failure defaults to 0 and
    # counts as a (wrong) prediction on y_true=1:
    headline = compute_metrics(responses, y_true)
    assert headline["accuracy"] == 0.5


def test_compute_metrics_rejects_misaligned_lengths():
    import pytest

    with pytest.raises(ValueError):
        compute_metrics(["YES"], [1, 0])
    with pytest.raises(ValueError):
        compute_metrics_excluding_failures(["YES"], [1, 0])


def test_compute_auc_perfect_separation():
    confidences = [0.9, 0.8, 0.1, 0.2]
    y_true = [1, 1, 0, 0]
    assert compute_auc(confidences, y_true) == 1.0


def test_compute_auc_drops_none_confidences():
    confidences = [0.9, None, 0.1, 0.2]
    y_true = [1, 1, 0, 0]
    # Row 2 (y=1) dropped; remaining (0.9,1),(0.1,0),(0.2,0) still perfectly separate.
    assert compute_auc(confidences, y_true) == 1.0


def test_compute_auc_nan_when_only_one_class_survives():
    confidences = [0.9, None, 0.1]
    y_true = [1, 0, 1]  # dropping the None leaves only y=1 rows
    assert math.isnan(compute_auc(confidences, y_true))


def test_compute_auc_nan_on_too_few_examples():
    assert math.isnan(compute_auc([], []))
    assert math.isnan(compute_auc([0.5], [1]))


def test_compute_metrics_without_confidences_omits_auc_key():
    metrics = compute_metrics(["YES", "NO"], [1, 0])
    assert "auc" not in metrics


def test_compute_metrics_with_confidences_adds_auc_key():
    responses = ["YES", "NO", "YES", "NO"]
    y_true = [1, 0, 1, 0]
    confidences = [0.9, 0.1, 0.8, 0.2]
    metrics = compute_metrics(responses, y_true, confidences=confidences)
    assert metrics["auc"] == 1.0
    assert metrics["accuracy"] == 1.0  # unaffected by adding confidences


def test_compute_metrics_rejects_misaligned_confidences():
    import pytest

    with pytest.raises(ValueError):
        compute_metrics(["YES", "NO"], [1, 0], confidences=[0.5])


def test_compute_metrics_with_abstention_zero_band_abstains_only_exact_ties():
    responses = ["YES", "NO", "YES", "YES"]
    y_true = [1, 0, 1, 1]
    confidences = [0.9, 0.1, 0.5, 0.7]
    m = compute_metrics_with_abstention(responses, y_true, confidences, band=0.0)
    assert m["n"] == 4
    assert m["n_abstained"] == 1
    assert m["coverage"] == 0.75
    assert m["accuracy"] == 1.0


def test_compute_metrics_with_abstention_wide_band_abstains_everything():
    responses = ["YES", "NO", "YES", "YES"]
    y_true = [1, 0, 1, 1]
    confidences = [0.9, 0.1, 0.5, 0.7]
    m = compute_metrics_with_abstention(responses, y_true, confidences, band=0.5)
    assert m["n"] == 4
    assert m["n_abstained"] == 4
    assert m["coverage"] == 0.0
    assert math.isnan(m["accuracy"])


def test_compute_metrics_with_abstention_none_confidence_counts_as_abstained():
    responses = ["YES", "NO", "YES"]
    y_true = [1, 0, 1]
    confidences = [0.9, None, 0.8]
    m = compute_metrics_with_abstention(responses, y_true, confidences, band=0.05)
    assert m["n"] == 3
    assert m["n_abstained"] == 1
    assert m["coverage"] == 2 / 3
    assert m["accuracy"] == 1.0


def test_compute_metrics_with_abstention_rejects_misaligned_lengths():
    import pytest

    with pytest.raises(ValueError):
        compute_metrics_with_abstention(["YES"], [1, 0], [0.9, 0.1], band=0.1)
    with pytest.raises(ValueError):
        compute_metrics_with_abstention(["YES", "NO"], [1, 0], [0.9], band=0.1)
