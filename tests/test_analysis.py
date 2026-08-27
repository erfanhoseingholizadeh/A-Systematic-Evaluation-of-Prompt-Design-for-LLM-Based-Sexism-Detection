import json
import math

import pandas as pd
import pytest

from sexism_prompting.analysis import (
    aggregate_confidence_capture,
    analyze_checkpoint,
    bootstrap_metric_cis,
    classify_cot_failure_causes,
    error_breakdown_by_category,
    holm_correction,
    mcnemar_exact_p,
    paired_bootstrap_metric_diff,
    summarize_confidence_capture_by_model,
)
from sexism_prompting.checkpoint import RunCheckpoint
from sexism_prompting.metrics import compute_metrics


def test_mcnemar_exact_p_hand_computed():
    assert mcnemar_exact_p(0, 0) == 1.0
    # 5 discordant pairs all favoring one side: 2 * 0.5^5 = 0.0625
    assert mcnemar_exact_p(5, 0) == pytest.approx(0.0625)
    assert mcnemar_exact_p(0, 5) == pytest.approx(0.0625)
    # Symmetric discordance is never significant (doubled tail clamps to 1)
    assert mcnemar_exact_p(1, 1) == 1.0


def test_holm_correction_hand_computed():
    # sorted: 0.01, 0.03, 0.04 -> adjusted: 3*0.01, 2*0.03, max(0.06, 1*0.04)
    assert holm_correction([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_correction([0.5]) == [0.5]


def test_bootstrap_cis_degenerate_perfect_predictions():
    preds = [1, 0] * 10
    cis = bootstrap_metric_cis(preds, preds, n_iters=200, seed=0)
    # accuracy is 1.0 in every possible resample; f1 can drop to 0 only in a
    # resample containing no positives (P = 0.5^20 here), so pin the upper
    # bound and sanity-check the lower one.
    assert cis["accuracy"] == (1.0, 1.0)
    assert cis["f1"][1] == 1.0
    assert cis["f1"][0] >= 0.0


def test_bootstrap_cis_are_deterministic_given_seed():
    a = bootstrap_metric_cis([1, 0, 1, 1], [1, 0, 0, 1], n_iters=100, seed=7)
    b = bootstrap_metric_cis([1, 0, 1, 1], [1, 0, 0, 1], n_iters=100, seed=7)
    assert a == b


def test_paired_bootstrap_metric_diff_identical_units_is_zero():
    preds = [1, 0, 1, 1, 0, 0, 1, 0]
    labels = [1, 0, 0, 1, 0, 1, 1, 0]
    result = paired_bootstrap_metric_diff(preds, preds, labels, n_iters=200, seed=0)
    assert result["diff"] == 0.0
    assert result["ci_low"] == 0.0
    assert result["ci_high"] == 0.0
    assert result["p_value"] == 1.0


def test_paired_bootstrap_metric_diff_detects_a_real_improvement():
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 3
    baseline_preds = [0] * len(labels)  # always predicts "not sexist"
    perfect_preds = labels  # perfect classifier
    result = paired_bootstrap_metric_diff(baseline_preds, perfect_preds, labels, n_iters=500, seed=0)
    assert result["diff"] > 0.5
    assert result["ci_low"] > 0.0  # CI excludes 0
    assert result["p_value"] < 0.05


def test_paired_bootstrap_metric_diff_rejects_misaligned_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap_metric_diff([1, 0], [1, 0, 1], [1, 0, 1])


def _make_checkpoint(tmp_path):
    cp = RunCheckpoint(tmp_path / "ckpt")
    labels = [1, 0, 1, 0, 1, 0]
    # Baseline: 4/6 correct. Variant: perfect.
    cp.mark_done(
        "V1__m", {"f1": 0.0},
        predictions=["YES", "NO", "NO", "YES", "YES", "NO"],
        rewire_ids=[f"id{i}" for i in range(6)], labels=labels,
    )
    cp.mark_done(
        "V2__m", {"f1": 0.0},
        predictions=["YES", "NO", "YES", "NO", "YES", "NO"],
        rewire_ids=[f"id{i}" for i in range(6)], labels=labels,
    )
    return cp


def test_analyze_checkpoint_end_to_end(tmp_path):
    _make_checkpoint(tmp_path)
    metrics_df, mcnemar_df = analyze_checkpoint(tmp_path / "ckpt", n_bootstrap=100, seed=0)

    assert set(metrics_df["unit_id"]) == {"V1__m", "V2__m"}
    v2 = metrics_df[metrics_df["unit_id"] == "V2__m"].iloc[0]
    assert v2["accuracy"] == 1.0
    assert v2["excl_coverage"] == 1.0
    assert v2["f1_ci_low"] <= v2["f1"] <= v2["f1_ci_high"]

    assert len(mcnemar_df) == 1
    row = mcnemar_df.iloc[0]
    assert row["variant_id"] == "V2"
    assert row["baseline_only_correct"] == 0
    assert row["variant_only_correct"] == 2  # the two rows V1 flipped
    assert 0 < row["p_exact"] <= 1
    assert row["p_holm"] >= row["p_exact"]
    # V2 is perfect, V1 (baseline) is 4/6: F1 diff should be positive and
    # its CI/p-value present alongside McNemar's accuracy-based test.
    assert row["f1_diff"] > 0
    assert row["f1_diff_ci_low"] <= row["f1_diff"] <= row["f1_diff_ci_high"]
    assert 0 < row["f1_diff_p_boot"] <= 1
    assert row["f1_diff_p_holm"] >= row["f1_diff_p_boot"]


def test_error_breakdown_by_category(tmp_path):
    cp = RunCheckpoint(tmp_path / "ckpt")
    rewire_ids = ["r1", "r2", "r3", "r4"]
    labels = [1, 1, 0, 0]
    # r1: sexist/derogation, predicted NO -> false negative
    # r2: sexist/animosity, predicted YES -> correct
    # r3: not sexist, predicted YES -> false positive
    # r4: not sexist, predicted NO -> correct
    cp.mark_done(
        "V1__m", {"f1": 0.0},
        predictions=["NO", "YES", "YES", "NO"],
        rewire_ids=rewire_ids, labels=labels,
    )
    full_df = pd.DataFrame({
        "rewire_id": rewire_ids,
        "label_category": ["2. derogation", "3. animosity", "none", "none"],
        "label_vector": ["x"] * 4,
    })
    examples_df = pd.DataFrame({"rewire_id": rewire_ids})

    breakdown = error_breakdown_by_category(tmp_path / "ckpt", examples_df, full_df)

    assert set(breakdown["category"]) == {"2. derogation", "3. animosity", "none"}
    derog = breakdown[breakdown["category"] == "2. derogation"].iloc[0]
    assert derog["false_negatives"] == 1
    assert derog["fn_rate"] == 1.0
    none_row = breakdown[breakdown["category"] == "none"].iloc[0]
    assert none_row["false_positives"] == 1
    assert none_row["fp_rate"] == 0.5


def test_error_breakdown_by_category_skips_legacy_units_without_rewire_ids(tmp_path):
    cp = RunCheckpoint(tmp_path / "ckpt")
    cp.mark_done("V1__m", {"f1": 0.0}, predictions=["YES", "NO"])  # legacy bare list, no rewire_ids

    breakdown = error_breakdown_by_category(
        tmp_path / "ckpt",
        pd.DataFrame({"rewire_id": []}),
        pd.DataFrame({"rewire_id": [], "label_category": [], "label_vector": []}),
        fallback_labels=[1, 0],
    )
    assert breakdown.empty


def test_analyze_checkpoint_legacy_unit_uses_fallback_labels(tmp_path):
    cp = RunCheckpoint(tmp_path / "ckpt")
    cp.mark_done("V1__m", {"f1": 0.0}, predictions=["YES", "NO"])  # legacy: no embedded labels

    # Without a fallback the unit is skipped...
    metrics_df, _ = analyze_checkpoint(tmp_path / "ckpt", fallback_labels=None)
    assert metrics_df.empty

    # ...with one, it is scored.
    metrics_df, _ = analyze_checkpoint(tmp_path / "ckpt", fallback_labels=[1, 0], n_bootstrap=50)
    assert len(metrics_df) == 1
    assert metrics_df.iloc[0]["accuracy"] == 1.0


def _write_confidence_capture_unit(out_dir, unit_id, model, variant, responses, confidences, y_true, live_agreement):
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(responses, y_true, confidences=confidences)
    with open(out_dir / f"{unit_id}.json", "w") as f:
        json.dump(
            {
                "unit_id": unit_id,
                "model": model,
                "variant": variant,
                "n": len(responses),
                "n_empty": sum(1 for r in responses if r == ""),
                "n_none_confidence": sum(1 for c in confidences if c is None),
                "agreement_rate": live_agreement,
                "metrics": metrics,
                "responses": responses,
                "confidences": confidences,
                "y_true": y_true,
            },
            f,
        )


def test_aggregate_confidence_capture_corrects_tie_and_failure_disagreement(tmp_path):
    # Row 0: exact confidence==0.5 tie where the model actually said YES -- the
    # documented "CHECK flag" case, miscounted as disagreement by the run
    # script's own strict `>` comparison. Row 3: a genuine parse failure
    # ("GARBLED" has no YES/NO), defaults to a NO prediction; its confidence
    # (restricted to YES/NO logits) still leans YES, a real but
    # already-understood limitation, not a bug.
    responses = ["YES", "NO", "YES", "GARBLED"]
    confidences = [0.5, 0.3, 0.9, 0.6]
    y_true = [1, 0, 1, 0]
    _write_confidence_capture_unit(
        tmp_path, "V1__testmodel", "testmodel", "V1", responses, confidences, y_true, live_agreement=0.6667
    )

    units_df = aggregate_confidence_capture(tmp_path)
    assert len(units_df) == 1
    row = units_df.iloc[0]

    assert row["agreement_rate_live"] == 0.6667  # passed through unchanged, not trusted as-is
    assert row["agreement_rate_corrected"] == pytest.approx(0.75)  # row 0's tie now counts as agreement
    assert row["agreement_rate_excl_failures"] == pytest.approx(1.0)  # row 3 dropped from the denominator
    assert row["auc"] == pytest.approx(compute_metrics(responses, y_true, confidences=confidences)["auc"])


def test_summarize_confidence_capture_by_model_selects_f1_argmax_and_flags_classical_beat(tmp_path):
    # modelA: V1's responses are all wrong (f1=0.0) but its confidence scores
    # rank perfectly (auc=1.0); V2's responses are all correct (f1=1.0) but
    # its confidence scores are uninformative ties (auc=0.5). Selection must
    # follow F1 (this project's existing "best variant by F1" convention),
    # not silently switch to AUC just because this is a new subsection --
    # picking by AUC here would wrongly choose V1.
    _write_confidence_capture_unit(
        tmp_path, "V1__modelA", "modelA", "V1",
        ["NO", "YES"], [0.9, 0.1], [1, 0], live_agreement=1.0,
    )
    _write_confidence_capture_unit(
        tmp_path, "V2__modelA", "modelA", "V2",
        ["YES", "NO", "YES", "NO"], [0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0], live_agreement=1.0,
    )
    _write_confidence_capture_unit(
        tmp_path, "V1__modelB", "modelB", "V1",
        ["YES", "NO", "YES", "NO"], [0.6, 0.4, 0.6, 0.4], [1, 0, 1, 0], live_agreement=1.0,
    )

    units_df = aggregate_confidence_capture(tmp_path)
    by_model = summarize_confidence_capture_by_model(units_df, classical_auc=0.70)

    assert list(by_model["model"]) == ["modelA", "modelB"]
    a_row = by_model[by_model["model"] == "modelA"].iloc[0]
    assert a_row["best_covered_variant"] == "V2"  # f1=1.0 beats V1's f1=0.0, despite V1's higher auc
    assert a_row["f1"] == pytest.approx(1.0)
    assert a_row["auc"] == pytest.approx(0.5)
    assert a_row["beats_classical_auc"] == (a_row["auc"] > 0.70)


def test_classify_cot_failure_causes_splits_refusal_vs_other(tmp_path):
    cp = RunCheckpoint(tmp_path / "ckpt")
    refusal_a = "I cannot create content that is discriminatory or hateful."
    refusal_b = "I'm not able to assist with that request."
    other_a = "1. Analyzing the text for sexist content: the phrase implies"
    responses = ["YES", "NO", refusal_a, refusal_b, other_a]
    cp.mark_done("V8__testmodel", {"f1": 0.0}, predictions=responses)

    result = classify_cot_failure_causes(tmp_path / "ckpt", ["V8__testmodel"])

    assert len(result) == 1
    row = result.iloc[0]
    assert row["n"] == 5
    assert row["n_fail"] == 3
    assert row["fail_ratio"] == pytest.approx(0.6)
    assert row["refusal_share"] == pytest.approx(2 / 3)
    assert row["other_share"] == pytest.approx(1 / 3)
    assert row["mean_refusal_length"] == pytest.approx((len(refusal_a) + len(refusal_b)) / 2)
    assert row["mean_other_length"] == pytest.approx(len(other_a))


def test_classify_cot_failure_causes_handles_zero_failures(tmp_path):
    cp = RunCheckpoint(tmp_path / "ckpt")
    cp.mark_done("V6__testmodel", {"f1": 0.5}, predictions=["YES", "NO", "YES"])

    result = classify_cot_failure_causes(tmp_path / "ckpt", ["V6__testmodel"])

    row = result.iloc[0]
    assert row["n_fail"] == 0
    assert row["fail_ratio"] == 0.0
    assert math.isnan(row["refusal_share"])
    assert math.isnan(row["other_share"])
