"""Post-hoc statistical analysis of saved Stage 1 predictions.

Computes statistical significance and confidence-interval reporting
directly from the checkpoint's raw per-example responses: no GPU, models,
or re-runs needed.

- per-unit metrics under BOTH failure policies (headline: failures default
  to "not sexist"; excluded: failures dropped, coverage reported); see
  ``metrics.compute_metrics_excluding_failures`` for why both matter;
- percentile bootstrap confidence intervals per unit;
- exact McNemar tests between each prompt variant and a baseline variant on
  the same test rows, Holm-corrected within each model.

Seed-stability analysis across few-shot demonstration seeds is the planned
next addition once multi-seed predictions exist; nothing here precludes it.

Pure Python + numpy/pandas, so the whole module is unit-testable in the
GPU-less dev sandbox.
"""

from __future__ import annotations

import json
import re
from math import comb
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .checkpoint import RunCheckpoint
from .metrics import (
    compute_metrics,
    compute_metrics_excluding_failures,
    compute_metrics_from_predictions,
    is_failure,
    process_response,
)


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant-pair counts.

    ``b`` = rows the baseline got right and the variant got wrong,
    ``c`` = rows the variant got right and the baseline got wrong.
    Exact binomial on b out of (b+c) with p=0.5, two-sided by doubling the
    smaller tail (the standard convention), clamped to 1.0.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * tail)


def holm_correction(p_values: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values, in the input order."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * p_values[idx])
        running_max = max(running_max, adj)  # enforce monotonicity
        adjusted[idx] = running_max
    return adjusted


def bootstrap_metric_cis(
    preds: Sequence[int],
    y_true: Sequence[int],
    metrics: Sequence[str] = ("f1", "accuracy"),
    n_iters: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Tuple[float, float]]:
    """Percentile bootstrap CIs for several metrics from one set of resamples."""
    preds_arr = np.asarray(list(preds), dtype=int)
    y_arr = np.asarray(list(y_true), dtype=int)
    n = len(y_arr)
    rng = np.random.default_rng(seed)

    samples: Dict[str, List[float]] = {m: [] for m in metrics}
    for _ in range(n_iters):
        idx = rng.integers(0, n, n)
        resampled = compute_metrics_from_predictions(preds_arr[idx], y_arr[idx])
        for m in metrics:
            samples[m].append(resampled[m])

    out = {}
    for m in metrics:
        lo, hi = np.percentile(samples[m], [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[m] = (float(lo), float(hi))
    return out


def paired_bootstrap_metric_diff(
    preds_a: Sequence[int],
    preds_b: Sequence[int],
    y_true: Sequence[int],
    metric: str = "f1",
    n_iters: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Paired percentile-bootstrap CI + two-sided p-value for the difference in
    one metric (``metric``, ``b - a``) between two units scored on the same
    test rows.

    McNemar (above) tests accuracy via discordant-pair counts; this is the
    complementary F1-aligned test. Each iteration resamples row indices
    ONCE and applies that same resample to both prediction sets, so the
    comparison stays paired (not an independent two-sample bootstrap), the
    same principle McNemar's pairing already relies on, just extended to a
    metric (F1) that isn't a simple pairwise correct/incorrect count.
    """
    a = np.asarray(list(preds_a), dtype=int)
    b = np.asarray(list(preds_b), dtype=int)
    y = np.asarray(list(y_true), dtype=int)
    if len(a) != len(b) or len(a) != len(y):
        raise ValueError("preds_a, preds_b, and y_true must be the same length (same test rows)")
    n = len(y)
    rng = np.random.default_rng(seed)

    diffs = np.empty(n_iters)
    for i in range(n_iters):
        idx = rng.integers(0, n, n)
        diffs[i] = (
            compute_metrics_from_predictions(b[idx], y[idx])[metric]
            - compute_metrics_from_predictions(a[idx], y[idx])[metric]
        )

    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Two-sided achieved significance: double whichever tail (below/above 0) is smaller,
    # the same doubling convention mcnemar_exact_p uses above.
    p_value = min(1.0, 2 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0))))
    return {"diff": float(np.mean(diffs)), "ci_low": float(lo), "ci_high": float(hi), "p_value": p_value}


def _load_units(
    checkpoint: RunCheckpoint, fallback_labels: Optional[List[int]]
) -> Dict[str, Dict]:
    """Load every completed unit's responses/preds/labels, preferring the
    labels embedded in the predictions file (exact alignment by construction)
    and falling back to ``fallback_labels`` for legacy bare-list files."""
    units: Dict[str, Dict] = {}
    for uid in checkpoint.completed_units():
        variant_id, model = uid.split("__", 1)
        try:
            norm = checkpoint.load_predictions_normalized(uid)
        except FileNotFoundError:
            print(f"Warning: no predictions file for '{uid}' (metrics-only unit?); skipping it.")
            continue
        responses = norm["responses"]
        labels = norm["labels"] if norm["labels"] is not None else fallback_labels
        if labels is None:
            print(f"Warning: '{uid}' has no embedded labels and no --data-dir fallback was usable; skipping it.")
            continue
        if len(labels) != len(responses):
            print(f"Warning: '{uid}' has {len(responses)} responses vs {len(labels)} labels; skipping it.")
            continue
        units[uid] = {
            "variant_id": variant_id,
            "model": model,
            "responses": responses,
            "labels": list(labels),
            "preds": [process_response(r) for r in responses],
            "rewire_ids": norm["rewire_ids"],
        }
    return units


def analyze_checkpoint(
    checkpoint_dir: str | Path,
    fallback_labels: Optional[List[int]] = None,
    baseline_variant: str = "V1",
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Full post-hoc analysis of one checkpoint directory.

    Returns ``(metrics_df, mcnemar_df)``:

    - ``metrics_df``: one row per unit, headline metrics, failures-excluded
      metrics (prefixed ``excl_``), coverage, and bootstrap CIs for f1 and
      accuracy under the headline policy.
    - ``mcnemar_df``: one row per (model, variant != baseline) pair with the
      accuracy-based discordant counts + exact McNemar p-value (Holm-adjusted
      within each model's family) AND the paired-bootstrap F1-difference
      (``f1_diff``, its CI, and its own Holm-adjusted p-value,
      ``f1_diff_p_holm``): two independent tests of the same
      baseline-vs-variant comparison, each Holm-corrected within its own
      family rather than pooled together.
    """
    checkpoint = RunCheckpoint(checkpoint_dir)
    units = _load_units(checkpoint, fallback_labels)

    metric_rows = []
    for uid, unit in sorted(units.items()):
        headline = compute_metrics(unit["responses"], unit["labels"])
        excluded = compute_metrics_excluding_failures(unit["responses"], unit["labels"])
        cis = bootstrap_metric_cis(unit["preds"], unit["labels"], n_iters=n_bootstrap, seed=seed)
        metric_rows.append({
            "unit_id": uid,
            "variant_id": unit["variant_id"],
            "model": unit["model"],
            **headline,
            **{f"excl_{k}": v for k, v in excluded.items() if k not in ("n",)},
            "f1_ci_low": cis["f1"][0],
            "f1_ci_high": cis["f1"][1],
            "accuracy_ci_low": cis["accuracy"][0],
            "accuracy_ci_high": cis["accuracy"][1],
        })
    metrics_df = pd.DataFrame(metric_rows)

    mcnemar_rows = []
    models = sorted({u["model"] for u in units.values()})
    for model in models:
        base_uid = f"{baseline_variant}__{model}"
        if base_uid not in units:
            print(f"Warning: baseline unit '{base_uid}' not available; skipping McNemar for model '{model}'.")
            continue
        base = units[base_uid]
        base_correct = np.asarray(base["preds"]) == np.asarray(base["labels"])

        family = []
        for uid, unit in sorted(units.items()):
            if unit["model"] != model or unit["variant_id"] == baseline_variant:
                continue
            if base["rewire_ids"] is None or unit["rewire_ids"] is None:
                print(f"Warning: '{uid}' or '{base_uid}' has no stored rewire_ids; cannot verify row alignment, skipping the pair.")
                continue
            if unit["rewire_ids"] != base["rewire_ids"]:
                print(f"Warning: '{uid}' rewire_ids differ from '{base_uid}'; not the same test rows, skipping the pair.")
                continue
            variant_correct = np.asarray(unit["preds"]) == np.asarray(unit["labels"])
            b = int(np.sum(base_correct & ~variant_correct))
            c = int(np.sum(~base_correct & variant_correct))
            f1_diff = paired_bootstrap_metric_diff(
                base["preds"], unit["preds"], unit["labels"],
                metric="f1", n_iters=n_bootstrap, seed=seed,
            )
            family.append({
                "model": model,
                "baseline": baseline_variant,
                "variant_id": unit["variant_id"],
                "baseline_only_correct": b,
                "variant_only_correct": c,
                "n_discordant": b + c,
                "p_exact": mcnemar_exact_p(b, c),
                "f1_diff": f1_diff["diff"],
                "f1_diff_ci_low": f1_diff["ci_low"],
                "f1_diff_ci_high": f1_diff["ci_high"],
                "f1_diff_p_boot": f1_diff["p_value"],
            })
        if family:
            # McNemar (accuracy) and the F1-diff bootstrap test the same
            # baseline-vs-variant hypothesis with different statistics.
            # Holm-corrected as two separate families within this model
            # rather than pooled, so neither test's correction is diluted by
            # the other's p-values.
            adjusted = holm_correction([row["p_exact"] for row in family])
            f1_adjusted = holm_correction([row["f1_diff_p_boot"] for row in family])
            for row, p_holm, f1_p_holm in zip(family, adjusted, f1_adjusted):
                row["p_holm"] = p_holm
                row["f1_diff_p_holm"] = f1_p_holm
            mcnemar_rows.extend(family)

    return metrics_df, pd.DataFrame(mcnemar_rows)


def error_breakdown_by_category(
    checkpoint_dir: str | Path,
    examples_df: pd.DataFrame,
    full_df: pd.DataFrame,
    fallback_labels: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Per (model, variant, EDOS Task-B category) false-positive/negative
    breakdown, from predictions already sitting in the checkpoint: no GPU,
    no new generation.

    Joins each unit's stored ``rewire_ids`` against
    ``edos_full.join_categories``'s ``label_category`` (``"none"`` for
    genuinely non-sexist rows, one of the 4 sexism subtypes otherwise), so
    the "none" rows carry every false positive (predicted sexist, actually
    not) and each real subtype carries its own false-negative rate
    (predicted not-sexist, actually that subtype): an error breakdown by
    EDOS subtype and common false-positive/false-negative pattern.

    Units whose predictions file has no stored ``rewire_ids`` (the legacy
    bare-list shape, from before aligned predictions existed) are skipped
    with a warning; there is no way to recover which row is which without
    them.
    """
    from .edos_full import join_categories

    checkpoint = RunCheckpoint(checkpoint_dir)
    units = _load_units(checkpoint, fallback_labels)
    category_by_rewire_id = join_categories(examples_df, full_df).set_index("rewire_id")["label_category"]

    rows = []
    for uid, unit in sorted(units.items()):
        if unit["rewire_ids"] is None:
            print(f"Warning: '{uid}' has no stored rewire_ids (legacy predictions file?); skipping category breakdown.")
            continue
        try:
            categories = category_by_rewire_id.loc[unit["rewire_ids"]].tolist()
        except KeyError as e:
            print(f"Warning: '{uid}' has a rewire_id not present in the full EDOS release ({e}); skipping.")
            continue

        unit_df = pd.DataFrame({"category": categories, "label": unit["labels"], "pred": unit["preds"]})
        for category, group in unit_df.groupby("category"):
            y = np.asarray(group["label"])
            p = np.asarray(group["pred"])
            n_sexist = int(np.sum(y == 1))
            n_not_sexist = int(np.sum(y == 0))
            fp = int(np.sum((p == 1) & (y == 0)))
            fn = int(np.sum((p == 0) & (y == 1)))
            rows.append({
                "variant_id": unit["variant_id"],
                "model": unit["model"],
                "category": category,
                "n": len(group),
                "false_positives": fp,
                "false_negatives": fn,
                "fp_rate": fp / n_not_sexist if n_not_sexist else float("nan"),
                "fn_rate": fn / n_sexist if n_sexist else float("nan"),
            })
    return pd.DataFrame(rows)


def aggregate_confidence_capture(checkpoint_dir: str | Path) -> pd.DataFrame:
    """One row per unit from ``scripts/run_confidence_capture.py``'s output
    directory (a flat dir of self-contained per-unit JSON files, a different
    and deliberately simpler shape than ``RunCheckpoint`` — see that script's
    own docstring for why).

    The live run script's own printed ``agreement_rate`` compares the
    confidence-implied call against the parsed decision with a strict ``>``
    (``(c > 0.5) == (pred == 1)``), which under-counts exact
    ``confidence == 0.5`` ties where the model actually generated "YES" as a
    disagreement even though the two signals agree at that boundary (see
    HANDOFF's "CHECK flag" investigation, traced to real per-row data rather
    than assumed). This recomputes agreement with ``>=`` instead
    (``agreement_rate_corrected``), and additionally reports a version that
    excludes genuine parse failures (``metrics.is_failure``) from the
    denominator (``agreement_rate_excl_failures``): a failure's confidence
    score is restricted to comparing only among YES/NO candidate logits even
    though the model's real top token was neither, a documented
    literature-standard limitation of first-token-restricted confidence, not
    a disagreement bug. The original live-printed value is kept as
    ``agreement_rate_live`` for reference/comparison, not because it should
    be trusted as-is.
    """
    checkpoint_dir = Path(checkpoint_dir)
    rows = []
    for path in sorted(checkpoint_dir.glob("*.json")):
        with open(path) as f:
            d = json.load(f)

        preds = [process_response(r) for r in d["responses"]]
        fails = [is_failure(r) for r in d["responses"]]
        pairs = [
            (c, p, fail)
            for c, p, fail in zip(d["confidences"], preds, fails)
            if c is not None
        ]
        agree_all = [(c >= 0.5) == (p == 1) for c, p, _ in pairs]
        agree_excl_fail = [(c >= 0.5) == (p == 1) for c, p, fail in pairs if not fail]

        rows.append({
            "unit_id": d["unit_id"],
            "model": d["model"],
            "variant": d["variant"],
            "n": d["n"],
            "n_empty": d["n_empty"],
            "n_none_confidence": d["n_none_confidence"],
            "agreement_rate_live": d["agreement_rate"],
            "agreement_rate_corrected": float(np.mean(agree_all)) if agree_all else float("nan"),
            "agreement_rate_excl_failures": float(np.mean(agree_excl_fail)) if agree_excl_fail else float("nan"),
            **d["metrics"],
        })
    return pd.DataFrame(rows)


def summarize_confidence_capture_by_model(
    units_df: pd.DataFrame, classical_auc: Optional[float] = None
) -> pd.DataFrame:
    """Per-model rollup of ``aggregate_confidence_capture``'s unit rows: the
    F1-best variant among each model's *covered* (non-CoT) units.

    Selecting by F1 rather than by AUC matches this project's existing
    "best variant by F1" convention used throughout the main grid (Table II),
    so this rollup stays comparable rather than silently switching selection
    criteria for one new subsection. For a model whose true overall-best
    variant (across all 18, including chain-of-thought) isn't in this run's
    7-variant coverage, this reports the best AUC-scorable stand-in, not the
    true best — callers that also know each model's true best variant should
    flag that gap explicitly rather than let this rollup imply otherwise.
    """
    rows = []
    for model, group in units_df.groupby("model"):
        best = group.loc[group["f1"].idxmax()]
        row = {
            "model": model,
            "best_covered_variant": best["variant"],
            "f1": float(best["f1"]),
            "auc": float(best["auc"]),
            "agreement_rate_corrected": float(best["agreement_rate_corrected"]),
            "agreement_rate_excl_failures": float(best["agreement_rate_excl_failures"]),
            "fail_ratio": float(best["fail_ratio"]),
        }
        if classical_auc is not None:
            row["beats_classical_auc"] = bool(best["auc"] > classical_auc)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


_REFUSAL_RE = re.compile(
    r"\b(i cannot|i can't|i'm not able to|not able to (?:create|generate|provide|assist)|"
    r"cannot (?:create|generate|provide|assist))\b",
    re.IGNORECASE,
)


def classify_cot_failure_causes(checkpoint_dir: str | Path, unit_ids: Sequence[str]) -> pd.DataFrame:
    """Split each unit's failed rows into explicit-refusal vs. other causes.

    The same regex-based check originally done ad hoc for llama3.1's
    chain-of-thought variants (Sec.~V-C / Table III), formalized here as
    reusable code so it can be re-run against other models' smaller
    elevated-``fail_ratio`` cells without re-deriving the method.

    A failed row (``metrics.is_failure``) counts as a refusal if it matches
    an explicit decline pattern (``I cannot``, ``I can't``, ``I'm not able
    to``, ``cannot``/``not able to`` + create/generate/provide/assist,
    case-insensitive); everything else counts as "other". For
    chain-of-thought variants specifically, "other" failures were, on
    inspection in the original llama3.1 analysis, genuine reasoning cut off
    by the token budget before reaching a decision, not malformed output —
    the per-unit mean length columns here are corroborating evidence for
    that reading (truncations cluster near the model's CoT token budget;
    refusals are short), not part of the classification rule itself.
    """
    checkpoint = RunCheckpoint(checkpoint_dir)
    rows = []
    for uid in unit_ids:
        responses = checkpoint.load_predictions_normalized(uid)["responses"]
        n = len(responses)
        fails = [r for r in responses if is_failure(r)]
        n_fail = len(fails)
        refusals = [r for r in fails if _REFUSAL_RE.search(r)]
        other = [r for r in fails if not _REFUSAL_RE.search(r)]
        rows.append({
            "unit_id": uid,
            "n": n,
            "n_fail": n_fail,
            "fail_ratio": n_fail / n if n else float("nan"),
            "refusal_share": len(refusals) / n_fail if n_fail else float("nan"),
            "other_share": len(other) / n_fail if n_fail else float("nan"),
            "mean_refusal_length": float(np.mean([len(r) for r in refusals])) if refusals else float("nan"),
            "mean_other_length": float(np.mean([len(r) for r in other])) if other else float("nan"),
        })
    return pd.DataFrame(rows)
