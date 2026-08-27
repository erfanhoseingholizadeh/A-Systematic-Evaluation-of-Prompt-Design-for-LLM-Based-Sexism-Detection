"""Parsing model responses into binary labels and computing classification metrics.

``_extract_answer_tail`` finds the *last* occurrence of an answer marker in
the response, which is what lets this same parser handle both the strict
2-token classification variants (marker inserted by the prompt itself) and
the long chain-of-thought variants (marker produced by the model after its
own reasoning) without any special-casing.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

_YES_RE = re.compile(r"\bYES\b", re.IGNORECASE)
_NO_RE = re.compile(r"\bNO\b", re.IGNORECASE)
_ANSWER_MARKERS = ("ANSWER:", "<|assistant|>", "Assistant:")


def _extract_answer_tail(text: str) -> str:
    if not isinstance(text, str):
        return ""
    best_pos, best_len = -1, 0
    for marker in _ANSWER_MARKERS:
        pos = text.rfind(marker)
        if pos > best_pos:
            best_pos, best_len = pos, len(marker)
    return text[best_pos + best_len :] if best_pos != -1 else text


def _decide(tail: str) -> Optional[int]:
    """Resolve an already-extracted answer tail to 1/0, or None if the
    response never reaches an unambiguous decision.

    A clean leading marker (``ANSWER: YES``/``NO``) always wins. Absent
    that, exactly one of YES/NO appearing is still treated as the answer
    (handles CoT responses that conclude in prose without a literal
    ``ANSWER:`` marker). If *both* appear with no clean leading marker, the
    response is ambiguous (e.g. a hedge like "I can't tell if this is YES
    or NO") and must NOT be resolved by incidental word order.
    """
    m = re.match(r"^(YES|NO)\b", tail, flags=re.IGNORECASE)
    if m:
        return 1 if m.group(1).upper() == "YES" else 0

    yes_m = _YES_RE.search(tail)
    no_m = _NO_RE.search(tail)
    if yes_m and no_m:
        return None
    if yes_m:
        return 1
    if no_m:
        return 0
    return None


def process_response(response: str) -> int:
    """Parse a raw model response into a binary label (1 = sexist, 0 = not/undecided).

    Ambiguous/undecided responses (see ``_decide``) default to 0, matching
    this project's "failures count as NO" headline-metrics convention (see
    ``compute_metrics``). Use ``is_failure`` to detect these separately.
    """
    tail = _extract_answer_tail(response).strip()
    decision = _decide(tail)
    return decision if decision is not None else 0


def is_failure(raw: str) -> bool:
    """True if the response never reaches an unambiguous YES/NO decision.

    Operates on the same extracted answer tail as ``process_response`` (not
    the full raw text) so both functions agree on what counts as "decided".
    Scanning the full raw text for either word appearing anywhere would let
    a hedge mentioning both ("I can't tell if this is YES or NO") go
    undetected as a failure and get silently scored as a confident
    prediction based on incidental word order.
    """
    if not isinstance(raw, str):
        return True
    tail = _extract_answer_tail(raw).strip()
    return _decide(tail) is None


def compute_metrics_from_predictions(preds: Sequence[int], y_true: Sequence[int]) -> Dict[str, float]:
    """Accuracy/precision/recall/F1 from already-decided binary predictions.

    Shared by ``compute_metrics`` (LLM text responses) and any classifier that
    already outputs 0/1 directly (encoder / classical baselines).
    """
    y_true_arr = np.asarray(list(y_true), dtype=int)
    preds_arr = np.asarray(list(preds), dtype=int)

    n = len(y_true_arr)
    if n == 0:
        return {"accuracy": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}

    tp = int(np.sum((preds_arr == 1) & (y_true_arr == 1)))
    fp = int(np.sum((preds_arr == 1) & (y_true_arr == 0)))
    fn = int(np.sum((preds_arr == 0) & (y_true_arr == 1)))

    accuracy = float((preds_arr == y_true_arr).mean())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"accuracy": accuracy, "precision": float(precision), "recall": float(recall), "f1": float(f1)}


def compute_auc(confidences: Sequence[Optional[float]], y_true: Sequence[int]) -> float:
    """ROC-AUC from a per-example P(sexist) confidence score, dropping any
    example whose confidence is ``None`` (extraction failed or wasn't
    attempted; see ``confidence.py``).

    Returns NaN rather than raising when there's nothing to score: too few
    examples survive the ``None`` filter, or (``roc_auc_score``'s own
    requirement) only one class is present among the survivors. Both are
    real possibilities on a small or heavily imbalanced batch, not bugs.
    """
    pairs = [(c, y) for c, y in zip(confidences, y_true) if c is not None]
    if len(pairs) < 2:
        return float("nan")
    conf_kept = [c for c, _ in pairs]
    y_kept = [y for _, y in pairs]
    if len(set(y_kept)) < 2:
        return float("nan")
    return float(roc_auc_score(y_kept, conf_kept))


def compute_metrics(
    responses: Sequence[str],
    y_true: Sequence[int],
    confidences: Optional[Sequence[Optional[float]]] = None,
) -> Dict[str, float]:
    """Accuracy/precision/recall/F1 plus YES/NO parse failure rate, from raw LLM text responses.

    Failed responses still count in the headline metrics as their defaulted
    prediction (0 = not sexist, see ``process_response``); use
    ``compute_metrics_excluding_failures`` for the complementary
    failures-excluded view.

    ``confidences``, when given (one P(sexist) score per response, ``None``
    where unavailable; see ``confidence.py`` and
    ``inference.generate_responses_with_confidence``), adds an ``"auc"`` key
    via ``compute_auc``. Omitted entirely when not given, so callers that
    don't capture confidence get the plain dict shape without an
    ``"auc"`` key.

    Returns a dict with keys: accuracy, precision, recall, f1, fail_ratio, n,
    n_fail, and (only when ``confidences`` is given) auc.
    """
    n = len(y_true)
    if n == 0:
        base = compute_metrics_from_predictions([], [])
        result = {**base, "fail_ratio": float("nan"), "n": 0, "n_fail": 0}
        if confidences is not None:
            result["auc"] = float("nan")
        return result
    if len(responses) != n:
        raise ValueError(f"got {len(responses)} responses for {n} labels; refusing to score misaligned data")
    if confidences is not None and len(confidences) != n:
        raise ValueError(f"got {len(confidences)} confidences for {n} labels; refusing to score misaligned data")

    preds = [process_response(r) for r in responses]
    fails = np.array([is_failure(r) for r in responses], dtype=bool)

    base = compute_metrics_from_predictions(preds, y_true)
    result = {**base, "fail_ratio": float(fails.mean()), "n": int(n), "n_fail": int(fails.sum())}
    if confidences is not None:
        result["auc"] = compute_auc(confidences, y_true)
    return result


def compute_metrics_excluding_failures(responses: Sequence[str], y_true: Sequence[int]) -> Dict[str, float]:
    """Metrics computed only over responses that actually contain a YES/NO decision.

    The headline ``compute_metrics`` silently scores every parse failure as a
    0 ("not sexist") prediction, which subsidizes variants with high failure
    rates (CoT) when compared against variants that structurally cannot fail
    (non-CoT, 2-token pre-filled answers). This view drops the failed rows
    instead and reports ``coverage`` (the answered fraction) alongside, so the
    two policies can be compared explicitly.

    Returns the ``compute_metrics_from_predictions`` keys plus coverage, n,
    n_answered.
    """
    n = len(y_true)
    if len(responses) != n:
        raise ValueError(f"got {len(responses)} responses for {n} labels; refusing to score misaligned data")

    kept = [(process_response(r), y) for r, y in zip(responses, y_true) if not is_failure(r)]
    preds_kept = [p for p, _ in kept]
    y_kept = [y for _, y in kept]

    base = compute_metrics_from_predictions(preds_kept, y_kept)
    coverage = (len(kept) / n) if n else float("nan")
    return {**base, "coverage": float(coverage), "n": int(n), "n_answered": int(len(kept))}


def compute_metrics_with_abstention(
    responses: Sequence[str],
    y_true: Sequence[int],
    confidences: Sequence[Optional[float]],
    band: float,
) -> Dict[str, float]:
    """Accuracy/precision/recall/F1 on the subset of examples the model was
    confident about, abstaining whenever P(sexist) falls within ``band`` of
    the 0.5 decision boundary (i.e. confidence in ``[0.5-band, 0.5+band]``)
    or is unavailable (``None``, e.g. a parse failure; see ``confidence.py``).

    A confident example's prediction still comes from the response's own
    parsed decision (``process_response``), not from thresholding the
    confidence score itself: confidence only gates whether an example is
    covered, mirroring how a real abstention policy would defer to, rather
    than override, the model's own generated answer. Complementary to
    ``compute_metrics_excluding_failures``'s parse-failure coverage, this
    reports the confidence-abstention coverage tradeoff instead.

    Returns the ``compute_metrics_from_predictions`` keys plus coverage, n,
    n_abstained.
    """
    n = len(y_true)
    if len(responses) != n:
        raise ValueError(f"got {len(responses)} responses for {n} labels; refusing to score misaligned data")
    if len(confidences) != n:
        raise ValueError(f"got {len(confidences)} confidences for {n} labels; refusing to score misaligned data")

    lo, hi = 0.5 - band, 0.5 + band
    kept = [
        (process_response(r), y)
        for r, y, c in zip(responses, y_true, confidences)
        if c is not None and not (lo <= c <= hi)
    ]
    preds_kept = [p for p, _ in kept]
    y_kept = [y for _, y in kept]

    base = compute_metrics_from_predictions(preds_kept, y_kept)
    coverage = (len(kept) / n) if n else float("nan")
    return {**base, "coverage": float(coverage), "n": int(n), "n_abstained": int(n - len(kept))}
