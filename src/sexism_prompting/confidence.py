"""Extracting a calibrated YES-probability from a model's first-token logits.

The generation pipeline otherwise only keeps a parsed YES/NO decision,
never a continuous confidence score, which a per-example AUC needs.

Scope, deliberately: this only covers the non-CoT variants. Their prompt
already ends in a forced ``ANSWER:`` cue (see ``prompts.py``), so the
model's very first generated token IS the decision: a well-established
technique ("first-token logit comparison") for reading a calibrated binary
probability off an unconstrained generative model. CoT variants don't have
this property (their first generated token starts the reasoning, not the
decision; the real decision appears wherever ``ANSWER:`` ends up in the
finished text), and pulling a confidence out of *that* would need a
materially different technique not implemented here. Left as a documented
gap rather than a guessed implementation.

Every function in this module is pure (numpy/list arithmetic or tokenizer
``encode`` calls only) and testable without a GPU or the ``torch``/
``transformers`` stack. The actual ``model.generate(..., output_scores=True)``
call that produces the logits this module consumes lives in
``inference.generate_responses_with_confidence``, which is NOT wired into
the main grid's default path (see that function's docstring for why).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

# Tried in order; the first token of each successful encoding is kept as a
# candidate. Multiple surface forms because tokenizers vary in whether a
# leading space or capitalization changes the token id (e.g. " YES" and
# "YES" are frequently different tokens under a BPE/SentencePiece vocab).
_YES_SURFACE_FORMS = (" YES", "YES", " Yes", "Yes", " yes", "yes")
_NO_SURFACE_FORMS = (" NO", "NO", " No", "No", " no", "no")


def resolve_yes_no_token_ids(tokenizer) -> Tuple[List[int], List[int]]:
    """Candidate token ids for a YES/NO decision, for this tokenizer specifically.

    Deduplicated but order-preserving; a tokenizer for which no surface form
    encodes as a single leading token still gets its first sub-token as a
    best-effort candidate (some tokenizers split "yes" into multiple
    pieces): better to compare *something* consistently than to raise and
    lose the whole unit's confidence capture over one edge case.

    Also drops any id that ended up a candidate for BOTH yes and no. This is
    possible under SentencePiece-style tokenizers, where a leading-space
    surface form (e.g. " YES", " NO") can encode as a shared prefix token
    (e.g. a bare ``▁``) followed by more sub-tokens rather than a single
    self-contained leading token; in that case the shared prefix is not
    real evidence for either label, and letting it stay a candidate for both
    would silently pull ``yes_probability_from_logits``'s comparison toward
    0.5 without any signal that a collision happened.
    """

    def _first_token_ids(forms: Sequence[str]) -> List[int]:
        seen = set()
        ids: List[int] = []
        for form in forms:
            try:
                encoded = tokenizer.encode(form, add_special_tokens=False)
            except Exception:  # noqa: BLE001
                continue
            if not encoded:
                continue
            first = encoded[0]
            if first not in seen:
                seen.add(first)
                ids.append(first)
        return ids

    yes_ids = _first_token_ids(_YES_SURFACE_FORMS)
    no_ids = _first_token_ids(_NO_SURFACE_FORMS)
    overlap = set(yes_ids) & set(no_ids)
    if overlap:
        yes_ids = [i for i in yes_ids if i not in overlap]
        no_ids = [i for i in no_ids if i not in overlap]
    return yes_ids, no_ids


def yes_probability_from_logits(
    logits: Sequence[float],
    yes_token_ids: Sequence[int],
    no_token_ids: Sequence[int],
) -> Optional[float]:
    """P(YES) vs P(NO), restricted to just the candidate YES/NO token ids.

    Softmax over the *full* vocabulary would let every other token's mass
    dilute the YES/NO comparison (e.g. punctuation or whitespace tokens
    absorbing probability that says nothing about the actual decision).
    Restricting to just these two token sets and renormalizing gives a
    cleaner binary confidence, the standard "logit comparison" reading of a
    forced-choice generation. Returns ``None`` (not 0.5) when neither
    token set resolved to anything usable, so a missing signal is never
    silently reported as a genuine 50/50 call.
    """
    if not yes_token_ids or not no_token_ids:
        return None
    n = len(logits)
    yes_ids = [i for i in yes_token_ids if 0 <= i < n]
    no_ids = [i for i in no_token_ids if 0 <= i < n]
    if not yes_ids or not no_ids:
        return None

    yes_logit = max(logits[i] for i in yes_ids)
    no_logit = max(logits[i] for i in no_ids)

    # Numerically stable 2-way softmax: subtract the max before exponentiating.
    top = max(yes_logit, no_logit)
    yes_exp = math.exp(yes_logit - top)
    no_exp = math.exp(no_logit - top)
    return yes_exp / (yes_exp + no_exp)
