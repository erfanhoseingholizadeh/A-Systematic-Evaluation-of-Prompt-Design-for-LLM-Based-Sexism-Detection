import math

from sexism_prompting.confidence import resolve_yes_no_token_ids, yes_probability_from_logits


class _FakeTokenizer:
    """Assigns a stable, distinct token id to every distinct string it's asked
    to encode (single-token vocabulary, like a real tokenizer's leading-token
    behavior for short words)."""

    def __init__(self):
        self._vocab = {}

    def encode(self, text, add_special_tokens=False):
        if text not in self._vocab:
            self._vocab[text] = len(self._vocab) + 1
        return [self._vocab[text]]


class _RaisingTokenizer:
    def encode(self, text, add_special_tokens=False):
        raise RuntimeError("boom")


def test_resolve_yes_no_token_ids_deduplicates_and_preserves_order():
    tok = _FakeTokenizer()
    yes_ids, no_ids = resolve_yes_no_token_ids(tok)
    assert len(yes_ids) == len(set(yes_ids))  # no duplicate ids
    assert len(no_ids) == len(set(no_ids))
    assert yes_ids  # at least one surface form resolved
    assert no_ids
    assert set(yes_ids).isdisjoint(no_ids)  # YES and NO never share a token id


def test_resolve_yes_no_token_ids_survives_a_raising_tokenizer():
    yes_ids, no_ids = resolve_yes_no_token_ids(_RaisingTokenizer())
    assert yes_ids == []
    assert no_ids == []


class _CollidingTokenizer:
    """Simulates a SentencePiece-style tokenizer where every leading-space
    surface form (" YES", " NO", ...) encodes to the SAME shared prefix
    token id rather than a fully distinct one, while the no-space forms
    ("YES", "NO", ...) still get their own distinct ids."""

    _SHARED_PREFIX_ID = 1

    def __init__(self):
        self._vocab = {}

    def encode(self, text, add_special_tokens=False):
        if text.startswith(" "):
            return [self._SHARED_PREFIX_ID]
        if text not in self._vocab:
            self._vocab[text] = len(self._vocab) + 100
        return [self._vocab[text]]


def test_resolve_yes_no_token_ids_drops_ids_shared_between_yes_and_no():
    """A shared leading-space prefix token carries no real signal for either
    label and must not stay a candidate for both. Left in, it would
    silently pull yes_probability_from_logits's comparison toward 0.5 with
    no indication a collision happened."""
    yes_ids, no_ids = resolve_yes_no_token_ids(_CollidingTokenizer())
    assert _CollidingTokenizer._SHARED_PREFIX_ID not in yes_ids
    assert _CollidingTokenizer._SHARED_PREFIX_ID not in no_ids
    assert set(yes_ids).isdisjoint(no_ids)
    assert yes_ids and no_ids  # the non-colliding surface forms still resolved


def test_yes_probability_from_logits_confident_yes():
    # index 0 = YES token, wildly higher logit than everything else
    logits = [10.0, -5.0, -5.0, -5.0]
    p = yes_probability_from_logits(logits, yes_token_ids=[0], no_token_ids=[1])
    assert p > 0.999


def test_yes_probability_from_logits_confident_no():
    logits = [-5.0, 10.0, -5.0, -5.0]
    p = yes_probability_from_logits(logits, yes_token_ids=[0], no_token_ids=[1])
    assert p < 0.001


def test_yes_probability_from_logits_tied_is_half():
    logits = [3.0, 3.0]
    p = yes_probability_from_logits(logits, yes_token_ids=[0], no_token_ids=[1])
    assert math.isclose(p, 0.5, abs_tol=1e-9)


def test_yes_probability_from_logits_takes_max_across_multiple_candidate_ids():
    # Two candidate YES tokens ("YES", " YES"); the stronger one should win,
    # not an average, since either one being emitted means "yes" was decided.
    logits = [1.0, 9.0, 2.0]  # yes candidates at 0,1; no candidate at 2
    p = yes_probability_from_logits(logits, yes_token_ids=[0, 1], no_token_ids=[2])
    assert p > 0.99  # dominated by logit 9.0, not diluted by logit 1.0


def test_yes_probability_from_logits_returns_none_when_no_candidates():
    assert yes_probability_from_logits([1.0, 2.0], yes_token_ids=[], no_token_ids=[0]) is None
    assert yes_probability_from_logits([1.0, 2.0], yes_token_ids=[0], no_token_ids=[]) is None


def test_yes_probability_from_logits_ignores_out_of_range_ids():
    """A candidate id from a different tokenizer's vocab size must not crash
    a lookup against a shorter logits vector; it should just get filtered
    out."""
    logits = [1.0, 2.0]
    p = yes_probability_from_logits(logits, yes_token_ids=[0, 999], no_token_ids=[1])
    assert p is not None
