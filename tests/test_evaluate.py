import json
import time
import weakref
from pathlib import Path

import pandas as pd
import pytest

from sexism_prompting.checkpoint import RunCheckpoint
from sexism_prompting.evaluate import (
    evaluate_all_variants_sequential,
    evaluate_variant,
    load_prompt_variants,
    predict_with_model,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class _FakeTokenizer:
    chat_template = None


class _FakePipe:
    """Stands in for a HF text-generation pipeline: always answers NO,
    with a call counter so tests can assert how many times inference ran."""

    def __init__(self):
        self.calls = 0

    def __call__(self, batch, **kwargs):
        self.calls += 1
        return [{"generated_text": "NO"} for _ in batch]


def _toy_data():
    test_df = pd.DataFrame({"text": ["hello", "world"], "label": [0, 1]})
    demos_df = pd.DataFrame({"text": [f"d{i}" for i in range(10)], "label": [0, 1] * 5})
    return test_df, demos_df


def test_load_prompt_variants_reads_real_config():
    variants, fewshot_n, fewshot_seed = load_prompt_variants(REPO_ROOT / "configs" / "prompt_variants.yaml")
    assert len(variants) == 18
    assert {v["id"] for v in variants} == {f"V{i}" for i in range(1, 19)}
    assert fewshot_n == 2
    assert fewshot_seed == 42


def test_predict_with_model_returns_one_response_per_row():
    test_df, demos_df = _toy_data()
    pipe = _FakePipe()
    responses = predict_with_model((pipe, _FakeTokenizer()), test_df, demos_df, {"cot": False}, batch_size=32)
    assert responses == ["NO", "NO"]
    assert pipe.calls == 1


class _KwargsCapturingPipe:
    """Like _FakePipe, but records the exact kwargs it was called with --
    for asserting what predict_with_model actually passes to the pipeline
    call, not just what comes back out."""

    def __init__(self):
        self.last_kwargs = None

    def __call__(self, batch, **kwargs):
        self.last_kwargs = kwargs
        return [{"generated_text": "ANSWER: NO"} for _ in batch]


def test_predict_with_model_injects_tokenizer_alongside_stop_strings():
    """The HF pipeline does NOT auto-forward its own constructor-time
    tokenizer to generate() for the stop_strings kwarg: it raises "we
    could not locate a tokenizer" unless `tokenizer` is also passed
    explicitly per-call. CoT variants resolve stop_strings via
    inference.COT_GEN_KWARGS (see metrics.py for the last-marker-wins
    hallucination issue this guards against), so predict_with_model must
    inject the tokenizer whenever stop_strings is present."""
    test_df, demos_df = _toy_data()
    pipe = _KwargsCapturingPipe()
    tokenizer = _FakeTokenizer()
    predict_with_model((pipe, tokenizer), test_df, demos_df, {"cot": True}, batch_size=32)
    assert pipe.last_kwargs["stop_strings"] == ["ANSWER: YES", "ANSWER: NO"]
    assert pipe.last_kwargs["tokenizer"] is tokenizer


def test_predict_with_model_does_not_inject_tokenizer_for_non_cot():
    """Non-CoT variants use DEFAULT_GEN_KWARGS (no stop_strings: the
    2-token cap already bounds generation), so no tokenizer kwarg should be
    added; passing one unconditionally would be untested surface area on
    the non-CoT path for no benefit."""
    test_df, demos_df = _toy_data()
    pipe = _KwargsCapturingPipe()
    predict_with_model((pipe, _FakeTokenizer()), test_df, demos_df, {"cot": False}, batch_size=32)
    assert "tokenizer" not in pipe.last_kwargs
    assert "stop_strings" not in pipe.last_kwargs


def test_evaluate_variant_scores_already_loaded_models():
    test_df, demos_df = _toy_data()
    models = {"fake_model": (_FakePipe(), _FakeTokenizer())}
    results = evaluate_variant(models, test_df, demos_df, {"cot": False})
    assert "fake_model" in results
    assert results["fake_model"]["n"] == 2


def test_evaluate_all_variants_sequential_computes_every_unit(tmp_path):
    test_df, demos_df = _toy_data()
    variants = [{"id": "V1", "cot": False}, {"id": "V6", "cot": False, "fewshot": True}]
    checkpoint = RunCheckpoint(tmp_path / "ckpt")
    load_calls = []

    def fake_load(key):
        load_calls.append(key)
        return _FakePipe(), _FakeTokenizer()

    df = evaluate_all_variants_sequential(
        ["fake_model"], test_df, demos_df, variants, checkpoint,
        load_fn=fake_load, unload_fn=lambda: None,
        model_registry={},
    )
    assert len(df) == 2
    assert set(df["variant_id"]) == {"V1", "V6"}
    assert load_calls == ["fake_model"]  # loaded exactly once, not once per variant


def test_unload_drops_reference_before_next_models_load(tmp_path):
    """The orchestrator's own `loaded` binding must be dropped independently
    of whatever `unload_fn`'s own del/gc/empty_cache clears (`unload_fn`
    only has access to its own local binding, not the orchestrator's
    `loaded` variable). Otherwise `loaded` would stay bound to the old model
    across the whole next model's `load_fn` call, so peak VRAM during every
    model transition would be old+new, not just new. A weakref proves the
    old model is actually collectible by the time the *next* model's
    load_fn runs, not just eventually by the time the sweep ends."""
    test_df, demos_df = _toy_data()
    variants = [{"id": "V1", "cot": False}]
    checkpoint = RunCheckpoint(tmp_path / "ckpt")

    class _Sentinel:
        pass

    weak_refs = []
    alive_when_second_load_starts = []

    def fake_load(key):
        if key == "model_b":
            alive_when_second_load_starts.append(weak_refs[0]() is not None)
        sentinel = _Sentinel()
        weak_refs.append(weakref.ref(sentinel))
        return _FakePipe(), sentinel  # tokenizer slot repurposed to carry the sentinel

    evaluate_all_variants_sequential(
        ["model_a", "model_b"], test_df, demos_df, variants, checkpoint,
        load_fn=fake_load, unload_fn=lambda: None,
        model_registry={},
    )
    assert alive_when_second_load_starts == [False]


def test_unload_fn_runs_only_after_reference_is_dropped(tmp_path):
    """`unload_fn` takes no arguments and is called only after this loop's
    own `loaded = None`. Calling it as `unload_fn(loaded)` instead would
    keep the argument itself as a live reference for the entire call, so
    any gc.collect()/empty_cache() unload_fn performs would run too early
    to free anything: the orchestrator's own `loaded` binding, and
    unload_fn's own parameter binding, would both still point at the
    object. This test proves the previous model's last reference is
    already gone by the time unload_fn is invoked, not just by the time
    the next model's load_fn eventually runs."""
    test_df, demos_df = _toy_data()
    variants = [{"id": "V1", "cot": False}]
    checkpoint = RunCheckpoint(tmp_path / "ckpt")

    class _Sentinel:
        pass

    weak_ref = []
    alive_when_unload_fn_runs = []

    def fake_load(key):
        sentinel = _Sentinel()
        weak_ref.append(weakref.ref(sentinel))
        return _FakePipe(), sentinel

    def fake_unload():
        alive_when_unload_fn_runs.append(weak_ref[-1]() is not None)

    evaluate_all_variants_sequential(
        ["fake_model"], test_df, demos_df, variants, checkpoint,
        load_fn=fake_load, unload_fn=fake_unload,
        model_registry={},
    )
    assert alive_when_unload_fn_runs == [False]


def test_evaluate_all_variants_sequential_skips_completed_units_on_resume(tmp_path):
    test_df, demos_df = _toy_data()
    variants = [{"id": "V1", "cot": False}, {"id": "V6", "cot": False, "fewshot": True}]
    ckpt_dir = tmp_path / "ckpt"

    load_calls = []

    def fake_load(key):
        load_calls.append(key)
        return _FakePipe(), _FakeTokenizer()

    checkpoint1 = RunCheckpoint(ckpt_dir)
    evaluate_all_variants_sequential(
        ["fake_model"], test_df, demos_df, variants, checkpoint1,
        load_fn=fake_load, unload_fn=lambda: None,
        model_registry={},
    )
    assert len(load_calls) == 1

    # Simulates a restarted run against the same checkpoint directory.
    checkpoint2 = RunCheckpoint(ckpt_dir)
    df_resumed = evaluate_all_variants_sequential(
        ["fake_model"], test_df, demos_df, variants, checkpoint2,
        load_fn=fake_load, unload_fn=lambda: None,
        model_registry={},
    )
    assert len(load_calls) == 1  # no new load: everything was already done
    assert len(df_resumed) == 2  # still returns both rows, from the manifest


def test_config_change_invalidates_stale_checkpoint_on_resume(tmp_path):
    """Keying a checkpoint only by variant_id__model_key gives no way to
    detect that the prompt/generation/data/model config changed since a
    unit was checkpointed, which would let an old result get silently
    reused forever. Changing gen_kwargs between two runs against the same
    checkpoint dir must cause a real re-run, not a silent skip."""
    test_df, demos_df = _toy_data()
    variants = [{"id": "V1", "cot": False}]
    ckpt_dir = tmp_path / "ckpt"
    load_calls = []

    def fake_load(key):
        load_calls.append(key)
        return _FakePipe(), _FakeTokenizer()

    checkpoint1 = RunCheckpoint(ckpt_dir)
    evaluate_all_variants_sequential(
        ["fake_model"], test_df, demos_df, variants, checkpoint1,
        load_fn=fake_load, unload_fn=lambda: None,
        model_registry={}, gen_kwargs={"max_new_tokens": 2},
    )
    assert len(load_calls) == 1

    # Same checkpoint dir, DIFFERENT generation config: must be treated as
    # pending again, not silently skipped as already-done.
    checkpoint2 = RunCheckpoint(ckpt_dir)
    evaluate_all_variants_sequential(
        ["fake_model"], test_df, demos_df, variants, checkpoint2,
        load_fn=fake_load, unload_fn=lambda: None,
        model_registry={}, gen_kwargs={"max_new_tokens": 999},
    )
    assert len(load_calls) == 2  # re-loaded and re-ran: config change was detected


def test_evaluate_all_variants_sequential_unloads_even_on_error(tmp_path):
    test_df, demos_df = _toy_data()
    variants = [{"id": "V1", "cot": False}]
    checkpoint = RunCheckpoint(tmp_path / "ckpt")
    unload_calls = []

    class _BoomPipe:
        def __call__(self, batch, **kwargs):
            raise RuntimeError("boom")

    def fake_load(key):
        return _BoomPipe(), _FakeTokenizer()

    # generate_responses catches per-batch failures internally, so this
    # shouldn't raise; confirm unload still runs regardless.
    evaluate_all_variants_sequential(
        ["fake_model"], test_df, demos_df, variants, checkpoint,
        load_fn=fake_load, unload_fn=lambda: unload_calls.append(True),
        model_registry={},
    )
    assert len(unload_calls) == 1


def test_evaluate_all_variants_sequential_skips_model_that_fails_to_load(tmp_path):
    """Regression test for a real failure: an unauthenticated gated model
    (llama3.1, no valid HF token) crashed the entire pilot run instead of
    being skipped, losing the chance to try the models queued after it."""
    test_df, demos_df = _toy_data()
    variants = [{"id": "V1", "cot": False}]
    checkpoint = RunCheckpoint(tmp_path / "ckpt")

    def fake_load(key):
        if key == "gated_model":
            raise OSError("You are trying to access a gated repo.")
        return _FakePipe(), _FakeTokenizer()

    df = evaluate_all_variants_sequential(
        ["gated_model", "open_model"], test_df, demos_df, variants, checkpoint,
        load_fn=fake_load, unload_fn=lambda: None,
        model_registry={},
    )
    assert set(df["model"]) == {"open_model"}  # gated_model skipped, open_model still ran
    assert not checkpoint.is_done(RunCheckpoint.unit_id("V1", "gated_model"))
    assert checkpoint.is_done(RunCheckpoint.unit_id("V1", "open_model"))


class _OOMPipe:
    """Raises like a CUDA OOM: generate_responses catches it per batch and
    fills the batch with empty strings. generate_responses must not let
    that get checkpointed as a completed all-'NO' unit."""

    def __call__(self, batch, **kwargs):
        raise RuntimeError("CUDA out of memory")


def test_high_fail_ratio_unit_is_not_checkpointed(tmp_path):
    """A batch-level generation failure must leave the unit pending for a
    retry in a later session, not freeze garbage metrics into the grid."""
    test_df, demos_df = _toy_data()
    variants = [{"id": "V1", "cot": False}]
    checkpoint = RunCheckpoint(tmp_path / "ckpt")

    df = evaluate_all_variants_sequential(
        ["fake_model"], test_df, demos_df, variants, checkpoint,
        load_fn=lambda key: (_OOMPipe(), _FakeTokenizer()), unload_fn=lambda: None,
        model_registry={},
    )
    assert not checkpoint.is_done(RunCheckpoint.unit_id("V1", "fake_model"))
    assert df.empty


def test_high_fail_ratio_unit_saves_failure_evidence(tmp_path):
    """A unit crossing failure_threshold must persist its raw responses
    rather than just `continue` and discard them entirely. Otherwise
    there is no way to inspect why it failed short of adding one-off
    instrumentation on a future retry."""
    test_df, demos_df = _toy_data()
    variants = [{"id": "V1", "cot": False}]
    checkpoint = RunCheckpoint(tmp_path / "ckpt")

    evaluate_all_variants_sequential(
        ["fake_model"], test_df, demos_df, variants, checkpoint,
        load_fn=lambda key: (_OOMPipe(), _FakeTokenizer()), unload_fn=lambda: None,
        model_registry={},
    )

    uid = RunCheckpoint.unit_id("V1", "fake_model")
    saved = json.loads((tmp_path / "ckpt" / "failures" / f"{uid}.json").read_text())
    assert saved["responses"] == ["", ""]  # the OOM-stand-in pipe's empty-string fill
    assert "fail_ratio" in saved["reason"]


def test_failure_threshold_is_configurable(tmp_path):
    """fail_ratio 0.5 on toy data: blocked at the default threshold (0.5),
    checkpointed when the caller raises the threshold deliberately."""
    test_df, demos_df = _toy_data()  # 2 rows -> one failed response = 0.5
    variants = [{"id": "V1", "cot": False}]

    class _HalfFailPipe:
        def __call__(self, batch, **kwargs):
            return [{"generated_text": "NO"}] + [{"generated_text": "hmm???"}] * (len(batch) - 1)

    checkpoint_default = RunCheckpoint(tmp_path / "default")
    evaluate_all_variants_sequential(
        ["fake_model"], test_df, demos_df, variants, checkpoint_default,
        load_fn=lambda key: (_HalfFailPipe(), _FakeTokenizer()), unload_fn=lambda: None,
        model_registry={},
    )
    assert not checkpoint_default.is_done(RunCheckpoint.unit_id("V1", "fake_model"))

    checkpoint_raised = RunCheckpoint(tmp_path / "raised")
    evaluate_all_variants_sequential(
        ["fake_model"], test_df, demos_df, variants, checkpoint_raised,
        load_fn=lambda key: (_HalfFailPipe(), _FakeTokenizer()), unload_fn=lambda: None,
        model_registry={}, failure_threshold=0.6,
    )
    assert checkpoint_raised.is_done(RunCheckpoint.unit_id("V1", "fake_model"))


def test_failure_threshold_override_is_scoped_to_its_unit(tmp_path):
    """A per-unit override in failure_threshold_overrides must free only that
    (model, variant) pair from the global threshold. A different model run
    against the same variant, with no override of its own, still blocks at
    the global default."""
    test_df, demos_df = _toy_data()  # 2 rows -> one failed response = 0.5
    variants = [{"id": "V1", "cot": False}]

    class _HalfFailPipe:
        def __call__(self, batch, **kwargs):
            return [{"generated_text": "NO"}] + [{"generated_text": "hmm???"}] * (len(batch) - 1)

    checkpoint = RunCheckpoint(tmp_path / "ckpt")
    evaluate_all_variants_sequential(
        ["overridden_model", "other_model"], test_df, demos_df, variants, checkpoint,
        load_fn=lambda key: (_HalfFailPipe(), _FakeTokenizer()), unload_fn=lambda: None,
        model_registry={},
        failure_threshold_overrides={"V1__overridden_model": 0.6},
    )
    assert checkpoint.is_done(RunCheckpoint.unit_id("V1", "overridden_model"))
    assert not checkpoint.is_done(RunCheckpoint.unit_id("V1", "other_model"))


def test_unit_exception_skips_unit_but_continues(tmp_path):
    """An exception inside one unit (here: too few demos for few-shot
    sampling) must neither abort the session nor checkpoint the unit."""
    test_df = pd.DataFrame({"text": ["hello", "world"], "label": [0, 1]})
    demos_df = pd.DataFrame({"text": ["a", "b"], "label": [0, 1]})  # 1 per class, sampler needs 2
    variants = [{"id": "V1", "cot": False}, {"id": "V6", "cot": False, "fewshot": True}]
    checkpoint = RunCheckpoint(tmp_path / "ckpt")

    df = evaluate_all_variants_sequential(
        ["fake_model"], test_df, demos_df, variants, checkpoint,
        load_fn=lambda key: (_FakePipe(), _FakeTokenizer()), unload_fn=lambda: None,
        model_registry={},
    )
    assert checkpoint.is_done(RunCheckpoint.unit_id("V1", "fake_model"))
    assert not checkpoint.is_done(RunCheckpoint.unit_id("V6", "fake_model"))
    assert set(df["variant_id"]) == {"V1"}


def test_should_continue_stops_cleanly_between_units(tmp_path):
    test_df, demos_df = _toy_data()
    variants = [{"id": "V1", "cot": False}, {"id": "V2", "cot": False, "role": True}]
    checkpoint = RunCheckpoint(tmp_path / "ckpt")
    unload_calls = []

    calls = {"n": 0}

    def should_continue():
        # Checked once before the model load, then once before each unit:
        # allow the load and V1, stop before V2.
        calls["n"] += 1
        return calls["n"] <= 2

    df = evaluate_all_variants_sequential(
        ["fake_model"], test_df, demos_df, variants, checkpoint,
        load_fn=lambda key: (_FakePipe(), _FakeTokenizer()),
        unload_fn=lambda: unload_calls.append(True),
        model_registry={},
        should_continue=should_continue,
    )
    assert checkpoint.is_done(RunCheckpoint.unit_id("V1", "fake_model"))
    assert not checkpoint.is_done(RunCheckpoint.unit_id("V2", "fake_model"))
    assert list(df["variant_id"]) == ["V1"]
    assert len(unload_calls) == 1  # the loaded model is still unloaded on an early stop


def test_cot_gen_kwargs_overrides_applies_only_to_its_model(tmp_path):
    """A reasoning-tuned model needs a much larger CoT token budget than the
    default calibrated for non-reasoning models. The override must reach
    that model's actual generation call, and must NOT leak into another
    model's CoT units running in the same sweep."""
    test_df, demos_df = _toy_data()
    variants = [{"id": "V5", "cot": True}]
    checkpoint = RunCheckpoint(tmp_path / "ckpt")
    seen_kwargs = []

    class _RecordingPipe:
        def __call__(self, batch, **kwargs):
            seen_kwargs.append(kwargs.get("max_new_tokens"))
            return [{"generated_text": "ANSWER: NO"} for _ in batch]

    evaluate_all_variants_sequential(
        ["reasoning_model", "plain_model"], test_df, demos_df, variants, checkpoint,
        load_fn=lambda key: (_RecordingPipe(), _FakeTokenizer()), unload_fn=lambda: None,
        model_registry={}, cot_gen_kwargs={"max_new_tokens": 220},
        cot_gen_kwargs_overrides={"reasoning_model": {"max_new_tokens": 2000}},
    )
    assert seen_kwargs == [2000, 220]


def test_batch_size_overrides_applies_only_to_its_model(tmp_path):
    """A model needing a smaller batch size than the global default (e.g. to
    avoid CUDA OOM under a larger per-unit token budget) must get that
    override applied to its own units' actual chunking, without changing
    another model's batch size in the same sweep."""
    test_df = pd.DataFrame({"text": ["a", "b", "c", "d"], "label": [0, 1, 0, 1]})
    demos_df = pd.DataFrame({"text": [f"d{i}" for i in range(10)], "label": [0, 1] * 5})
    variants = [{"id": "V1", "cot": False}]
    checkpoint = RunCheckpoint(tmp_path / "ckpt")
    seen_batch_sizes = {}

    def make_pipe(model_key):
        sizes = []
        seen_batch_sizes[model_key] = sizes

        class _RecordingPipe:
            def __call__(self, batch, **kwargs):
                sizes.append(len(batch))
                return [{"generated_text": "NO"} for _ in batch]

        return _RecordingPipe()

    evaluate_all_variants_sequential(
        ["small_batch_model", "plain_model"], test_df, demos_df, variants, checkpoint,
        load_fn=lambda key: (make_pipe(key), _FakeTokenizer()), unload_fn=lambda: None,
        model_registry={}, batch_size=4,
        batch_size_overrides={"small_batch_model": 1},
    )
    assert seen_batch_sizes["small_batch_model"] == [1, 1, 1, 1]  # 4 rows at batch_size=1 -> 4 chunks
    assert seen_batch_sizes["plain_model"] == [4]  # global default, unaffected by the other model's override


def test_component_order_reaches_the_generation_call(tmp_path):
    test_df, demos_df = _toy_data()
    variants = [{"id": "V16", "role": True, "aspects": True, "context": True, "cot": False}]
    seen_prompts = []

    class _RecordingPipe:
        def __call__(self, batch, **kwargs):
            seen_prompts.extend(batch)
            return [{"generated_text": "NO"} for _ in batch]

    checkpoint_a = RunCheckpoint(tmp_path / "order_a")
    evaluate_all_variants_sequential(
        ["fake_model"], test_df, demos_df, variants, checkpoint_a,
        load_fn=lambda key: (_RecordingPipe(), _FakeTokenizer()), unload_fn=lambda: None,
        model_registry={}, component_order=["aspects", "context"],
    )
    prompt_order_a = seen_prompts[0]
    seen_prompts.clear()

    checkpoint_b = RunCheckpoint(tmp_path / "order_b")
    evaluate_all_variants_sequential(
        ["fake_model"], test_df, demos_df, variants, checkpoint_b,
        load_fn=lambda key: (_RecordingPipe(), _FakeTokenizer()), unload_fn=lambda: None,
        model_registry={}, component_order=["context", "aspects"],
    )
    prompt_order_b = seen_prompts[0]

    assert prompt_order_a != prompt_order_b  # different order -> different rendered prompt


def test_units_are_timed_with_elapsed_and_per_example_seconds(tmp_path):
    test_df, demos_df = _toy_data()  # 2 rows
    variants = [{"id": "V1", "cot": False}]
    checkpoint = RunCheckpoint(tmp_path / "ckpt")

    class _SlowPipe:
        def __call__(self, batch, **kwargs):
            time.sleep(0.02)
            return [{"generated_text": "NO"} for _ in batch]

    df = evaluate_all_variants_sequential(
        ["fake_model"], test_df, demos_df, variants, checkpoint,
        load_fn=lambda key: (_SlowPipe(), _FakeTokenizer()), unload_fn=lambda: None,
        model_registry={},
    )
    row = df.iloc[0]
    assert row["elapsed_seconds"] >= 0.02
    assert row["seconds_per_example"] == pytest.approx(row["elapsed_seconds"] / 2)

    uid = RunCheckpoint.unit_id("V1", "fake_model")
    assert "elapsed_seconds" in checkpoint.metrics_for(uid)


def test_early_stop_preserves_completed_rows_for_unreached_models(tmp_path):
    """When `should_continue` stops the sweep mid-grid, a model later in
    `model_keys` that the loop's iteration never reaches must still appear
    in the returned DataFrame if it was already fully checkpointed in an
    earlier session. Reusing already-completed rows lazily inside the same
    loop that `break`s early would drop that model instead: the manifest
    still has it, but a caller trusting this return value (e.g.
    `save_results(df, ..., append=False)`) would silently lose it."""
    test_df, demos_df = _toy_data()
    variants = [{"id": "V1", "cot": False}]
    ckpt_dir = tmp_path / "ckpt"

    # Session 1: only "model_b" runs, completes, gets checkpointed.
    checkpoint1 = RunCheckpoint(ckpt_dir)
    evaluate_all_variants_sequential(
        ["model_b"], test_df, demos_df, variants, checkpoint1,
        load_fn=lambda key: (_FakePipe(), _FakeTokenizer()), unload_fn=lambda: None,
        model_registry={},
    )
    assert checkpoint1.is_done(RunCheckpoint.unit_id("V1", "model_b"))

    # Session 2: "model_a" (still pending) comes first in model_keys,
    # "model_b" (already done) comes after. should_continue trips false
    # immediately, before the loop ever reaches model_b's iteration.
    checkpoint2 = RunCheckpoint(ckpt_dir)
    df = evaluate_all_variants_sequential(
        ["model_a", "model_b"], test_df, demos_df, variants, checkpoint2,
        load_fn=lambda key: (_FakePipe(), _FakeTokenizer()), unload_fn=lambda: None,
        model_registry={}, should_continue=lambda: False,
    )
    assert not checkpoint2.is_done(RunCheckpoint.unit_id("V1", "model_a"))
    assert len(df) == 1
    assert df.iloc[0]["model"] == "model_b"


def test_should_continue_false_prevents_model_load(tmp_path):
    test_df, demos_df = _toy_data()
    variants = [{"id": "V1", "cot": False}]
    checkpoint = RunCheckpoint(tmp_path / "ckpt")
    load_calls = []

    df = evaluate_all_variants_sequential(
        ["fake_model"], test_df, demos_df, variants, checkpoint,
        load_fn=lambda key: load_calls.append(key) or (_FakePipe(), _FakeTokenizer()),
        unload_fn=lambda: None,
        model_registry={},
        should_continue=lambda: False,
    )
    assert load_calls == []  # never paid the ~minutes-long model load
    assert df.empty
