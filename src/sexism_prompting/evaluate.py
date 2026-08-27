"""Run prompt variants across models, with a model-outer loop for sequential
loading (see ``models.load_one``/``unload_one``) and resumable checkpointing
(see ``checkpoint.RunCheckpoint``).

A single variant-flag-driven path handles every variant combination.
Looping model-outer keeps at most one model's weights in GPU memory at a
time.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from .checkpoint import RunCheckpoint
from .inference import gen_kwargs_for, generate_responses
from .metrics import compute_metrics
from .prompts import build_few_shot_demonstrations, build_variant_messages, render_prompts, variant_name

# `models.py` imports torch/transformers at module level, which aren't
# installed in GPU-less environments (this dev sandbox included). Importing
# it lazily, only inside the function that actually loads a model, keeps the
# rest of this module (load_prompt_variants, predict_with_model, evaluate_variant)
# importable and testable without the GPU stack.


def load_prompt_variants(path: str | Path) -> Tuple[List[Dict], int, int]:
    """Load the variant flag-grid and few-shot sampling params from a YAML config.

    Returns (variants, fewshot_num_per_class, fewshot_seed).
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg["variants"], cfg.get("fewshot_num_per_class", 2), cfg.get("fewshot_seed", 42)


def predict_with_model(
    pipe_tokenizer: Tuple,
    test_df: pd.DataFrame,
    demos_df: pd.DataFrame,
    flags: Dict[str, bool],
    fewshot_num_per_class: int = 2,
    fewshot_seed: int = 42,
    batch_size: int = 32,
    gen_kwargs: Optional[Dict[str, Any]] = None,
    cot_gen_kwargs: Optional[Dict[str, Any]] = None,
    component_order: Optional[List[str]] = None,
) -> List[str]:
    """Generate raw responses for one (model, variant) pair, without scoring.

    Kept separate from scoring so raw per-example responses are always
    available for later statistical analysis, not just aggregate metrics.
    """
    pipe, tokenizer = pipe_tokenizer
    texts = test_df["text"].tolist()
    is_cot = flags.get("cot", False)

    examples = None
    if flags.get("fewshot", False):
        examples = build_few_shot_demonstrations(demos_df, num_per_class=fewshot_num_per_class, seed=fewshot_seed)

    template = build_variant_messages(flags, examples=examples, component_order=component_order)
    prompts = render_prompts(template, texts, tokenizer, enable_thinking=is_cot)

    kwargs = gen_kwargs_for(is_cot, cot_gen_kwargs if is_cot else gen_kwargs)
    if "stop_strings" in kwargs:
        # generate()'s stop_strings needs the tokenizer to match candidate
        # stop strings against generated tokens; the pipeline doesn't
        # auto-forward its own constructor-time tokenizer for this even
        # though it already has one internally (see
        # inference.COT_GEN_KWARGS's comment).
        kwargs = {**kwargs, "tokenizer": tokenizer}
    return generate_responses(pipe, prompts, batch_size=batch_size, gen_kwargs=kwargs)


def _data_fingerprint(test_df: pd.DataFrame) -> str:
    """A short hash of the test set's actual content (text + label), so a
    changed data file is detected instead of silently reused under an
    unchanged variant_id/model_key checkpoint key."""
    payload = test_df[["text", "label"]].to_json(orient="records")
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _config_fingerprint(
    template: List[Dict[str, str]],
    resolved_gen_kwargs: Dict[str, Any],
    model_entry: Any,
    data_hash: str,
) -> str:
    """A short hash of everything that determines a unit's expected output:
    the rendered prompt template (flags + component wording + the concrete
    few-shot examples used), the resolved generation kwargs, the model
    registry's target (id/revision), and the test data's content. Stored
    alongside each checkpointed unit so a later prompt/generation/data/model
    change is detected instead of an old result being silently reused (see
    ``RunCheckpoint.is_done``).
    """
    payload = json.dumps(
        {
            "template": template,
            "gen_kwargs": resolved_gen_kwargs,
            "model_entry": model_entry,
            "data_hash": data_hash,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def evaluate_all_variants_sequential(
    model_keys: List[str],
    test_df: pd.DataFrame,
    demos_df: pd.DataFrame,
    variants: List[Dict],
    checkpoint: RunCheckpoint,
    fewshot_num_per_class: int = 2,
    fewshot_seed: int = 42,
    batch_size: int = 32,
    gen_kwargs: Optional[Dict[str, Any]] = None,
    cot_gen_kwargs: Optional[Dict[str, Any]] = None,
    on_unit_done: Optional[Callable[[str, Dict], None]] = None,
    load_fn: Optional[Callable[[str], Tuple]] = None,
    unload_fn: Optional[Callable[[], None]] = None,
    model_registry: Optional[Dict[str, Any]] = None,
    should_continue: Optional[Callable[[], bool]] = None,
    failure_threshold: float = 0.5,
    failure_threshold_overrides: Optional[Dict[str, float]] = None,
    cot_gen_kwargs_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    batch_size_overrides: Optional[Dict[str, int]] = None,
    component_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Model-outer sweep of every variant, resumable via ``checkpoint``.

    ``component_order``: overrides ``prompts.DEFAULT_COMPONENT_ORDER`` for
    every unit in this call. Used by the prompt-order sensitivity sweep
    (``sensitivity_sweeps.py``), which calls this once per (model, order)
    pair with a single-variant list. Left ``None`` for the main grid, which
    always wants the default order.

    For each model: load once, run every variant not already in the
    checkpoint's manifest, then unload before moving to the next model. A
    killed/restarted run re-enters this loop, skips every ``(variant, model)``
    unit already marked done, and picks up exactly where it left off.

    ``load_fn``/``unload_fn``/``model_registry`` default to
    ``models.load_one``/``unload_one``/``models.MODEL_REGISTRY`` (imported
    lazily so this module stays importable without the GPU stack);
    overriding them with fakes is how tests exercise the resume/skip
    orchestration logic without needing torch installed. ``unload_fn`` takes
    no arguments; it's called only after this loop has already dropped its
    own reference to the just-finished model (see the ``finally`` block
    below), since reclaiming GPU memory only works once nothing still
    references what's being reclaimed.

    Each unit's checkpoint entry is tagged with a config fingerprint (hash of
    the rendered prompt template, resolved generation kwargs, the model
    registry's target, and the test data's content) so a later prompt/
    generation/data/model change is detected as a resume-invalidating change
    instead of an old result being silently reused (see
    ``RunCheckpoint.is_done``). Manifest entries from before this existed
    have no stored fingerprint and are trusted as-is.

    ``should_continue`` (if given) is checked before loading each model and
    before starting each unit; once it returns False the sweep stops cleanly
    instead of letting the platform hard-kill the session mid-unit. The
    caller can then run its final checkpoint sync and exit with a normal
    terminal state. Units are never left half-recorded by this: the check
    happens only at unit boundaries.

    ``failure_threshold``: a unit whose ``fail_ratio`` is at or above this is
    NOT checkpointed (it stays pending for a retry in a later session). This
    is what stops a transient batch-level generation failure (e.g. CUDA OOM
    filling whole batches with empty responses) from being frozen into the
    grid as an all-"NO" unit that every future resume silently skips. If a
    (model, variant) pair *legitimately* fails this often, raise the
    threshold in the config; the warning it prints says exactly that.

    ``failure_threshold_overrides``: an optional ``{unit_id: threshold}``
    map (unit_id is ``RunCheckpoint.unit_id(variant_id, model_key)``, e.g.
    ``"V8__llama3.1"``) for (model, variant) pairs whose genuine failure
    rate has already been confirmed rather than guessed at. Raising the
    single global ``failure_threshold`` would also loosen the guard for
    every other unit, most of which sit near a 0 fail_ratio with no need
    for slack. A unit not listed here still uses the global default.

    ``cot_gen_kwargs_overrides``: an optional ``{model_key: gen_kwargs}`` map,
    merged on top of the global ``cot_gen_kwargs`` for that model's CoT
    units only (non-CoT units are unaffected). Exists for reasoning-tuned
    models (e.g. a Qwen3-class model), whose own extended "thinking" can run
    far longer than the ~220-token budget calibrated for non-reasoning
    models' brief step-by-step CoT. Without a larger budget, the model's
    reasoning gets cut off before it ever reaches the required trailing
    ``ANSWER: YES``/``ANSWER: NO``, which is exactly the "forced cue leaves
    no room to reason" bug this project already fixed once, recurring for a
    new model class if left unhandled.

    ``batch_size_overrides``: an optional ``{model_key: batch_size}`` map,
    replacing (not merging with) the global ``batch_size`` for that model's
    units (all of them, CoT and non-CoT alike: unlike
    ``cot_gen_kwargs_overrides``, batch size isn't CoT-specific). Exists for
    the same reason as the CoT budget override: a model whose per-unit
    memory footprint is unusually large for the global default (e.g. a
    reasoning-tuned model generating a much longer CoT budget) risks a
    CUDA-OOM-driven ``fail_ratio`` spike at the default batch size, which
    would otherwise just retry forever against the same OOM every session
    (see ``failure_threshold``) rather than actually succeeding at a
    smaller batch size. A model not listed here still uses the global
    default.

    A unit that raises is likewise skipped (warned, not checkpointed) rather
    than aborting the session; the model-load try/except already gave models
    the same courtesy.

    Every unit's metrics also include ``elapsed_seconds`` (wall-clock time
    for that unit's ``predict_with_model`` call) and ``seconds_per_example``
    (``elapsed_seconds`` divided by row count), for computational-cost
    reporting. Wall-clock rather than a token-count-based cost proxy: it's
    what a moderation team
    actually cares about ("how long until I have a decision"), and doesn't
    need a second tokenizer pass to compute. Timed around generation only
    (not the surrounding metric computation, which is negligible and
    CPU-only), so this reflects real GPU inference cost, not bookkeeping.
    Units already done from a prior run keep whatever ``elapsed_seconds``
    was recorded when they first completed (not re-measured on resume).

    Returns a tidy DataFrame with one row per (variant, model), including
    units that were already done from a prior run, not just ones computed
    now. If the sweep stopped early via ``should_continue``, rows for models
    it never reached are not included (the manifest still has them).
    """
    if load_fn is None or unload_fn is None:
        from .models import load_one, unload_one

        load_fn = load_fn or load_one
        unload_fn = unload_fn or unload_one
    if model_registry is None:
        from .models import MODEL_REGISTRY

        model_registry = MODEL_REGISTRY

    y_true = test_df["label"].tolist()
    rewire_ids = test_df["rewire_id"].tolist() if "rewire_id" in test_df.columns else None
    rows: List[Dict] = []
    stopped_early = False
    data_hash = _data_fingerprint(test_df)

    def _fingerprint_for(variant: Dict, model_entry: Any, cot_kwargs: Dict[str, Any]) -> Optional[str]:
        # Best-effort: computing this reuses the same prompt-building
        # calls predict_with_model makes (e.g. few-shot sampling), which
        # can raise for the same reasons a unit itself can fail (too few
        # demos, etc). That must not abort the whole model's loop here:
        # None falls back to presence-only is_done checking below, and a
        # not-yet-done variant still reaches the real per-unit try/except
        # in the generation loop, where the same failure is handled
        # properly (warned, skipped, not checkpointed).
        try:
            flags = {k: v for k, v in variant.items() if k != "id"}
            is_cot = flags.get("cot", False)
            examples = None
            if flags.get("fewshot", False):
                examples = build_few_shot_demonstrations(
                    demos_df, num_per_class=fewshot_num_per_class, seed=fewshot_seed
                )
            template = build_variant_messages(flags, examples=examples, component_order=component_order)
            resolved_kwargs = gen_kwargs_for(is_cot, cot_kwargs if is_cot else gen_kwargs)
            return _config_fingerprint(template, resolved_kwargs, model_entry, data_hash)
        except Exception:  # noqa: BLE001
            return None

    # Fingerprints (and any already-completed rows) are computed for every
    # model up front, not lazily as the loop below reaches each one. An
    # early `should_continue` stop `break`s out of that loop before it ever
    # visits later models; computing this lazily inside that same loop
    # would mean a stopped-early run silently drops historical rows for
    # every model it never got back to, even though the checkpoint manifest
    # still has them (a caller trusting this return value, e.g.
    # `save_results(df, ..., append=False)`, would silently lose them).
    fingerprints_by_model: Dict[str, Dict[str, Optional[str]]] = {}
    for model_key in model_keys:
        model_entry = model_registry.get(model_key, model_key)
        effective_cot_gen_kwargs = {
            **(cot_gen_kwargs or {}),
            **((cot_gen_kwargs_overrides or {}).get(model_key, {})),
        }
        fingerprints = {
            variant["id"]: _fingerprint_for(variant, model_entry, effective_cot_gen_kwargs) for variant in variants
        }
        fingerprints_by_model[model_key] = fingerprints
        for variant in variants:
            uid = RunCheckpoint.unit_id(variant["id"], model_key)
            if checkpoint.is_done(uid, fingerprints[variant["id"]]):
                metrics = checkpoint.metrics_for(uid)
                rows.append({"variant_id": variant["id"], "variant_name": variant_name(
                    {k: v for k, v in variant.items() if k != "id"}), "model": model_key, **metrics})

    for model_key in model_keys:
        effective_batch_size = (batch_size_overrides or {}).get(model_key, batch_size)
        effective_cot_gen_kwargs = {
            **(cot_gen_kwargs or {}),
            **((cot_gen_kwargs_overrides or {}).get(model_key, {})),
        }
        fingerprints = fingerprints_by_model[model_key]

        pending = [
            v for v in variants
            if not checkpoint.is_done(RunCheckpoint.unit_id(v["id"], model_key), fingerprints[v["id"]])
        ]
        if not pending:
            continue

        if should_continue is not None and not should_continue():
            print(f"Time budget reached. Stopping before loading '{model_key}'. A resumed run picks up here.")
            stopped_early = True
            break

        print(f"Loading model: {model_key}")
        try:
            loaded = load_fn(model_key)
        except Exception as e:  # noqa: BLE001
            # A single model failing to load (e.g. a gated model with no
            # valid HF token) must not abort the whole grid: every other
            # model's progress up to this point is already checkpointed,
            # and every model after this one should still get its turn.
            print(f"Warning: failed to load '{model_key}' ({e!r}). Skipping this model for this pass.")
            continue

        try:
            for variant in pending:
                if should_continue is not None and not should_continue():
                    print(f"Time budget reached. Stopping before unit '{variant['id']}__{model_key}'. A resumed run picks up here.")
                    stopped_early = True
                    break

                flags = {k: v for k, v in variant.items() if k != "id"}
                uid = RunCheckpoint.unit_id(variant["id"], model_key)
                fp = fingerprints[variant["id"]]

                try:
                    unit_start = time.monotonic()
                    responses = predict_with_model(
                        loaded,
                        test_df,
                        demos_df,
                        flags,
                        fewshot_num_per_class=fewshot_num_per_class,
                        fewshot_seed=fewshot_seed,
                        batch_size=effective_batch_size,
                        gen_kwargs=gen_kwargs,
                        cot_gen_kwargs=effective_cot_gen_kwargs,
                        component_order=component_order,
                    )
                    elapsed_seconds = time.monotonic() - unit_start
                    metrics = compute_metrics(responses, y_true)
                    metrics["elapsed_seconds"] = elapsed_seconds
                    metrics["seconds_per_example"] = elapsed_seconds / len(responses) if responses else float("nan")
                except Exception as e:  # noqa: BLE001
                    print(f"Warning: unit '{uid}' failed ({e!r}). Not checkpointing it; it stays pending for a future run. Continuing with the next unit.")
                    continue

                effective_threshold = (failure_threshold_overrides or {}).get(uid, failure_threshold)
                if metrics["fail_ratio"] >= effective_threshold:
                    reason = (
                        f"fail_ratio={metrics['fail_ratio']:.3f} >= failure_threshold={effective_threshold}"
                    )
                    print(
                        f"Warning: unit '{uid}' {reason}: "
                        f"treating this as a generation failure and NOT checkpointing it, so a future session retries it. "
                        f"If this rate is genuine for this (model, variant), raise `failure_threshold` in the config "
                        f"(or add a per-unit override in `failure_threshold_overrides`)."
                    )
                    checkpoint.save_failure_evidence(
                        uid, responses, reason=reason, rewire_ids=rewire_ids, labels=y_true, config_fingerprint=fp
                    )
                    continue

                checkpoint.mark_done(
                    uid, metrics, predictions=responses, rewire_ids=rewire_ids, labels=y_true, config_fingerprint=fp
                )
                rows.append({"variant_id": variant["id"], "variant_name": variant_name(flags), "model": model_key, **metrics})
                if on_unit_done is not None:
                    on_unit_done(uid, metrics)
        finally:
            # Drop this loop's own reference BEFORE reclaiming memory, not
            # after: gc.collect()/empty_cache() can only free objects that
            # are already unreachable, and passing `loaded` into unload_fn
            # itself keeps it reachable for the entire duration of that
            # call. Without dropping it first, the previous model would
            # stay fully resident (and VRAM unreleased) for the entire
            # duration of the *next* model's from_pretrained() call.
            loaded = None
            unload_fn()

        if stopped_early:
            break

    return pd.DataFrame(rows)


# --- Simultaneous-loading path: kept for local/debug use on a small model
# subset that fits in memory at once. Not used for the full 6-model grid;
# see evaluate_all_variants_sequential above for that.


def evaluate_variant(
    models: Dict[str, Tuple],
    test_df: pd.DataFrame,
    demos_df: pd.DataFrame,
    flags: Dict[str, bool],
    fewshot_num_per_class: int = 2,
    fewshot_seed: int = 42,
    batch_size: int = 32,
    gen_kwargs: Optional[Dict[str, Any]] = None,
    cot_gen_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict]:
    """Evaluate every already-loaded model on a single prompt variant."""
    y_true = test_df["label"].tolist()
    results = {}
    for model_key, loaded in models.items():
        responses = predict_with_model(
            loaded,
            test_df,
            demos_df,
            flags,
            fewshot_num_per_class=fewshot_num_per_class,
            fewshot_seed=fewshot_seed,
            batch_size=batch_size,
            gen_kwargs=gen_kwargs,
            cot_gen_kwargs=cot_gen_kwargs,
        )
        results[model_key] = compute_metrics(responses, y_true)
    return results
