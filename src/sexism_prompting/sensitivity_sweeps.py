"""Prompt-component-order and few-shot-size sensitivity sweeps.

Two robustness checks on the main grid's "best" result per model:

- **Prompt-component-order**: ``generate_shuffled_orders()`` (in
  ``prompts.py``) renders each model's best-performing variant under
  multiple orderings of its active prompt components, to check whether
  component position influences the result.
- **Few-shot size**: sweeps ``fewshot_num_per_class`` itself, as opposed to
  which demonstrations are selected or their seed, against the grid's
  fixed 2-per-class choice.

Both sweeps target each model's own best-performing variant from the main
grid: re-running the full 18-variant grid along a second axis would be
prohibitively expensive, and the question these sweeps ask ("is the
headline result fragile to this one design choice?") is specifically about
that best variant, not the whole grid. Neither sweep can know in advance
*which* variant is best for a given model; that only exists once the real
full-EDOS main grid has actually produced results, so
``best_variant_per_model`` reads it live from the main grid's results CSV
rather than a hardcoded guess.

Both sweeps run via the same ``evaluate_all_variants_sequential`` engine as
the main grid, scoped to one (model, variant) pair per call, and always
write to a checkpoint sub-directory distinct from both the main grid's and
each other's, never touching or colliding with the main grid's manifest,
and never colliding with each other despite reusing the same
``variant_id``/``model_key`` (which alone is not a unique checkpoint key
once the same variant is evaluated under multiple orders/sizes; see each
``run_*_sweep_unit`` function for the directory-per-condition scheme this
relies on instead of extending ``RunCheckpoint``'s unit-id format).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .checkpoint import RunCheckpoint
from .evaluate import evaluate_all_variants_sequential
from .prompts import DEFAULT_COMPONENT_ORDER, generate_shuffled_orders

DEFAULT_FEWSHOT_SIZES: Tuple[int, ...] = (1, 3, 4)


def results_df_from_checkpoint(checkpoint: RunCheckpoint) -> pd.DataFrame:
    """Reconstruct a ``[variant_id, model, <metrics...>]`` DataFrame directly
    from a main-grid checkpoint's manifest, for use with
    ``best_variant_per_model`` when a saved ``results.csv`` isn't available.
    Parses ``unit_id`` back into ``(variant_id, model_key)`` via
    ``RunCheckpoint.unit_id``'s own ``f"{variant_id}__{model_key}"`` format.
    """
    rows = []
    for unit_id in checkpoint.completed_units():
        variant_id, _, model_key = unit_id.partition("__")
        metrics = checkpoint.metrics_for(unit_id) or {}
        rows.append({"variant_id": variant_id, "model": model_key, **metrics})
    return pd.DataFrame(rows)


def best_variant_per_model(results_df: pd.DataFrame, metric: str = "f1") -> Dict[str, str]:
    """``{model_key: variant_id}`` for each model's single best-scoring row
    in a main-grid results DataFrame (as returned by
    ``evaluate_all_variants_sequential`` or read back from its saved CSV).
    Empty dict for an empty/missing DataFrame; callers should treat that as
    "the main grid hasn't produced results yet", not silently sweep nothing.
    """
    if results_df.empty:
        return {}
    idx = results_df.groupby("model")[metric].idxmax()
    best = results_df.loc[idx]
    return dict(zip(best["model"], best["variant_id"]))


def is_order_sensitive(flags: Dict[str, bool], component_order: Optional[Sequence[str]] = None) -> bool:
    """True iff at least 2 of the shufflable components are active.

    With 0 or 1 active components from ``component_order``, every ordering
    renders an IDENTICAL prompt (inactive components never render
    regardless of their position; see ``prompts.build_variant_messages``),
    so shuffling would spend real GPU time measuring exactly zero signal.
    """
    order = component_order if component_order is not None else DEFAULT_COMPONENT_ORDER
    return sum(1 for name in order if flags.get(name, False)) >= 2


def uses_fewshot(flags: Dict[str, bool]) -> bool:
    return bool(flags.get("fewshot", False))


def _flags_of(variants_by_id: Dict[str, Dict], variant_id: str) -> Dict[str, bool]:
    return {k: v for k, v in variants_by_id[variant_id].items() if k != "id"}


def _induced_order(order: Sequence[str], flags: Dict[str, bool]) -> Tuple[str, ...]:
    """``order`` restricted to the components actually active in ``flags``.

    Two full component orders that differ only in the relative position of
    INACTIVE components render an identical prompt (see
    ``prompts.build_variant_messages``: an inactive component never
    renders regardless of where it sits), so this is the real dimension a
    "distinct ordering" sweep needs to vary, not the raw permutation.
    """
    return tuple(name for name in order if flags.get(name, False))


def plan_order_sweep(
    results_df: pd.DataFrame,
    variants_by_id: Dict[str, Dict],
    n_orders: int = 5,
    order_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Planned (model, order) units plus the list of models skipped because
    their best variant isn't order-sensitive (see ``is_order_sensitive``).
    Each planned unit: ``{model_key, variant_id, order_index, component_order}``.

    Deduplicates by ``_induced_order`` before capping at ``n_orders``: a raw
    shuffle of all 4 components in ``DEFAULT_COMPONENT_ORDER`` collapses to
    far fewer distinct RENDERED prompts once restricted to a variant's own
    active components (e.g. only 2 or 3 of the 4 are active for most
    variants). Without deduping, most of ``n_orders`` planned "different
    orderings" would actually re-measure the identical prompt under a
    different label, burning real GPU time for zero additional signal.
    Draws from the full permutation space (``generate_shuffled_orders``'
    own ``n`` cap, ``math.factorial(len(DEFAULT_COMPONENT_ORDER))``) so
    every achievable distinct induced order has a chance to be found; if a
    variant's active-component count is small enough that fewer than
    ``n_orders`` distinct induced orders exist at all (e.g. 2 active
    components -> only 2 possible orderings), the plan is capped there
    rather than padded with a repeat.
    """
    plan: List[Dict[str, Any]] = []
    skipped: List[str] = []
    max_candidates = math.factorial(len(DEFAULT_COMPONENT_ORDER))
    for model_key, variant_id in sorted(best_variant_per_model(results_df).items()):
        flags = _flags_of(variants_by_id, variant_id)
        if not is_order_sensitive(flags):
            skipped.append(model_key)
            continue
        candidates = generate_shuffled_orders(DEFAULT_COMPONENT_ORDER, n=max_candidates, seed=order_seed)
        seen_induced = set()
        i = 0
        for order in candidates:
            induced = _induced_order(order, flags)
            if induced in seen_induced:
                continue
            seen_induced.add(induced)
            plan.append(
                {"model_key": model_key, "variant_id": variant_id, "order_index": i, "component_order": order}
            )
            i += 1
            if i >= n_orders:
                break
    return plan, skipped


def plan_size_sweep(
    results_df: pd.DataFrame,
    variants_by_id: Dict[str, Dict],
    sizes: Sequence[int] = DEFAULT_FEWSHOT_SIZES,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Planned (model, size) units plus the list of models skipped because
    their best variant doesn't use few-shot at all.
    Each planned unit: ``{model_key, variant_id, fewshot_num_per_class}``.
    """
    plan: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for model_key, variant_id in sorted(best_variant_per_model(results_df).items()):
        flags = _flags_of(variants_by_id, variant_id)
        if not uses_fewshot(flags):
            skipped.append(model_key)
            continue
        for size in sizes:
            plan.append({"model_key": model_key, "variant_id": variant_id, "fewshot_num_per_class": size})
    return plan, skipped


DEFAULT_FEWSHOT_SEEDS: Tuple[int, ...] = (1, 7, 123)


def plan_seed_sweep(
    results_df: pd.DataFrame,
    variants_by_id: Dict[str, Dict],
    seeds: Sequence[int] = DEFAULT_FEWSHOT_SEEDS,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Planned (model, seed) units plus the list of models skipped because
    their best variant doesn't use few-shot at all.

    Complementary to ``plan_size_sweep``: that sweep varies how many
    demonstrations are shown, this one varies which specific
    demonstrations are drawn (``build_few_shot_demonstrations``'s own
    ``random_state=seed``), holding count fixed at the main grid's default
    2 per class. Neither sweep answers the other's question; the main
    grid's fixed ``seed=42`` selection is otherwise never varied anywhere
    in this study.

    Each planned unit: ``{model_key, variant_id, fewshot_seed}``.
    """
    plan: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for model_key, variant_id in sorted(best_variant_per_model(results_df).items()):
        flags = _flags_of(variants_by_id, variant_id)
        if not uses_fewshot(flags):
            skipped.append(model_key)
            continue
        for seed in seeds:
            plan.append({"model_key": model_key, "variant_id": variant_id, "fewshot_seed": seed})
    return plan, skipped


def run_order_sweep_unit(
    unit: Dict[str, Any],
    test_df: pd.DataFrame,
    demos_df: pd.DataFrame,
    variants_by_id: Dict[str, Dict],
    checkpoint_root: str | Path,
    fewshot_num_per_class: int = 2,
    fewshot_seed: int = 42,
    **evaluate_kwargs: Any,
) -> pd.DataFrame:
    """Run one planned order-sweep unit (see ``plan_order_sweep``) against its
    own checkpoint sub-directory, ``<root>/order_sweep/<model>/order_<i>/``,
    distinct per (model, order_index) so re-running the same variant under a
    different order is never mistaken for, and never overwrites, another
    order's result (the two runs share a ``variant_id``/``model_key`` but
    that alone is not a unique checkpoint key across different orders).
    """
    variant = variants_by_id[unit["variant_id"]]
    checkpoint = RunCheckpoint(
        Path(checkpoint_root) / "order_sweep" / unit["model_key"] / f"order_{unit['order_index']}"
    )
    return evaluate_all_variants_sequential(
        [unit["model_key"]], test_df, demos_df, [variant], checkpoint,
        fewshot_num_per_class=fewshot_num_per_class, fewshot_seed=fewshot_seed,
        component_order=unit["component_order"],
        **evaluate_kwargs,
    )


def run_size_sweep_unit(
    unit: Dict[str, Any],
    test_df: pd.DataFrame,
    demos_df: pd.DataFrame,
    variants_by_id: Dict[str, Dict],
    checkpoint_root: str | Path,
    fewshot_seed: int = 42,
    **evaluate_kwargs: Any,
) -> pd.DataFrame:
    """Run one planned size-sweep unit (see ``plan_size_sweep``) against its
    own checkpoint sub-directory, ``<root>/size_sweep/<model>/size_<n>/``,
    distinct per (model, size) for the same reason ``run_order_sweep_unit``
    uses a directory per order.
    """
    variant = variants_by_id[unit["variant_id"]]
    checkpoint = RunCheckpoint(
        Path(checkpoint_root) / "size_sweep" / unit["model_key"] / f"size_{unit['fewshot_num_per_class']}"
    )
    return evaluate_all_variants_sequential(
        [unit["model_key"]], test_df, demos_df, [variant], checkpoint,
        fewshot_num_per_class=unit["fewshot_num_per_class"], fewshot_seed=fewshot_seed,
        **evaluate_kwargs,
    )


def run_seed_sweep_unit(
    unit: Dict[str, Any],
    test_df: pd.DataFrame,
    demos_df: pd.DataFrame,
    variants_by_id: Dict[str, Dict],
    checkpoint_root: str | Path,
    fewshot_num_per_class: int = 2,
    **evaluate_kwargs: Any,
) -> pd.DataFrame:
    """Run one planned seed-sweep unit (see ``plan_seed_sweep``) against its
    own checkpoint sub-directory, ``<root>/seed_sweep/<model>/seed_<n>/``,
    distinct per (model, seed) for the same reason ``run_size_sweep_unit``
    uses a directory per size.
    """
    variant = variants_by_id[unit["variant_id"]]
    checkpoint = RunCheckpoint(
        Path(checkpoint_root) / "seed_sweep" / unit["model_key"] / f"seed_{unit['fewshot_seed']}"
    )
    return evaluate_all_variants_sequential(
        [unit["model_key"]], test_df, demos_df, [variant], checkpoint,
        fewshot_num_per_class=fewshot_num_per_class, fewshot_seed=unit["fewshot_seed"],
        **evaluate_kwargs,
    )
