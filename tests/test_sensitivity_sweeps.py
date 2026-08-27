import pandas as pd

from sexism_prompting.checkpoint import RunCheckpoint
from sexism_prompting.sensitivity_sweeps import (
    best_variant_per_model,
    is_order_sensitive,
    plan_order_sweep,
    plan_seed_sweep,
    plan_size_sweep,
    results_df_from_checkpoint,
    run_order_sweep_unit,
    run_seed_sweep_unit,
    run_size_sweep_unit,
    uses_fewshot,
)

VARIANTS_BY_ID = {
    "V1": {"id": "V1", "role": False, "aspects": False, "context": False, "cot": False, "fewshot": False},
    "V6": {"id": "V6", "role": False, "aspects": False, "context": False, "cot": False, "fewshot": True},
    "V16": {"id": "V16", "role": True, "aspects": True, "context": True, "cot": True, "fewshot": False},
    "V18": {"id": "V18", "role": True, "aspects": True, "context": True, "cot": True, "fewshot": True},
}


def _results_df():
    # model_a's best is V1 (baseline, order-insensitive, no fewshot).
    # model_b's best is V6 (fewshot only -> not order-sensitive, but uses fewshot).
    # model_c's best is V16 (3 active shufflable components -> order-sensitive, no fewshot).
    return pd.DataFrame(
        [
            {"model": "model_a", "variant_id": "V1", "f1": 0.5},
            {"model": "model_a", "variant_id": "V6", "f1": 0.4},
            {"model": "model_b", "variant_id": "V6", "f1": 0.7},
            {"model": "model_b", "variant_id": "V1", "f1": 0.6},
            {"model": "model_c", "variant_id": "V16", "f1": 0.8},
            {"model": "model_c", "variant_id": "V1", "f1": 0.3},
        ]
    )


def test_best_variant_per_model_picks_max_f1_row():
    best = best_variant_per_model(_results_df())
    assert best == {"model_a": "V1", "model_b": "V6", "model_c": "V16"}


def test_best_variant_per_model_empty_results_returns_empty():
    assert best_variant_per_model(pd.DataFrame()) == {}


def test_is_order_sensitive_requires_at_least_two_active_components():
    assert is_order_sensitive({"context": False, "cot": False, "aspects": False, "fewshot": False}) is False
    assert is_order_sensitive({"context": True, "cot": False, "aspects": False, "fewshot": False}) is False
    assert is_order_sensitive({"context": True, "cot": True, "aspects": False, "fewshot": False}) is True


def test_is_order_sensitive_fewshot_only_is_not_order_sensitive():
    """V6-shaped variant: only 'fewshot' active among the shufflable
    components -> every shuffled order renders identically, no signal."""
    assert is_order_sensitive({"fewshot": True}) is False


def test_uses_fewshot():
    assert uses_fewshot({"fewshot": True}) is True
    assert uses_fewshot({"fewshot": False}) is False
    assert uses_fewshot({}) is False


def test_plan_order_sweep_skips_non_order_sensitive_models():
    plan, skipped = plan_order_sweep(_results_df(), VARIANTS_BY_ID, n_orders=3)
    assert skipped == ["model_a", "model_b"]  # V1 and V6 both order-insensitive
    assert {u["model_key"] for u in plan} == {"model_c"}
    assert len(plan) == 3  # n_orders=3, one model qualified
    assert {u["order_index"] for u in plan} == {0, 1, 2}
    assert all(u["variant_id"] == "V16" for u in plan)


def test_plan_order_sweep_dedupes_by_induced_active_component_order():
    """A variant with fewer active shufflable components than n_orders must
    not get n_orders raw permutations of the FULL 4-component order:
    several of those permutations render an IDENTICAL prompt once
    restricted to just the active components, which would waste real GPU
    time re-measuring the same rendering under a different order_index
    label instead of a genuinely different one."""
    variants_by_id = {
        **VARIANTS_BY_ID,
        "V_TF": {"id": "V_TF", "role": False, "aspects": False, "context": False, "cot": True, "fewshot": True},
    }
    results = pd.DataFrame([{"model": "model_d", "variant_id": "V_TF", "f1": 0.9}])
    plan, skipped = plan_order_sweep(results, variants_by_id, n_orders=5)
    assert skipped == []
    assert len(plan) == 2  # only 2 distinct orderings of {cot, fewshot} exist at all
    induced = [tuple(c for c in u["component_order"] if c in ("cot", "fewshot")) for u in plan]
    assert len(set(induced)) == 2  # every planned unit is a genuinely distinct rendered prompt
    assert {u["order_index"] for u in plan} == {0, 1}


def test_plan_order_sweep_caps_at_n_orders_when_enough_distinct_orders_exist():
    """A 3-active-component variant (3! = 6 possible distinct orderings)
    requesting n_orders=3 should still land exactly 3 distinct planned
    units: dedup must not under-plan when there was never a duplication
    problem to begin with."""
    plan, skipped = plan_order_sweep(_results_df(), VARIANTS_BY_ID, n_orders=3)
    assert skipped == ["model_a", "model_b"]
    assert len(plan) == 3
    induced = [tuple(c for c in u["component_order"] if c in ("context", "cot", "aspects")) for u in plan]
    assert len(set(induced)) == 3


def test_plan_order_sweep_no_qualifying_models_returns_empty_plan():
    results = pd.DataFrame([{"model": "model_a", "variant_id": "V1", "f1": 0.5}])
    plan, skipped = plan_order_sweep(results, VARIANTS_BY_ID)
    assert plan == []
    assert skipped == ["model_a"]


def test_plan_size_sweep_skips_models_without_fewshot():
    plan, skipped = plan_size_sweep(_results_df(), VARIANTS_BY_ID, sizes=(1, 3))
    assert skipped == ["model_a", "model_c"]  # V1 and V16 don't use fewshot
    assert {u["model_key"] for u in plan} == {"model_b"}
    assert sorted(u["fewshot_num_per_class"] for u in plan) == [1, 3]
    assert all(u["variant_id"] == "V6" for u in plan)


def test_plan_seed_sweep_skips_models_without_fewshot():
    plan, skipped = plan_seed_sweep(_results_df(), VARIANTS_BY_ID, seeds=(1, 7))
    assert skipped == ["model_a", "model_c"]  # V1 and V16 don't use fewshot
    assert {u["model_key"] for u in plan} == {"model_b"}
    assert sorted(u["fewshot_seed"] for u in plan) == [1, 7]
    assert all(u["variant_id"] == "V6" for u in plan)


def test_results_df_from_checkpoint_reconstructs_model_and_variant(tmp_path):
    checkpoint = RunCheckpoint(tmp_path / "ckpt")
    checkpoint.mark_done(RunCheckpoint.unit_id("V1", "mistral"), {"f1": 0.5, "accuracy": 0.6})
    checkpoint.mark_done(RunCheckpoint.unit_id("V16", "mistral"), {"f1": 0.7, "accuracy": 0.7})
    checkpoint.mark_done(RunCheckpoint.unit_id("V6", "phi3"), {"f1": 0.8, "accuracy": 0.75})

    results_df = results_df_from_checkpoint(checkpoint)
    assert set(results_df["variant_id"]) == {"V1", "V16", "V6"}
    assert set(results_df["model"]) == {"mistral", "phi3"}
    best = best_variant_per_model(results_df)
    assert best == {"mistral": "V16", "phi3": "V6"}


def test_results_df_from_checkpoint_empty_manifest_returns_empty_df(tmp_path):
    checkpoint = RunCheckpoint(tmp_path / "ckpt")
    results_df = results_df_from_checkpoint(checkpoint)
    assert results_df.empty
    assert best_variant_per_model(results_df) == {}


def _toy_data():
    test_df = pd.DataFrame({"text": ["hello", "world"], "label": [0, 1]})
    demos_df = pd.DataFrame({"text": [f"d{i}" for i in range(10)], "label": [0, 1] * 5})
    return test_df, demos_df


class _FakeTokenizer:
    chat_template = None


class _FakePipe:
    def __call__(self, batch, **kwargs):
        return [{"generated_text": "NO"} for _ in batch]


def test_run_order_sweep_unit_writes_to_its_own_checkpoint_dir(tmp_path):
    test_df, demos_df = _toy_data()
    unit = {"model_key": "model_c", "variant_id": "V16", "order_index": 2, "component_order": ["cot", "context", "aspects"]}
    df = run_order_sweep_unit(
        unit, test_df, demos_df, VARIANTS_BY_ID, tmp_path,
        load_fn=lambda key: (_FakePipe(), _FakeTokenizer()), unload_fn=lambda: None,
        model_registry={},
    )
    assert len(df) == 1
    assert df.iloc[0]["variant_id"] == "V16"
    expected_dir = tmp_path / "order_sweep" / "model_c" / "order_2"
    assert expected_dir.exists()
    assert (expected_dir / "manifest.json").exists()


def test_run_order_sweep_unit_different_orders_do_not_collide(tmp_path):
    """Two different orders for the same (model, variant) must land in
    separate checkpoint directories: reusing one directory would make the
    second order's run overwrite the first's result under the same
    variant_id__model_key unit id."""
    test_df, demos_df = _toy_data()
    base_unit = {"model_key": "model_c", "variant_id": "V16"}

    run_order_sweep_unit(
        {**base_unit, "order_index": 0, "component_order": ["context", "cot", "aspects"]},
        test_df, demos_df, VARIANTS_BY_ID, tmp_path,
        load_fn=lambda key: (_FakePipe(), _FakeTokenizer()), unload_fn=lambda: None,
        model_registry={},
    )
    run_order_sweep_unit(
        {**base_unit, "order_index": 1, "component_order": ["aspects", "cot", "context"]},
        test_df, demos_df, VARIANTS_BY_ID, tmp_path,
        load_fn=lambda key: (_FakePipe(), _FakeTokenizer()), unload_fn=lambda: None,
        model_registry={},
    )
    ckpt0 = RunCheckpoint(tmp_path / "order_sweep" / "model_c" / "order_0")
    ckpt1 = RunCheckpoint(tmp_path / "order_sweep" / "model_c" / "order_1")
    assert ckpt0.is_done(RunCheckpoint.unit_id("V16", "model_c"))
    assert ckpt1.is_done(RunCheckpoint.unit_id("V16", "model_c"))


def test_run_size_sweep_unit_writes_to_its_own_checkpoint_dir(tmp_path):
    test_df, demos_df = _toy_data()
    unit = {"model_key": "model_b", "variant_id": "V6", "fewshot_num_per_class": 4}
    df = run_size_sweep_unit(
        unit, test_df, demos_df, VARIANTS_BY_ID, tmp_path,
        load_fn=lambda key: (_FakePipe(), _FakeTokenizer()), unload_fn=lambda: None,
        model_registry={},
    )
    assert len(df) == 1
    expected_dir = tmp_path / "size_sweep" / "model_b" / "size_4"
    assert expected_dir.exists()
    assert (expected_dir / "manifest.json").exists()


def test_run_size_sweep_unit_different_sizes_do_not_collide(tmp_path):
    test_df, demos_df = _toy_data()
    base_unit = {"model_key": "model_b", "variant_id": "V6"}

    run_size_sweep_unit(
        {**base_unit, "fewshot_num_per_class": 1}, test_df, demos_df, VARIANTS_BY_ID, tmp_path,
        load_fn=lambda key: (_FakePipe(), _FakeTokenizer()), unload_fn=lambda: None,
        model_registry={},
    )
    run_size_sweep_unit(
        {**base_unit, "fewshot_num_per_class": 3}, test_df, demos_df, VARIANTS_BY_ID, tmp_path,
        load_fn=lambda key: (_FakePipe(), _FakeTokenizer()), unload_fn=lambda: None,
        model_registry={},
    )
    ckpt1 = RunCheckpoint(tmp_path / "size_sweep" / "model_b" / "size_1")
    ckpt3 = RunCheckpoint(tmp_path / "size_sweep" / "model_b" / "size_3")
    assert ckpt1.is_done(RunCheckpoint.unit_id("V6", "model_b"))
    assert ckpt3.is_done(RunCheckpoint.unit_id("V6", "model_b"))


def test_run_seed_sweep_unit_writes_to_its_own_checkpoint_dir(tmp_path):
    test_df, demos_df = _toy_data()
    unit = {"model_key": "model_b", "variant_id": "V6", "fewshot_seed": 7}
    df = run_seed_sweep_unit(
        unit, test_df, demos_df, VARIANTS_BY_ID, tmp_path,
        load_fn=lambda key: (_FakePipe(), _FakeTokenizer()), unload_fn=lambda: None,
        model_registry={},
    )
    assert len(df) == 1
    expected_dir = tmp_path / "seed_sweep" / "model_b" / "seed_7"
    assert expected_dir.exists()
    assert (expected_dir / "manifest.json").exists()


def test_run_seed_sweep_unit_different_seeds_do_not_collide(tmp_path):
    test_df, demos_df = _toy_data()
    base_unit = {"model_key": "model_b", "variant_id": "V6"}

    run_seed_sweep_unit(
        {**base_unit, "fewshot_seed": 1}, test_df, demos_df, VARIANTS_BY_ID, tmp_path,
        load_fn=lambda key: (_FakePipe(), _FakeTokenizer()), unload_fn=lambda: None,
        model_registry={},
    )
    run_seed_sweep_unit(
        {**base_unit, "fewshot_seed": 7}, test_df, demos_df, VARIANTS_BY_ID, tmp_path,
        load_fn=lambda key: (_FakePipe(), _FakeTokenizer()), unload_fn=lambda: None,
        model_registry={},
    )
    ckpt1 = RunCheckpoint(tmp_path / "seed_sweep" / "model_b" / "seed_1")
    ckpt7 = RunCheckpoint(tmp_path / "seed_sweep" / "model_b" / "seed_7")
    assert ckpt1.is_done(RunCheckpoint.unit_id("V6", "model_b"))
    assert ckpt7.is_done(RunCheckpoint.unit_id("V6", "model_b"))
