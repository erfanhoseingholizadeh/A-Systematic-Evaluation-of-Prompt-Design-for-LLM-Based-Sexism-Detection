#!/usr/bin/env python3
"""Run the full model x variant grid locally on any CUDA machine.

Sequential model loading (one model resident in GPU memory at a time) and
resumable checkpointing are always on: a killed/restarted run picks up
where it left off. See scripts/run_on_cluster.py for the Slurm-cluster
entry point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml  # noqa: E402

from sexism_prompting.checkpoint import RunCheckpoint  # noqa: E402
from sexism_prompting.edos_full import download_edos_full, get_split, load_edos_full  # noqa: E402
from sexism_prompting.evaluate import evaluate_all_variants_sequential, load_prompt_variants  # noqa: E402
from sexism_prompting.io_utils import pivot_metric, save_results  # noqa: E402
from sexism_prompting.seeding import set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--checkpoint-dir", default="./checkpoints/experiment")
    parser.add_argument("--results-path", default="./results/results.csv")
    parser.add_argument("--models-config", default=None)
    parser.add_argument("--variants-config", default=None)
    parser.add_argument("--subsample", type=int, default=None, help="evaluate only the first N test rows (for a quick pilot)")
    parser.add_argument("--only-models", default=None, help="comma-separated model keys to restrict to, e.g. 'mistral' (for a quick pilot)")
    parser.add_argument("--only-variants", default=None, help="comma-separated variant ids to restrict to, e.g. 'V1,V6' (for a quick pilot)")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    models_config = args.models_config or repo_root / "configs" / "models.yaml"
    variants_config = args.variants_config or repo_root / "configs" / "prompt_variants.yaml"

    with open(models_config) as f:
        models_cfg = yaml.safe_load(f)
    set_seed(models_cfg.get("seed", 42))

    edos_full_dir = Path(args.data_dir) / "edos_full"
    download_edos_full(edos_full_dir)
    full_df = load_edos_full(edos_full_dir)
    test_df = get_split(full_df, "test")
    demos_df = get_split(full_df, "train")
    if args.subsample:
        test_df = test_df.iloc[: args.subsample].reset_index(drop=True)
        print(f"Subsampled test set to {len(test_df)} rows for a quick pilot.")

    variants, fewshot_n, fewshot_seed = load_prompt_variants(variants_config)
    model_keys = models_cfg["models"]
    if args.only_models:
        model_keys = [k.strip() for k in args.only_models.split(",")]
    if args.only_variants:
        wanted = {v.strip() for v in args.only_variants.split(",")}
        variants = [v for v in variants if v["id"] in wanted]

    checkpoint = RunCheckpoint(args.checkpoint_dir)

    def on_unit_done(unit_id: str, metrics: dict) -> None:
        print(f"[{unit_id}] " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items() if isinstance(v, float)))

    grid_df = evaluate_all_variants_sequential(
        model_keys,
        test_df,
        demos_df,
        variants,
        checkpoint,
        fewshot_num_per_class=fewshot_n,
        fewshot_seed=fewshot_seed,
        batch_size=models_cfg.get("batch_size", 32),
        batch_size_overrides=models_cfg.get("batch_size_overrides", {}),
        gen_kwargs=models_cfg.get("generation"),
        cot_gen_kwargs=models_cfg.get("generation_cot"),
        on_unit_done=on_unit_done,
        failure_threshold=models_cfg.get("failure_threshold", 0.5),
        failure_threshold_overrides=models_cfg.get("failure_threshold_overrides", {}),
        cot_gen_kwargs_overrides=models_cfg.get("generation_cot_overrides", {}),
    )

    save_results(grid_df, args.results_path, append=False)
    print(f"\nSaved {len(grid_df)} rows to {args.results_path}")
    print(pivot_metric(grid_df, "f1").to_string())


if __name__ == "__main__":
    main()
