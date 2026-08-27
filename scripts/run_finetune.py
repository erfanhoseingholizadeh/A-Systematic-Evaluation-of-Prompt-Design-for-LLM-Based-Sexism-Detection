#!/usr/bin/env python3
"""Train and evaluate the fine-tuned baselines (encoder + QLoRA) on the full EDOS train split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml  # noqa: E402

from sexism_prompting.edos_full import download_edos_full, get_split, load_edos_full  # noqa: E402
from sexism_prompting.io_utils import save_results  # noqa: E402
from sexism_prompting.seeding import set_seed  # noqa: E402

import pandas as pd  # noqa: E402

# `finetune.py` imports torch at module level (unlike evaluate.py's lazy
# `models.load_one`), so it's deferred into main(), after argument parsing:
# `--help` and bad-argument errors then work without the GPU extra installed.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--config", default=None)
    parser.add_argument("--which", choices=["encoder", "qlora", "both"], default="both")
    parser.add_argument("--results-path", default="./results/finetune_results.csv")
    parser.add_argument(
        "--only-models", default=None,
        help="comma-separated model keys to restrict qlora.model_keys to, e.g. 'mistral' "
             "(one Slurm job per model, mirroring run_on_cluster.py's --only-models)",
    )
    args = parser.parse_args()

    from sexism_prompting.finetune import (
        evaluate_encoder,
        evaluate_qlora,
        select_best_qlora_epoch,
        train_encoder_baseline,
        train_qlora,
    )

    repo_root = Path(args.repo_root)
    config_path = args.config or repo_root / "configs" / "finetune.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    edos_full_dir = Path(args.data_dir) / "edos_full"
    download_edos_full(edos_full_dir)
    full_df = load_edos_full(edos_full_dir)
    train_df = get_split(full_df, "train")
    dev_df = get_split(full_df, "dev")

    # Score on the official EDOS test split (same source train_df/dev_df above
    # already came from).
    test_df = get_split(full_df, "test")

    rows = []

    if args.which in ("encoder", "both"):
        enc_cfg = cfg["encoder"]
        trainer = train_encoder_baseline(
            train_df,
            dev_df,
            model_id=enc_cfg["model_id"],
            output_dir=enc_cfg["output_dir"],
            seed=seed,
            num_epochs=enc_cfg["num_epochs"],
            learning_rate=enc_cfg["learning_rate"],
            batch_size=enc_cfg["batch_size"],
            max_length=enc_cfg["max_length"],
        )
        metrics = evaluate_encoder(trainer, test_df, max_length=enc_cfg["max_length"])
        print(f"[encoder:{enc_cfg['model_id']}] {metrics}")
        rows.append({"baseline": "encoder", "model": enc_cfg["model_id"], **metrics})

    if args.which in ("qlora", "both"):
        qlora_cfg = cfg["qlora"]
        model_keys = qlora_cfg["model_keys"]
        if args.only_models:
            model_keys = [k.strip() for k in args.only_models.split(",")]
            unknown = [k for k in model_keys if k not in qlora_cfg["model_keys"]]
            if unknown:
                raise SystemExit(
                    f"--only-models: unknown model key(s) {unknown}; valid keys are {qlora_cfg['model_keys']}"
                )
        for model_key in model_keys:
            adapter_dir = Path(qlora_cfg["output_dir"]) / model_key
            train_qlora(
                model_key,
                train_df,
                output_dir=adapter_dir,
                seed=seed,
                num_epochs=qlora_cfg["num_epochs"],
                learning_rate=qlora_cfg["learning_rate"],
                batch_size=qlora_cfg["batch_size"],
                max_length=qlora_cfg["max_length"],
                lora_r=qlora_cfg["lora_r"],
                lora_alpha=qlora_cfg["lora_alpha"],
                lora_dropout=qlora_cfg["lora_dropout"],
                save_epoch_checkpoints=True,
            )
            selection = select_best_qlora_epoch(model_key, adapter_dir, dev_df)
            print(f"[qlora:{model_key}] selected epoch {selection['selected_epoch']} by dev F1: "
                  f"{selection['per_epoch']}")
            metrics = evaluate_qlora(model_key, adapter_dir, test_df)
            print(f"[qlora:{model_key}] {metrics}")
            rows.append({"baseline": "qlora", "model": model_key, **metrics})

    save_results(pd.DataFrame(rows), args.results_path, append=True)
    print(f"\nSaved {len(rows)} rows to {args.results_path}")


if __name__ == "__main__":
    main()
