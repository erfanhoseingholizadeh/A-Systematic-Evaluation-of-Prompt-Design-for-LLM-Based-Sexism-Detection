#!/usr/bin/env python3
"""Prediction-level diff for the cross-hardware determinism check.

Aggregate F1 matching between two hardware runs is not proof of
determinism: two runs can hit an identical F1 with different underlying
predictions if errors happen to cancel out. This compares, per model and
per test example (aligned by ``rewire_id``, not list position), whether
the two runs' raw generated text is byte-identical and whether their
parsed decisions (``metrics.process_response``) agree, across
``checkpoints/crosshw_<node_a>/<model>/`` and
``checkpoints/crosshw_<node_b>/<model>/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402

from sexism_prompting.checkpoint import RunCheckpoint  # noqa: E402
from sexism_prompting.metrics import process_response  # noqa: E402

MODELS_AND_VARIANTS = {
    "mistral": "V7",
    "phi3": "V9",
    "llama3.1": "V6",
    "qwen2.5": "V6",
    "gemma2": "V7",
    # qwen3 isn't a gorina8-vs-gorina9 comparison like the rest: its cross-
    # hardware leg used gorina9's A100 and H100 cards on the same node
    # instead, so run it separately with --models qwen3
    # --node-a gorina9a100 --node-b gorina9h100.
    "qwen3": "V8",
}


def diff_one_model(checkpoint_root: Path, node_a: str, node_b: str, model: str, variant: str) -> dict:
    unit_id = RunCheckpoint.unit_id(variant, model)
    ckpt_a = RunCheckpoint(checkpoint_root / f"crosshw_{node_a}" / model)
    ckpt_b = RunCheckpoint(checkpoint_root / f"crosshw_{node_b}" / model)
    pred_a = ckpt_a.load_predictions_normalized(unit_id)
    pred_b = ckpt_b.load_predictions_normalized(unit_id)

    ids_a, ids_b = pred_a["rewire_ids"], pred_b["rewire_ids"]
    if ids_a is None or ids_b is None:
        raise ValueError(f"{model}: predictions not id-aligned (rewire_ids missing)")
    if set(ids_a) != set(ids_b):
        raise ValueError(f"{model}: the two runs cover different example sets")

    by_id_a = dict(zip(ids_a, zip(pred_a["responses"], pred_a["labels"])))
    by_id_b = dict(zip(ids_b, zip(pred_b["responses"], pred_b["labels"])))

    n = len(by_id_a)
    text_matches = 0
    decision_matches = 0
    label_mismatches = 0
    decision_mismatch_ids = []
    for rid, (resp_a, label_a) in by_id_a.items():
        resp_b, label_b = by_id_b[rid]
        if label_a != label_b:
            label_mismatches += 1
        if resp_a == resp_b:
            text_matches += 1
        dec_a, dec_b = process_response(resp_a), process_response(resp_b)
        if dec_a == dec_b:
            decision_matches += 1
        else:
            decision_mismatch_ids.append(rid)

    return {
        "model": model,
        "variant": variant,
        "n": n,
        "gold_label_mismatches": label_mismatches,
        "text_identical": text_matches,
        "text_identical_frac": text_matches / n,
        "decision_identical": decision_matches,
        "decision_identical_frac": decision_matches / n,
        "decision_mismatch_ids": ";".join(decision_mismatch_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", default="./checkpoints")
    parser.add_argument("--node-a", default="gorina8")
    parser.add_argument("--node-b", default="gorina9")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[m for m in MODELS_AND_VARIANTS if m != "qwen3"],
        help="Model keys to diff (must share the --node-a/--node-b checkpoint tag pair). "
        "qwen3 uses a different tag pair (gorina9a100/gorina9h100) -- pass it separately.",
    )
    parser.add_argument("--out", default="./results/crosshw_prediction_diff.csv")
    args = parser.parse_args()

    checkpoint_root = Path(args.checkpoint_root)
    rows = [
        diff_one_model(checkpoint_root, args.node_a, args.node_b, model, MODELS_AND_VARIANTS[model])
        for model in args.models
    ]

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}\n")
    print(out_df.drop(columns=["decision_mismatch_ids"]).to_string(index=False))


if __name__ == "__main__":
    main()
