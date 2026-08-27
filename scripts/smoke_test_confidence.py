#!/usr/bin/env python3
"""Smoke-test ``inference.generate_responses_with_confidence`` against a real
model, on a small stratified slice of the real EDOS test split.

This function has never been exercised against a real model (see its
docstring in ``src/sexism_prompting/inference.py``): it bypasses the HF
pipeline wrapper to reach ``model.generate(..., output_scores=True)``
directly, and this project's dev sandbox has no GPU, so only its pure math
(``confidence.py``) has been unit-tested so far. Run this once on the
cluster before trusting it for the real 42-unit non-CoT confidence-capture
run (see paper Sec. V-I / Limitations).

Loads exactly like the real grid does (``cluster_setup.load_one_for_cluster``,
bf16, no quantization) and builds the V1 baseline prompt (no role/aspects/
context/cot/fewshot) via the same ``build_variant_messages``/``render_prompts``
path ``predict_with_model`` uses, so this is a faithful small-scale preview
of the real run, not a separate ad hoc path.

Usage (on a cluster GPU node, inside the repo's venv):
    python scripts/smoke_test_confidence.py --model phi3 --n-per-class 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model", default="phi3", help="MODEL_REGISTRY key; phi3 is the smallest/fastest to load")
    parser.add_argument("--n-per-class", type=int, default=15, help="rows sampled from each of sexist/not-sexist")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    from sexism_prompting.cluster_setup import load_one_for_cluster, unload_one
    from sexism_prompting.edos_full import download_edos_full, get_split, load_edos_full
    from sexism_prompting.inference import generate_responses_with_confidence
    from sexism_prompting.metrics import compute_metrics, is_failure, process_response
    from sexism_prompting.prompts import build_variant_messages, render_prompts
    from sexism_prompting.seeding import set_seed

    set_seed(42)

    edos_full_dir = Path(args.data_dir) / "edos_full"
    download_edos_full(edos_full_dir)
    full_df = load_edos_full(edos_full_dir)
    test_df = get_split(full_df, "test")

    sexist = test_df[test_df["label"] == 1].head(args.n_per_class)
    not_sexist = test_df[test_df["label"] == 0].head(args.n_per_class)
    sample_df = pd.concat([sexist, not_sexist]).reset_index(drop=True)
    print(f"Sample: {len(sample_df)} rows ({args.n_per_class} sexist + {args.n_per_class} not-sexist, first-N, deterministic)")

    # V1: every component flag off. Same call predict_with_model makes for
    # the grid's own V1 unit.
    template = build_variant_messages({})

    print(f"Loading {args.model} via load_one_for_cluster(quantize=False) ...")
    pipe_tokenizer = load_one_for_cluster(args.model, batch_size=args.batch_size, quantize=False)
    _, tokenizer = pipe_tokenizer

    texts = sample_df["text"].tolist()
    y_true = sample_df["label"].tolist()
    prompts = render_prompts(template, texts, tokenizer, enable_thinking=False)

    print(f"Generating {len(prompts)} responses with confidence capture ...")
    responses, confidences = generate_responses_with_confidence(
        pipe_tokenizer, prompts, batch_size=args.batch_size
    )
    unload_one()

    print("\n--- per-example ---")
    print(f"{'true':>4}  {'pred':>4}  {'fail':>5}  {'P(sexist)':>10}   raw response")
    for text_preview, y, resp, conf in zip(
        [t[:40].replace(chr(10), " ") for t in texts], y_true, responses, confidences
    ):
        pred = process_response(resp)
        fail = is_failure(resp)
        conf_str = f"{conf:.3f}" if conf is not None else "None"
        print(f"{y:>4}  {pred:>4}  {str(fail):>5}  {conf_str:>10}   {resp!r}  # {text_preview}")

    n_none = sum(1 for c in confidences if c is None)
    print(f"\nconfidences: {len(confidences)} total, {n_none} None (extraction failed/unresolved)")

    metrics = compute_metrics(responses, y_true, confidences=confidences)
    print("\n--- compute_metrics (this small sample only, not a real result) ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
