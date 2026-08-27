#!/usr/bin/env python3
"""Real confidence-capture run: the 7 non-chain-of-thought variants
(V1, V2, V3, V4, V6, V7, V9), across all 6 models, against the full
official EDOS test split -- the run that exercises
``inference.generate_responses_with_confidence`` for real, after the
tokenizer preflight (``confidence_preflight_check.py``) and the small-sample
smoke test both passed clean on phi3.

Deliberately a separate, standalone script, not a code path added to
``evaluate.evaluate_all_variants_sequential``/``predict_with_model``: this
keeps the never-before-exercised confidence-capture bypass (see
``inference.generate_responses_with_confidence``'s own docstring) fully
isolated from the tested pipeline that produced every number already in the
paper. It reuses that pipeline's own prompt-building functions
(``build_few_shot_demonstrations``, ``build_variant_messages``,
``render_prompts``) unchanged, so every rendered prompt here is byte-for-byte
identical to the one the original grid run used for the same
(model, variant) unit -- only the generation call differs.

Per-unit output is a JSON file under ``--output-dir`` (skipped if it already
exists, for a cheap resume): raw responses, confidences, true labels, and
computed metrics including AUC (``metrics.compute_metrics``). Also reports,
per unit: how many responses came back empty (the per-batch ``except``
fallback in ``generate_responses_with_confidence``, which pads silently
rather than raising -- a sign of a degraded batch, e.g. a quiet OOM) and how
many confidences came back ``None``, plus an agreement rate between the
confidence-implied call (``P(sexist) > 0.5``) and the actual parsed decision
(``metrics.process_response``) -- these two should agree on every row by
construction (same generated first token), so anything below ~100% flags a
real bug, not noise.

Usage (on a cluster GPU node, inside the repo's venv, HF_TOKEN exported for
the two gated models):
    python scripts/run_confidence_capture.py --models mistral,phi3,llama3.1,qwen2.5,gemma2,qwen3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

NON_COT_VARIANTS = {"V1", "V2", "V3", "V4", "V6", "V7", "V9"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--models", default="mistral,phi3,llama3.1,qwen2.5,gemma2,qwen3")
    parser.add_argument("--models-config", default=None)
    parser.add_argument("--variants-config", default=None)
    parser.add_argument("--cluster-config", default=None)
    parser.add_argument(
        "--output-dir", default=None,
        help="default: ./checkpoints/confidence, or ./checkpoints/confidence_dryrun if --subsample is set "
             "(kept separate so a dry run's small-sample output can never be mistaken for a completed "
             "full-sample unit by the skip-if-exists resume check)",
    )
    parser.add_argument(
        "--subsample", type=int, default=None,
        help="evaluate only the first N test rows (for a dry run of this script's own logic "
             "before committing GPU time to the full 4,000-row run)",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = "./checkpoints/confidence_dryrun" if args.subsample else "./checkpoints/confidence"

    import yaml

    from sexism_prompting.cluster_setup import load_one_for_cluster, unload_one
    from sexism_prompting.edos_full import download_edos_full, get_split, load_edos_full
    from sexism_prompting.evaluate import load_prompt_variants
    from sexism_prompting.inference import generate_responses_with_confidence
    from sexism_prompting.metrics import compute_metrics, process_response
    from sexism_prompting.prompts import build_few_shot_demonstrations, build_variant_messages, render_prompts
    from sexism_prompting.seeding import set_seed

    repo_root = Path(args.repo_root)
    models_config = args.models_config or repo_root / "configs" / "models.yaml"
    variants_config = args.variants_config or repo_root / "configs" / "prompt_variants.yaml"
    cluster_config = args.cluster_config or repo_root / "configs" / "cluster.yaml"

    with open(models_config) as f:
        models_cfg = yaml.safe_load(f)
    with open(cluster_config) as f:
        cluster_cfg = yaml.safe_load(f)
    set_seed(models_cfg.get("seed", 42))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    edos_full_dir = Path(args.data_dir) / "edos_full"
    download_edos_full(edos_full_dir)
    full_df = load_edos_full(edos_full_dir)
    test_df = get_split(full_df, "test")
    demos_df = get_split(full_df, "train")
    if args.subsample:
        test_df = test_df.iloc[: args.subsample].reset_index(drop=True)
        print(f"Subsampled test set to {len(test_df)} rows (dry run).")
    texts = test_df["text"].tolist()
    y_true = test_df["label"].tolist()

    all_variants, fewshot_n, fewshot_seed = load_prompt_variants(variants_config)
    variants = [v for v in all_variants if v["id"] in NON_COT_VARIANTS]
    assert {v["id"] for v in variants} == NON_COT_VARIANTS, "non-CoT variant set mismatch vs. prompt_variants.yaml"
    assert all(not v["cot"] for v in variants), "a variant marked cot:true leaked into the non-CoT run"

    default_batch = cluster_cfg.get("batch_size", 32)
    batch_overrides = cluster_cfg.get("batch_size_overrides", {})

    model_keys = [k.strip() for k in args.models.split(",")]

    summary_rows = []

    for model_key in model_keys:
        batch_size = batch_overrides.get(model_key, default_batch)
        print(f"\n=== {model_key} (batch_size={batch_size}) ===")

        pending = [v for v in variants if not (output_dir / f"{v['id']}__{model_key}.json").exists()]
        if not pending:
            print("  all units already done, skipping model load entirely")
        else:
            t_load = time.time()
            pipe_tokenizer = load_one_for_cluster(model_key, batch_size=batch_size, quantize=False)
            print(f"  loaded in {time.time() - t_load:.1f}s")
            _, tokenizer = pipe_tokenizer

            for variant in pending:
                unit_id = f"{variant['id']}__{model_key}"
                t0 = time.time()

                flags = {
                    "role": variant["role"],
                    "aspects": variant["aspects"],
                    "context": variant["context"],
                    "cot": variant["cot"],
                    "fewshot": variant["fewshot"],
                }
                examples = None
                if flags["fewshot"]:
                    examples = build_few_shot_demonstrations(demos_df, num_per_class=fewshot_n, seed=fewshot_seed)
                template = build_variant_messages(flags, examples=examples)
                prompts = render_prompts(template, texts, tokenizer, enable_thinking=False)

                responses, confidences = generate_responses_with_confidence(
                    pipe_tokenizer, prompts, batch_size=batch_size
                )
                elapsed = time.time() - t0

                n_empty = sum(1 for r in responses if r == "")
                n_none_conf = sum(1 for c in confidences if c is None)
                agree_pairs = [
                    (c, process_response(r))
                    for r, c in zip(responses, confidences)
                    if c is not None and r != ""
                ]
                agreement_rate = (
                    sum(1 for c, pred in agree_pairs if (c > 0.5) == (pred == 1)) / len(agree_pairs)
                    if agree_pairs
                    else float("nan")
                )

                metrics = compute_metrics(responses, y_true, confidences=confidences)

                flag_note = ""
                if n_empty > 0 or (agree_pairs and agreement_rate < 0.999):
                    flag_note = "  <-- CHECK: empty responses and/or confidence/decision disagreement"
                print(
                    f"  [{unit_id}] {elapsed:.1f}s  n_empty={n_empty}  n_none_conf={n_none_conf}  "
                    f"agreement={agreement_rate:.4f}  auc={metrics.get('auc'):.4f}  "
                    f"f1={metrics['f1']:.3f}  fail_ratio={metrics['fail_ratio']:.3f}{flag_note}"
                )

                out_path = output_dir / f"{unit_id}.json"
                with open(out_path, "w") as f:
                    json.dump(
                        {
                            "unit_id": unit_id,
                            "model": model_key,
                            "variant": variant["id"],
                            "flags": flags,
                            "n": len(texts),
                            "n_empty": n_empty,
                            "n_none_confidence": n_none_conf,
                            "agreement_rate": agreement_rate,
                            "metrics": metrics,
                            "responses": responses,
                            "confidences": confidences,
                            "y_true": y_true,
                        },
                        f,
                    )

                summary_rows.append(
                    {
                        "model": model_key,
                        "variant": variant["id"],
                        "n_empty": n_empty,
                        "n_none_conf": n_none_conf,
                        "agreement_rate": agreement_rate,
                        **metrics,
                    }
                )

            del pipe_tokenizer
            unload_one()

    # Reload every completed unit (including ones skipped this run) so the
    # summary always covers the full set on disk, not just this invocation.
    print("\n=== summary (all completed units on disk) ===")
    header = f"{'unit':<14} {'n_empty':>7} {'n_none':>7} {'agree':>7} {'auc':>7} {'f1':>7} {'fail_ratio':>10}"
    print(header)
    for variant in variants:
        for model_key in model_keys:
            unit_id = f"{variant['id']}__{model_key}"
            path = output_dir / f"{unit_id}.json"
            if not path.exists():
                continue
            with open(path) as f:
                d = json.load(f)
            m = d["metrics"]
            print(
                f"{unit_id:<14} {d['n_empty']:>7} {d['n_none_confidence']:>7} "
                f"{d['agreement_rate']:>7.4f} {m.get('auc', float('nan')):>7.4f} "
                f"{m['f1']:>7.3f} {m['fail_ratio']:>10.3f}"
            )


if __name__ == "__main__":
    main()
