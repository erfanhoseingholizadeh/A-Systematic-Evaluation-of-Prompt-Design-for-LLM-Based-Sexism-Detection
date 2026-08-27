#!/usr/bin/env python3
"""Pure-tokenizer sanity gate for the confidence-capture run, no GPU needed.

Two checks per model, both able to silently produce plausible-looking wrong
numbers downstream if skipped:

1. ``resolve_yes_no_token_ids`` can return an empty candidate list for a
   tokenizer where " YES"/" NO" collapse to a shared SentencePiece prefix
   token that then gets dropped as ambiguous overlap (see its docstring in
   ``confidence.py``). An empty list makes ``yes_probability_from_logits``
   return ``None`` for every single row of that model, silently.
2. ``generate_responses_with_confidence`` hardcodes ``out.scores[0]``, the
   first newly generated token, as THE decision token. True by construction
   for a plain "ANSWER:"-cued prompt (confirmed on phi3). Not obviously true
   for qwen3, whose chat template may inject a closed ``<think>\\n\\n</think>``
   block as part of what the model has to emit before the real answer even
   with ``enable_thinking=False``. This prints the exact rendered prompt
   tail and, where possible, the template's own boilerplate so a human can
   confirm the very next generated token is really the YES/NO decision.

Run on any node with network access to the HF Hub (a login node or
gorina11 is fine; no GPU allocation needed). Gated models (llama3.1,
gemma2) need HF_TOKEN set from ~/.hf_token first, same as the real grid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", default="mistral,phi3,llama3.1,qwen2.5,gemma2,qwen3",
        help="comma-separated MODEL_REGISTRY keys to check",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from sexism_prompting.confidence import resolve_yes_no_token_ids
    from sexism_prompting.models import MODEL_REGISTRY
    from sexism_prompting.prompts import build_variant_messages, render_prompts

    template = build_variant_messages({})  # V1: no role/aspects/context/cot/fewshot
    sample_text = "This is a sample post used only to render a prompt tail for inspection."

    keys = [k.strip() for k in args.models.split(",")]
    problems = []

    for key in keys:
        print(f"\n=== {key} ===")
        if key not in MODEL_REGISTRY:
            print(f"  SKIP: {key!r} not in MODEL_REGISTRY")
            continue
        entry = MODEL_REGISTRY[key]
        try:
            tokenizer = AutoTokenizer.from_pretrained(entry["model_id"], revision=entry["revision"], use_fast=True)
        except Exception as e:  # noqa: BLE001
            print(f"  LOAD FAILED: {e}")
            problems.append(f"{key}: tokenizer load failed ({e})")
            continue

        yes_ids, no_ids = resolve_yes_no_token_ids(tokenizer)
        print(f"  yes_token_ids={yes_ids}  no_token_ids={no_ids}")
        if not yes_ids or not no_ids:
            print("  PROBLEM: empty candidate list -> every confidence will be None for this model")
            problems.append(f"{key}: empty yes/no candidate token ids")

        rendered = render_prompts(template, [sample_text], tokenizer, enable_thinking=False)[0]
        tail = rendered[-200:]
        print(f"  rendered prompt, last 200 chars:\n  {tail!r}")
        if "<think>" in rendered or "</think>" in rendered:
            print("  NOTE: prompt contains <think> boilerplate -- inspect the tail above to confirm")
            print("  the very next generated token is the real YES/NO decision, not inside/after an")
            print("  unclosed think block.")

    print("\n=== summary ===")
    if problems:
        print("PROBLEMS FOUND (fix before running the real confidence-capture pass):")
        for p in problems:
            print(f"  - {p}")
    else:
        print("No problems found in the automated checks. Still eyeball each model's rendered")
        print("prompt tail above, especially qwen3, before trusting the real run.")


if __name__ == "__main__":
    main()
