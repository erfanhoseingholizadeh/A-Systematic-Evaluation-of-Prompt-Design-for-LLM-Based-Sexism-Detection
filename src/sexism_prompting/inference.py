"""Batched text generation against a HuggingFace text-generation pipeline.

Two default generation configs: a strict 2-token config for direct YES/NO
variants, and a much larger budget for chain-of-thought variants that need
room to actually reason before answering (see ``prompts.py`` module
docstring for why a 2-token cap makes CoT structurally impossible).
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from tqdm.auto import tqdm

DEFAULT_GEN_KWARGS: Dict[str, Any] = dict(
    max_new_tokens=2,  # forces a strict, near-instant YES/NO decision
    temperature=None,
    do_sample=False,
    top_p=1.0,
    repetition_penalty=1.0,
    return_full_text=False,
)

# For cot=True variants: enough tokens to reason before answering. Greedy
# decoding stays deterministic (do_sample=False), so this is still fully
# reproducible given the same seed/model/hardware.
#
# stop_strings halts generation the moment the model emits its required
# "ANSWER: YES"/"ANSWER: NO" (see prompts.py's CoT instruction). Without
# it, generation runs to the full max_new_tokens budget even after a real
# decision is reached, which lets a model ramble past its own answer and
# hallucinate a second, contradicting "ANSWER:" block that
# metrics._extract_answer_tail's last-marker-wins parsing would then
# prefer over the correct one. The pipeline needs `tokenizer` passed
# explicitly alongside `stop_strings` (it doesn't auto-forward its own
# constructor-time tokenizer for this); predict_with_model in evaluate.py
# injects it. Adding stop_strings also changes _config_fingerprint for
# every CoT unit, which is intentional: it correctly invalidates any
# already-checkpointed CoT result instead of silently trusting output from
# before this behavior changed.
COT_GEN_KWARGS: Dict[str, Any] = dict(
    max_new_tokens=220,
    temperature=None,
    do_sample=False,
    top_p=1.0,
    repetition_penalty=1.0,
    return_full_text=False,
    stop_strings=["ANSWER: YES", "ANSWER: NO"],
)


def gen_kwargs_for(is_cot: bool, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Pick the right base generation config for a variant, then apply overrides.

    Merging (not replacing) means a partial override from a config file (e.g.
    only ``max_new_tokens``) doesn't silently drop other defaults that
    downstream parsing relies on (like ``return_full_text=False``).
    """
    base = COT_GEN_KWARGS if is_cot else DEFAULT_GEN_KWARGS
    return {**base, **(overrides or {})}


def _batched(seq: Sequence[str], batch_size: int = 16) -> Iterator[Sequence[str]]:
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def generate_responses_with_confidence(
    pipe_tokenizer,
    prompts: Sequence[str],
    batch_size: int = 32,
    gen_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[Optional[float]]]:
    """Like ``generate_responses``, but also returns a per-example P(YES)
    confidence, for non-CoT variants only (see ``confidence.py``'s module
    docstring for why CoT isn't covered).

    NOT wired into ``evaluate.predict_with_model``/
    ``evaluate_all_variants_sequential``'s default path: opt-in only,
    called directly by whatever entry point wants confidence capture. Two
    reasons: (1) it bypasses the HF ``pipeline`` wrapper to reach
    ``model.generate(..., output_scores=True)`` directly (the pipeline's own
    ``postprocess`` step discards per-token scores, keeping only decoded
    text), which is real additional surface area on the exact code path the
    full grid depends on; (2) **this function has not been exercised
    against a real model**. This project's dev sandbox has no GPU, so
    unlike the rest of the tested surface, this one could only be verified
    via ``confidence.py``'s pure-math unit tests (token-id resolution,
    softmax-restricted probability), not an actual `generate()` call.
    Smoke-test this specifically (small model, few rows) before relying on
    it for a real run.

    Left-pads the tokenizer for the duration of the call and restores
    whatever `padding_side` it had before returning: batched causal-LM
    generation needs left-padding (a right-padded shorter sequence in a
    batch would generate its continuation after padding tokens, not after
    its real content), which ``pipeline.__call__`` normally handles
    internally but this bypass must do explicitly.
    """
    from .confidence import resolve_yes_no_token_ids, yes_probability_from_logits

    pipe, tokenizer = pipe_tokenizer
    if not prompts:
        return [], []

    import torch  # lazy: keeps this module importable without the GPU stack (see module docstring)

    kwargs = gen_kwargs if gen_kwargs is not None else DEFAULT_GEN_KWARGS
    yes_ids, no_ids = resolve_yes_no_token_ids(tokenizer)

    prompts = list(prompts)
    texts: List[str] = []
    confidences: List[Optional[float]] = []
    n_batches = (len(prompts) + batch_size - 1) // batch_size

    original_padding_side = getattr(tokenizer, "padding_side", None)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        for batch in tqdm(_batched(prompts, batch_size), total=n_batches):
            try:
                encoded = tokenizer(list(batch), return_tensors="pt", padding=True).to(pipe.model.device)
                with torch.no_grad():
                    out = pipe.model.generate(
                        **encoded,
                        max_new_tokens=kwargs.get("max_new_tokens", 2),
                        do_sample=kwargs.get("do_sample", False),
                        top_p=kwargs.get("top_p", 1.0),
                        repetition_penalty=kwargs.get("repetition_penalty", 1.0),
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                prompt_len = encoded["input_ids"].shape[1]
                new_token_ids = out.sequences[:, prompt_len:]
                decoded = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
                first_step_logits = out.scores[0]  # (batch, vocab): logits for the first NEW token only

                for i, text in enumerate(decoded):
                    texts.append(text)
                    confidences.append(
                        yes_probability_from_logits(first_step_logits[i].tolist(), yes_ids, no_ids)
                    )
            except Exception as e:  # noqa: BLE001 - a single bad batch shouldn't kill a long run
                texts.extend([""] * len(batch))
                confidences.extend([None] * len(batch))
                print(f"Batch generation-with-confidence failed: {e!r}")
    finally:
        if original_padding_side is not None:
            tokenizer.padding_side = original_padding_side

    return texts, confidences


def generate_responses(
    model,
    prompts: Sequence[str],
    batch_size: int = 32,
    gen_kwargs: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Run batched, deterministic inference over a list of rendered prompts.

    On a batch failure, fills that batch with empty strings and continues
    rather than aborting the whole run; failures then show up via
    ``metrics.compute_metrics``'s ``fail_ratio``, not a crashed multi-hour job.
    """
    if not prompts:
        return []

    kwargs = gen_kwargs if gen_kwargs is not None else DEFAULT_GEN_KWARGS
    prompts = list(prompts)
    outputs: List[str] = []

    n_batches = (len(prompts) + batch_size - 1) // batch_size
    for batch in tqdm(_batched(prompts, batch_size), total=n_batches):
        try:
            gen = model(batch, **kwargs)
            if isinstance(gen, list) and gen and isinstance(gen[0], list):
                gen = list(itertools.chain.from_iterable(gen))
            for g in gen:
                text = g.get("generated_text", "") if isinstance(g, dict) else str(g)
                outputs.append(text)
        except Exception as e:  # noqa: BLE001 - a single bad batch shouldn't kill a long run
            outputs.extend([""] * len(batch))
            print(f"Batch generation failed: {e!r}")

    return outputs
