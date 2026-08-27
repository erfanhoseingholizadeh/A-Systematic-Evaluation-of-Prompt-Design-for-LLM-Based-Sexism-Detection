"""Model registry and loading for 4-bit quantized instruction-tuned LLMs.

Requires a CUDA GPU (bitsandbytes 4-bit quantization). Not imported by the
prompt/metrics/data modules so those stay testable without a GPU or the
``torch``/``transformers`` stack installed.

Built for sequential use rather than loading everything up front:
``load_one`` / ``unload_one`` around each model's evaluation pass, driven by
``evaluate.py``'s model-outer loop, so at most one model's weights sit in
GPU memory at a time.
"""

from __future__ import annotations

import gc
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

# Add new models here: {key: {"model_id": ..., "revision": ...}}.
# Gated (require accepting a license on huggingface.co + a valid HF token):
# llama3.1, gemma2. Open: mistral, phi3, qwen2.5, qwen3.
#
# `revision` pins an exact commit so a rerun can't silently load different
# weights/tokenizer if the upstream repo's default branch moves.
MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "mistral": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "revision": "c170c708c41dac9275d15a8fff4eca08d52bab71",
    },
    "phi3": {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "revision": "f39ac1d28e925b323eae81227eaba4464caced4e",
    },
    "llama3.1": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
    },
    "qwen2.5": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
    },
    "gemma2": {
        "model_id": "google/gemma-2-9b-it",
        "revision": "11c9b309abf73637e4b6f9a3fa1e92e615547819",
    },
    "qwen3": {
        # The reasoning-tuned model in the roster, 8B to keep it in the
        # same size class as qwen2.5/llama3.1. Ships under the bare size
        # name, with no separate "-Instruct" repo: one model with a runtime
        # `enable_thinking` toggle instead (see `REASONING_MODELS` below).
        # Requires transformers>=4.51.0 and tokenizers>=0.21.4.
        "model_id": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
    },
}

GATED_MODELS = {"llama3.1", "gemma2"}

# Reasoning-tuned models: their chat template accepts an `enable_thinking`
# kwarg, wired to the variant's own `cot` flag the same way every other
# model's `cot` flag controls behavior via prompt text and generation
# budget alone. Passing `enable_thinking` to a tokenizer whose template
# doesn't reference it is harmless, so this doesn't need to be conditional
# at the call site. The caller does still need a generous `max_new_tokens`
# for these models (see `evaluate.py`'s `cot_gen_kwargs_overrides` and
# `configs/models.yaml`'s `generation_cot_overrides`): thinking mode can
# run far longer than the budget calibrated for non-reasoning models'
# brief prose CoT, and a truncated `<think>` block never reaches the
# required answer cue.
REASONING_MODELS = {"qwen3"}


def create_quantization_config() -> BitsAndBytesConfig:
    """Standard 4-bit NF4 quantization config used across all models."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_one(key: str) -> Tuple:
    """Load a single model by its ``MODEL_REGISTRY`` key. Returns (pipeline, tokenizer)."""
    if key not in MODEL_REGISTRY:
        raise KeyError(f"'{key}' not in MODEL_REGISTRY: {sorted(MODEL_REGISTRY)}")
    model_id = MODEL_REGISTRY[key]["model_id"]
    revision = MODEL_REGISTRY[key]["revision"]

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        quantization_config=create_quantization_config(),
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
    return pipe, tokenizer


def unload_one() -> None:
    """Reclaim GPU memory after the caller has already dropped its last
    reference to a (pipeline, tokenizer) pair loaded via ``load_one``.

    Must be called only once the caller's own binding is gone (e.g.
    immediately after ``loaded = None``): ``gc.collect()``/``empty_cache()``
    can only free objects that are already unreachable. Taking the pair as
    an argument instead would not work: the argument itself is a live
    reference for the entire duration of the call, so the object is still
    reachable exactly when these two calls run, and they end up doing
    nothing. Necessary to fit 6 quantized 7-9B models sequentially on one
    GPU. Without this, VRAM from each prior model would accumulate until
    CUDA OOMs partway through the roster.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_models(model_keys: List[str]) -> Dict[str, Tuple]:
    """Load multiple models from ``MODEL_REGISTRY`` by key, all at once.

    Kept for small subsets / local debugging where simultaneous loading still
    fits in memory. For a full multi-model sweep, use ``load_one``/
    ``unload_one`` in a per-model loop instead (see ``evaluate.py``'s
    ``evaluate_all_variants_sequential``).

    A single model failing to load (e.g. a gated model with no HF token
    configured) is skipped with a warning rather than aborting the whole
    batch, so one bad model doesn't silently discard every other model's
    results on a long unattended run.
    """
    models = {}
    for key in model_keys:
        if key not in MODEL_REGISTRY:
            print(f"Warning: '{key}' not found in MODEL_REGISTRY. Skipping.")
            continue
        try:
            models[key] = load_one(key)
        except Exception as e:  # noqa: BLE001
            print(f"Warning: failed to load '{key}' ({MODEL_REGISTRY[key]['model_id']}): {e!r}. Skipping.")
    return models
