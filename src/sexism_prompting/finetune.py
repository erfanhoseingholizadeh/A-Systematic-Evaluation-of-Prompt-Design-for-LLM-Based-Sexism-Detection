"""Fine-tuned baselines, giving the zero-/few-shot prompting grid a
training-based comparison point.

Two independent baselines, both trained on the EDOS train split (see
``edos_full.get_split``) and scored with the same ``metrics.compute_metrics``
used everywhere else in this repo, so numbers are directly comparable:

- ``train_encoder_baseline``: a small supervised encoder classifier
  (RoBERTa/DistilBERT). Cheap (~20-40 min GPU), the conventional "what does
  real supervision buy you" reference point.
- ``train_qlora`` / ``evaluate_qlora``: QLoRA (4-bit base + LoRA adapters) on
  the same instruction-tuned LLMs used for zero-/few-shot prompting, trained
  as supervised fine-tuning on (prompt, "YES"/"NO") pairs built from the same
  prompt templates as the rest of this repo. Shows what these exact models do
  once tuned, with a low VRAM footprint since only the LoRA adapters are
  trained. ``select_best_qlora_epoch`` optionally picks the best-on-dev epoch
  afterward, the same model-selection idea the encoder baseline gets from
  ``load_best_model_at_end``.

Both accept a ``seed`` and pass it through to every relevant source of
randomness (``Trainer``'s ``seed``, train/val splits) per this repo's
reproducibility requirement (see ``seeding.py``). NOT ``data_seed``:
``transformers``' ``TrainingArguments.__post_init__`` gates ``data_seed`` on
``accelerate>=1.1.0``, and passing it under an older ``accelerate`` raises
``NotImplementedError``. ``seed`` alone still covers model init and
shuffling; only the data-loader RNG stream is left coupled to it. Bitwise
determinism is not enforced anywhere in this repo regardless (see
HANDOFF.md), so this is not a new reproducibility gap.

Requires torch/transformers (+peft for QLoRA); not imported by the
prompt/metrics/data modules, so those stay testable without a GPU.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .metrics import compute_metrics, compute_metrics_from_predictions
from .models import MODEL_REGISTRY, create_quantization_config
from .prompts import build_variant_messages, render_prompts
from .seeding import set_seed
from .sft import balance_classes, build_masked_example

# ---------------------------------------------------------------------------
# Encoder baseline (RoBERTa / DistilBERT via HF Trainer)
# ---------------------------------------------------------------------------


class _EncoderDataset(Dataset):
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: Sequence[int]):
        self.encodings = encodings
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_encoder_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_id: str = "roberta-base",
    output_dir: str | Path = "./checkpoints/encoder",
    seed: int = 42,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    max_length: int = 128,
    resume_from_checkpoint: Optional[str] = None,
):
    """Fine-tune a supervised binary sequence classifier. Returns the trained ``Trainer``.

    Checkpointed every epoch by ``Trainer`` itself (``save_strategy="epoch"``)
    under ``output_dir``: pass that same path back as ``resume_from_checkpoint``
    to continue a killed run instead of restarting training from scratch.
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

    def _encode(df: pd.DataFrame) -> _EncoderDataset:
        enc = tokenizer(
            df["text"].tolist(), truncation=True, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        return _EncoderDataset(enc, df["label"].tolist())

    train_ds = _encode(train_df)
    val_ds = _encode(val_df)

    def _compute_metrics(eval_pred) -> Dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return compute_metrics_from_predictions(preds, labels)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=seed,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    return trainer


def evaluate_encoder(trainer, test_df: pd.DataFrame, max_length: int = 128) -> Dict[str, float]:
    """Score a trained encoder ``Trainer`` on held-out data with the same metric set as the LLM path."""
    tokenizer = trainer.tokenizer if hasattr(trainer, "tokenizer") else trainer.processing_class
    enc = tokenizer(
        test_df["text"].tolist(), truncation=True, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    ds = _EncoderDataset(enc, test_df["label"].tolist())
    preds = np.argmax(trainer.predict(ds).predictions, axis=-1)
    return compute_metrics_from_predictions(preds, test_df["label"].tolist())


# ---------------------------------------------------------------------------
# QLoRA (4-bit base + LoRA adapters) supervised fine-tuning of the LLMs
# ---------------------------------------------------------------------------

DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def release_cuda_memory() -> None:
    """gc + empty the CUDA cache. Callers must drop their own references
    first (``del trainer`` etc.). This is the between-train-and-eval /
    between-models hygiene that keeps sequential 7-9B loads fitting on one
    ~16GB GPU (mirrors ``models.unload_one``)."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class _PromptCompletionDataset(Dataset):
    """Tokenized (prompt, completion) pairs with the prompt span's loss masked out.

    Only the completion tokens (" YES"/" NO" + EOS) contribute to the LM loss,
    so the adapter learns to answer, and to stop after answering. Prompt and
    completion are tokenized separately (completion without special tokens)
    and concatenated as ids; see ``build_masked_example`` for why.
    """

    def __init__(self, prompts: Sequence[str], completions: Sequence[str], tokenizer, max_length: int = 512):
        self.examples = []
        eos_id = getattr(tokenizer, "eos_token_id", None)
        for prompt, completion in zip(prompts, completions):
            prompt_ids = list(tokenizer(prompt)["input_ids"])
            completion_ids = list(tokenizer(completion, add_special_tokens=False)["input_ids"])
            if eos_id is not None:
                completion_ids.append(eos_id)
            input_ids, labels = build_masked_example(prompt_ids, completion_ids, max_length)
            self.examples.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


def _pad_collate(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids, labels, attention_mask = [], [], []
    for item in batch:
        pad_len = max_len - item["input_ids"].size(0)
        input_ids.append(torch.nn.functional.pad(item["input_ids"], (0, pad_len), value=pad_token_id))
        labels.append(torch.nn.functional.pad(item["labels"], (0, pad_len), value=-100))
        attention_mask.append(torch.cat([torch.ones_like(item["input_ids"]), torch.zeros(pad_len, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


def build_sft_examples(
    train_df: pd.DataFrame, flags: Optional[Dict[str, bool]] = None
) -> Tuple[List[Dict[str, str]], List[str]]:
    """Build a (chat message template, completions) pair for QLoRA SFT from the train split.

    Uses the same variant-flag prompt builder as zero-/few-shot evaluation
    (default: no components, matching the baseline variant) so the fine-tuned
    model is directly comparable to the prompting-only results at inference
    time: same instruction, same format, just tuned weights. The returned
    template still has a ``{text}`` placeholder; ``train_qlora`` fills it in
    per-example via ``render_prompts`` (which needs the model's tokenizer).
    """
    flags = flags or {}
    template = build_variant_messages(flags, examples=None)
    completions = [" YES" if label == 1 else " NO" for label in train_df["label"]]
    return template, completions


def train_qlora(
    model_key: str,
    train_df: pd.DataFrame,
    output_dir: str | Path = "./checkpoints/qlora",
    seed: int = 42,
    num_epochs: int = 2,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    max_length: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    flags: Optional[Dict[str, bool]] = None,
    resume_from_checkpoint: Optional[str] = None,
    save_epoch_checkpoints: bool = False,
):
    """QLoRA-fine-tune one registry model on EDOS train. Returns the trained ``Trainer``.

    Checkpointed by ``Trainer`` (``save_strategy="steps"``) under
    ``output_dir``: pass that path as ``resume_from_checkpoint`` to continue
    a killed run.

    ``save_epoch_checkpoints=True`` additionally snapshots the adapter at the
    end of every epoch into ``<output_dir>/epoch_checkpoints/epoch_<n>/``,
    alongside (not instead of) the ``resume_from_checkpoint`` ones above.
    Pass these to ``select_best_qlora_epoch`` afterward to pick the epoch
    with the best dev F1, the same model-selection idea
    ``train_encoder_baseline`` already gets from ``load_best_model_at_end``.
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

    set_seed(seed)
    model_id = MODEL_REGISTRY[model_key]["model_id"]
    revision = MODEL_REGISTRY[model_key]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # device_map={"": 0}, not "auto": same silent-CPU-fallback bad-GPU
    # issue as cluster_setup.load_one_for_cluster (see its docstring for the
    # full mechanism): "auto" silently falls back to full CPU placement on
    # gorina8's known-bad GPU index 3 instead of raising.
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision,
        quantization_config=create_quantization_config(), device_map={"": 0}, torch_dtype=torch.bfloat16
    )
    base_model = prepare_model_for_kbit_training(base_model)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules or DEFAULT_LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)

    # EDOS's train split is naturally ~76/24 not-sexist/sexist; this SFT
    # objective has no built-in class weighting, so an unbalanced split
    # collapses the adapter to always predicting the majority label (see
    # sft.balance_classes docstring). Balance before building examples.
    train_df = balance_classes(train_df, seed=seed)
    template, raw_completions = build_sft_examples(train_df, flags=flags)
    # Non-chat-template models render "...\n\nAssistant:" as the prompt tail already;
    # chat-template models get add_generation_prompt=True, so both are ready for a
    # completion to be appended directly.
    prompts = render_prompts(template, train_df["text"].tolist(), tokenizer, enable_thinking=(flags or {}).get("cot", False))

    dataset = _PromptCompletionDataset(prompts, raw_completions, tokenizer, max_length=max_length)

    def collate(batch):
        return _pad_collate(batch, tokenizer.pad_token_id)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        seed=seed,
        report_to=[],
    )
    callbacks = None
    if save_epoch_checkpoints:
        epoch_checkpoints_dir = Path(output_dir) / "epoch_checkpoints"

        class _SaveEpochAdapterCallback(TrainerCallback):
            def on_epoch_end(self, args, state, control, model=None, **kwargs):
                model.save_pretrained(str(epoch_checkpoints_dir / f"epoch_{round(state.epoch)}"))
                return control

        callbacks = [_SaveEpochAdapterCallback()]

    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=collate, callbacks=callbacks)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    # Explicit final save: Trainer only leaves checkpoint-*/ subdirs on its
    # own, but evaluate_qlora loads the adapter from output_dir's ROOT.
    # Without this line it crashes with "adapter_config.json not found".
    # (When save_epoch_checkpoints=True, select_best_qlora_epoch overwrites
    # this last-epoch save with whichever epoch scored best on dev.)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return trainer, tokenizer


def evaluate_qlora(
    model_key: str,
    adapter_dir: str | Path,
    test_df: pd.DataFrame,
    flags: Optional[Dict[str, bool]] = None,
    batch_size: int = 16,
) -> Dict[str, float]:
    """Load a trained LoRA adapter onto the 4-bit base model and score it like any other variant.

    Reuses ``metrics.compute_metrics`` so QLoRA results are directly
    comparable, row-for-row, with the zero-/few-shot prompting results.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    from .inference import DEFAULT_GEN_KWARGS, generate_responses

    model_id = MODEL_REGISTRY[model_key]["model_id"]
    revision = MODEL_REGISTRY[model_key]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision,
        quantization_config=create_quantization_config(), device_map={"": 0}, torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    # Without this, LoRA's lora_dropout stays active (nn.Module defaults to
    # training mode) and silently breaks the determinism greedy decoding is
    # supposed to guarantee (see inference.py's DEFAULT_GEN_KWARGS docstring).
    model.eval()
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

    template = build_variant_messages(flags or {}, examples=None)
    prompts = render_prompts(template, test_df["text"].tolist(), tokenizer, enable_thinking=(flags or {}).get("cot", False))
    responses = generate_responses(pipe, prompts, batch_size=batch_size, gen_kwargs=DEFAULT_GEN_KWARGS)
    metrics = compute_metrics(responses, test_df["label"].tolist())
    # Free this base model's VRAM before the caller loads the next one:
    # two 4-bit 7-9B bases resident at once can OOM a VRAM-constrained GPU.
    del pipe, model, base_model
    release_cuda_memory()
    return metrics


def select_best_qlora_epoch(
    model_key: str,
    adapter_dir: str | Path,
    dev_df: pd.DataFrame,
    flags: Optional[Dict[str, bool]] = None,
    batch_size: int = 16,
) -> Dict[str, object]:
    """Score every per-epoch adapter snapshot from ``train_qlora(...,
    save_epoch_checkpoints=True)`` against ``dev_df`` and promote the epoch
    with the best F1 to ``adapter_dir``'s root, the path ``evaluate_qlora``
    actually loads from. Mirrors ``train_encoder_baseline``'s
    ``metric_for_best_model="f1"`` selection so the two fine-tuned baselines
    are chosen by the same criterion.

    Writes ``adapter_dir/dev_selection.json`` (every epoch's dev metrics plus
    which one was picked) for the same auditability this project keeps for
    every other run artifact.

    Call this only after releasing the training model's CUDA memory (as
    every caller already does before its own ``evaluate_qlora`` call): each
    epoch here loads its own fresh 4-bit base model via ``evaluate_qlora``,
    and two resident copies of a 7-9B model can OOM a VRAM-constrained GPU.
    """
    adapter_dir = Path(adapter_dir)
    epoch_checkpoints_dir = adapter_dir / "epoch_checkpoints"
    epoch_dirs = sorted(epoch_checkpoints_dir.iterdir(), key=lambda p: int(p.name.rsplit("_", 1)[1]))
    if not epoch_dirs:
        raise ValueError(
            f"no epoch checkpoints found under {epoch_checkpoints_dir}: "
            "train_qlora must be called with save_epoch_checkpoints=True first"
        )

    per_epoch = []
    best_epoch, best_f1, best_dir = None, float("-inf"), None
    for epoch_dir in epoch_dirs:
        epoch_num = int(epoch_dir.name.rsplit("_", 1)[1])
        epoch_metrics = evaluate_qlora(model_key, epoch_dir, dev_df, flags=flags, batch_size=batch_size)
        per_epoch.append({"epoch": epoch_num, **epoch_metrics})
        if epoch_metrics["f1"] > best_f1:
            best_f1, best_epoch, best_dir = epoch_metrics["f1"], epoch_num, epoch_dir

    for item in best_dir.iterdir():
        shutil.copy2(item, adapter_dir / item.name)

    selection = {"selected_epoch": best_epoch, "per_epoch": per_epoch}
    with open(adapter_dir / "dev_selection.json", "w") as f:
        json.dump(selection, f, indent=2)
    return selection
