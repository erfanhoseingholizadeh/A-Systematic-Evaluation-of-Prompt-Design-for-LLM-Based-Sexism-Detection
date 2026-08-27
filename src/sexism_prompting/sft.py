"""Torch-free helpers for building supervised fine-tuning examples.

Kept out of ``finetune.py`` (which imports torch at module level) so this
logic stays unit-testable in the GPU-less dev sandbox, per this repo's
pure-Python-tests rule.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import pandas as pd


def build_masked_example(
    prompt_ids: Sequence[int], completion_ids: Sequence[int], max_length: int
) -> Tuple[List[int], List[int]]:
    """(input_ids, labels) for one SFT example, with the prompt span masked to -100.

    Works on already-tokenized id lists and simply concatenates them, so the
    prompt/completion boundary is exact by construction. Tokenizing
    ``prompt + completion`` as one string and masking ``len(tokenize(prompt))``
    tokens would instead be off by one whenever the tokenizer merges across
    the boundary, silently masking part of the completion or training on
    the last prompt token. This also matches inference exactly:
    the model really is fed the prompt's own tokenization and must continue it.

    If the pair exceeds ``max_length`` the PROMPT is truncated from the LEFT,
    so the answer cue at its tail ("… ANSWER:") and the completion survive;
    right-truncation would cut the cue and could leave every label -100
    (a NaN loss).
    """
    completion_ids = list(completion_ids)
    budget = max_length - len(completion_ids)
    if budget < 0:
        completion_ids = completion_ids[:max_length]
        budget = 0
    kept_prompt = list(prompt_ids)[-budget:] if budget > 0 else []
    input_ids = kept_prompt + completion_ids
    labels = [-100] * len(kept_prompt) + completion_ids
    return input_ids, labels


def balance_classes(df: pd.DataFrame, label_col: str = "label", seed: int = 42) -> pd.DataFrame:
    """Oversample the minority class (with replacement) to match the majority count.

    EDOS's real train split is ~76/24 not-sexist/sexist. QLoRA's SFT
    objective (next-token prediction on a single " YES"/" NO" completion)
    has no built-in class weighting the way a classification head does:
    trained on that raw imbalance it collapses to unconditionally
    predicting the majority label (confirmed: mistral's first QLoRA run
    scored accuracy=0.5/f1=0.0 on the balanced held-out set, i.e. always
    "NO"). Resampling to balance at the data level, before any tokenization,
    sidesteps that without touching the training loop.
    """
    majority_n = df[label_col].value_counts().max()
    parts = []
    for _, group in df.groupby(label_col):
        if len(group) < majority_n:
            extra = group.sample(n=majority_n - len(group), replace=True, random_state=seed)
            group = pd.concat([group, extra], axis=0)
        parts.append(group)
    balanced = pd.concat(parts, axis=0)
    return balanced.sample(frac=1, random_state=seed).reset_index(drop=True)
