"""Modular prompt construction for the R / A / C / T / F component grid.

R = role specification, A = aspect-focused linguistic markers, C = contextual
framing, T = chain-of-thought, F = few-shot demonstrations.

``cot`` variants omit a forced trailing ``ANSWER:`` cue and instead ask the
model to end its own response with ``ANSWER: YES``/``ANSWER: NO`` once it
has reasoned. Appending that cue immediately before generation, the way
non-CoT variants do, would leave the model zero room to actually reason
even when the CoT instructions ask it to. The metrics module's tail parser
already finds the *last* ``ANSWER:`` occurrence, so this works whether that
marker was inserted by the prompt (non-CoT) or produced by the model after
reasoning (CoT). Pair this with a real ``max_new_tokens`` budget for ``cot``
variants (see ``configs/models.yaml``): 2 tokens still won't reason.
"""

from __future__ import annotations

import math
import random as _random
import copy
from typing import Dict, List, Optional, Sequence

import pandas as pd

PROMPT_COMPONENTS: Dict[str, Dict[str, str]] = {
    "role": {
        "type": "system",
        "content": "You are an annotator for sexism detection.",
    },
    "context": {
        "type": "section",
        "content": """CONTEXT:
Platform: Social media
Content type: Public post
Language: English
""",
    },
    "cot": {
        "type": "section",
        "content": """INSTRUCTIONS:
1. First, briefly analyze the text for sexist content, reasoning step by step.
2. Then, on a new line, give your final decision in exactly this format: ANSWER: YES or ANSWER: NO
""",
    },
    "aspects": {
        "type": "section",
        "content": """LINGUISTIC MARKERS TO CONSIDER:
- Objectification: Reducing women to physical appearance or sexual objects
- Gender stereotypes: Reinforcing limiting roles based on gender
- Hostile language: Aggressive, derogatory, or threatening language toward women
- Benevolent sexism: Seemingly positive but patronizing statements
- Dehumanization: Using demeaning terms or animal comparisons
- Double standards: Different expectations or judgments based on gender
- Dismissiveness: Trivializing women's experiences, opinions, or capabilities
""",
    },
}

# Order components are concatenated into the user message body, before the
# TEXT/ANSWER block. This is the order the 18-variant grid uses.
DEFAULT_COMPONENT_ORDER = ["context", "cot", "aspects", "fewshot"]

_BASE_INSTRUCTION = "Your task is to classify input text as containing sexism or not."


def _render_section(name: str, examples: Optional[List[str]]) -> str:
    if name == "fewshot":
        return "EXAMPLES:\n" + "\n\n".join(examples or [])
    return PROMPT_COMPONENTS[name]["content"]


def generate_shuffled_orders(components: Sequence[str], n: int = 5, seed: int = 42) -> List[List[str]]:
    """Up to ``n`` distinct random orderings of ``components``. Deterministic given ``seed``."""
    components = list(components)
    n = min(n, math.factorial(len(components)))
    rng = _random.Random(seed)

    seen = set()
    orders: List[List[str]] = []
    while len(orders) < n:
        order = components[:]
        rng.shuffle(order)
        key = tuple(order)
        if key not in seen:
            seen.add(key)
            orders.append(order)
    return orders


def build_few_shot_demonstrations(
    demonstrations: pd.DataFrame, num_per_class: int = 2, seed: int = 42
) -> List[str]:
    """Sample a balanced, seeded set of demonstrations formatted as TEXT/ANSWER pairs."""
    d0 = demonstrations[demonstrations["label"] == 0].sample(n=num_per_class, random_state=seed, replace=False)
    d1 = demonstrations[demonstrations["label"] == 1].sample(n=num_per_class, random_state=seed, replace=False)

    demos = []
    for _, row in pd.concat([d1, d0], axis=0).iterrows():
        answer = "YES" if int(row["label"]) == 1 else "NO"
        demos.append(f"TEXT: {row['text']}\nANSWER: {answer}")
    return demos


def build_variant_messages(
    flags: Dict[str, bool],
    examples: Optional[List[str]] = None,
    component_order: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Build a chat-message template (with a ``{text}`` placeholder) for one variant.

    Args:
        flags: subset of {"role", "context", "aspects", "cot", "fewshot"} -> bool.
            Unset keys default to False.
        examples: required when ``flags["fewshot"]`` is True; formatted
            demonstration strings (see ``build_few_shot_demonstrations``).
        component_order: order to concatenate active (non-role) components in,
            before the TEXT block. Defaults to ``DEFAULT_COMPONENT_ORDER``.

    Returns:
        A list of ``{"role": ..., "content": ...}`` messages; ``content`` still
        contains a literal ``{text}`` placeholder filled per-example by
        ``render_prompts``.
    """
    if flags.get("fewshot", False) and not examples:
        raise ValueError("flags['fewshot'] is True but no `examples` were provided")

    order = component_order if component_order is not None else DEFAULT_COMPONENT_ORDER
    is_cot = flags.get("cot", False)

    response_format = "" if is_cot else " Respond only YES or NO."
    sections = [_BASE_INSTRUCTION + response_format, ""]

    for name in order:
        if flags.get(name, False):
            sections.append(_render_section(name, examples))
            sections.append("")

    sections += ["TEXT:", "{text}"]

    # Non-CoT variants get an explicit trailing cue to force an immediate,
    # short (max_new_tokens=2) YES/NO. CoT variants omit it deliberately:
    # appending "ANSWER:" here would leave the model no room to reason
    # before answering.
    if not is_cot:
        sections += ["", "ANSWER:"]

    user_content = "\n".join(sections)

    messages: List[Dict[str, str]] = []
    if flags.get("role", False):
        messages.append({"role": "system", "content": PROMPT_COMPONENTS["role"]["content"]})
    messages.append({"role": "user", "content": user_content})
    return messages


def render_prompts(
    messages_template: Sequence[Dict[str, str]],
    texts: Sequence[str],
    tokenizer,
    enable_thinking: Optional[bool] = None,
) -> List[str]:
    """Fill the ``{text}`` placeholder for each input and render via the chat template.

    Falls back to a plain "System/User/Assistant" layout for tokenizers
    without a chat template. Some chat templates (e.g. Gemma-2's) reject a
    separate system-role message outright; when that happens, the system
    content is folded into the leading user turn instead of dropped or
    left to crash the whole (model, variant) unit.

    ``enable_thinking``, when not ``None``, is passed straight through to
    ``apply_chat_template``: reasoning-tuned models (e.g. Qwen3) use it to
    toggle genuine extended reasoning on/off per call. Templates that don't
    reference this kwarg simply ignore it (an unused Jinja template
    variable is not an error), so passing it is safe for every other model
    too. Left unset (``None``) by default for call sites that don't need it.
    """
    chat_template_kwargs: Dict[str, bool] = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    has_chat = hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None)
    has_system = any(m.get("role") == "system" for m in messages_template)
    fold_system_into_user = False

    if has_chat and has_system:
        probe = copy.deepcopy(list(messages_template))
        for m in probe:
            if isinstance(m.get("content"), str):
                m["content"] = m["content"].replace("{text}", texts[0] if texts else "")
        try:
            tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=True, **chat_template_kwargs)
        except Exception:
            fold_system_into_user = True

    rendered: List[str] = []

    for text in texts:
        messages = copy.deepcopy(list(messages_template))
        for m in messages:
            if isinstance(m.get("content"), str):
                # str.replace, NOT str.format: few-shot demonstration texts are
                # part of the template and may contain literal braces, which
                # .format would treat as replacement fields and crash on
                # (KeyError); the full EDOS train split really contains such
                # texts. The input text itself is only ever the replacement
                # value, so its braces are safe either way.
                m["content"] = m["content"].replace("{text}", text)

        if fold_system_into_user:
            sys_msg = next((m["content"] for m in messages if m.get("role") == "system"), "")
            messages = [m for m in messages if m.get("role") != "system"]
            for m in messages:
                if m.get("role") == "user":
                    m["content"] = f"{sys_msg}\n\n{m['content']}"
                    break

        if has_chat:
            prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
            )
        else:
            sys_msg = next((m["content"] for m in messages if m.get("role") == "system"), "")
            user_msg = next((m["content"] for m in messages if m.get("role") == "user"), "")
            prompt_str = f"System:\n{sys_msg}\n\nUser:\n{user_msg}\n\nAssistant:"
        rendered.append(prompt_str)

    return rendered


def variant_name(flags: Dict[str, bool]) -> str:
    """Short human-readable name for a flag set, e.g. {"role": True, "fewshot": True} -> 'R+F'."""
    letters = {"role": "R", "aspects": "A", "context": "C", "cot": "T", "fewshot": "F"}
    active = [letters[k] for k in ["role", "aspects", "context", "cot", "fewshot"] if flags.get(k, False)]
    return "+".join(active) if active else "baseline"
