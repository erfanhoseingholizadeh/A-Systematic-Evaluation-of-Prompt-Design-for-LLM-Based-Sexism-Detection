import pandas as pd
import pytest

from sexism_prompting.prompts import (
    PROMPT_COMPONENTS,
    build_few_shot_demonstrations,
    build_variant_messages,
    generate_shuffled_orders,
    render_prompts,
    variant_name,
)


def test_non_cot_variant_ends_with_answer_cue():
    messages = build_variant_messages({"cot": False})
    user_content = messages[-1]["content"]
    assert user_content.rstrip().endswith("ANSWER:")
    assert "Respond only YES or NO." in user_content


def test_cot_variant_has_no_forced_answer_cue():
    """The CoT fix: the prompt must not force an immediate ANSWER: cue, or the
    model has no room to reason before answering."""
    messages = build_variant_messages({"cot": True})
    user_content = messages[-1]["content"]
    assert not user_content.rstrip().endswith("ANSWER:")
    assert "Respond only YES or NO." not in user_content
    assert "ANSWER: YES" in user_content  # instructs the model to self-produce the marker


def test_role_flag_adds_system_message():
    without_role = build_variant_messages({"role": False})
    with_role = build_variant_messages({"role": True})
    assert without_role[0]["role"] == "user"
    assert with_role[0]["role"] == "system"
    assert with_role[-1]["role"] == "user"


def test_fewshot_requires_examples():
    with pytest.raises(ValueError):
        build_variant_messages({"fewshot": True}, examples=None)


def test_fewshot_with_examples_included_in_content():
    messages = build_variant_messages({"fewshot": True}, examples=["TEXT: foo\nANSWER: YES"])
    assert "TEXT: foo" in messages[-1]["content"]


def test_variant_name_labels():
    assert variant_name({}) == "baseline"
    assert variant_name({"role": True, "fewshot": True}) == "R+F"
    assert variant_name({"role": True, "aspects": True, "context": True, "cot": True, "fewshot": True}) == "R+A+C+T+F"


def test_build_few_shot_demonstrations_is_balanced_and_seeded():
    demos = pd.DataFrame(
        {"text": [f"t{i}" for i in range(10)], "label": [0, 1] * 5}
    )
    result_a = build_few_shot_demonstrations(demos, num_per_class=2, seed=42)
    result_b = build_few_shot_demonstrations(demos, num_per_class=2, seed=42)
    assert result_a == result_b  # deterministic given the same seed
    assert len(result_a) == 4
    assert sum("ANSWER: YES" in d for d in result_a) == 2
    assert sum("ANSWER: NO" in d for d in result_a) == 2


def test_generate_shuffled_orders_deterministic_and_distinct():
    orders_a = generate_shuffled_orders(["context", "cot", "aspects"], n=3, seed=1)
    orders_b = generate_shuffled_orders(["context", "cot", "aspects"], n=3, seed=1)
    assert orders_a == orders_b
    assert len({tuple(o) for o in orders_a}) == len(orders_a)  # all distinct


class _FakeTokenizerNoChat:
    chat_template = None


def test_render_prompts_fallback_without_chat_template():
    template = build_variant_messages({"role": True})
    rendered = render_prompts(template, ["hello world"], _FakeTokenizerNoChat())
    assert len(rendered) == 1
    assert "hello world" in rendered[0]
    assert "System:" in rendered[0] and "User:" in rendered[0]


class _FakeTokenizerAcceptsSystemRole:
    chat_template = "fake"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"


class _FakeTokenizerRejectsSystemRole:
    """Simulates Gemma-2's real chat template, which raises a TemplateError
    for any system-role message rather than rendering one."""

    chat_template = "fake"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        if any(m.get("role") == "system" for m in messages):
            raise ValueError("System role not supported")
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"


def test_render_prompts_keeps_system_role_when_template_supports_it():
    template = build_variant_messages({"role": True})
    rendered = render_prompts(template, ["hello world"], _FakeTokenizerAcceptsSystemRole())
    assert "system:" in rendered[0]
    assert "user:" in rendered[0]
    assert "hello world" in rendered[0]


def test_render_prompts_folds_system_into_user_when_template_rejects_it():
    """Regression: Gemma-2's real chat template raised
    TemplateError('System role not supported') on every role:true variant
    (9 of 18), permanently blocking those units. The system content must be
    folded into the user turn, not dropped or left to crash the unit."""
    template = build_variant_messages({"role": True})
    rendered = render_prompts(template, ["hello world"], _FakeTokenizerRejectsSystemRole())
    assert len(rendered) == 1
    assert "hello world" in rendered[0]
    assert PROMPT_COMPONENTS["role"]["content"] in rendered[0]
    assert "system:" not in rendered[0]  # no separate system turn was sent


class _FakeTokenizerRecordsThinkingKwarg:
    """Simulates a reasoning-tuned model's chat template (e.g. Qwen3), whose
    template accepts an `enable_thinking` kwarg and records what it got."""

    chat_template = "fake"

    def __init__(self):
        self.seen_kwargs = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        self.seen_kwargs.append(kwargs.get("enable_thinking"))
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"


def test_render_prompts_passes_enable_thinking_through_when_given():
    template = build_variant_messages({"cot": True})
    tokenizer = _FakeTokenizerRecordsThinkingKwarg()
    render_prompts(template, ["hello"], tokenizer, enable_thinking=True)
    assert tokenizer.seen_kwargs == [True]


def test_render_prompts_omits_enable_thinking_by_default():
    """A tokenizer with a strict apply_chat_template signature (no **kwargs,
    like every non-reasoning-model fake in this file) must not break when
    enable_thinking isn't explicitly requested: the kwarg must not be sent
    at all, not sent as None."""
    template = build_variant_messages({"role": True})
    rendered = render_prompts(template, ["hello world"], _FakeTokenizerAcceptsSystemRole())
    assert "hello world" in rendered[0]  # no TypeError from an unexpected kwarg


def test_render_prompts_survives_braces_in_fewshot_demos():
    """A few-shot demonstration text containing literal braces must render
    correctly, not raise KeyError from a naive .format(text=...) call. The
    full EDOS train split really does contain such texts, so a demo-seed
    variation run could otherwise crash mid-session on them."""
    demos = [
        "TEXT: women be like {angry} lol\nANSWER: YES",
        "TEXT: nice {day} today }{\nANSWER: NO",
    ]
    template = build_variant_messages({"fewshot": True}, examples=demos)
    rendered = render_prompts(template, ["some {input} }{ text"], _FakeTokenizerNoChat())
    assert "{angry}" in rendered[0]  # demo braces survive verbatim
    assert "some {input} }{ text" in rendered[0]  # input text placed, braces intact
    assert "{text}" not in rendered[0]  # the placeholder itself was replaced
