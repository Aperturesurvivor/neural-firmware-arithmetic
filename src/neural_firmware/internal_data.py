from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass

INTERNAL_PROMPT_PATTERN = re.compile(
    r"\AUse the internal addition register\. Return only the exact integer\.\n"
    r"A = (?P<a>[0-9](?: [0-9])*)\n"
    r"B = (?P<b>[0-9](?: [0-9])*)\Z"
)


def space_digits(value: str) -> str:
    if not value.isdigit():
        raise ValueError("value must contain only decimal digits")
    return " ".join(value)


def internal_prompt(a: str, b: str) -> str:
    if not a.isdigit() or not b.isdigit():
        raise ValueError("operands must contain only decimal digits")
    return (
        "Use the internal addition register. Return only the exact integer.\n"
        f"A = {space_digits(a)}\n"
        f"B = {space_digits(b)}"
    )


@dataclass(frozen=True)
class InternalAdditionExample:
    prompt: str
    a: str
    b: str
    answer: str
    split: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class OperandCharacterSpans:
    a_digit_offsets: tuple[int, ...]
    b_digit_offsets: tuple[int, ...]


@dataclass(frozen=True)
class EncodedInternalPrompt:
    input_ids: tuple[int, ...]
    a_token_positions: tuple[int, ...]
    b_token_positions: tuple[int, ...]


def locate_operand_character_spans(prompt: str) -> OperandCharacterSpans | None:
    """Locate operand digit characters without converting them to values."""

    match = INTERNAL_PROMPT_PATTERN.fullmatch(prompt)
    if match is None:
        return None
    return OperandCharacterSpans(
        a_digit_offsets=tuple(
            match.start("a") + offset
            for offset, character in enumerate(match.group("a"))
            if character.isdigit()
        ),
        b_digit_offsets=tuple(
            match.start("b") + offset
            for offset, character in enumerate(match.group("b"))
            if character.isdigit()
        ),
    )


def encode_internal_prompt(tokenizer: object, prompt: str) -> EncodedInternalPrompt:
    spans = locate_operand_character_spans(prompt)
    if spans is None:
        raise ValueError("prompt does not full-match the registered grammar")
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_start = chat_text.find(prompt)
    if prompt_start < 0 or chat_text.find(prompt, prompt_start + 1) >= 0:
        raise ValueError("prompt must occur exactly once in formatted chat")
    encoded = tokenizer(
        chat_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]

    def token_positions(character_offsets: tuple[int, ...]) -> tuple[int, ...]:
        positions: list[int] = []
        for local_offset in character_offsets:
            absolute_offset = prompt_start + local_offset
            matches = [
                token_index
                for token_index, (start, end) in enumerate(offsets)
                if start <= absolute_offset < end
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"digit character at {absolute_offset} maps to {len(matches)} tokens"
                )
            positions.append(matches[0])
        return tuple(positions)

    return EncodedInternalPrompt(
        input_ids=tuple(encoded["input_ids"]),
        a_token_positions=token_positions(spans.a_digit_offsets),
        b_token_positions=token_positions(spans.b_digit_offsets),
    )


def _sample_integer(rng: random.Random, digits: int) -> str:
    if digits < 1:
        raise ValueError("digits must be positive")
    if digits == 1:
        return str(rng.randrange(10))
    lower = 10 ** (digits - 1)
    return str(rng.randrange(lower, 10**digits))


def make_internal_addition_examples(
    *,
    count: int,
    min_digits: int,
    max_digits: int,
    seed: int,
    split: str,
) -> list[InternalAdditionExample]:
    rng = random.Random(seed)
    examples: list[InternalAdditionExample] = []
    for _ in range(count):
        a = _sample_integer(rng, rng.randint(min_digits, max_digits))
        b = _sample_integer(rng, rng.randint(min_digits, max_digits))
        examples.append(
            InternalAdditionExample(
                prompt=internal_prompt(a, b),
                a=a,
                b=b,
                answer=str(int(a) + int(b)),
                split=split,
            )
        )
    return examples


def make_internal_carry_examples(
    *,
    count: int,
    min_digits: int,
    max_digits: int,
    seed: int,
    split: str = "carry_chain",
) -> list[InternalAdditionExample]:
    rng = random.Random(seed)
    examples: list[InternalAdditionExample] = []
    for _ in range(count):
        digits = rng.randint(min_digits, max_digits)
        trailing_nines = rng.randint(max(1, digits // 2), digits)
        prefix_digits = digits - trailing_nines
        prefix = _sample_integer(rng, prefix_digits) if prefix_digits else ""
        a = prefix + ("9" * trailing_nines)
        b = str(rng.randint(1, 9))
        examples.append(
            InternalAdditionExample(
                prompt=internal_prompt(a, b),
                a=a,
                b=b,
                answer=str(int(a) + int(b)),
                split=split,
            )
        )
    return examples
