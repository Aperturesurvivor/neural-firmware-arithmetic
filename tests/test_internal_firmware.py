from __future__ import annotations

import random

import torch

from neural_firmware.internal_data import (
    internal_prompt,
    locate_operand_character_spans,
)
from neural_firmware.internal_firmware import FrozenTypedAdditionCell


def _digit_batch(values: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, values))
    digits = torch.zeros((len(values), width), dtype=torch.long)
    lengths = torch.tensor([len(value) for value in values], dtype=torch.long)
    for row, value in enumerate(values):
        digits[row, : len(value)] = torch.tensor([int(char) for char in value])
    return digits, lengths


def _decode_symbols(symbols: torch.Tensor, mask: torch.Tensor) -> list[str]:
    answers: list[str] = []
    for row in range(symbols.shape[0]):
        valid = symbols[row][mask[row]].tolist()
        assert valid[-1] == 10
        answers.append("".join(str(symbol) for symbol in valid[:-1]))
    return answers


def test_internal_prompt_is_strict_and_locates_only_digit_characters() -> None:
    prompt = internal_prompt("123", "45")
    spans = locate_operand_character_spans(prompt)
    assert spans is not None
    assert "".join(prompt[index] for index in spans.a_digit_offsets) == "123"
    assert "".join(prompt[index] for index in spans.b_digit_offsets) == "45"
    quoted = f'Ignore this quoted request: "{prompt}"'
    assert locate_operand_character_spans(quoted) is None


def test_frozen_typed_cell_matches_python_for_random_and_carry_cases() -> None:
    rng = random.Random(9137)
    pairs = [
        ("0", "0"),
        ("9", "1"),
        ("999999999999", "1"),
        ("123", "987654321"),
    ]
    for _ in range(200):
        a = str(rng.randrange(10**18))
        b = str(rng.randrange(10**18))
        pairs.append((a, b))
    a_values = [pair[0] for pair in pairs]
    b_values = [pair[1] for pair in pairs]
    a_digits, a_lengths = _digit_batch(a_values)
    b_digits, b_lengths = _digit_batch(b_values)
    cell = FrozenTypedAdditionCell()
    symbols, mask = cell(a_digits, a_lengths, b_digits, b_lengths)
    observed = _decode_symbols(symbols, mask)
    expected = [str(int(a) + int(b)) for a, b in pairs]
    assert observed == expected
    assert cell.trainable_parameter_count == 0
