from __future__ import annotations

import random

import torch
from torch import nn

from neural_firmware.internal_data import (
    internal_prompt,
    locate_operand_character_spans,
)
from neural_firmware.internal_firmware import (
    FrozenTypedAdditionCell,
    InternalArithmeticUnit,
    InternalFirmwareContext,
    InternalFirmwareLayer,
    InternalLearnedControlLayer,
    LearnedControlContext,
    ParameterMatchedResidualAdapter,
)


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


class _IdentityLayer(nn.Module):
    attention_type = "full_attention"

    def forward(self, hidden_states: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        return hidden_states


def test_internal_layer_runs_cell_and_injects_inside_wrapped_forward() -> None:
    unit = InternalArithmeticUnit(hidden_size=10, strength=3.0)
    with torch.no_grad():
        unit.digit_encoder.classifier.weight.copy_(torch.eye(10))
        unit.digit_encoder.classifier.bias.zero_()
    wrapper = InternalFirmwareLayer(
        _IdentityLayer(),
        unit,
        depth_after_blocks=6,
    )
    hidden = torch.zeros((1, 8, 10))
    hidden[0, 1, 1] = 10
    hidden[0, 2, 2] = 10
    hidden[0, 4, 3] = 10
    context = InternalFirmwareContext(
        a_positions=torch.tensor([[1, 2]]),
        a_lengths=torch.tensor([2]),
        b_positions=torch.tensor([[4]]),
        b_lengths=torch.tensor([1]),
        output_positions=torch.tensor([[5, 6, 7, 0]]),
    )
    wrapper.set_context(context)
    output = wrapper(hidden)
    assert context.diagnostics["a_predictions"].tolist() == [[1, 2]]
    assert context.diagnostics["b_predictions"].tolist() == [[3]]
    assert context.diagnostics["planned_symbols"].tolist() == [[1, 5, 10, -1]]
    assert context.diagnostics["planned_symbol_mask"].tolist() == [
        [True, True, True, False]
    ]
    assert not torch.equal(output[:, 5:, :], hidden[:, 5:, :])
    assert torch.equal(output[:, :5, :], hidden[:, :5, :])


def test_internal_layer_off_is_exact_identity_after_base_layer() -> None:
    unit = InternalArithmeticUnit(hidden_size=10, strength=3.0)
    wrapper = InternalFirmwareLayer(
        _IdentityLayer(),
        unit,
        depth_after_blocks=12,
    )
    hidden = torch.randn((2, 4, 10))
    wrapper.set_context(None)
    assert torch.equal(wrapper(hidden), hidden)


def test_learned_control_matches_interface_parameter_count_and_is_gated() -> None:
    hidden_size = 896
    unit = InternalArithmeticUnit(hidden_size=hidden_size, strength=64.0)
    adapter = ParameterMatchedResidualAdapter(hidden_size=hidden_size, rank=10)
    assert unit.interface_parameter_count == 18_826
    assert sum(parameter.numel() for parameter in adapter.parameters()) == 18_826
    wrapper = InternalLearnedControlLayer(
        _IdentityLayer(),
        adapter,
        depth_after_blocks=6,
    )
    hidden = torch.randn((1, 4, hidden_size))
    wrapper.set_context(None)
    assert torch.equal(wrapper(hidden), hidden)
    wrapper.set_context(
        LearnedControlContext(
            output_positions=torch.tensor([[2, 0]]),
            output_mask=torch.tensor([[True, False]]),
        )
    )
    with torch.no_grad():
        adapter.up.weight.fill_(0.01)
    changed = wrapper(hidden)
    assert not torch.equal(changed[:, 2, :], hidden[:, 2, :])
    assert torch.equal(changed[:, :2, :], hidden[:, :2, :])
