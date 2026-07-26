from __future__ import annotations

import torch
from torch import nn

from neural_firmware.phase6_firmware import (
    FrozenAdditionProgram,
    NeuralRegisterMapper,
    install_neural_firmware,
    register_program_call_counts,
)


def _register_logits(rows: list[list[list[int]]], max_digits: int) -> torch.Tensor:
    logits = torch.full((len(rows), 3, max_digits, 11), -10.0)
    for batch, operands in enumerate(rows):
        for operand_index, digits in enumerate(operands):
            values = digits + [10] * (max_digits - len(digits))
            for position, value in enumerate(values):
                logits[batch, operand_index, position, value] = 10.0
    return logits


def _symbol_text(symbols: torch.Tensor, mask: torch.Tensor) -> str:
    return "".join(
        str(int(symbol))
        for symbol, active in zip(symbols, mask, strict=True)
        if bool(active) and int(symbol) != 10
    )


def test_frozen_program_reuses_cell_for_two_calls() -> None:
    logits = _register_logits(
        [
            [[1, 2], [8], [9]],
            [[1, 2], [8], [5]],
        ],
        max_digits=3,
    )
    execution = FrozenAdditionProgram()(logits, torch.tensor([1, 2]))
    assert _symbol_text(execution.final_symbols[0], execution.final_mask[0]) == "20"
    assert _symbol_text(execution.call_symbols[1, 0], execution.call_masks[1, 0]) == "20"
    assert _symbol_text(execution.call_symbols[1, 1], execution.call_masks[1, 1]) == "25"
    assert _symbol_text(execution.final_symbols[1], execution.final_mask[1]) == "25"
    assert list(FrozenAdditionProgram().cell.parameters()) == []


def test_zero_call_masks_all_calculator_output() -> None:
    logits = _register_logits([[[4], [5], [6]]], max_digits=2)
    execution = FrozenAdditionProgram()(logits, torch.tensor([0]))
    assert not bool(execution.final_mask.any())
    assert not bool(execution.call_masks.any())


def test_register_occupancy_selects_program_length() -> None:
    logits = _register_logits(
        [
            [[1], [2], []],
            [[1], [2], [3]],
        ],
        max_digits=2,
    )
    assert register_program_call_counts(logits).tolist() == [1, 2]


def test_neural_register_mapper_shapes() -> None:
    mapper = NeuralRegisterMapper(
        8,
        max_digits=4,
        model_width=8,
        attention_heads=2,
        decoder_layers=1,
    )
    hidden = torch.randn(2, 6, 8)
    mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]],
    )
    anchors = torch.tensor([3, 5])
    logits, control_logits = mapper.forward_with_control(hidden, mask, anchors)
    assert logits.shape == (2, 3, 4, 11)
    assert control_logits.shape == (2, 5)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(control_logits).all()


class _FakeLayer(nn.Module):
    attention_type = "full_attention"

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        hidden: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        return hidden


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 32})()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_FakeLayer() for _ in range(24)])


def test_installation_freezes_base_and_keeps_learned_interfaces() -> None:
    model = _FakeModel()
    installation = install_neural_firmware(
        model,
        max_digits=4,
        model_width=16,
        attention_heads=4,
        decoder_layers=1,
        controller_width=8,
    )
    assert installation.learned_parameter_count > 0
    assert all(
        not parameter.requires_grad
        for parameter in installation.capture.base_layer.parameters()
    )
    assert any(
        parameter.requires_grad
        for parameter in installation.capture.mapper.parameters()
    )
    assert list(installation.final.program.cell.parameters()) == []
