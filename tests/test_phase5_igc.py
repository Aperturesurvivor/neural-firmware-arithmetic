from __future__ import annotations

import torch
from torch import nn

from neural_firmware.phase5_igc import (
    IGCArithmeticUnit,
    IGCInputMapping,
    IGCOutputMapping,
    categorical_digits_to_register,
    install_dual_depth_igc,
)


def test_matched_igc_parameter_count_is_exact() -> None:
    unit = IGCArithmeticUnit(
        896,
        max_digits=12,
        attention_width=3,
        attention_heads=1,
        output_width=577,
        initial_strength=64.0,
    )
    assert unit.learned_parameter_count == 24_231


def test_native_igc_parameter_count() -> None:
    unit = IGCArithmeticUnit(
        896,
        max_digits=12,
        attention_width=64,
        attention_heads=8,
        output_width=896,
        initial_strength=64.0,
    )
    assert unit.learned_parameter_count == 583_450


class _FakeLayer(nn.Module):
    attention_type = "full_attention"

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, hidden: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        return hidden


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 896})()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_FakeLayer() for _ in range(24)])
        self.anchor = nn.Parameter(torch.zeros(()))


def test_dual_depth_parameter_budgets() -> None:
    matched = install_dual_depth_igc(
        _FakeModel(),
        input_depth_after_blocks=1,
        output_depth_after_blocks=24,
        max_digits=12,
        attention_width=3,
        attention_heads=1,
        output_width=495,
        router_hidden_width=0,
        learn_output_strength=False,
    )
    assert matched.learned_parameter_count == 24_225
    native = install_dual_depth_igc(
        _FakeModel(),
        input_depth_after_blocks=1,
        output_depth_after_blocks=24,
        max_digits=12,
        attention_width=64,
        attention_heads=8,
        output_width=896,
    )
    assert native.learned_parameter_count == 597_819


def test_categorical_digits_use_first_pad_as_length() -> None:
    logits = torch.full((2, 4, 11), -10.0)
    for row, values in enumerate(((1, 2, 10, 9), (10, 3, 4, 10))):
        for column, value in enumerate(values):
            logits[row, column, value] = 10.0
    digits, lengths = categorical_digits_to_register(logits)
    assert lengths.tolist() == [2, 1]
    assert digits[0, :2].tolist() == [1, 2]
    assert digits[1, 0].item() == 0


def test_input_mapping_shapes_and_padding_mask() -> None:
    mapping = IGCInputMapping(
        8,
        max_digits=3,
        attention_width=4,
        attention_heads=1,
    )
    hidden = torch.randn(2, 5, 8)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    anchors = torch.tensor([2, 4])
    a_logits, b_logits, operation_logits = mapping(hidden, mask, anchors)
    assert a_logits.shape == (2, 3, 11)
    assert b_logits.shape == (2, 3, 11)
    assert operation_logits.shape == (2, 2)
    assert torch.isfinite(a_logits).all()


def test_output_mapping_zero_gate_is_half_strength() -> None:
    mapping = IGCOutputMapping(
        8,
        output_width=6,
        initial_strength=4.0,
    )
    hidden = torch.zeros(2, 8)
    symbols = torch.tensor([1, 10])
    residuals = mapping(hidden, symbols)
    assert residuals.shape == (2, 8)
    assert torch.allclose(residuals[:, 6:], torch.zeros(2, 2))
    assert torch.allclose(residuals.norm(dim=-1), torch.full((2,), 2.0), atol=1e-5)


def test_calculator_remains_parameter_free() -> None:
    unit = IGCArithmeticUnit(
        8,
        max_digits=3,
        attention_width=4,
        attention_heads=1,
        output_width=8,
        initial_strength=4.0,
    )
    assert list(unit.calculator.parameters()) == []
    assert isinstance(unit.calculator, nn.Module)
