from __future__ import annotations

import torch
from torch import nn

from neural_firmware.phase10_training import (
    PHASE10_CONDITIONS,
    install_phase10_implant,
    phase10_condition,
)


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(32, 64, bias=False)
        self.up_proj = nn.Linear(32, 64, bias=False)
        self.down_proj = nn.Linear(64, 32, bias=False)
        self.act_fn = nn.SiLU()


class TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = TinyMLP()


class TinyInner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TinyLayer()])


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = TinyInner()


class TinyBundle:
    def __init__(self) -> None:
        self.model = TinyModel()
        self.device = torch.device("cpu")


def source_checkpoint() -> dict[str, object]:
    return {
        "layout": {
            "max_digits": 1,
            "digit_classes": 11,
            "result_classes": 12,
            "learned_step": False,
        },
        "layer_index": 0,
        "selected_indices": torch.arange(28),
        "output_strength": 16.0,
        "route_threshold": 0.5,
        "digit_threshold": 0.8,
        "input_rows": torch.randn(16, 32) * 0.01,
        "result_columns": torch.randn(32, 12),
    }


def test_phase10_factorial_conditions_are_declared() -> None:
    assert [condition.name for condition in PHASE10_CONDITIONS] == [
        "linear",
        "nonlinear",
        "linear_representation",
        "nonlinear_representation",
    ]
    assert phase10_condition("nonlinear").interface_kind == "bottleneck_silu"


def test_phase10_nonlinear_interface_is_exactly_input_budget_matched() -> None:
    linear = install_phase10_implant(
        TinyBundle(),
        source_checkpoint(),
        condition=phase10_condition("linear"),
        seed=16_199,
    )
    nonlinear = install_phase10_implant(
        TinyBundle(),
        source_checkpoint(),
        condition=phase10_condition("nonlinear"),
        seed=16_199,
    )
    assert linear.input_rows.numel() == nonlinear.input_rows.numel() == 512
    nonlinear_budget = (
        nonlinear.bottleneck_rows.numel()
        + nonlinear.bottleneck_mix.numel()
    )
    assert nonlinear_budget == linear.input_rows.numel()
    assert linear.trainable_parameter_count == nonlinear.trainable_parameter_count
    assert nonlinear.interface_kind == "bottleneck_silu"


def test_phase10_representation_condition_discloses_added_parameters() -> None:
    implant = install_phase10_implant(
        TinyBundle(),
        source_checkpoint(),
        condition=phase10_condition("linear_representation"),
        seed=16_199,
    )
    assert implant.representation_rank == 4
    assert implant.adapt_base_mlp is False
    assert implant.representation_down.numel() == 128
    assert implant.representation_up.numel() == 128
