from __future__ import annotations

import torch
from torch import nn

from neural_firmware.phase7_implant import (
    FrozenNeuronAddition,
    ImplantRuntimeContext,
    NeuronImplantLayout,
    NeuronImplantMLP,
)


class TinyMLP(nn.Module):
    def __init__(self, hidden_size: int = 8, intermediate_size: int = 128) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden)) * self.up_proj(hidden))


def typed_digits(values: list[list[int]], width: int, pad: int = 10) -> torch.Tensor:
    result = torch.full((len(values), width), pad, dtype=torch.long)
    for row, digits in enumerate(values):
        result[row, : len(digits)] = torch.tensor(digits)
    return result


def test_layout_occupies_108_channels() -> None:
    layout = NeuronImplantLayout(max_digits=4)
    assert layout.input_width == 96
    assert layout.result_width == 12
    assert layout.total_width == 108


def test_frozen_neuron_addition_emits_exact_requested_symbols() -> None:
    layout = NeuronImplantLayout(max_digits=4)
    calculator = FrozenNeuronAddition(layout)
    a = typed_digits([[9, 9], [1, 2, 3], [7]], layout.max_digits)
    b = typed_digits([[1], [8, 7, 7], [5]], layout.max_digits)
    # 99+1=100, 123+877=1000, 7+5=12. Select result positions 0, 3, 1.
    execution = calculator(
        route=torch.ones(3, dtype=torch.long),
        a_digits=a,
        b_digits=b,
        step=torch.tensor([0, 3, 1]),
        eligible_mask=torch.ones(3, dtype=torch.bool),
    )
    assert execution.result_symbols.tolist() == [1, 0, 2]
    assert execution.route_active.tolist() == [True, True, True]
    assert calculator.trainable_parameter_count == 0


def test_invalid_or_off_interfaces_do_not_activate() -> None:
    layout = NeuronImplantLayout(max_digits=4)
    calculator = FrozenNeuronAddition(layout)
    a = typed_digits([[1, 2], [1, 2]], layout.max_digits)
    b = typed_digits([[3], [3]], layout.max_digits)
    a[1] = torch.tensor([1, layout.pad_digit, 2, layout.pad_digit])
    execution = calculator(
        route=torch.tensor([0, 1]),
        a_digits=a,
        b_digits=b,
        step=torch.zeros(2, dtype=torch.long),
        eligible_mask=torch.ones(2, dtype=torch.bool),
    )
    assert execution.route_active.tolist() == [False, False]
    assert execution.result_activations.count_nonzero().item() == 0


def test_wrapper_keeps_width_and_uses_only_replacement_parameters() -> None:
    torch.manual_seed(4)
    layout = NeuronImplantLayout(max_digits=4)
    base = TinyMLP(intermediate_size=128)
    implant = NeuronImplantMLP(
        base,
        torch.arange(layout.total_width),
        layout=layout,
        output_strength=1.0,
    )
    hidden = torch.randn(2, 3, 8)
    output = implant(hidden)
    assert output.shape == hidden.shape
    assert implant.mlp_width == 128
    assert implant.trainable_parameter_count == (layout.input_width + 12) * 8
    assert all(not parameter.requires_grad for parameter in base.parameters())


def test_teacher_forced_exact_symbol_flows_through_result_columns() -> None:
    torch.manual_seed(5)
    layout = NeuronImplantLayout(max_digits=4)
    base = TinyMLP(intermediate_size=128)
    for parameter in base.parameters():
        nn.init.zeros_(parameter)
    implant = NeuronImplantMLP(
        base,
        torch.arange(layout.total_width),
        layout=layout,
        output_strength=1.0,
    )
    with torch.no_grad():
        implant.result_columns.zero_()
        implant.result_columns[0, 5] = 2.0

    hidden = torch.zeros(1, 1, 8)
    eligible = torch.ones(1, 1, dtype=torch.bool)
    implant.set_context(
        ImplantRuntimeContext(
            eligible_mask=eligible,
            teacher_route=torch.ones(1, 1, dtype=torch.long),
            teacher_a_digits=typed_digits([[2]], layout.max_digits).reshape(1, 1, -1),
            teacher_b_digits=typed_digits([[3]], layout.max_digits).reshape(1, 1, -1),
            teacher_step=torch.zeros(1, 1, dtype=torch.long),
        )
    )
    output = implant(hidden)
    assert output[0, 0, 0].item() == 2.0
    assert output[0, 0, 1:].count_nonzero().item() == 0

