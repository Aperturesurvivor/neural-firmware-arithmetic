from __future__ import annotations

import torch
from torch import nn

from neural_firmware.phase7_sequence_implant import (
    FrozenSequenceAddition,
    SequenceImplantContext,
    SequenceImplantLayout,
    SequenceInterface,
    SequenceNeuronImplantMLP,
)


class TinyMLP(nn.Module):
    def __init__(self, hidden_size: int = 8, intermediate_size: int = 64) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden)) * self.up_proj(hidden))


def test_sequence_layout_uses_34_existing_channels() -> None:
    layout = SequenceImplantLayout(max_digits=4)
    assert layout.input_width == 22
    assert layout.result_width == 12
    assert layout.total_width == 34


def test_fixed_step_layout_uses_28_existing_channels() -> None:
    layout = SequenceImplantLayout(max_digits=4, learned_step=False)
    assert layout.input_width == 16
    assert layout.result_width == 12
    assert layout.total_width == 28


def test_digit_confidence_handshake_rejects_uncertain_digit() -> None:
    layout = SequenceImplantLayout(max_digits=2)
    implant = SequenceNeuronImplantMLP(
        TinyMLP(intermediate_size=64),
        torch.arange(layout.total_width),
        layout=layout,
        digit_threshold=0.9,
    )
    digit_logits = torch.zeros(1, 2, layout.digit_width)
    digit_logits[0, 0, 7] = 20
    digit_logits[0, 1, 7] = 2
    hard = implant.hard_interface(
        SequenceInterface(
            route_logits=torch.zeros(1, 2, layout.route_width),
            role_logits=torch.zeros(1, 2, layout.role_width),
            digit_logits=digit_logits,
            step_logits=torch.zeros(1, 2, layout.step_width),
        )
    )
    assert hard.digits.tolist() == [[7, layout.non_digit]]
    assert hard.digit_probability[0, 0] > 0.9
    assert hard.digit_probability[0, 1] < 0.9


def test_sequence_calculator_scans_roles_and_adds_exactly() -> None:
    layout = SequenceImplantLayout(max_digits=4)
    calculator = FrozenSequenceAddition(layout)
    # Token roles encode A=99 and B=1 amid ordinary tokens.
    roles = torch.tensor([[0, 1, 1, 0, 2, 0, 0]])
    digits = torch.tensor([[10, 9, 9, 10, 1, 10, 10]])
    route = torch.zeros_like(roles)
    route[0, -1] = 1
    step = torch.zeros_like(roles)
    eligible = torch.zeros_like(roles, dtype=torch.bool)
    eligible[0, -1] = True
    sequence = torch.ones_like(eligible)
    execution = calculator(
        route=route,
        roles=roles,
        digits=digits,
        step=step,
        eligible_mask=eligible,
        sequence_mask=sequence,
    )
    assert execution.a_digits[0].tolist() == [9, 9, 10, 10]
    assert execution.b_digits[0].tolist() == [1, 10, 10, 10]
    assert execution.result_symbols[0, -1].item() == 1
    assert execution.route_active[0, -1].item() is True
    assert calculator.trainable_parameter_count == 0


def test_sequence_calculator_requires_role_and_digit_type_agreement() -> None:
    layout = SequenceImplantLayout(max_digits=4)
    calculator = FrozenSequenceAddition(layout)
    # The first token is spuriously tagged as operand A, but its digit type is
    # NON_DIGIT. The typed handshake must ignore it.
    roles = torch.tensor([[1, 1, 2, 0]])
    digits = torch.tensor([[10, 7, 5, 10]])
    route = torch.tensor([[0, 0, 0, 1]])
    eligible = torch.tensor([[False, False, False, True]])
    execution = calculator(
        route=route,
        roles=roles,
        digits=digits,
        step=torch.zeros_like(roles),
        eligible_mask=eligible,
        sequence_mask=torch.ones_like(eligible),
    )
    assert execution.operands_valid.tolist() == [True]
    assert execution.a_digits[0].tolist() == [7, 10, 10, 10]
    assert execution.b_digits[0].tolist() == [5, 10, 10, 10]
    assert execution.result_symbols[0, -1].item() == 1


def test_sequence_calculator_selects_later_result_digit_and_eos() -> None:
    layout = SequenceImplantLayout(max_digits=4)
    calculator = FrozenSequenceAddition(layout)
    roles = torch.tensor(
        [
            [1, 1, 2, 0],
            [1, 1, 2, 0],
        ]
    )
    digits = torch.tensor(
        [
            [9, 9, 1, 10],
            [9, 9, 1, 10],
        ]
    )
    route = torch.tensor([[0, 0, 0, 1], [0, 0, 0, 1]])
    step = torch.tensor([[0, 0, 0, 2], [0, 0, 0, 3]])
    eligible = torch.zeros_like(route, dtype=torch.bool)
    eligible[:, -1] = True
    execution = calculator(
        route=route,
        roles=roles,
        digits=digits,
        step=step,
        eligible_mask=eligible,
        sequence_mask=torch.ones_like(eligible),
    )
    assert execution.result_symbols[:, -1].tolist() == [0, 10]


def test_sequence_calculator_rejects_overwide_or_missing_operands() -> None:
    layout = SequenceImplantLayout(max_digits=2)
    calculator = FrozenSequenceAddition(layout)
    roles = torch.tensor([[1, 1, 1, 2, 0], [1, 0, 0, 0, 0]])
    digits = torch.tensor([[1, 2, 3, 4, 10], [1, 10, 10, 10, 10]])
    route = torch.zeros_like(roles)
    route[:, -1] = 1
    eligible = torch.zeros_like(roles, dtype=torch.bool)
    eligible[:, -1] = True
    execution = calculator(
        route=route,
        roles=roles,
        digits=digits,
        step=torch.zeros_like(roles),
        eligible_mask=eligible,
        sequence_mask=torch.ones_like(eligible),
    )
    assert execution.operands_valid.tolist() == [False, False]
    assert execution.route_active.count_nonzero().item() == 0


def test_gated_implant_exactly_preserves_base_when_route_is_off() -> None:
    torch.manual_seed(10)
    layout = SequenceImplantLayout(max_digits=2)
    base = TinyMLP(intermediate_size=64)
    implant = SequenceNeuronImplantMLP(
        base,
        torch.arange(layout.total_width),
        layout=layout,
    )
    hidden = torch.randn(1, 5, 8)
    eligible = torch.zeros(1, 5, dtype=torch.bool)
    eligible[:, -1] = True
    teacher_route = torch.full((1, 5), -1, dtype=torch.long)
    teacher_route[:, -1] = 0
    implant.set_context(
        SequenceImplantContext(
            eligible_mask=eligible,
            sequence_mask=torch.ones_like(eligible),
            teacher_route=teacher_route,
            preserve_base_when_off=True,
        )
    )
    assert torch.allclose(implant(hidden), base(hidden), atol=1e-6)


def test_sequence_implant_exactly_preserves_base_without_runtime_context() -> None:
    torch.manual_seed(11)
    layout = SequenceImplantLayout(max_digits=2)
    base = TinyMLP(intermediate_size=64)
    implant = SequenceNeuronImplantMLP(
        base,
        torch.arange(layout.total_width),
        layout=layout,
    )
    hidden = torch.randn(2, 4, 8)
    assert implant.runtime_context is None
    assert torch.allclose(implant(hidden), base(hidden), atol=1e-6)
