from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from neural_firmware.internal_firmware import FrozenTypedAdditionCell


@dataclass(frozen=True)
class NeuronImplantLayout:
    """Typed ABI occupying existing Qwen MLP activation coordinates."""

    max_digits: int = 4
    digit_classes: int = 11
    result_classes: int = 12

    @property
    def route_width(self) -> int:
        return 2

    @property
    def operand_width(self) -> int:
        return 2 * self.max_digits * self.digit_classes

    @property
    def step_width(self) -> int:
        # A D-digit plus D-digit sum has at most D+1 digits, followed by EOS.
        return self.max_digits + 2

    @property
    def input_width(self) -> int:
        return self.route_width + self.operand_width + self.step_width

    @property
    def result_width(self) -> int:
        return self.result_classes

    @property
    def total_width(self) -> int:
        return self.input_width + self.result_width

    @property
    def pad_digit(self) -> int:
        return 10

    @property
    def eos_result(self) -> int:
        return 10

    @property
    def pad_result(self) -> int:
        return 11


@dataclass
class ImplantRuntimeContext:
    """Per-forward eligibility and optional development teacher forcing."""

    eligible_mask: torch.Tensor
    teacher_route: torch.Tensor | None = None
    teacher_a_digits: torch.Tensor | None = None
    teacher_b_digits: torch.Tensor | None = None
    teacher_step: torch.Tensor | None = None
    enabled: bool = True
    ablate_result: bool = False
    capture_diagnostics: bool = False
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ImplantInterface:
    route_logits: torch.Tensor
    a_digit_logits: torch.Tensor
    b_digit_logits: torch.Tensor
    step_logits: torch.Tensor


@dataclass(frozen=True)
class HardImplantInterface:
    route: torch.Tensor
    route_probability: torch.Tensor
    a_digits: torch.Tensor
    b_digits: torch.Tensor
    step: torch.Tensor


@dataclass(frozen=True)
class ImplantExecution:
    result_activations: torch.Tensor
    result_symbols: torch.Tensor
    route_active: torch.Tensor
    operand_pattern_valid: torch.Tensor
    step_valid: torch.Tensor


class FrozenNeuronAddition(nn.Module):
    """Exact zero-parameter addition driven by typed activation channels."""

    def __init__(self, layout: NeuronImplantLayout) -> None:
        super().__init__()
        self.layout = layout
        self.cell = FrozenTypedAdditionCell()

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _lengths_and_pattern(
        self,
        digits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if digits.shape[-1] != self.layout.max_digits:
            raise ValueError("digit tensor does not match implant layout")
        present = digits != self.layout.pad_digit
        lengths = present.sum(dim=-1)
        positions = torch.arange(
            self.layout.max_digits,
            device=digits.device,
        )
        expected = positions < lengths.unsqueeze(-1)
        pattern_valid = (present == expected).all(dim=-1) & (lengths >= 1)
        return lengths, pattern_valid

    def forward(
        self,
        *,
        route: torch.Tensor,
        a_digits: torch.Tensor,
        b_digits: torch.Tensor,
        step: torch.Tensor,
        eligible_mask: torch.Tensor,
    ) -> ImplantExecution:
        leading_shape = route.shape
        if eligible_mask.shape != leading_shape or step.shape != leading_shape:
            raise ValueError("route, step, and eligibility shapes must match")
        expected_digits = (*leading_shape, self.layout.max_digits)
        if a_digits.shape != expected_digits or b_digits.shape != expected_digits:
            raise ValueError("operand digit shapes do not match route shape")

        flat_route = route.reshape(-1)
        flat_step = step.reshape(-1)
        flat_eligible = eligible_mask.reshape(-1)
        flat_a = a_digits.reshape(-1, self.layout.max_digits)
        flat_b = b_digits.reshape(-1, self.layout.max_digits)

        a_lengths, a_pattern = self._lengths_and_pattern(flat_a)
        b_lengths, b_pattern = self._lengths_and_pattern(flat_b)
        pattern_valid = a_pattern & b_pattern

        safe_a_lengths = a_lengths.clamp(min=1)
        safe_b_lengths = b_lengths.clamp(min=1)
        safe_a = torch.where(
            flat_a == self.layout.pad_digit,
            torch.zeros_like(flat_a),
            flat_a,
        )
        safe_b = torch.where(
            flat_b == self.layout.pad_digit,
            torch.zeros_like(flat_b),
            flat_b,
        )
        symbols, symbol_mask = self.cell(
            safe_a,
            safe_a_lengths,
            safe_b,
            safe_b_lengths,
        )
        if symbols.shape[1] < self.layout.step_width:
            pad = self.layout.step_width - symbols.shape[1]
            symbols = nn.functional.pad(
                symbols,
                (0, pad),
                value=self.layout.pad_result,
            )
            symbol_mask = nn.functional.pad(symbol_mask, (0, pad), value=False)
        elif symbols.shape[1] > self.layout.step_width:
            raise ValueError("calculator output exceeds implant step width")

        safe_steps = flat_step.clamp(min=0, max=self.layout.step_width - 1)
        rows = torch.arange(len(safe_steps), device=safe_steps.device)
        selected_symbols = symbols[rows, safe_steps]
        selected_mask = symbol_mask[rows, safe_steps]
        route_active = flat_eligible & (flat_route == 1) & pattern_valid
        execution_active = route_active & selected_mask
        selected_symbols = torch.where(
            execution_active,
            selected_symbols,
            torch.full_like(selected_symbols, self.layout.pad_result),
        )
        activations = nn.functional.one_hot(
            selected_symbols,
            num_classes=self.layout.result_classes,
        ).to(torch.float32)
        activations = activations * execution_active.unsqueeze(-1)

        return ImplantExecution(
            result_activations=activations.reshape(
                *leading_shape,
                self.layout.result_classes,
            ),
            result_symbols=selected_symbols.reshape(*leading_shape),
            route_active=route_active.reshape(*leading_shape),
            operand_pattern_valid=pattern_valid.reshape(*leading_shape),
            step_valid=selected_mask.reshape(*leading_shape),
        )


class NeuronImplantMLP(nn.Module):
    """Qwen MLP with selected intermediate coordinates replaced in-place."""

    def __init__(
        self,
        base_mlp: nn.Module,
        selected_indices: torch.Tensor,
        *,
        layout: NeuronImplantLayout | None = None,
        output_strength: float = 16.0,
        route_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.base_mlp = base_mlp
        self.layout = layout or NeuronImplantLayout()
        if selected_indices.ndim != 1:
            raise ValueError("selected_indices must be one-dimensional")
        if len(selected_indices) != self.layout.total_width:
            raise ValueError(
                f"implant needs {self.layout.total_width} channels, "
                f"received {len(selected_indices)}"
            )
        if len(torch.unique(selected_indices)) != len(selected_indices):
            raise ValueError("selected channel indices must be unique")
        intermediate_size = base_mlp.up_proj.out_features
        if bool((selected_indices < 0).any()) or bool(
            (selected_indices >= intermediate_size).any()
        ):
            raise ValueError("selected channel index is out of range")
        self.register_buffer(
            "selected_indices",
            selected_indices.to(dtype=torch.long),
            persistent=True,
        )
        self.output_strength = float(output_strength)
        self.route_threshold = float(route_threshold)
        hidden_size = base_mlp.up_proj.in_features

        # These parameters are semantically replacement rows and columns of the
        # existing MLP matrices. Keeping them compact avoids allocating optimizer
        # state for all of Qwen during the local experiment.
        self.input_rows = nn.Parameter(
            torch.empty(self.layout.input_width, hidden_size)
        )
        self.result_columns = nn.Parameter(
            torch.zeros(hidden_size, self.layout.result_width)
        )
        nn.init.normal_(self.input_rows, std=hidden_size**-0.5)
        self.calculator = FrozenNeuronAddition(self.layout)
        self.runtime_context: ImplantRuntimeContext | None = None

        for parameter in self.base_mlp.parameters():
            parameter.requires_grad_(False)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @property
    def mlp_width(self) -> int:
        return self.base_mlp.up_proj.out_features

    def set_context(self, context: ImplantRuntimeContext | None) -> None:
        self.runtime_context = context

    def interface_logits(self, hidden: torch.Tensor) -> ImplantInterface:
        raw = nn.functional.linear(hidden.float(), self.input_rows.float())
        cursor = 0
        route_logits = raw[..., cursor : cursor + self.layout.route_width]
        cursor += self.layout.route_width
        operand_values = (
            raw[..., cursor : cursor + self.layout.operand_width]
            .reshape(
                *hidden.shape[:-1],
                2,
                self.layout.max_digits,
                self.layout.digit_classes,
            )
        )
        cursor += self.layout.operand_width
        step_logits = raw[..., cursor : cursor + self.layout.step_width]
        cursor += self.layout.step_width
        if cursor != self.layout.input_width:
            raise AssertionError("implant interface layout is inconsistent")
        return ImplantInterface(
            route_logits=route_logits,
            a_digit_logits=operand_values[..., 0, :, :],
            b_digit_logits=operand_values[..., 1, :, :],
            step_logits=step_logits,
        )

    def hard_interface(self, interface: ImplantInterface) -> HardImplantInterface:
        route_probability = interface.route_logits.softmax(dim=-1)[..., 1]
        return HardImplantInterface(
            route=(route_probability >= self.route_threshold).to(torch.long),
            route_probability=route_probability,
            a_digits=interface.a_digit_logits.argmax(dim=-1),
            b_digits=interface.b_digit_logits.argmax(dim=-1),
            step=interface.step_logits.argmax(dim=-1),
        )

    def _apply_teachers(
        self,
        hard: HardImplantInterface,
        context: ImplantRuntimeContext,
    ) -> HardImplantInterface:
        route = hard.route
        a_digits = hard.a_digits
        b_digits = hard.b_digits
        step = hard.step
        if context.teacher_route is not None:
            route = torch.where(
                context.teacher_route >= 0,
                context.teacher_route,
                route,
            )
        if context.teacher_a_digits is not None:
            a_digits = torch.where(
                context.teacher_a_digits >= 0,
                context.teacher_a_digits,
                a_digits,
            )
        if context.teacher_b_digits is not None:
            b_digits = torch.where(
                context.teacher_b_digits >= 0,
                context.teacher_b_digits,
                b_digits,
            )
        if context.teacher_step is not None:
            step = torch.where(context.teacher_step >= 0, context.teacher_step, step)
        return HardImplantInterface(
            route=route,
            route_probability=hard.route_probability,
            a_digits=a_digits,
            b_digits=b_digits,
            step=step,
        )

    def _base_without_selected_channels(self, hidden: torch.Tensor) -> torch.Tensor:
        intermediate = self.base_mlp.act_fn(
            self.base_mlp.gate_proj(hidden)
        ) * self.base_mlp.up_proj(hidden)
        intermediate = intermediate.clone()
        intermediate.index_fill_(-1, self.selected_indices, 0)
        return self.base_mlp.down_proj(intermediate)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        base_output = self._base_without_selected_channels(hidden)
        context = self.runtime_context
        if context is None or not context.enabled:
            return base_output
        if context.eligible_mask.shape != hidden.shape[:-1]:
            raise ValueError(
                "implant eligibility mask must match MLP batch and sequence"
            )

        interface = self.interface_logits(hidden)
        hard = self._apply_teachers(self.hard_interface(interface), context)
        execution = self.calculator(
            route=hard.route,
            a_digits=hard.a_digits,
            b_digits=hard.b_digits,
            step=hard.step,
            eligible_mask=context.eligible_mask,
        )
        result_activations = execution.result_activations
        if context.ablate_result:
            result_activations = torch.zeros_like(result_activations)
        implant_output = nn.functional.linear(
            result_activations.to(self.result_columns.dtype) * self.output_strength,
            self.result_columns,
        ).to(base_output.dtype)

        if context.capture_diagnostics:
            eligible = context.eligible_mask
            context.diagnostics = {
                "route_probability": hard.route_probability[eligible].detach().cpu(),
                "route": hard.route[eligible].detach().cpu(),
                "a_digits": hard.a_digits[eligible].detach().cpu(),
                "b_digits": hard.b_digits[eligible].detach().cpu(),
                "step": hard.step[eligible].detach().cpu(),
                "result_symbols": execution.result_symbols[eligible].detach().cpu(),
                "route_active": execution.route_active[eligible].detach().cpu(),
                "operand_pattern_valid": (
                    execution.operand_pattern_valid[eligible].detach().cpu()
                ),
                "step_valid": execution.step_valid[eligible].detach().cpu(),
            }
        return base_output + implant_output


def install_neuron_implant(
    model: nn.Module,
    *,
    layer_index: int,
    selected_indices: torch.Tensor,
    layout: NeuronImplantLayout | None = None,
    output_strength: float = 16.0,
    route_threshold: float = 0.5,
) -> NeuronImplantMLP:
    layers = model.model.layers
    if not 0 <= layer_index < len(layers):
        raise ValueError("layer_index is out of range")
    if isinstance(layers[layer_index].mlp, NeuronImplantMLP):
        raise ValueError("target layer already contains a neuron implant")
    base_mlp = layers[layer_index].mlp
    implant = NeuronImplantMLP(
        base_mlp,
        selected_indices,
        layout=layout,
        output_strength=output_strength,
        route_threshold=route_threshold,
    )
    reference = next(base_mlp.parameters())
    implant.to(device=reference.device, dtype=reference.dtype)
    layers[layer_index].mlp = implant
    return implant


def uninstall_neuron_implant(
    model: nn.Module,
    *,
    layer_index: int,
) -> nn.Module:
    implant = model.model.layers[layer_index].mlp
    if not isinstance(implant, NeuronImplantMLP):
        raise ValueError("target layer does not contain a neuron implant")
    model.model.layers[layer_index].mlp = implant.base_mlp
    return implant.base_mlp

