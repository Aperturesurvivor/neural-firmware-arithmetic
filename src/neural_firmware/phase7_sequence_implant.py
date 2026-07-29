from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from neural_firmware.internal_firmware import FrozenTypedAdditionCell


def normalized_hadamard(width: int) -> torch.Tensor:
    """Return a deterministic orthogonal Hadamard matrix."""
    if width < 1 or width & (width - 1):
        raise ValueError("Hadamard width must be a positive power of two")
    matrix = torch.ones(1, 1)
    while matrix.shape[0] < width:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix / width**0.5


@dataclass(frozen=True)
class SequenceImplantLayout:
    """Compact sequence-distributed typed ABI for the in-place implant."""

    max_digits: int = 4
    digit_classes: int = 11
    result_classes: int = 12
    learned_step: bool = True

    @property
    def route_width(self) -> int:
        return 2

    @property
    def role_width(self) -> int:
        # NONE, operand A, operand B.
        return 3

    @property
    def digit_width(self) -> int:
        # 0-9 and NON_DIGIT.
        return self.digit_classes

    @property
    def step_width(self) -> int:
        return self.max_digits + 2

    @property
    def input_width(self) -> int:
        return (
            self.route_width
            + self.role_width
            + self.digit_width
            + (self.step_width if self.learned_step else 0)
        )

    @property
    def result_width(self) -> int:
        return self.result_classes

    @property
    def total_width(self) -> int:
        return self.input_width + self.result_width

    @property
    def non_digit(self) -> int:
        return 10

    @property
    def eos_result(self) -> int:
        return 10

    @property
    def pad_result(self) -> int:
        return 11


@dataclass
class SequenceImplantContext:
    eligible_mask: torch.Tensor
    sequence_mask: torch.Tensor
    request_pool_mask: torch.Tensor | None = None
    teacher_route: torch.Tensor | None = None
    teacher_roles: torch.Tensor | None = None
    teacher_digits: torch.Tensor | None = None
    teacher_step: torch.Tensor | None = None
    enabled: bool = True
    ablate_result: bool = False
    preserve_base_when_off: bool = False
    capture_diagnostics: bool = False
    register_a_digits: torch.Tensor | None = None
    register_b_digits: torch.Tensor | None = None
    register_a_lengths: torch.Tensor | None = None
    register_b_lengths: torch.Tensor | None = None
    register_valid: torch.Tensor | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SequenceInterface:
    route_logits: torch.Tensor
    role_logits: torch.Tensor
    digit_logits: torch.Tensor
    step_logits: torch.Tensor


@dataclass(frozen=True)
class HardSequenceInterface:
    route: torch.Tensor
    route_probability: torch.Tensor
    roles: torch.Tensor
    digits: torch.Tensor
    digit_probability: torch.Tensor
    step: torch.Tensor


@dataclass(frozen=True)
class SequenceExecution:
    result_activations: torch.Tensor
    result_symbols: torch.Tensor
    route_active: torch.Tensor
    operands_valid: torch.Tensor
    a_digits: torch.Tensor
    b_digits: torch.Tensor
    a_lengths: torch.Tensor
    b_lengths: torch.Tensor


class FrozenSequenceAddition(nn.Module):
    """Scan typed per-token neuron states and execute exact addition."""

    def __init__(self, layout: SequenceImplantLayout) -> None:
        super().__init__()
        self.layout = layout
        self.cell = FrozenTypedAdditionCell()

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _extract_operands(
        self,
        roles: torch.Tensor,
        digits: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, _sequence = roles.shape
        a = torch.full(
            (batch, self.layout.max_digits),
            self.layout.non_digit,
            dtype=torch.long,
            device=roles.device,
        )
        b = torch.full_like(a, self.layout.non_digit)
        a_lengths = torch.zeros(batch, dtype=torch.long, device=roles.device)
        b_lengths = torch.zeros_like(a_lengths)
        valid = torch.ones(batch, dtype=torch.bool, device=roles.device)
        for row in range(batch):
            for role_value, target, lengths in (
                (1, a, a_lengths),
                (2, b, b_lengths),
            ):
                positions = torch.where(
                    (roles[row] == role_value)
                    & (digits[row] != self.layout.non_digit)
                    & sequence_mask[row]
                )[0]
                count = len(positions)
                if count < 1 or count > self.layout.max_digits:
                    valid[row] = False
                    continue
                values = digits[row, positions]
                if bool((values == self.layout.non_digit).any()):
                    valid[row] = False
                    continue
                target[row, :count] = values
                lengths[row] = count
        return a, a_lengths, b, b_lengths, valid

    def forward(
        self,
        *,
        route: torch.Tensor,
        roles: torch.Tensor,
        digits: torch.Tensor,
        step: torch.Tensor,
        eligible_mask: torch.Tensor,
        sequence_mask: torch.Tensor,
        register_a_digits: torch.Tensor | None = None,
        register_b_digits: torch.Tensor | None = None,
        register_a_lengths: torch.Tensor | None = None,
        register_b_lengths: torch.Tensor | None = None,
        register_valid: torch.Tensor | None = None,
    ) -> SequenceExecution:
        if not (
            route.shape
            == roles.shape
            == digits.shape
            == step.shape
            == eligible_mask.shape
            == sequence_mask.shape
        ):
            raise ValueError("all sequence interface tensors must share [B,S]")
        register_values = (
            register_a_digits,
            register_b_digits,
            register_a_lengths,
            register_b_lengths,
            register_valid,
        )
        if any(value is not None for value in register_values):
            if not all(value is not None for value in register_values):
                raise ValueError("operand register must provide all five tensors")
            a = register_a_digits
            b = register_b_digits
            a_lengths = register_a_lengths
            b_lengths = register_b_lengths
            operands_valid = register_valid
            if a.shape != (roles.shape[0], self.layout.max_digits):
                raise ValueError("registered operand A has the wrong shape")
            if b.shape != a.shape:
                raise ValueError("registered operand B has the wrong shape")
            if a_lengths.shape != (roles.shape[0],):
                raise ValueError("registered operand A lengths have the wrong shape")
            if b_lengths.shape != a_lengths.shape:
                raise ValueError("registered operand B lengths have the wrong shape")
            if operands_valid.shape != a_lengths.shape:
                raise ValueError("registered validity has the wrong shape")
        else:
            a, a_lengths, b, b_lengths, operands_valid = self._extract_operands(
                roles,
                digits,
                sequence_mask,
            )
        safe_a_lengths = a_lengths.clamp_min(1)
        safe_b_lengths = b_lengths.clamp_min(1)
        safe_a = torch.where(
            a == self.layout.non_digit,
            torch.zeros_like(a),
            a,
        )
        safe_b = torch.where(
            b == self.layout.non_digit,
            torch.zeros_like(b),
            b,
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
            raise ValueError("calculator result exceeds step width")

        safe_step = step.clamp(min=0, max=self.layout.step_width - 1)
        batch_rows = torch.arange(route.shape[0], device=route.device)[:, None]
        selected_symbols = symbols[batch_rows, safe_step]
        selected_mask = symbol_mask[batch_rows, safe_step]
        active = (
            eligible_mask
            & (route == 1)
            & operands_valid[:, None]
            & selected_mask
        )
        selected_symbols = torch.where(
            active,
            selected_symbols,
            torch.full_like(selected_symbols, self.layout.pad_result),
        )
        activations = nn.functional.one_hot(
            selected_symbols,
            num_classes=self.layout.result_classes,
        ).to(torch.float32)
        activations = activations * active.unsqueeze(-1)
        return SequenceExecution(
            result_activations=activations,
            result_symbols=selected_symbols,
            route_active=active,
            operands_valid=operands_valid,
            a_digits=a,
            b_digits=b,
            a_lengths=a_lengths,
            b_lengths=b_lengths,
        )


class SequenceNeuronImplantMLP(nn.Module):
    """In-place Qwen MLP implant reading distributed token activations."""

    def __init__(
        self,
        base_mlp: nn.Module,
        selected_indices: torch.Tensor,
        *,
        layout: SequenceImplantLayout | None = None,
        output_strength: float = 16.0,
        route_threshold: float = 0.5,
        route_temperature: float = 1.0,
        digit_threshold: float = 0.0,
        use_swiglu_interface: bool = False,
        interface_kind: str = "linear",
        representation_rank: int = 0,
        adapt_base_mlp: bool = True,
        request_router_kind: str = "interface",
        request_route_threshold: float | None = None,
        request_route_temperature: float = 2.0,
        request_tail_tokens: int = 8,
    ) -> None:
        super().__init__()
        self.base_mlp = base_mlp
        self.layout = layout or SequenceImplantLayout()
        if selected_indices.ndim != 1 or len(selected_indices) != self.layout.total_width:
            raise ValueError("selected channel bank does not match sequence ABI")
        if len(torch.unique(selected_indices)) != len(selected_indices):
            raise ValueError("selected channel indices must be unique")
        intermediate_size = base_mlp.up_proj.out_features
        if bool((selected_indices < 0).any()) or bool(
            (selected_indices >= intermediate_size).any()
        ):
            raise ValueError("selected channel index is out of range")
        self.register_buffer(
            "selected_indices",
            selected_indices.to(torch.long),
            persistent=True,
        )
        hidden_size = base_mlp.up_proj.in_features
        self.input_rows = nn.Parameter(
            torch.empty(self.layout.input_width, hidden_size)
        )
        self.gate_rows = nn.Parameter(
            torch.empty(self.layout.input_width, hidden_size)
        )
        if interface_kind not in {
            "linear",
            "swiglu",
            "fixed_mix_silu",
            "bottleneck_silu",
        }:
            raise ValueError(f"unknown sequence interface kind: {interface_kind}")
        if use_swiglu_interface:
            if interface_kind != "linear":
                raise ValueError(
                    "legacy use_swiglu_interface cannot be combined with "
                    "interface_kind"
                )
            interface_kind = "swiglu"
        self.interface_kind = interface_kind
        self.result_columns = nn.Parameter(
            torch.zeros(hidden_size, self.layout.result_width)
        )
        nn.init.normal_(self.input_rows, std=hidden_size**-0.5)
        nn.init.normal_(self.gate_rows, std=hidden_size**-0.5)
        self.use_swiglu_interface = self.interface_kind == "swiglu"
        self.gate_rows.requires_grad_(self.use_swiglu_interface)
        if self.interface_kind == "fixed_mix_silu":
            fixed_mix = normalized_hadamard(self.layout.input_width)
        else:
            fixed_mix = torch.empty(0, 0)
        self.register_buffer("fixed_interface_mix", fixed_mix, persistent=True)
        if self.interface_kind == "bottleneck_silu":
            projected_width = hidden_size - self.layout.input_width
            if projected_width < 1:
                raise ValueError(
                    "hidden width must exceed the bottleneck interface width"
                )
            self.bottleneck_rows = nn.Parameter(
                torch.empty(self.layout.input_width, projected_width)
            )
            self.bottleneck_mix = nn.Parameter(
                torch.empty(
                    self.layout.input_width,
                    self.layout.input_width,
                )
            )
            nn.init.normal_(
                self.bottleneck_rows,
                std=projected_width**-0.5,
            )
            nn.init.eye_(self.bottleneck_mix)
            self.input_rows.requires_grad_(False)
        else:
            self.register_parameter("bottleneck_rows", None)
            self.register_parameter("bottleneck_mix", None)
        if representation_rank < 0:
            raise ValueError("representation rank cannot be negative")
        self.representation_rank = int(representation_rank)
        self.adapt_base_mlp = bool(adapt_base_mlp)
        if self.representation_rank:
            self.representation_down = nn.Parameter(
                torch.empty(self.representation_rank, hidden_size)
            )
            self.representation_up = nn.Parameter(
                torch.zeros(hidden_size, self.representation_rank)
            )
            nn.init.normal_(
                self.representation_down,
                std=hidden_size**-0.5,
            )
        else:
            self.register_parameter("representation_down", None)
            self.register_parameter("representation_up", None)
        if request_router_kind not in {
            "interface",
            "last",
            "sequence_mean",
            "sequence_max",
            "user_mean",
            "user_max",
            "user_tail_mean",
        }:
            raise ValueError(
                f"unknown request router kind: {request_router_kind}"
            )
        self.request_router_kind = request_router_kind
        if self.request_router_kind == "interface":
            self.register_parameter("request_route_rows", None)
        else:
            self.request_route_rows = nn.Parameter(torch.empty(2, hidden_size))
            nn.init.normal_(self.request_route_rows, std=hidden_size**-0.5)
        self.request_route_threshold = float(
            route_threshold
            if request_route_threshold is None
            else request_route_threshold
        )
        self.request_route_temperature = float(request_route_temperature)
        if self.request_route_temperature <= 0:
            raise ValueError("request route temperature must be positive")
        self.request_tail_tokens = int(request_tail_tokens)
        if self.request_tail_tokens < 1:
            raise ValueError("request tail token count must be positive")
        self.output_strength = float(output_strength)
        self.route_threshold = float(route_threshold)
        self.route_temperature = float(route_temperature)
        if self.route_temperature <= 0:
            raise ValueError("route temperature must be positive")
        self.digit_threshold = float(digit_threshold)
        if not 0.0 <= self.digit_threshold <= 1.0:
            raise ValueError("digit threshold must be in [0, 1]")
        self.calculator = FrozenSequenceAddition(self.layout)
        self.runtime_context: SequenceImplantContext | None = None
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

    def set_context(self, context: SequenceImplantContext | None) -> None:
        self.runtime_context = context

    def adapted_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.representation_rank == 0:
            return hidden
        residual = nn.functional.linear(
            hidden.float(),
            self.representation_down.float(),
        )
        residual = nn.functional.silu(residual)
        residual = nn.functional.linear(
            residual,
            self.representation_up.float(),
        )
        return hidden + (residual / self.representation_rank).to(hidden.dtype)

    def _interface_logits_from_adapted(
        self,
        hidden: torch.Tensor,
    ) -> SequenceInterface:
        raw = nn.functional.linear(hidden.float(), self.input_rows.float())
        if self.interface_kind == "swiglu":
            gate = nn.functional.silu(
                nn.functional.linear(hidden.float(), self.gate_rows.float())
            )
            raw = gate * raw
        elif self.interface_kind == "fixed_mix_silu":
            raw = nn.functional.linear(
                nn.functional.silu(raw),
                self.fixed_interface_mix.float(),
            )
        elif self.interface_kind == "bottleneck_silu":
            bottleneck = nn.functional.linear(
                hidden.float()[..., : self.bottleneck_rows.shape[1]],
                self.bottleneck_rows.float(),
            )
            raw = nn.functional.linear(
                nn.functional.silu(bottleneck),
                self.bottleneck_mix.float(),
            )
        cursor = 0
        route = raw[..., cursor : cursor + self.layout.route_width]
        cursor += self.layout.route_width
        roles = raw[..., cursor : cursor + self.layout.role_width]
        cursor += self.layout.role_width
        digits = raw[..., cursor : cursor + self.layout.digit_width]
        cursor += self.layout.digit_width
        if self.layout.learned_step:
            step = raw[..., cursor : cursor + self.layout.step_width]
            cursor += self.layout.step_width
        else:
            step = torch.zeros(
                (*raw.shape[:-1], self.layout.step_width),
                dtype=raw.dtype,
                device=raw.device,
            )
        if cursor != self.layout.input_width:
            raise AssertionError("sequence implant layout is inconsistent")
        return SequenceInterface(
            route_logits=route,
            role_logits=roles,
            digit_logits=digits,
            step_logits=step,
        )

    def interface_logits(self, hidden: torch.Tensor) -> SequenceInterface:
        return self._interface_logits_from_adapted(self.adapted_hidden(hidden))

    def hard_interface(self, interface: SequenceInterface) -> HardSequenceInterface:
        probability = (
            interface.route_logits / self.route_temperature
        ).softmax(dim=-1)[..., 1]
        digit_probabilities = interface.digit_logits.softmax(dim=-1)
        digit_probability, digits = digit_probabilities.max(dim=-1)
        digits = torch.where(
            digit_probability >= self.digit_threshold,
            digits,
            torch.full_like(digits, self.layout.non_digit),
        )
        return HardSequenceInterface(
            route=(probability >= self.route_threshold).to(torch.long),
            route_probability=probability,
            roles=interface.role_logits.argmax(dim=-1),
            digits=digits,
            digit_probability=digit_probability,
            step=interface.step_logits.argmax(dim=-1),
        )

    def request_route_features(
        self,
        adapted_hidden: torch.Tensor,
        context: SequenceImplantContext,
    ) -> torch.Tensor:
        """Pool prompt representations for a separate request-level router."""
        if self.request_router_kind == "interface":
            raise ValueError("interface routing has no request-level features")
        sequence_mask = context.sequence_mask.to(torch.bool)
        if self.request_router_kind == "last":
            positions = sequence_mask.sum(dim=-1).clamp_min(1) - 1
            rows = torch.arange(
                adapted_hidden.shape[0],
                device=adapted_hidden.device,
            )
            return adapted_hidden[rows, positions]
        if self.request_router_kind.startswith("user_"):
            if context.request_pool_mask is None:
                raise ValueError("user-pooled router requires a request mask")
            pool_mask = context.request_pool_mask.to(torch.bool) & sequence_mask
        else:
            pool_mask = sequence_mask
        if not bool(pool_mask.any(dim=-1).all()):
            raise ValueError("every request router row needs a pooled token")
        if self.request_router_kind == "user_tail_mean":
            tail_mask = torch.zeros_like(pool_mask)
            for row in range(pool_mask.shape[0]):
                positions = torch.where(pool_mask[row])[0]
                tail_mask[row, positions[-self.request_tail_tokens :]] = True
            pool_mask = tail_mask
        masked = pool_mask.unsqueeze(-1)
        if self.request_router_kind in {
            "sequence_max",
            "user_max",
        }:
            return adapted_hidden.float().masked_fill(
                ~masked,
                float("-inf"),
            ).amax(dim=1)
        return (
            (adapted_hidden.float() * masked).sum(dim=1)
            / pool_mask.sum(dim=1, keepdim=True).clamp_min(1)
        )

    def apply_request_router(
        self,
        hard: HardSequenceInterface,
        adapted_hidden: torch.Tensor,
        context: SequenceImplantContext,
    ) -> HardSequenceInterface:
        if self.request_router_kind == "interface":
            return hard
        features = self.request_route_features(adapted_hidden, context)
        logits = nn.functional.linear(
            features.float(),
            self.request_route_rows.float(),
        )
        probability = (
            logits / self.request_route_temperature
        ).softmax(dim=-1)[..., 1]
        decision = (
            probability >= self.request_route_threshold
        ).to(torch.long)
        eligible = context.eligible_mask
        route = torch.where(eligible, decision[:, None], hard.route)
        route_probability = torch.where(
            eligible,
            probability[:, None],
            hard.route_probability,
        )
        return HardSequenceInterface(
            route=route,
            route_probability=route_probability,
            roles=hard.roles,
            digits=hard.digits,
            digit_probability=hard.digit_probability,
            step=hard.step,
        )

    def _apply_teachers(
        self,
        hard: HardSequenceInterface,
        context: SequenceImplantContext,
    ) -> HardSequenceInterface:
        route = hard.route
        roles = hard.roles
        digits = hard.digits
        step = hard.step
        if context.teacher_route is not None:
            route = torch.where(context.teacher_route >= 0, context.teacher_route, route)
        if context.teacher_roles is not None:
            roles = torch.where(context.teacher_roles >= 0, context.teacher_roles, roles)
        if context.teacher_digits is not None:
            digits = torch.where(
                context.teacher_digits >= 0,
                context.teacher_digits,
                digits,
            )
        if context.teacher_step is not None:
            step = torch.where(context.teacher_step >= 0, context.teacher_step, step)
        return HardSequenceInterface(
            route=route,
            route_probability=hard.route_probability,
            roles=roles,
            digits=digits,
            digit_probability=hard.digit_probability,
            step=step,
        )

    def _base_components(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        intermediate = self.base_mlp.act_fn(
            self.base_mlp.gate_proj(hidden)
        ) * self.base_mlp.up_proj(hidden)
        full_output = self.base_mlp.down_proj(intermediate)
        selected_values = intermediate.index_select(-1, self.selected_indices)
        selected_columns = self.base_mlp.down_proj.weight.index_select(
            1,
            self.selected_indices,
        )
        selected_contribution = nn.functional.linear(
            selected_values,
            selected_columns,
        )
        return full_output - selected_contribution, selected_contribution

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        context = self.runtime_context
        if context is None or not context.enabled:
            base_output, selected_base_contribution = self._base_components(hidden)
            return base_output + selected_base_contribution
        if context.eligible_mask.shape != hidden.shape[:-1]:
            raise ValueError("eligibility mask does not match MLP input")
        if context.sequence_mask.shape != hidden.shape[:-1]:
            raise ValueError("sequence mask does not match MLP input")
        if (
            context.request_pool_mask is not None
            and context.request_pool_mask.shape != hidden.shape[:-1]
        ):
            raise ValueError("request pool mask does not match MLP input")
        adapted_hidden = self.adapted_hidden(hidden)
        base_output, selected_base_contribution = self._base_components(
            adapted_hidden if self.adapt_base_mlp else hidden
        )
        interface = self._interface_logits_from_adapted(adapted_hidden)
        hard = self.hard_interface(interface)
        hard = self.apply_request_router(hard, adapted_hidden, context)
        hard = self._apply_teachers(hard, context)
        execution = self.calculator(
            route=hard.route,
            roles=hard.roles,
            digits=hard.digits,
            step=hard.step,
            eligible_mask=context.eligible_mask,
            sequence_mask=context.sequence_mask,
            register_a_digits=context.register_a_digits,
            register_b_digits=context.register_b_digits,
            register_a_lengths=context.register_a_lengths,
            register_b_lengths=context.register_b_lengths,
            register_valid=context.register_valid,
        )
        activations = execution.result_activations
        if context.ablate_result:
            activations = torch.zeros_like(activations)
        implant_output = nn.functional.linear(
            activations.to(self.result_columns.dtype) * self.output_strength,
            self.result_columns,
        ).to(base_output.dtype)
        if context.preserve_base_when_off:
            replacement_active = execution.route_active.unsqueeze(-1)
            implant_output = implant_output + torch.where(
                replacement_active,
                torch.zeros_like(selected_base_contribution),
                selected_base_contribution,
            )
        if context.capture_diagnostics:
            eligible = context.eligible_mask
            context.diagnostics = {
                "route_probability": hard.route_probability[eligible].detach().cpu(),
                "route": hard.route[eligible].detach().cpu(),
                "step": hard.step[eligible].detach().cpu(),
                "result_symbols": execution.result_symbols[eligible].detach().cpu(),
                "route_active": execution.route_active[eligible].detach().cpu(),
                "operands_valid": execution.operands_valid.detach().cpu(),
                "a_digits": execution.a_digits.detach().cpu(),
                "b_digits": execution.b_digits.detach().cpu(),
                "a_lengths": execution.a_lengths.detach().cpu(),
                "b_lengths": execution.b_lengths.detach().cpu(),
            }
        return base_output + implant_output


def install_sequence_neuron_implant(
    model: nn.Module,
    *,
    layer_index: int,
    selected_indices: torch.Tensor,
    layout: SequenceImplantLayout | None = None,
    output_strength: float = 16.0,
    route_threshold: float = 0.5,
    route_temperature: float = 1.0,
    digit_threshold: float = 0.0,
    use_swiglu_interface: bool = False,
    interface_kind: str = "linear",
    representation_rank: int = 0,
    adapt_base_mlp: bool = True,
    request_router_kind: str = "interface",
    request_route_threshold: float | None = None,
    request_route_temperature: float = 2.0,
    request_tail_tokens: int = 8,
) -> SequenceNeuronImplantMLP:
    layer = model.model.layers[layer_index]
    if isinstance(layer.mlp, SequenceNeuronImplantMLP):
        raise ValueError("target layer already contains a sequence implant")
    base_mlp = layer.mlp
    implant = SequenceNeuronImplantMLP(
        base_mlp,
        selected_indices,
        layout=layout,
        output_strength=output_strength,
        route_threshold=route_threshold,
        route_temperature=route_temperature,
        digit_threshold=digit_threshold,
        use_swiglu_interface=use_swiglu_interface,
        interface_kind=interface_kind,
        representation_rank=representation_rank,
        adapt_base_mlp=adapt_base_mlp,
        request_router_kind=request_router_kind,
        request_route_threshold=request_route_threshold,
        request_route_temperature=request_route_temperature,
        request_tail_tokens=request_tail_tokens,
    )
    reference = next(base_mlp.parameters())
    implant.to(device=reference.device, dtype=reference.dtype)
    layer.mlp = implant
    return implant
