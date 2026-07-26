from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn

from neural_firmware.internal_firmware import FrozenTypedAdditionCell
from neural_firmware.semantic_firmware import SemanticRouter

IGCRouteMode = Literal["learned", "force_on", "force_off"]
PAD_DIGIT = 10


class IGCInputMapping(nn.Module):
    """Anchor-query attention that maps a token sequence to typed operands.

    This follows the central IGC design: a learned mapping consumes all
    pre-anchor residuals, emits fixed-length categorical operands and an
    operation decision, and is trained with an auxiliary supervised loss.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        max_digits: int,
        attention_width: int,
        attention_heads: int,
    ) -> None:
        super().__init__()
        if max_digits < 1:
            raise ValueError("max_digits must be positive")
        if attention_width < 1:
            raise ValueError("attention_width must be positive")
        if attention_heads < 1 or attention_width % attention_heads:
            raise ValueError("attention_heads must divide attention_width")
        self.max_digits = max_digits
        self.attention_width = attention_width
        contextual_width = 2 * attention_width
        if contextual_width % attention_heads:
            raise ValueError("attention_heads must divide bidirectional width")
        self.sequence_encoder = nn.GRU(
            hidden_size,
            attention_width,
            batch_first=True,
            bidirectional=True,
        )
        self.slot_queries = nn.Parameter(
            torch.empty(2 * max_digits + 1, contextual_width)
        )
        self.attention = nn.MultiheadAttention(
            contextual_width,
            attention_heads,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(contextual_width)
        self.feed_forward = nn.Sequential(
            nn.Linear(contextual_width, 4 * contextual_width),
            nn.SiLU(),
            nn.Linear(4 * contextual_width, contextual_width),
        )
        self.output_norm = nn.LayerNorm(contextual_width)
        self.digit_classifier = nn.Linear(contextual_width, 11)
        self.operation_classifier = nn.Linear(contextual_width, 2)
        self.position_scale = nn.Parameter(torch.ones(()))
        self.log_digit_temperature = nn.Parameter(torch.zeros(()))
        self.log_operation_temperature = nn.Parameter(torch.zeros(()))
        self.direction_scale = nn.Parameter(torch.ones(2))
        self.operand_query_scale = nn.Parameter(torch.ones(2, attention_width))
        self.log_register_temperature = nn.Parameter(torch.zeros(2, 3))
        nn.init.normal_(self.slot_queries, std=contextual_width**-0.5)

    @staticmethod
    def _relative_position_encoding(
        hidden: torch.Tensor,
        anchor_positions: torch.Tensor,
    ) -> torch.Tensor:
        sequence = hidden.shape[1]
        width = hidden.shape[2]
        positions = torch.arange(sequence, device=hidden.device)[None, :]
        relative = (anchor_positions[:, None] - positions).float()
        frequencies = torch.exp(
            torch.arange(0, width, 2, device=hidden.device).float()
            * (-math.log(10_000.0) / max(width, 2))
        )
        angles = relative[:, :, None] * frequencies[None, None, :]
        encoding = torch.zeros_like(hidden.float())
        encoding[:, :, 0::2] = torch.sin(angles)
        if width > 1:
            encoding[:, :, 1::2] = torch.cos(angles[:, :, : encoding[:, :, 1::2].shape[2]])
        return encoding

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [batch, sequence, hidden]")
        if attention_mask.shape != hidden.shape[:2]:
            raise ValueError("attention_mask shape must match hidden sequence")
        if anchor_positions.shape != (hidden.shape[0],):
            raise ValueError("anchor_positions must have shape [batch]")

        rows = torch.arange(hidden.shape[0], device=hidden.device)
        lengths = (anchor_positions + 1).detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            hidden.float(),
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_contextual, _ = self.sequence_encoder(packed)
        contextual, _ = nn.utils.rnn.pad_packed_sequence(
            packed_contextual,
            batch_first=True,
            total_length=hidden.shape[1],
        )
        contextual = contextual.reshape(
            hidden.shape[0],
            hidden.shape[1],
            2,
            self.attention_width,
        )
        contextual = contextual * self.direction_scale[None, None, :, None]
        contextual = contextual.flatten(2)
        anchors = contextual[rows, anchor_positions]
        slot_queries = anchors[:, None, :] + self.slot_queries[None, :, :]
        operand_scales = self.operand_query_scale.repeat_interleave(
            self.max_digits,
            dim=0,
        )
        operation_scale = torch.ones(
            (1, self.attention_width),
            device=hidden.device,
            dtype=operand_scales.dtype,
        )
        query_scale = torch.cat([operand_scales, operation_scale], dim=0)
        query_scale = query_scale.repeat_interleave(2, dim=1)
        slot_queries = slot_queries * query_scale[None, :, :]

        positions = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
        valid = attention_mask.bool() & (positions <= anchor_positions[:, None])
        position_encoding = self._relative_position_encoding(
            contextual,
            anchor_positions,
        )
        key_values = contextual + self.position_scale * position_encoding
        attended, _ = self.attention(
            slot_queries,
            key_values,
            key_values,
            key_padding_mask=~valid,
            need_weights=False,
        )
        attended = self.attention_norm(slot_queries + attended)
        attended = self.output_norm(attended + self.feed_forward(attended))

        digit_states = attended[:, : 2 * self.max_digits].reshape(
            hidden.shape[0],
            2,
            self.max_digits,
            2 * self.attention_width,
        )
        digit_temperature = self.log_digit_temperature.exp().clamp(0.1, 10.0)
        operation_temperature = self.log_operation_temperature.exp().clamp(0.1, 10.0)
        digit_logits = self.digit_classifier(digit_states) / digit_temperature
        register_temperature = self.log_register_temperature.exp().clamp(0.1, 10.0)
        register_temperature = register_temperature.repeat_interleave(
            math.ceil(self.max_digits / 3),
            dim=1,
        )[:, : self.max_digits]
        digit_logits = digit_logits / register_temperature[None, :, :, None]
        operation_logits = (
            self.operation_classifier(attended[:, -1]) / operation_temperature
        )
        return digit_logits[:, 0], digit_logits[:, 1], operation_logits


def categorical_digits_to_register(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Discretize left-aligned digit/PAD predictions for the exact calculator."""

    predictions = logits.argmax(dim=-1)
    pad = predictions == PAD_DIGIT
    has_pad = pad.any(dim=1)
    first_pad = pad.float().argmax(dim=1).long()
    lengths = torch.where(
        has_pad,
        first_pad,
        torch.full_like(first_pad, predictions.shape[1]),
    )
    lengths = lengths.clamp_min(1)
    digits = predictions.clamp_max(9)
    first_was_pad = pad[:, 0]
    if bool(first_was_pad.any()):
        digits = digits.clone()
        digits[first_was_pad, 0] = 0
    return digits, lengths


class IGCOutputMapping(nn.Module):
    """Learned gated categorical-result-to-residual mapping."""

    def __init__(
        self,
        hidden_size: int,
        *,
        output_width: int,
        initial_strength: float,
        learn_strength: bool = True,
    ) -> None:
        super().__init__()
        if output_width < 1 or output_width > hidden_size:
            raise ValueError("output_width must be in [1, hidden_size]")
        self.hidden_size = hidden_size
        self.output_width = output_width
        self.codebook = nn.Embedding(11, output_width)
        self.gate = nn.Linear(hidden_size, 1)
        log_strength = torch.tensor(math.log(initial_strength))
        if learn_strength:
            self.log_strength = nn.Parameter(log_strength)
        else:
            self.register_buffer("log_strength", log_strength, persistent=True)
        nn.init.normal_(self.codebook.weight, std=output_width**-0.5)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, hidden: torch.Tensor, symbols: torch.Tensor) -> torch.Tensor:
        vectors = self.codebook(symbols)
        vectors = nn.functional.normalize(vectors, dim=-1)
        if self.output_width < self.hidden_size:
            vectors = nn.functional.pad(
                vectors,
                (0, self.hidden_size - self.output_width),
            )
        strength = self.log_strength.exp().clamp(max=256.0)
        gates = torch.sigmoid(self.gate(hidden.float()))
        return vectors.to(hidden.dtype) * gates.to(hidden.dtype) * strength.to(hidden.dtype)


class IGCArithmeticUnit(nn.Module):
    """Learned I/O mappings around a frozen addition calculator."""

    def __init__(
        self,
        hidden_size: int,
        *,
        max_digits: int,
        attention_width: int,
        attention_heads: int,
        output_width: int,
        initial_strength: float,
        learn_output_strength: bool = True,
    ) -> None:
        super().__init__()
        self.input_mapping = IGCInputMapping(
            hidden_size,
            max_digits=max_digits,
            attention_width=attention_width,
            attention_heads=attention_heads,
        )
        self.calculator = FrozenTypedAdditionCell()
        self.output_mapping = IGCOutputMapping(
            hidden_size,
            output_width=output_width,
            initial_strength=initial_strength,
            learn_strength=learn_output_strength,
        )

    @property
    def learned_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@dataclass
class IGCContext:
    attention_mask: torch.Tensor
    anchor_positions: torch.Tensor
    output_positions: torch.Tensor | None = None
    teacher_symbols: torch.Tensor | None = None
    teacher_symbol_mask: torch.Tensor | None = None
    generation_index: int | None = None
    route_mode: IGCRouteMode = "learned"
    route_threshold: float = 0.5
    route_active: torch.Tensor | None = None
    route_probabilities: torch.Tensor | None = None
    planned_symbols: torch.Tensor | None = None
    planned_symbol_mask: torch.Tensor | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)


class IGCFirmwareLayer(nn.Module):
    """A decoder block followed by an IGC-style learned calculator interface."""

    def __init__(
        self,
        base_layer: nn.Module,
        unit: IGCArithmeticUnit,
        *,
        depth_after_blocks: int,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.unit = unit
        self.depth_after_blocks = depth_after_blocks
        self.runtime_context: IGCContext | None = None

    @property
    def attention_type(self) -> str:
        return self.base_layer.attention_type

    def set_context(self, context: IGCContext | None) -> None:
        self.runtime_context = context

    def _plan(
        self,
        hidden: torch.Tensor,
        context: IGCContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context.teacher_symbols is not None:
            if context.teacher_symbol_mask is None:
                raise ValueError("teacher symbols require a teacher mask")
            batch = hidden.shape[0]
            context.route_active = torch.ones(
                batch,
                dtype=torch.bool,
                device=hidden.device,
            )
            context.route_probabilities = torch.ones(batch, device=hidden.device)
            symbols = context.teacher_symbols
            mask = context.teacher_symbol_mask
        else:
            a_logits, b_logits, operation_logits = self.unit.input_mapping(
                hidden,
                context.attention_mask,
                context.anchor_positions,
            )
            probabilities = torch.softmax(operation_logits, dim=-1)[:, 1]
            if context.route_mode == "force_on":
                active = torch.ones_like(probabilities, dtype=torch.bool)
            elif context.route_mode == "force_off":
                active = torch.zeros_like(probabilities, dtype=torch.bool)
            elif context.route_mode == "learned":
                active = probabilities >= context.route_threshold
            else:
                raise ValueError(f"unknown route mode: {context.route_mode}")
            a_digits, a_lengths = categorical_digits_to_register(a_logits)
            b_digits, b_lengths = categorical_digits_to_register(b_logits)
            symbols, mask = self.unit.calculator(
                a_digits,
                a_lengths,
                b_digits,
                b_lengths,
            )
            mask = mask & active[:, None]
            context.route_active = active.detach()
            context.route_probabilities = probabilities.detach()
            context.diagnostics["a_predictions"] = a_logits.argmax(dim=-1).detach().cpu()
            context.diagnostics["b_predictions"] = b_logits.argmax(dim=-1).detach().cpu()
            context.diagnostics["a_lengths"] = a_lengths.detach().cpu()
            context.diagnostics["b_lengths"] = b_lengths.detach().cpu()
            context.diagnostics["operation_logits"] = operation_logits.detach().cpu()

        context.planned_symbols = symbols.detach()
        context.planned_symbol_mask = mask.detach()
        context.diagnostics["route_active"] = context.route_active.detach().cpu()
        context.diagnostics["route_probabilities"] = (
            context.route_probabilities.detach().cpu()
        )
        context.diagnostics["planned_symbols"] = symbols.detach().cpu()
        context.diagnostics["planned_symbol_mask"] = mask.detach().cpu()
        return symbols, mask

    def _inject(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        symbols: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_grid = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
        batch_indices = batch_grid.expand_as(positions)[mask]
        token_positions = positions[mask]
        if batch_indices.numel() == 0:
            return hidden
        selected_hidden = hidden[batch_indices, token_positions]
        residuals = self.unit.output_mapping(selected_hidden, symbols[mask])
        additions = torch.zeros_like(hidden)
        additions.index_put_(
            (batch_indices, token_positions),
            residuals,
            accumulate=True,
        )
        return hidden + additions

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        hidden = self.base_layer(hidden_states, *args, **kwargs)
        context = self.runtime_context
        if context is None:
            return hidden
        if context.planned_symbols is None or context.planned_symbol_mask is None:
            symbols, mask = self._plan(hidden, context)
        else:
            symbols = context.planned_symbols
            mask = context.planned_symbol_mask

        if context.generation_index is None:
            if context.output_positions is None:
                raise ValueError("teacher-forced context requires output positions")
            return self._inject(
                hidden,
                context.output_positions,
                symbols,
                mask,
            )

        index = context.generation_index
        if index < 0:
            raise ValueError("generation_index must be nonnegative")
        if index >= symbols.shape[1]:
            return hidden
        active = mask[:, index]
        if not bool(active.any()):
            return hidden
        positions = torch.full(
            (hidden.shape[0], 1),
            hidden.shape[1] - 1,
            dtype=torch.long,
            device=hidden.device,
        )
        return self._inject(
            hidden,
            positions,
            symbols[:, index : index + 1],
            active[:, None],
        )


def install_igc_firmware(
    model: nn.Module,
    *,
    depth_after_blocks: int,
    max_digits: int,
    attention_width: int,
    attention_heads: int,
    output_width: int,
    initial_strength: float = 64.0,
    learn_output_strength: bool = True,
) -> IGCFirmwareLayer:
    layers = model.model.layers
    if depth_after_blocks < 1 or depth_after_blocks > len(layers):
        raise ValueError("depth_after_blocks is outside the transformer stack")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layer_index = depth_after_blocks - 1
    base_layer = layers[layer_index]
    unit = IGCArithmeticUnit(
        model.config.hidden_size,
        max_digits=max_digits,
        attention_width=attention_width,
        attention_heads=attention_heads,
        output_width=output_width,
        initial_strength=initial_strength,
        learn_output_strength=learn_output_strength,
    )
    reference = next(base_layer.parameters())
    unit.to(device=reference.device, dtype=reference.dtype)
    wrapper = IGCFirmwareLayer(
        base_layer,
        unit,
        depth_after_blocks=depth_after_blocks,
    )
    layers[layer_index] = wrapper
    return wrapper


@dataclass
class IGCDualContext:
    attention_mask: torch.Tensor
    anchor_positions: torch.Tensor
    output_positions: torch.Tensor | None = None
    teacher_symbols: torch.Tensor | None = None
    teacher_symbol_mask: torch.Tensor | None = None
    generation_index: int | None = None
    route_mode: IGCRouteMode = "learned"
    route_threshold: float = 0.5
    a_logits: torch.Tensor | None = None
    b_logits: torch.Tensor | None = None
    early_operation_logits: torch.Tensor | None = None
    route_active: torch.Tensor | None = None
    route_probabilities: torch.Tensor | None = None
    planned_symbols: torch.Tensor | None = None
    planned_symbol_mask: torch.Tensor | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)


class IGCDigitCaptureLayer(nn.Module):
    """Early block wrapper that learns operand registers without modifying state."""

    def __init__(
        self,
        base_layer: nn.Module,
        input_mapping: IGCInputMapping,
        *,
        depth_after_blocks: int,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.input_mapping = input_mapping
        self.depth_after_blocks = depth_after_blocks
        self.runtime_context: IGCDualContext | None = None

    @property
    def attention_type(self) -> str:
        return self.base_layer.attention_type

    def set_context(self, context: IGCDualContext | None) -> None:
        self.runtime_context = context

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        hidden = self.base_layer(hidden_states, *args, **kwargs)
        context = self.runtime_context
        if (
            context is not None
            and context.teacher_symbols is None
            and context.a_logits is None
        ):
            a_logits, b_logits, operation_logits = self.input_mapping(
                hidden,
                context.attention_mask,
                context.anchor_positions,
            )
            context.a_logits = a_logits
            context.b_logits = b_logits
            context.early_operation_logits = operation_logits
        return hidden


class IGCFinalCalculatorLayer(nn.Module):
    """Late semantic router, frozen calculator, and gated output mapping."""

    def __init__(
        self,
        base_layer: nn.Module,
        router: SemanticRouter,
        output_mapping: IGCOutputMapping,
        *,
        depth_after_blocks: int,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.router = router
        self.calculator = FrozenTypedAdditionCell()
        self.output_mapping = output_mapping
        self.depth_after_blocks = depth_after_blocks
        self.runtime_context: IGCDualContext | None = None

    @property
    def attention_type(self) -> str:
        return self.base_layer.attention_type

    def set_context(self, context: IGCDualContext | None) -> None:
        self.runtime_context = context

    def _plan(
        self,
        hidden: torch.Tensor,
        context: IGCDualContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context.teacher_symbols is not None:
            if context.teacher_symbol_mask is None:
                raise ValueError("teacher symbols require a teacher mask")
            active = torch.ones(
                hidden.shape[0],
                dtype=torch.bool,
                device=hidden.device,
            )
            probabilities = torch.ones(hidden.shape[0], device=hidden.device)
            symbols = context.teacher_symbols
            mask = context.teacher_symbol_mask
        else:
            if context.a_logits is None or context.b_logits is None:
                raise RuntimeError("early IGC operand capture did not run")
            route_logits = self.router(hidden[:, -1, :])
            probabilities = torch.sigmoid(route_logits)
            if context.route_mode == "force_on":
                active = torch.ones_like(probabilities, dtype=torch.bool)
            elif context.route_mode == "force_off":
                active = torch.zeros_like(probabilities, dtype=torch.bool)
            elif context.route_mode == "learned":
                active = probabilities >= context.route_threshold
            else:
                raise ValueError(f"unknown route mode: {context.route_mode}")
            a_digits, a_lengths = categorical_digits_to_register(context.a_logits)
            b_digits, b_lengths = categorical_digits_to_register(context.b_logits)
            symbols, mask = self.calculator(
                a_digits,
                a_lengths,
                b_digits,
                b_lengths,
            )
            mask = mask & active[:, None]
            context.diagnostics["a_predictions"] = (
                context.a_logits.argmax(dim=-1).detach().cpu()
            )
            context.diagnostics["b_predictions"] = (
                context.b_logits.argmax(dim=-1).detach().cpu()
            )
            context.diagnostics["a_lengths"] = a_lengths.detach().cpu()
            context.diagnostics["b_lengths"] = b_lengths.detach().cpu()

        context.route_active = active.detach()
        context.route_probabilities = probabilities.detach()
        context.planned_symbols = symbols.detach()
        context.planned_symbol_mask = mask.detach()
        context.diagnostics["route_active"] = active.detach().cpu()
        context.diagnostics["route_probabilities"] = probabilities.detach().cpu()
        context.diagnostics["planned_symbols"] = symbols.detach().cpu()
        context.diagnostics["planned_symbol_mask"] = mask.detach().cpu()
        return symbols, mask

    def _inject(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        symbols: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_grid = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
        batch_indices = batch_grid.expand_as(positions)[mask]
        token_positions = positions[mask]
        if batch_indices.numel() == 0:
            return hidden
        selected = hidden[batch_indices, token_positions]
        residuals = self.output_mapping(selected, symbols[mask])
        additions = torch.zeros_like(hidden)
        additions.index_put_(
            (batch_indices, token_positions),
            residuals,
            accumulate=True,
        )
        return hidden + additions

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        hidden = self.base_layer(hidden_states, *args, **kwargs)
        context = self.runtime_context
        if context is None:
            return hidden
        if context.planned_symbols is None or context.planned_symbol_mask is None:
            symbols, mask = self._plan(hidden, context)
        else:
            symbols = context.planned_symbols
            mask = context.planned_symbol_mask
        if context.generation_index is None:
            if context.output_positions is None:
                raise ValueError("teacher-forced context requires output positions")
            return self._inject(
                hidden,
                context.output_positions,
                symbols,
                mask,
            )
        index = context.generation_index
        if index >= symbols.shape[1]:
            return hidden
        active = mask[:, index]
        positions = torch.full(
            (hidden.shape[0], 1),
            hidden.shape[1] - 1,
            dtype=torch.long,
            device=hidden.device,
        )
        return self._inject(
            hidden,
            positions,
            symbols[:, index : index + 1],
            active[:, None],
        )


@dataclass
class IGCDualInstallation:
    capture: IGCDigitCaptureLayer
    final: IGCFinalCalculatorLayer

    def set_context(self, context: IGCDualContext | None) -> None:
        self.capture.set_context(context)
        self.final.set_context(context)

    @property
    def learned_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for module in (
                self.capture.input_mapping,
                self.final.router,
                self.final.output_mapping,
            )
            for parameter in module.parameters()
        )

    def state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "input_mapping": self.capture.input_mapping.state_dict(),
            "router": self.final.router.state_dict(),
            "output_mapping": self.final.output_mapping.state_dict(),
        }

    def load_state_dict(self, state: dict[str, dict[str, torch.Tensor]]) -> None:
        self.capture.input_mapping.load_state_dict(state["input_mapping"])
        self.final.router.load_state_dict(state["router"])
        self.final.output_mapping.load_state_dict(state["output_mapping"])


def install_dual_depth_igc(
    model: nn.Module,
    *,
    input_depth_after_blocks: int,
    output_depth_after_blocks: int,
    max_digits: int,
    attention_width: int,
    attention_heads: int,
    output_width: int,
    router_hidden_width: int = 16,
    initial_strength: float = 64.0,
    learn_output_strength: bool = True,
) -> IGCDualInstallation:
    if input_depth_after_blocks >= output_depth_after_blocks:
        raise ValueError("IGC input depth must precede output depth")
    layers = model.model.layers
    for depth in (input_depth_after_blocks, output_depth_after_blocks):
        if depth < 1 or depth > len(layers):
            raise ValueError("IGC depth is outside the transformer stack")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    input_index = input_depth_after_blocks - 1
    output_index = output_depth_after_blocks - 1
    input_base = layers[input_index]
    output_base = layers[output_index]
    input_mapping = IGCInputMapping(
        model.config.hidden_size,
        max_digits=max_digits,
        attention_width=attention_width,
        attention_heads=attention_heads,
    )
    router = SemanticRouter(model.config.hidden_size, router_hidden_width)
    output_mapping = IGCOutputMapping(
        model.config.hidden_size,
        output_width=output_width,
        initial_strength=initial_strength,
        learn_strength=learn_output_strength,
    )
    reference = next(input_base.parameters())
    for module in (input_mapping, router, output_mapping):
        module.to(device=reference.device, dtype=reference.dtype)
    capture = IGCDigitCaptureLayer(
        input_base,
        input_mapping,
        depth_after_blocks=input_depth_after_blocks,
    )
    final = IGCFinalCalculatorLayer(
        output_base,
        router,
        output_mapping,
        depth_after_blocks=output_depth_after_blocks,
    )
    final.calculator.to(device=reference.device)
    layers[input_index] = capture
    layers[output_index] = final
    return IGCDualInstallation(capture=capture, final=final)
