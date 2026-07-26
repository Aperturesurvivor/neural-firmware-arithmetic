from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn

from neural_firmware.internal_firmware import (
    FrozenTypedAdditionCell,
    SymbolResidualDecoder,
)

PAD_DIGIT = 10
Phase6RouteMode = Literal["learned", "force_on", "force_off"]


class NeuralRegisterMapper(nn.Module):
    """Map early residual sequences to three explicit decimal registers."""

    def __init__(
        self,
        hidden_size: int,
        *,
        max_digits: int,
        model_width: int = 256,
        attention_heads: int = 8,
        decoder_layers: int = 2,
    ) -> None:
        super().__init__()
        if max_digits < 1:
            raise ValueError("max_digits must be positive")
        if model_width < 2 or model_width % 2:
            raise ValueError("model_width must be positive and even")
        if attention_heads < 1 or model_width % attention_heads:
            raise ValueError("attention_heads must divide model_width")
        if decoder_layers < 1:
            raise ValueError("decoder_layers must be positive")
        self.max_digits = max_digits
        self.model_width = model_width
        self.source_projection = nn.Linear(hidden_size, model_width)
        self.sequence_encoder = nn.GRU(
            model_width,
            model_width // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.source_norm = nn.LayerNorm(model_width)
        self.operand_embeddings = nn.Parameter(torch.empty(3, model_width))
        self.digit_embeddings = nn.Parameter(
            torch.empty(max_digits, model_width)
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_width,
            nhead=attention_heads,
            dim_feedforward=4 * model_width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.slot_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_layers,
            norm=nn.LayerNorm(model_width),
        )
        self.digit_classifier = nn.Linear(model_width, 11)
        self.control_query = nn.Parameter(torch.empty(1, model_width))
        self.control_attention = nn.MultiheadAttention(
            model_width,
            attention_heads,
            batch_first=True,
        )
        self.control_norm = nn.LayerNorm(model_width)
        self.control_classifier = nn.Linear(model_width, 5)
        self.position_scale = nn.Parameter(torch.ones(()))
        self.anchor_scale = nn.Parameter(torch.ones(()))
        self.log_temperature = nn.Parameter(torch.zeros(3, 3))
        nn.init.normal_(self.operand_embeddings, std=model_width**-0.5)
        nn.init.normal_(self.digit_embeddings, std=model_width**-0.5)
        nn.init.normal_(self.control_query, std=model_width**-0.5)

    @staticmethod
    def _relative_position_encoding(
        source: torch.Tensor,
        anchor_positions: torch.Tensor,
    ) -> torch.Tensor:
        sequence = source.shape[1]
        width = source.shape[2]
        positions = torch.arange(sequence, device=source.device)[None, :]
        relative = (anchor_positions[:, None] - positions).float()
        frequencies = torch.exp(
            torch.arange(0, width, 2, device=source.device).float()
            * (-math.log(10_000.0) / max(width, 2))
        )
        angles = relative[:, :, None] * frequencies[None, None, :]
        encoding = torch.zeros_like(source.float())
        encoding[:, :, 0::2] = torch.sin(angles)
        if width > 1:
            encoding[:, :, 1::2] = torch.cos(
                angles[:, :, : encoding[:, :, 1::2].shape[2]]
            )
        return encoding

    def _encode_source(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [batch, sequence, hidden]")
        if attention_mask.shape != hidden.shape[:2]:
            raise ValueError("attention mask must match the hidden sequence")
        if anchor_positions.shape != (hidden.shape[0],):
            raise ValueError("anchor positions must have shape [batch]")
        source = self.source_projection(hidden.float())
        lengths = (anchor_positions + 1).detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            source,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_context, _ = self.sequence_encoder(packed)
        context, _ = nn.utils.rnn.pad_packed_sequence(
            packed_context,
            batch_first=True,
            total_length=hidden.shape[1],
        )
        context = self.source_norm(context)
        context = context + self.position_scale * self._relative_position_encoding(
            context,
            anchor_positions,
        )
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        anchors = context[rows, anchor_positions]
        positions = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
        valid = attention_mask.bool() & (positions <= anchor_positions[:, None])
        return context, anchors, valid

    def _digit_logits(
        self,
        context: torch.Tensor,
        anchors: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        hidden_batch = context.shape[0]
        queries = (
            self.operand_embeddings[:, None, :]
            + self.digit_embeddings[None, :, :]
        ).reshape(3 * self.max_digits, self.model_width)
        queries = queries[None, :, :].expand(hidden_batch, -1, -1)
        queries = queries + self.anchor_scale * anchors[:, None, :]
        decoded = self.slot_decoder(
            queries,
            context,
            memory_key_padding_mask=~valid,
        )
        logits = self.digit_classifier(decoded).reshape(
            hidden_batch,
            3,
            self.max_digits,
            11,
        )
        temperature = self.log_temperature.exp().clamp(0.1, 10.0)
        temperature = temperature.repeat_interleave(
            math.ceil(self.max_digits / 3),
            dim=1,
        )[:, : self.max_digits]
        return logits / temperature[None, :, :, None]

    def _control_logits(
        self,
        context: torch.Tensor,
        anchors: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        query = self.control_query[None, :, :].expand(
            context.shape[0],
            -1,
            -1,
        )
        query = query + self.anchor_scale * anchors[:, None, :]
        attended, _ = self.control_attention(
            query,
            context,
            context,
            key_padding_mask=~valid,
            need_weights=False,
        )
        state = self.control_norm(query + attended)[:, 0]
        return self.control_classifier(state)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
    ) -> torch.Tensor:
        context, anchors, valid = self._encode_source(
            hidden,
            attention_mask,
            anchor_positions,
        )
        return self._digit_logits(context, anchors, valid)

    def forward_with_control(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context, anchors, valid = self._encode_source(
            hidden,
            attention_mask,
            anchor_positions,
        )
        return (
            self._digit_logits(context, anchors, valid),
            self._control_logits(context, anchors, valid),
        )

    def control(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Predict call semantics from the full early residual sequence."""

        context, anchors, valid = self._encode_source(
            hidden,
            attention_mask,
            anchor_positions,
        )
        return self._control_logits(context, anchors, valid)


class NeuralCallController(nn.Module):
    """Factor operation identity into no/add/unsupported call classes.

    Classes are no call, one ADD, two ADDs, one unsupported operation, and an
    unsupported multi-operation request. Only the ADD classes can activate
    firmware.
    """

    def __init__(self, hidden_size: int, hidden_width: int = 64) -> None:
        super().__init__()
        if hidden_width < 1:
            raise ValueError("hidden_width must be positive")
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, 5),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.classifier(hidden.float())


def categorical_registers(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Discretize left-aligned digit/PAD predictions."""

    if logits.ndim != 4 or logits.shape[1] != 3 or logits.shape[-1] != 11:
        raise ValueError("register logits must have shape [batch, 3, digits, 11]")
    predictions = logits.argmax(dim=-1)
    pad = predictions == PAD_DIGIT
    has_pad = pad.any(dim=2)
    first_pad = pad.float().argmax(dim=2).long()
    lengths = torch.where(
        has_pad,
        first_pad,
        torch.full_like(first_pad, predictions.shape[2]),
    )
    lengths = lengths.clamp_min(1)
    digits = predictions.clamp_max(9)
    first_was_pad = pad[:, :, 0]
    if bool(first_was_pad.any()):
        digits = digits.clone()
        digits[:, :, 0] = torch.where(
            first_was_pad,
            torch.zeros_like(digits[:, :, 0]),
            digits[:, :, 0],
        )
    return digits, lengths


def register_program_call_counts(register_logits: torch.Tensor) -> torch.Tensor:
    """Infer one versus two calls from learned third-register occupancy."""

    if (
        register_logits.ndim != 4
        or register_logits.shape[1] != 3
        or register_logits.shape[-1] != 11
    ):
        raise ValueError("register logits must have shape [batch, 3, digits, 11]")
    third_register_present = (
        register_logits[:, 2, 0].argmax(dim=-1) != PAD_DIGIT
    )
    return third_register_present.long() + 1


def symbols_to_register(
    symbols: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert calculator result symbols back to a typed operand register."""

    digit_mask = mask & (symbols >= 0) & (symbols <= 9)
    lengths = digit_mask.sum(dim=1).clamp_min(1)
    digits = symbols.clamp(0, 9)
    return digits, lengths


def _right_pad_symbols(
    symbols: torch.Tensor,
    mask: torch.Tensor,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if symbols.shape[1] > width:
        raise ValueError("cannot pad symbols to a shorter width")
    if symbols.shape[1] == width:
        return symbols, mask
    padding = width - symbols.shape[1]
    return (
        nn.functional.pad(symbols, (0, padding), value=-1),
        nn.functional.pad(mask, (0, padding), value=False),
    )


@dataclass
class ProgramExecution:
    register_digits: torch.Tensor
    register_lengths: torch.Tensor
    call_symbols: torch.Tensor
    call_masks: torch.Tensor
    final_symbols: torch.Tensor
    final_mask: torch.Tensor


class FrozenAdditionProgram(nn.Module):
    """Reuse one frozen ripple-carry cell for one- or two-call programs."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = FrozenTypedAdditionCell()

    def forward(
        self,
        register_logits: torch.Tensor,
        call_counts: torch.Tensor,
    ) -> ProgramExecution:
        if call_counts.shape != (register_logits.shape[0],):
            raise ValueError("call counts must have shape [batch]")
        digits, lengths = categorical_registers(register_logits)
        first_symbols, first_mask = self.cell(
            digits[:, 0],
            lengths[:, 0],
            digits[:, 1],
            lengths[:, 1],
        )
        intermediate_digits, intermediate_lengths = symbols_to_register(
            first_symbols,
            first_mask,
        )
        second_symbols, second_mask = self.cell(
            intermediate_digits,
            intermediate_lengths,
            digits[:, 2],
            lengths[:, 2],
        )
        width = max(first_symbols.shape[1], second_symbols.shape[1])
        first_symbols, first_mask = _right_pad_symbols(
            first_symbols,
            first_mask,
            width,
        )
        second_symbols, second_mask = _right_pad_symbols(
            second_symbols,
            second_mask,
            width,
        )
        call_symbols = torch.stack((first_symbols, second_symbols), dim=1)
        call_masks = torch.stack(
            (
                first_mask & (call_counts >= 1)[:, None],
                second_mask & (call_counts >= 2)[:, None],
            ),
            dim=1,
        )
        use_second = call_counts == 2
        final_symbols = torch.where(
            use_second[:, None],
            second_symbols,
            first_symbols,
        )
        final_mask = torch.where(
            use_second[:, None],
            second_mask,
            first_mask,
        )
        final_mask = final_mask & (call_counts > 0)[:, None]
        return ProgramExecution(
            register_digits=digits,
            register_lengths=lengths,
            call_symbols=call_symbols,
            call_masks=call_masks,
            final_symbols=final_symbols,
            final_mask=final_mask,
        )


@dataclass
class NeuralFirmwareContext:
    attention_mask: torch.Tensor
    anchor_positions: torch.Tensor
    output_positions: torch.Tensor | None = None
    teacher_symbols: torch.Tensor | None = None
    teacher_symbol_mask: torch.Tensor | None = None
    generation_index: int | None = None
    route_mode: Phase6RouteMode = "learned"
    route_threshold: float = 0.5
    register_logits: torch.Tensor | None = None
    early_control_logits: torch.Tensor | None = None
    call_logits: torch.Tensor | None = None
    call_counts: torch.Tensor | None = None
    route_probabilities: torch.Tensor | None = None
    planned_symbols: torch.Tensor | None = None
    planned_symbol_mask: torch.Tensor | None = None
    program_call_symbols: torch.Tensor | None = None
    program_call_masks: torch.Tensor | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)


class NeuralRegisterCaptureLayer(nn.Module):
    """Early wrapper that reads residuals without modifying the base state."""

    def __init__(
        self,
        base_layer: nn.Module,
        mapper: NeuralRegisterMapper,
        *,
        depth_after_blocks: int,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.mapper = mapper
        self.depth_after_blocks = depth_after_blocks
        self.runtime_context: NeuralFirmwareContext | None = None

    @property
    def attention_type(self) -> str:
        return self.base_layer.attention_type

    def set_context(self, context: NeuralFirmwareContext | None) -> None:
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
            and context.register_logits is None
        ):
            register_logits, control_logits = self.mapper.forward_with_control(
                hidden,
                context.attention_mask,
                context.anchor_positions,
            )
            context.register_logits = register_logits
            context.early_control_logits = control_logits
        return hidden


class NeuralProgramLayer(nn.Module):
    """Late call controller, repeated exact cell, and residual return path."""

    def __init__(
        self,
        base_layer: nn.Module,
        controller: NeuralCallController,
        output_decoder: SymbolResidualDecoder,
        *,
        depth_after_blocks: int,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.controller = controller
        self.program = FrozenAdditionProgram()
        self.output_decoder = output_decoder
        self.depth_after_blocks = depth_after_blocks
        self.runtime_context: NeuralFirmwareContext | None = None

    @property
    def attention_type(self) -> str:
        return self.base_layer.attention_type

    def set_context(self, context: NeuralFirmwareContext | None) -> None:
        self.runtime_context = context

    def _select_calls(
        self,
        hidden: torch.Tensor,
        context: NeuralFirmwareContext,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        late_logits = self.controller(hidden[:, -1, :])
        logits = late_logits
        if context.early_control_logits is not None:
            logits = logits + context.early_control_logits
        probabilities = torch.softmax(logits, dim=-1)
        route_probabilities = probabilities[:, 1:3].sum(dim=-1)
        if context.register_logits is None:
            raise RuntimeError("register logits are required for call selection")
        positive_counts = register_program_call_counts(
            context.register_logits
        )
        if context.route_mode == "force_off":
            counts = torch.zeros_like(positive_counts)
        elif context.route_mode == "force_on":
            counts = positive_counts
        elif context.route_mode == "learned":
            counts = torch.where(
                route_probabilities >= context.route_threshold,
                positive_counts,
                torch.zeros_like(positive_counts),
            )
        else:
            raise ValueError(f"unknown route mode: {context.route_mode}")
        context.diagnostics["late_call_logits"] = late_logits.detach().cpu()
        if context.early_control_logits is not None:
            context.diagnostics["early_call_logits"] = (
                context.early_control_logits.detach().cpu()
            )
        return counts, route_probabilities, logits

    def _plan(
        self,
        hidden: torch.Tensor,
        context: NeuralFirmwareContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context.teacher_symbols is not None:
            if context.teacher_symbol_mask is None:
                raise ValueError("teacher symbols require a teacher mask")
            symbols = context.teacher_symbols
            mask = context.teacher_symbol_mask
            call_counts = torch.ones(
                hidden.shape[0],
                dtype=torch.long,
                device=hidden.device,
            )
            route_probabilities = torch.ones(
                hidden.shape[0],
                device=hidden.device,
            )
        else:
            if context.register_logits is None:
                raise RuntimeError("neural register capture did not run")
            call_counts, route_probabilities, call_logits = self._select_calls(
                hidden,
                context,
            )
            execution = self.program(context.register_logits, call_counts)
            symbols = execution.final_symbols
            mask = execution.final_mask
            context.call_logits = call_logits.detach()
            context.program_call_symbols = execution.call_symbols.detach()
            context.program_call_masks = execution.call_masks.detach()
            context.diagnostics["register_predictions"] = (
                context.register_logits.argmax(dim=-1).detach().cpu()
            )
            context.diagnostics["register_lengths"] = (
                execution.register_lengths.detach().cpu()
            )
            context.diagnostics["program_call_symbols"] = (
                execution.call_symbols.detach().cpu()
            )
            context.diagnostics["program_call_masks"] = (
                execution.call_masks.detach().cpu()
            )
            context.diagnostics["call_logits"] = call_logits.detach().cpu()
        context.call_counts = call_counts.detach()
        context.route_probabilities = route_probabilities.detach()
        context.planned_symbols = symbols.detach()
        context.planned_symbol_mask = mask.detach()
        context.diagnostics["call_counts"] = call_counts.detach().cpu()
        context.diagnostics["route_probabilities"] = (
            route_probabilities.detach().cpu()
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
        residuals = self.output_decoder(symbols[mask]).to(hidden.dtype)
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
            raise ValueError("generation index must be nonnegative")
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
class NeuralFirmwareInstallation:
    capture: NeuralRegisterCaptureLayer
    final: NeuralProgramLayer

    def set_context(self, context: NeuralFirmwareContext | None) -> None:
        self.capture.set_context(context)
        self.final.set_context(context)

    @property
    def learned_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for module in (
                self.capture.mapper,
                self.final.controller,
                self.final.output_decoder,
            )
            for parameter in module.parameters()
        )

    def select_program_call(
        self,
        context: NeuralFirmwareContext,
        call_index: int,
    ) -> None:
        """Expose an intermediate call through the same autoregressive bridge."""

        if call_index not in (0, 1):
            raise ValueError("call index must be zero or one")
        if context.program_call_symbols is None or context.program_call_masks is None:
            raise RuntimeError("program has not been planned")
        context.planned_symbols = context.program_call_symbols[:, call_index]
        context.planned_symbol_mask = context.program_call_masks[:, call_index]
        context.generation_index = 0

    def state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "mapper": self.capture.mapper.state_dict(),
            "controller": self.final.controller.state_dict(),
            "output_decoder": self.final.output_decoder.state_dict(),
        }

    def load_state_dict(self, state: dict[str, dict[str, torch.Tensor]]) -> None:
        self.capture.mapper.load_state_dict(state["mapper"])
        self.final.controller.load_state_dict(state["controller"])
        self.final.output_decoder.load_state_dict(state["output_decoder"])


def install_neural_firmware(
    model: nn.Module,
    *,
    input_depth_after_blocks: int = 1,
    output_depth_after_blocks: int = 24,
    max_digits: int = 8,
    model_width: int = 256,
    attention_heads: int = 8,
    decoder_layers: int = 2,
    controller_width: int = 64,
    output_strength: float = 64.0,
) -> NeuralFirmwareInstallation:
    if input_depth_after_blocks >= output_depth_after_blocks:
        raise ValueError("input depth must precede output depth")
    layers = model.model.layers
    for depth in (input_depth_after_blocks, output_depth_after_blocks):
        if depth < 1 or depth > len(layers):
            raise ValueError("firmware depth is outside the transformer stack")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    input_index = input_depth_after_blocks - 1
    output_index = output_depth_after_blocks - 1
    input_base = layers[input_index]
    output_base = layers[output_index]
    mapper = NeuralRegisterMapper(
        model.config.hidden_size,
        max_digits=max_digits,
        model_width=model_width,
        attention_heads=attention_heads,
        decoder_layers=decoder_layers,
    )
    controller = NeuralCallController(
        model.config.hidden_size,
        hidden_width=controller_width,
    )
    output_decoder = SymbolResidualDecoder(
        model.config.hidden_size,
        output_strength,
    )
    reference = next(input_base.parameters())
    for module in (mapper, controller, output_decoder):
        module.to(device=reference.device, dtype=reference.dtype)
    capture = NeuralRegisterCaptureLayer(
        input_base,
        mapper,
        depth_after_blocks=input_depth_after_blocks,
    )
    final = NeuralProgramLayer(
        output_base,
        controller,
        output_decoder,
        depth_after_blocks=output_depth_after_blocks,
    )
    final.program.to(device=reference.device)
    layers[input_index] = capture
    layers[output_index] = final
    return NeuralFirmwareInstallation(capture=capture, final=final)
