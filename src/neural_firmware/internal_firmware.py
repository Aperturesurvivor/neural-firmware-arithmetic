from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from neural_firmware.pretrained_firmware import CARRY_TABLE, SUM_TABLE


class FrozenTypedAdditionCell(nn.Module):
    """Zero-parameter decimal ripple-carry cell over typed digit tensors."""

    def __init__(self) -> None:
        super().__init__()
        sum_table = torch.tensor(SUM_TABLE, dtype=torch.long).reshape(10, 10, 2)
        carry_table = torch.tensor(CARRY_TABLE, dtype=torch.long).reshape(10, 10, 2)
        self.register_buffer("sum_table", sum_table, persistent=True)
        self.register_buffer("carry_table", carry_table, persistent=True)

    @property
    def trainable_parameter_count(self) -> int:
        return 0

    def forward(
        self,
        a_digits: torch.Tensor,
        a_lengths: torch.Tensor,
        b_digits: torch.Tensor,
        b_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return left-to-right result symbols followed by symbol 10 (EOS).

        Digit tensors are left aligned and padded arbitrarily after their
        declared lengths. The returned tensor is padded with -1 and accompanied
        by a boolean validity mask.
        """

        if a_digits.ndim != 2 or b_digits.ndim != 2:
            raise ValueError("digit tensors must have shape [batch, positions]")
        if a_digits.shape[0] != b_digits.shape[0]:
            raise ValueError("operand batches must have the same size")
        if a_lengths.shape != b_lengths.shape or a_lengths.ndim != 1:
            raise ValueError("length tensors must have shape [batch]")
        if a_lengths.shape[0] != a_digits.shape[0]:
            raise ValueError("length batch does not match digit batch")
        if bool((a_lengths < 1).any()) or bool((b_lengths < 1).any()):
            raise ValueError("every operand must contain at least one digit")
        if bool((a_lengths > a_digits.shape[1]).any()):
            raise ValueError("a_lengths exceed available digit positions")
        if bool((b_lengths > b_digits.shape[1]).any()):
            raise ValueError("b_lengths exceed available digit positions")

        device = a_digits.device
        batch = a_digits.shape[0]
        widths = torch.maximum(a_lengths, b_lengths)
        maximum_width = int(widths.max().item())
        carry = torch.zeros(batch, dtype=torch.long, device=device)
        reversed_digits = torch.full(
            (batch, maximum_width + 1),
            -1,
            dtype=torch.long,
            device=device,
        )
        batch_indices = torch.arange(batch, device=device)

        for offset in range(maximum_width):
            active = offset < widths
            a_valid = offset < a_lengths
            b_valid = offset < b_lengths
            a_positions = (a_lengths - 1 - offset).clamp_min(0)
            b_positions = (b_lengths - 1 - offset).clamp_min(0)
            a_values = a_digits[batch_indices, a_positions]
            b_values = b_digits[batch_indices, b_positions]
            a_values = torch.where(a_valid, a_values, torch.zeros_like(a_values))
            b_values = torch.where(b_valid, b_values, torch.zeros_like(b_values))
            if bool(((a_values < 0) | (a_values > 9)).any()):
                raise ValueError("a_digits contain a non-decimal value")
            if bool(((b_values < 0) | (b_values > 9)).any()):
                raise ValueError("b_digits contain a non-decimal value")
            summed = self.sum_table[a_values, b_values, carry]
            next_carry = self.carry_table[a_values, b_values, carry]
            reversed_digits[:, offset] = torch.where(
                active,
                summed,
                reversed_digits[:, offset],
            )
            carry = torch.where(active, next_carry, carry)

        output = torch.full(
            (batch, maximum_width + 2),
            -1,
            dtype=torch.long,
            device=device,
        )
        mask = torch.zeros_like(output, dtype=torch.bool)
        for row in range(batch):
            width = int(widths[row].item())
            has_carry = int(carry[row].item())
            if has_carry:
                reversed_digits[row, width] = carry[row]
            result_length = width + has_carry
            result = reversed_digits[row, :result_length].flip(0)
            output[row, :result_length] = result
            output[row, result_length] = 10
            mask[row, : result_length + 1] = True
        return output, mask


class ResidualDigitEncoder(nn.Module):
    """Learned translator from an intermediate residual to a typed digit."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 10)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.classifier(hidden.float())


class SymbolResidualDecoder(nn.Module):
    """Learned translator from eleven typed firmware symbols to residuals."""

    def __init__(self, hidden_size: int, strength: float) -> None:
        super().__init__()
        self.strength = strength
        self.codebook = nn.Embedding(11, hidden_size)
        nn.init.normal_(self.codebook.weight, std=hidden_size**-0.5)

    def forward(self, symbols: torch.Tensor) -> torch.Tensor:
        vectors = self.codebook(symbols)
        return nn.functional.normalize(vectors, dim=-1) * self.strength


class InternalArithmeticUnit(nn.Module):
    """Learned interfaces around a frozen typed addition cell."""

    def __init__(self, hidden_size: int, strength: float = 32.0) -> None:
        super().__init__()
        self.digit_encoder = ResidualDigitEncoder(hidden_size)
        self.cell = FrozenTypedAdditionCell()
        self.symbol_decoder = SymbolResidualDecoder(hidden_size, strength)

    @property
    def interface_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def gather_digit_logits(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.ndim != 3 or positions.ndim != 2:
            raise ValueError("expected hidden [B,S,H] and positions [B,D]")
        if hidden.shape[0] != positions.shape[0]:
            raise ValueError("hidden and position batches differ")
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
        gathered = hidden[batch_indices, positions.clamp_min(0)]
        return self.digit_encoder(gathered)

    def plan_from_hidden(
        self,
        hidden: torch.Tensor,
        a_positions: torch.Tensor,
        a_lengths: torch.Tensor,
        b_positions: torch.Tensor,
        b_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        a_logits = self.gather_digit_logits(hidden, a_positions)
        b_logits = self.gather_digit_logits(hidden, b_positions)
        a_digits = a_logits.argmax(dim=-1)
        b_digits = b_logits.argmax(dim=-1)
        symbols, symbol_mask = self.cell(
            a_digits,
            a_lengths,
            b_digits,
            b_lengths,
        )
        return symbols, symbol_mask, a_logits, b_logits

    def inject_symbols(
        self,
        hidden: torch.Tensor,
        output_positions: torch.Tensor,
        symbols: torch.Tensor,
        symbol_mask: torch.Tensor,
    ) -> torch.Tensor:
        if output_positions.shape != symbols.shape:
            raise ValueError("output positions and symbols must have equal shape")
        if symbol_mask.shape != symbols.shape:
            raise ValueError("symbol mask and symbols must have equal shape")
        valid_symbols = symbols[symbol_mask]
        residuals = self.symbol_decoder(valid_symbols).to(hidden.dtype)
        batch_grid = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
        batch_indices = batch_grid.expand_as(symbols)[symbol_mask]
        token_positions = output_positions[symbol_mask]
        additions = torch.zeros_like(hidden)
        additions.index_put_(
            (batch_indices, token_positions),
            residuals,
            accumulate=True,
        )
        return hidden + additions


@dataclass
class InternalFirmwareContext:
    """Per-forward typed-register locations and autoregressive state."""

    a_positions: torch.Tensor
    a_lengths: torch.Tensor
    b_positions: torch.Tensor
    b_lengths: torch.Tensor
    output_positions: torch.Tensor | None = None
    generation_index: int | None = None
    enabled: bool = True
    symbol_batch_permutation: torch.Tensor | None = None
    symbol_override: torch.Tensor | None = None
    planned_symbols: torch.Tensor | None = None
    planned_symbol_mask: torch.Tensor | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)


class InternalFirmwareLayer(nn.Module):
    """A Qwen decoder block followed by a native deterministic arithmetic unit."""

    def __init__(
        self,
        base_layer: nn.Module,
        unit: InternalArithmeticUnit,
        *,
        depth_after_blocks: int,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.unit = unit
        self.depth_after_blocks = depth_after_blocks
        self.runtime_context: InternalFirmwareContext | None = None

    @property
    def attention_type(self) -> str:
        return self.base_layer.attention_type

    def set_context(self, context: InternalFirmwareContext | None) -> None:
        self.runtime_context = context

    def _plan(
        self,
        hidden: torch.Tensor,
        context: InternalFirmwareContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        symbols, mask, a_logits, b_logits = self.unit.plan_from_hidden(
            hidden,
            context.a_positions,
            context.a_lengths,
            context.b_positions,
            context.b_lengths,
        )
        context.diagnostics["a_predictions"] = a_logits.argmax(dim=-1).detach().cpu()
        context.diagnostics["b_predictions"] = b_logits.argmax(dim=-1).detach().cpu()
        if context.symbol_batch_permutation is not None:
            permutation = context.symbol_batch_permutation
            symbols = symbols[permutation]
            mask = mask[permutation]
        if context.symbol_override is not None:
            override_mask = context.symbol_override >= 0
            symbols = torch.where(override_mask, context.symbol_override, symbols)
        context.planned_symbols = symbols.detach()
        context.planned_symbol_mask = mask.detach()
        context.diagnostics["planned_symbols"] = symbols.detach().cpu()
        context.diagnostics["planned_symbol_mask"] = mask.detach().cpu()
        return symbols, mask

    def forward(self, hidden_states: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        hidden = self.base_layer(hidden_states, *args, **kwargs)
        context = self.runtime_context
        if context is None or not context.enabled:
            return hidden

        if context.planned_symbols is None or context.planned_symbol_mask is None:
            symbols, symbol_mask = self._plan(hidden, context)
        else:
            symbols = context.planned_symbols
            symbol_mask = context.planned_symbol_mask

        if context.generation_index is None:
            if context.output_positions is None:
                raise ValueError("teacher-forced context requires output positions")
            return self.unit.inject_symbols(
                hidden,
                context.output_positions,
                symbols,
                symbol_mask,
            )

        index = context.generation_index
        if index < 0:
            raise ValueError("generation index must be nonnegative")
        if index >= symbols.shape[1]:
            return hidden
        active = symbol_mask[:, index]
        if not bool(active.any()):
            return hidden
        step_symbols = symbols[:, index : index + 1]
        step_mask = active[:, None]
        step_positions = torch.full(
            step_symbols.shape,
            hidden.shape[1] - 1,
            dtype=torch.long,
            device=hidden.device,
        )
        return self.unit.inject_symbols(
            hidden,
            step_positions,
            step_symbols,
            step_mask,
        )


def install_internal_firmware_layer(
    model: nn.Module,
    *,
    depth_after_blocks: int,
    strength: float,
) -> InternalFirmwareLayer:
    """Replace one Qwen layer entry with a block-plus-firmware wrapper."""

    layers = model.model.layers
    if depth_after_blocks < 1 or depth_after_blocks > len(layers):
        raise ValueError("depth_after_blocks is outside the transformer stack")
    layer_index = depth_after_blocks - 1
    base_layer = layers[layer_index]
    if isinstance(base_layer, InternalFirmwareLayer):
        raise ValueError("selected layer is already wrapped")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    unit = InternalArithmeticUnit(
        hidden_size=model.config.hidden_size,
        strength=strength,
    )
    reference_parameter = next(base_layer.parameters())
    unit.to(device=reference_parameter.device, dtype=reference_parameter.dtype)
    wrapper = InternalFirmwareLayer(
        base_layer,
        unit,
        depth_after_blocks=depth_after_blocks,
    )
    layers[layer_index] = wrapper
    return wrapper


class ParameterMatchedResidualAdapter(nn.Module):
    """Learned same-depth control with no access to deterministic state."""

    def __init__(self, hidden_size: int, rank: int = 10) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, rank)
        self.up = nn.Linear(rank, hidden_size)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.up(nn.functional.silu(self.down(hidden.float()))).to(hidden.dtype)


@dataclass
class LearnedControlContext:
    output_positions: torch.Tensor | None = None
    output_mask: torch.Tensor | None = None
    generation: bool = False
    enabled: bool = True


class InternalLearnedControlLayer(nn.Module):
    """Original Qwen block followed by a learned bottleneck residual only."""

    def __init__(
        self,
        base_layer: nn.Module,
        adapter: ParameterMatchedResidualAdapter,
        *,
        depth_after_blocks: int,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.adapter = adapter
        self.depth_after_blocks = depth_after_blocks
        self.runtime_context: LearnedControlContext | None = None

    @property
    def attention_type(self) -> str:
        return self.base_layer.attention_type

    def set_context(self, context: LearnedControlContext | None) -> None:
        self.runtime_context = context

    def forward(self, hidden_states: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        hidden = self.base_layer(hidden_states, *args, **kwargs)
        context = self.runtime_context
        if context is None or not context.enabled:
            return hidden
        if context.generation:
            positions = torch.full(
                (hidden.shape[0], 1),
                hidden.shape[1] - 1,
                dtype=torch.long,
                device=hidden.device,
            )
        else:
            if context.output_positions is None:
                raise ValueError("teacher-forced control requires output positions")
            positions = context.output_positions
        if context.generation:
            active = torch.ones_like(positions, dtype=torch.bool)
        else:
            if context.output_mask is None:
                raise ValueError("teacher-forced control requires an output mask")
            active = context.output_mask
        batch_grid = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
        batch_indices = batch_grid.expand_as(positions)[active]
        token_positions = positions[active]
        selected = hidden[batch_indices, token_positions]
        residuals = self.adapter(selected)
        additions = torch.zeros_like(hidden)
        additions.index_put_(
            (batch_indices, token_positions),
            residuals,
            accumulate=True,
        )
        return hidden + additions


def install_internal_learned_control(
    model: nn.Module,
    *,
    depth_after_blocks: int,
    rank: int = 10,
) -> InternalLearnedControlLayer:
    layers = model.model.layers
    if depth_after_blocks < 1 or depth_after_blocks > len(layers):
        raise ValueError("depth_after_blocks is outside the transformer stack")
    layer_index = depth_after_blocks - 1
    base_layer = layers[layer_index]
    if isinstance(base_layer, (InternalFirmwareLayer, InternalLearnedControlLayer)):
        raise ValueError("selected layer is already wrapped")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapter = ParameterMatchedResidualAdapter(model.config.hidden_size, rank=rank)
    reference_parameter = next(base_layer.parameters())
    adapter.to(device=reference_parameter.device, dtype=reference_parameter.dtype)
    wrapper = InternalLearnedControlLayer(
        base_layer,
        adapter,
        depth_after_blocks=depth_after_blocks,
    )
    layers[layer_index] = wrapper
    return wrapper
