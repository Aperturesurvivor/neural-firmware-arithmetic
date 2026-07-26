from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn

from neural_firmware.internal_firmware import (
    FrozenTypedAdditionCell,
    SymbolResidualDecoder,
)

RouteMode = Literal["learned", "force_on", "force_off"]


class SemanticRouter(nn.Module):
    """Learned request-level decision to invoke the registered addition unit."""

    def __init__(self, hidden_size: int, hidden_width: int = 16) -> None:
        super().__init__()
        if hidden_width < 0:
            raise ValueError("hidden_width must be nonnegative")
        if hidden_width == 0:
            self.classifier = nn.Linear(hidden_size, 1)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_width),
                nn.SiLU(),
                nn.Linear(hidden_width, 1),
            )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.classifier(hidden.float()).squeeze(-1)


class SemanticInternalArithmeticUnit(nn.Module):
    """Semantic router and residual decoder around a frozen typed addition cell."""

    def __init__(
        self,
        hidden_size: int,
        strength: float,
        router_hidden_width: int = 16,
    ) -> None:
        super().__init__()
        self.router = SemanticRouter(hidden_size, router_hidden_width)
        self.cell = FrozenTypedAdditionCell()
        self.symbol_decoder = SymbolResidualDecoder(hidden_size, strength)

    @property
    def interface_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def plan_from_digits(
        self,
        a_digits: torch.Tensor,
        a_lengths: torch.Tensor,
        b_digits: torch.Tensor,
        b_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cell(a_digits, a_lengths, b_digits, b_lengths)

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
        if valid_symbols.numel() == 0:
            return hidden
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
class SemanticFirmwareContext:
    a_digits: torch.Tensor
    a_lengths: torch.Tensor
    b_digits: torch.Tensor
    b_lengths: torch.Tensor
    output_positions: torch.Tensor | None = None
    generation_index: int | None = None
    route_mode: RouteMode = "learned"
    route_threshold: float = 0.5
    route_active: torch.Tensor | None = None
    route_probabilities: torch.Tensor | None = None
    symbol_override: torch.Tensor | None = None
    planned_symbols: torch.Tensor | None = None
    planned_symbol_mask: torch.Tensor | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)


def _route_decision(
    router: SemanticRouter,
    hidden: torch.Tensor,
    *,
    mode: RouteMode,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = router(hidden[:, -1, :])
    probabilities = torch.sigmoid(logits)
    if mode == "force_on":
        active = torch.ones_like(probabilities, dtype=torch.bool)
    elif mode == "force_off":
        active = torch.zeros_like(probabilities, dtype=torch.bool)
    elif mode == "learned":
        active = probabilities >= threshold
    else:
        raise ValueError(f"unknown route mode: {mode}")
    return active, probabilities


class SemanticInternalFirmwareLayer(nn.Module):
    """A Qwen block followed by learned semantic routing and exact addition."""

    def __init__(
        self,
        base_layer: nn.Module,
        unit: SemanticInternalArithmeticUnit,
        *,
        depth_after_blocks: int,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.unit = unit
        self.depth_after_blocks = depth_after_blocks
        self.runtime_context: SemanticFirmwareContext | None = None

    @property
    def attention_type(self) -> str:
        return self.base_layer.attention_type

    def set_context(self, context: SemanticFirmwareContext | None) -> None:
        self.runtime_context = context

    def _plan(
        self,
        hidden: torch.Tensor,
        context: SemanticFirmwareContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context.route_active is None:
            active, probabilities = _route_decision(
                self.unit.router,
                hidden,
                mode=context.route_mode,
                threshold=context.route_threshold,
            )
            context.route_active = active.detach()
            context.route_probabilities = probabilities.detach()
            context.diagnostics["route_active"] = active.detach().cpu()
            context.diagnostics["route_probabilities"] = probabilities.detach().cpu()
        symbols, mask = self.unit.plan_from_digits(
            context.a_digits,
            context.a_lengths,
            context.b_digits,
            context.b_lengths,
        )
        if context.symbol_override is not None:
            override_mask = context.symbol_override >= 0
            symbols = torch.where(override_mask, context.symbol_override, symbols)
        mask = mask & context.route_active[:, None]
        context.planned_symbols = symbols.detach()
        context.planned_symbol_mask = mask.detach()
        context.diagnostics["planned_symbols"] = symbols.detach().cpu()
        context.diagnostics["planned_symbol_mask"] = mask.detach().cpu()
        return symbols, mask

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
        positions = torch.full(
            step_symbols.shape,
            hidden.shape[1] - 1,
            dtype=torch.long,
            device=hidden.device,
        )
        return self.unit.inject_symbols(hidden, positions, step_symbols, step_mask)


def install_semantic_internal_firmware(
    model: nn.Module,
    *,
    depth_after_blocks: int,
    strength: float,
    router_hidden_width: int = 16,
) -> SemanticInternalFirmwareLayer:
    layers = model.model.layers
    if depth_after_blocks < 1 or depth_after_blocks > len(layers):
        raise ValueError("depth_after_blocks is outside the transformer stack")
    layer_index = depth_after_blocks - 1
    base_layer = layers[layer_index]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    unit = SemanticInternalArithmeticUnit(
        hidden_size=model.config.hidden_size,
        strength=strength,
        router_hidden_width=router_hidden_width,
    )
    reference = next(base_layer.parameters())
    unit.to(device=reference.device, dtype=reference.dtype)
    wrapper = SemanticInternalFirmwareLayer(
        base_layer,
        unit,
        depth_after_blocks=depth_after_blocks,
    )
    layers[layer_index] = wrapper
    return wrapper


class SemanticMatchedResidualAdapter(nn.Module):
    """Rank-five learned control exactly matching the 11-symbol codebook."""

    def __init__(self, hidden_size: int, rank: int = 5) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=True)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.up(nn.functional.silu(self.down(hidden.float()))).to(hidden.dtype)


@dataclass
class SemanticControlContext:
    output_positions: torch.Tensor | None = None
    output_mask: torch.Tensor | None = None
    generation: bool = False
    route_mode: RouteMode = "learned"
    route_threshold: float = 0.5
    route_active: torch.Tensor | None = None
    route_probabilities: torch.Tensor | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)


class SemanticLearnedControlLayer(nn.Module):
    """Same router, location, and trainable count without deterministic state."""

    def __init__(
        self,
        base_layer: nn.Module,
        router: SemanticRouter,
        adapter: SemanticMatchedResidualAdapter,
        *,
        depth_after_blocks: int,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.router = router
        self.adapter = adapter
        self.depth_after_blocks = depth_after_blocks
        self.runtime_context: SemanticControlContext | None = None

    @property
    def attention_type(self) -> str:
        return self.base_layer.attention_type

    @property
    def interface_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for module in (self.router, self.adapter)
            for parameter in module.parameters()
        )

    def set_context(self, context: SemanticControlContext | None) -> None:
        self.runtime_context = context

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
        if context.route_active is None:
            active, probabilities = _route_decision(
                self.router,
                hidden,
                mode=context.route_mode,
                threshold=context.route_threshold,
            )
            context.route_active = active.detach()
            context.route_probabilities = probabilities.detach()
            context.diagnostics["route_active"] = active.detach().cpu()
            context.diagnostics["route_probabilities"] = probabilities.detach().cpu()
        if not bool(context.route_active.any()):
            return hidden

        if context.generation:
            positions = torch.full(
                (hidden.shape[0], 1),
                hidden.shape[1] - 1,
                dtype=torch.long,
                device=hidden.device,
            )
            active_positions = context.route_active[:, None]
        else:
            if context.output_positions is None or context.output_mask is None:
                raise ValueError("teacher-forced control requires positions and mask")
            positions = context.output_positions
            active_positions = context.output_mask & context.route_active[:, None]
        batch_grid = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
        batch_indices = batch_grid.expand_as(positions)[active_positions]
        token_positions = positions[active_positions]
        selected = hidden[batch_indices, token_positions]
        residuals = self.adapter(selected)
        additions = torch.zeros_like(hidden)
        additions.index_put_(
            (batch_indices, token_positions),
            residuals,
            accumulate=True,
        )
        return hidden + additions


def install_semantic_learned_control(
    model: nn.Module,
    *,
    depth_after_blocks: int,
    rank: int = 5,
    router_hidden_width: int = 16,
) -> SemanticLearnedControlLayer:
    layers = model.model.layers
    if depth_after_blocks < 1 or depth_after_blocks > len(layers):
        raise ValueError("depth_after_blocks is outside the transformer stack")
    layer_index = depth_after_blocks - 1
    base_layer = layers[layer_index]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    router = SemanticRouter(model.config.hidden_size, router_hidden_width)
    adapter = SemanticMatchedResidualAdapter(model.config.hidden_size, rank=rank)
    reference = next(base_layer.parameters())
    router.to(device=reference.device, dtype=reference.dtype)
    adapter.to(device=reference.device, dtype=reference.dtype)
    wrapper = SemanticLearnedControlLayer(
        base_layer,
        router,
        adapter,
        depth_after_blocks=depth_after_blocks,
    )
    layers[layer_index] = wrapper
    return wrapper
