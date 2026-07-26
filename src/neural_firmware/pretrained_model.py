from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class BridgeConfig:
    hidden_size: int
    strength: float = 32.0
    router_threshold: float = 0.5


class FirmwareBridge(nn.Module):
    """Learned residual interface from eleven immutable firmware symbols."""

    def __init__(self, config: BridgeConfig) -> None:
        super().__init__()
        self.config = config
        self.symbol_vectors = nn.Embedding(11, config.hidden_size)
        self.router = nn.Linear(config.hidden_size, 1)
        nn.init.normal_(
            self.symbol_vectors.weight,
            mean=0.0,
            std=1.0 / math.sqrt(config.hidden_size),
        )
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def router_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.router(hidden.float()).squeeze(-1)

    def routed_hidden(
        self,
        hidden: torch.Tensor,
        symbols: torch.Tensor,
        *,
        hard_route: bool,
        eligible: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = self.router_logits(hidden)
        gate = torch.sigmoid(router_logits)
        if hard_route:
            gate = (gate >= self.config.router_threshold).to(hidden.dtype)
        if eligible is not None:
            gate = gate * eligible.to(gate.dtype)
        vectors = self.symbol_vectors(symbols).to(hidden.dtype)
        vectors = nn.functional.normalize(vectors, dim=-1)
        steered = hidden + gate.unsqueeze(-1) * self.config.strength * vectors
        return steered, router_logits


class LowRankLinear(nn.Module):
    """A frozen linear layer with a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.lora_a.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_b.to(device=base.weight.device, dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.lora_b(self.lora_a(inputs)) * self.scale


def install_last_layer_lora(
    model: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 16.0,
) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layer = model.model.layers[-1]
    layer.self_attn.q_proj = LowRankLinear(layer.self_attn.q_proj, rank=rank, alpha=alpha)
    layer.self_attn.v_proj = LowRankLinear(layer.self_attn.v_proj, rank=rank, alpha=alpha)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def install_all_layer_lora(
    model: nn.Module,
    *,
    rank: int = 4,
    alpha: float = 8.0,
) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer in model.model.layers:
        layer.self_attn.q_proj = LowRankLinear(
            layer.self_attn.q_proj,
            rank=rank,
            alpha=alpha,
        )
        layer.self_attn.v_proj = LowRankLinear(
            layer.self_attn.v_proj,
            rank=rank,
            alpha=alpha,
        )
    return [parameter for parameter in model.parameters() if parameter.requires_grad]
