from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn


class MatchedResidualAdapterMLP(nn.Module):
    """Frozen decoder MLP plus an ordinary learned bottleneck residual."""

    def __init__(self, base_mlp: nn.Module, *, hidden_size: int, rank: int) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("adapter rank must be positive")
        self.base_mlp = base_mlp
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.activation = nn.GELU()
        self.enabled = True
        nn.init.normal_(self.down.weight, std=hidden_size**-0.5)
        nn.init.zeros_(self.up.weight)
        for parameter in self.base_mlp.parameters():
            parameter.requires_grad_(False)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        base = self.base_mlp(hidden)
        if not self.enabled:
            return base
        residual = self.up(self.activation(self.down(hidden)))
        return base + residual.to(base.dtype)


def matched_adapter_rank(hidden_size: int, learned_parameter_count: int) -> int:
    denominator = 2 * hidden_size
    if learned_parameter_count % denominator:
        raise ValueError(
            "learned parameter count cannot be matched by a bias-free "
            "square residual adapter"
        )
    return learned_parameter_count // denominator


def install_matched_residual_adapter(
    model: nn.Module,
    *,
    layer_index: int,
    learned_parameter_count: int,
) -> MatchedResidualAdapterMLP:
    layer = model.model.layers[layer_index]
    if isinstance(layer.mlp, MatchedResidualAdapterMLP):
        raise ValueError("target layer already contains a matched adapter")
    hidden_size = int(model.config.hidden_size)
    rank = matched_adapter_rank(hidden_size, learned_parameter_count)
    wrapper = MatchedResidualAdapterMLP(
        layer.mlp,
        hidden_size=hidden_size,
        rank=rank,
    )
    reference = next(layer.mlp.parameters())
    wrapper.to(device=reference.device, dtype=reference.dtype)
    layer.mlp = wrapper
    if wrapper.trainable_parameter_count != learned_parameter_count:
        raise AssertionError("matched adapter parameter count is not exact")
    return wrapper


@contextmanager
def adapter_enabled(
    adapter: MatchedResidualAdapterMLP,
    enabled: bool,
):
    previous = adapter.enabled
    adapter.enabled = enabled
    try:
        yield
    finally:
        adapter.enabled = previous
