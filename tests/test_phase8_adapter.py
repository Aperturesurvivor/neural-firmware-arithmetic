from __future__ import annotations

import torch
from torch import nn

from neural_firmware.phase8_adapter import (
    MatchedResidualAdapterMLP,
    matched_adapter_rank,
)


class TinyMLP(nn.Module):
    def __init__(self, hidden_size: int = 8, intermediate_size: int = 32) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        intermediate = self.act_fn(self.gate_proj(hidden)) * self.up_proj(hidden)
        return self.down_proj(intermediate)


def test_tinyllama_implant_budget_has_exact_rank14_match() -> None:
    assert matched_adapter_rank(2_048, 57_344) == 14


def test_matched_adapter_is_base_identical_at_initialization() -> None:
    torch.manual_seed(4)
    base = TinyMLP(hidden_size=8, intermediate_size=32)
    adapter = MatchedResidualAdapterMLP(base, hidden_size=8, rank=3)
    hidden = torch.randn(2, 5, 8)
    assert adapter.trainable_parameter_count == 48
    assert torch.equal(adapter(hidden), base(hidden))
