from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from neural_firmware.firmware import FrozenRippleCarryAdder
from neural_firmware.tokenizer import ArithmeticTokenizer


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 384
    dropout: float = 0.0
    max_sequence_length: int = 96
    firmware_strength: float = 8.0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def sinusoidal_positions(max_length: int, d_model: int) -> torch.Tensor:
    positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    encoding = torch.zeros(max_length, d_model, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies)
    return encoding


class CausalArithmeticTransformer(nn.Module):
    VALID_MODES = {"baseline", "latent_firmware", "direct_firmware"}

    def __init__(
        self,
        tokenizer: ArithmeticTokenizer,
        config: ModelConfig,
        mode: str,
    ) -> None:
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {sorted(self.VALID_MODES)}")
        self.tokenizer = tokenizer
        self.config = config
        self.mode = mode

        self.token_embedding = nn.Embedding(tokenizer.vocab_size, config.d_model)
        self.register_buffer(
            "position_encoding",
            sinusoidal_positions(config.max_sequence_length, config.d_model),
            persistent=True,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, tokenizer.vocab_size, bias=False)
        self.firmware = FrozenRippleCarryAdder(tokenizer, config.d_model)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, sequence_length = input_ids.shape
        if sequence_length > self.config.max_sequence_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"{self.config.max_sequence_length}"
            )
        hidden = self.token_embedding(input_ids)
        hidden = hidden + self.position_encoding[:sequence_length].unsqueeze(0)
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=input_ids.device,
            ),
            diagonal=1,
        )
        padding_mask = None if attention_mask is None else ~attention_mask.bool()
        hidden = self.transformer(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        hidden = self.final_norm(hidden)

        targets: torch.Tensor | None = None
        valid: torch.Tensor | None = None
        if self.mode == "latent_firmware":
            signal, targets, valid = self.firmware.latent_signal(input_ids)
            hidden = hidden + self.config.firmware_strength * signal

        logits = self.output_head(hidden)

        if self.mode == "direct_firmware":
            targets, valid = self.firmware.next_token_targets(input_ids)
            direct = torch.zeros_like(logits)
            direct.scatter_(-1, targets.unsqueeze(-1), self.config.firmware_strength * 10.0)
            logits = logits + direct * valid.unsqueeze(-1)

        return logits

