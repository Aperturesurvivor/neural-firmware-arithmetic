from __future__ import annotations

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
