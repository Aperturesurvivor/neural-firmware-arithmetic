from __future__ import annotations

import time

import torch
from torch import nn

from neural_firmware.internal_data import (
    InternalAdditionExample,
    encode_internal_prompt,
)
from neural_firmware.internal_firmware import ResidualDigitEncoder
from neural_firmware.internal_training import set_seed
from neural_firmware.pretrained_training import ModelBundle


def _padded_batch(
    sequences: list[tuple[int, ...]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(map(len, sequences))
    input_ids = torch.full(
        (len(sequences), maximum),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        input_ids[row, : len(sequence)] = torch.tensor(
            sequence,
            dtype=torch.long,
            device=device,
        )
        attention_mask[row, : len(sequence)] = 1
    return input_ids, attention_mask


@torch.inference_mode()
def collect_digit_features(
    bundle: ModelBundle,
    examples: list[InternalAdditionExample],
    *,
    depth_after_blocks: int,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    encoded = [
        (example, encode_internal_prompt(bundle.tokenizer, example.prompt))
        for example in examples
    ]
    feature_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    example_id_chunks: list[torch.Tensor] = []
    started = time.perf_counter()
    for start in range(0, len(encoded), batch_size):
        batch = encoded[start : start + batch_size]
        input_ids, attention_mask = _padded_batch(
            [row[1].input_ids for row in batch],
            pad_token_id=bundle.tokenizer.pad_token_id,
            device=bundle.device,
        )
        outputs = bundle.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
        positions: list[tuple[int, int]] = []
        labels: list[int] = []
        example_ids: list[int] = []
        for local_row, (example, prompt) in enumerate(batch):
            digit_positions = (
                list(prompt.a_token_positions) + list(prompt.b_token_positions)
            )
            positions.extend(
                (local_row, position) for position in digit_positions
            )
            labels.extend(int(character) for character in example.a + example.b)
            example_ids.extend([start + local_row] * len(digit_positions))
        batch_indices = torch.tensor(
            [position[0] for position in positions],
            dtype=torch.long,
            device=bundle.device,
        )
        token_indices = torch.tensor(
            [position[1] for position in positions],
            dtype=torch.long,
            device=bundle.device,
        )
        feature_chunks.append(
            outputs.hidden_states[depth_after_blocks][
                batch_indices,
                token_indices,
            ]
            .float()
            .cpu()
        )
        label_chunks.append(torch.tensor(labels, dtype=torch.long))
        example_id_chunks.append(torch.tensor(example_ids, dtype=torch.long))
    return {
        "features": torch.cat(feature_chunks),
        "labels": torch.cat(label_chunks),
        "example_ids": torch.cat(example_id_chunks),
        "examples": torch.tensor(len(examples)),
        "extraction_seconds": torch.tensor(time.perf_counter() - started),
    }


def train_digit_encoder(
    feature_set: dict[str, torch.Tensor],
    *,
    hidden_size: int,
    device: torch.device,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[ResidualDigitEncoder, dict[str, float | int]]:
    set_seed(seed)
    encoder = ResidualDigitEncoder(hidden_size).to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    features = feature_set["features"]
    labels = feature_set["labels"]
    generator = torch.Generator().manual_seed(seed)
    initial_loss = float("nan")
    started = time.perf_counter()
    for step in range(steps):
        indices = torch.randint(
            len(labels),
            (batch_size,),
            generator=generator,
        )
        logits = encoder(features[indices].to(device))
        loss = loss_function(logits, labels[indices].to(device))
        if step == 0:
            initial_loss = loss.item()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if device.type == "mps":
        torch.mps.synchronize()
    return encoder, {
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "trainable_parameters": sum(
            parameter.numel() for parameter in encoder.parameters()
        ),
        "initial_loss": initial_loss,
        "final_loss": loss.item(),
        "wall_time_seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def evaluate_digit_encoder(
    encoder: ResidualDigitEncoder,
    feature_set: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> dict[str, float | int]:
    logits = encoder(feature_set["features"].to(device))
    predicted = logits.argmax(dim=-1).cpu()
    labels = feature_set["labels"]
    correct = predicted == labels
    example_ids = feature_set["example_ids"]
    example_count = int(feature_set["examples"].item())
    exact_examples = sum(
        int(bool(correct[example_ids == example_id].all()))
        for example_id in range(example_count)
    )
    return {
        "digits": len(labels),
        "correct_digits": int(correct.sum().item()),
        "digit_accuracy": float(correct.float().mean().item()),
        "examples": example_count,
        "exact_registers": exact_examples,
        "exact_register_accuracy": exact_examples / example_count,
    }
