from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from neural_firmware.internal_data import (
    InternalAdditionExample,
    encode_internal_prompt,
)
from neural_firmware.internal_firmware import (
    InternalFirmwareContext,
    InternalFirmwareLayer,
    InternalLearnedControlLayer,
    LearnedControlContext,
)
from neural_firmware.pretrained_data import answer_token_ids
from neural_firmware.pretrained_evaluation import GenerationResult
from neural_firmware.pretrained_training import ModelBundle


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass(frozen=True)
class InternalDecoderTrainConfig:
    seed: int
    steps: int
    batch_size: int
    learning_rate: float


@dataclass(frozen=True)
class InternalDecoderTrainResult:
    config: dict[str, int | float]
    trainable_parameters: int
    initial_loss: float
    final_loss: float
    wall_time_seconds: float


@dataclass
class InternalTrainingBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_ids: torch.Tensor
    target_mask: torch.Tensor
    context: InternalFirmwareContext


def make_internal_training_batch(
    bundle: ModelBundle,
    examples: list[InternalAdditionExample],
) -> InternalTrainingBatch:
    encoded_rows = [
        (example, encode_internal_prompt(bundle.tokenizer, example.prompt))
        for example in examples
    ]
    targets = [
        answer_token_ids(bundle.tokenizer, example.answer)
        for example, _ in encoded_rows
    ]
    sequences = [
        list(encoded.input_ids) + target[:-1]
        for (_, encoded), target in zip(encoded_rows, targets, strict=True)
    ]
    maximum_sequence = max(map(len, sequences))
    input_ids = torch.full(
        (len(examples), maximum_sequence),
        bundle.tokenizer.pad_token_id,
        dtype=torch.long,
        device=bundle.device,
    )
    attention_mask = torch.zeros_like(input_ids)
    maximum_a = max(len(row[1].a_token_positions) for row in encoded_rows)
    maximum_b = max(len(row[1].b_token_positions) for row in encoded_rows)
    maximum_symbols = max(
        max(len(row[0].a), len(row[0].b)) + 2 for row in encoded_rows
    )
    a_positions = torch.zeros(
        (len(examples), maximum_a),
        dtype=torch.long,
        device=bundle.device,
    )
    b_positions = torch.zeros(
        (len(examples), maximum_b),
        dtype=torch.long,
        device=bundle.device,
    )
    a_lengths = torch.zeros(len(examples), dtype=torch.long, device=bundle.device)
    b_lengths = torch.zeros(len(examples), dtype=torch.long, device=bundle.device)
    output_positions = torch.zeros(
        (len(examples), maximum_symbols),
        dtype=torch.long,
        device=bundle.device,
    )
    target_ids = torch.full(
        (len(examples), maximum_symbols),
        bundle.tokenizer.eos_token_id,
        dtype=torch.long,
        device=bundle.device,
    )
    target_mask = torch.zeros_like(target_ids, dtype=torch.bool)
    for row, ((_, encoded), sequence, target) in enumerate(
        zip(encoded_rows, sequences, targets, strict=True)
    ):
        input_ids[row, : len(sequence)] = torch.tensor(
            sequence,
            dtype=torch.long,
            device=bundle.device,
        )
        attention_mask[row, : len(sequence)] = 1
        a_length = len(encoded.a_token_positions)
        b_length = len(encoded.b_token_positions)
        a_positions[row, :a_length] = torch.tensor(
            encoded.a_token_positions,
            dtype=torch.long,
            device=bundle.device,
        )
        b_positions[row, :b_length] = torch.tensor(
            encoded.b_token_positions,
            dtype=torch.long,
            device=bundle.device,
        )
        a_lengths[row] = a_length
        b_lengths[row] = b_length
        first_output_position = len(encoded.input_ids) - 1
        positions = torch.arange(
            first_output_position,
            first_output_position + len(target),
            dtype=torch.long,
            device=bundle.device,
        )
        output_positions[row, : len(target)] = positions
        target_ids[row, : len(target)] = torch.tensor(
            target,
            dtype=torch.long,
            device=bundle.device,
        )
        target_mask[row, : len(target)] = True
    return InternalTrainingBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_ids=target_ids,
        target_mask=target_mask,
        context=InternalFirmwareContext(
            a_positions=a_positions,
            a_lengths=a_lengths,
            b_positions=b_positions,
            b_lengths=b_lengths,
            output_positions=output_positions,
        ),
    )


def initialize_internal_decoder_from_output_head(
    bundle: ModelBundle,
    wrapper: InternalFirmwareLayer,
) -> None:
    token_ids = [
        bundle.tokenizer.encode(str(digit), add_special_tokens=False)[0]
        for digit in range(10)
    ] + [bundle.tokenizer.eos_token_id]
    with torch.no_grad():
        vectors = bundle.model.lm_head.weight[token_ids].detach().float()
        wrapper.unit.symbol_decoder.codebook.weight.copy_(
            nn.functional.normalize(vectors, dim=-1)
        )


def train_internal_decoder(
    bundle: ModelBundle,
    wrapper: InternalFirmwareLayer,
    examples: list[InternalAdditionExample],
    config: InternalDecoderTrainConfig,
) -> InternalDecoderTrainResult:
    set_seed(config.seed)
    for parameter in bundle.model.parameters():
        parameter.requires_grad_(False)
    for parameter in wrapper.unit.digit_encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in wrapper.unit.symbol_decoder.parameters():
        parameter.requires_grad_(True)
    initialize_internal_decoder_from_output_head(bundle, wrapper)
    parameters = [
        parameter
        for parameter in wrapper.unit.symbol_decoder.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
    loss_function = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(config.seed)
    initial_loss = float("nan")
    final_loss = float("nan")
    started = time.perf_counter()
    for step in range(config.steps):
        indices = torch.randint(
            len(examples),
            (config.batch_size,),
            generator=generator,
        ).tolist()
        batch = make_internal_training_batch(
            bundle,
            [examples[index] for index in indices],
        )
        batch.context.planned_symbols = None
        batch.context.planned_symbol_mask = None
        wrapper.set_context(batch.context)
        outputs = bundle.model.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            use_cache=False,
        )
        logits = bundle.model.lm_head(outputs.last_hidden_state)
        batch_grid = torch.arange(
            batch.input_ids.shape[0],
            device=bundle.device,
        )[:, None].expand_as(batch.target_ids)
        selected_logits = logits[
            batch_grid[batch.target_mask],
            batch.context.output_positions[batch.target_mask],
        ]
        selected_targets = batch.target_ids[batch.target_mask]
        loss = loss_function(selected_logits.float(), selected_targets)
        if step == 0:
            initial_loss = loss.item()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
    wrapper.set_context(None)
    if bundle.device.type == "mps":
        torch.mps.synchronize()
    return InternalDecoderTrainResult(
        config=asdict(config),
        trainable_parameters=sum(parameter.numel() for parameter in parameters),
        initial_loss=initial_loss,
        final_loss=final_loss,
        wall_time_seconds=time.perf_counter() - started,
    )


@torch.inference_mode()
def generate_internal(
    bundle: ModelBundle,
    wrapper: InternalFirmwareLayer,
    example: InternalAdditionExample,
    *,
    enabled: bool,
    symbol_batch_permutation: torch.Tensor | None = None,
    symbol_override: torch.Tensor | None = None,
) -> GenerationResult:
    encoded = encode_internal_prompt(bundle.tokenizer, example.prompt)
    input_ids = torch.tensor(
        [encoded.input_ids],
        dtype=torch.long,
        device=bundle.device,
    )
    attention_mask = torch.ones_like(input_ids)
    context = InternalFirmwareContext(
        a_positions=torch.tensor(
            [encoded.a_token_positions],
            dtype=torch.long,
            device=bundle.device,
        ),
        a_lengths=torch.tensor(
            [len(encoded.a_token_positions)],
            dtype=torch.long,
            device=bundle.device,
        ),
        b_positions=torch.tensor(
            [encoded.b_token_positions],
            dtype=torch.long,
            device=bundle.device,
        ),
        b_lengths=torch.tensor(
            [len(encoded.b_token_positions)],
            dtype=torch.long,
            device=bundle.device,
        ),
        generation_index=0,
        enabled=enabled,
        symbol_batch_permutation=symbol_batch_permutation,
        symbol_override=symbol_override,
    )
    wrapper.set_context(context if enabled else None)
    started = time.perf_counter()
    outputs = bundle.model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    hidden = outputs.last_hidden_state[:, -1, :]
    generated: list[int] = []
    maximum_tokens = len(example.answer) + 3
    for step in range(maximum_tokens):
        logits = bundle.model.lm_head(hidden)
        next_token = logits.argmax(dim=-1)
        token_id = next_token.item()
        generated.append(token_id)
        if token_id == bundle.tokenizer.eos_token_id:
            break
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (1, 1),
                    dtype=attention_mask.dtype,
                    device=bundle.device,
                ),
            ],
            dim=1,
        )
        if enabled:
            context.generation_index = step + 1
        outputs = bundle.model.model(
            input_ids=next_token.unsqueeze(0),
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        hidden = outputs.last_hidden_state[:, -1, :]
    if bundle.device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    wrapper.set_context(None)
    text = bundle.tokenizer.decode(generated, skip_special_tokens=True).strip()
    return GenerationResult(
        prompt=example.prompt,
        expected=example.answer,
        generated_text=text,
        generated_token_ids=generated,
        route_probabilities=[],
        exact=text == example.answer,
        latency_seconds=elapsed,
    )


def train_internal_learned_control(
    bundle: ModelBundle,
    wrapper: InternalLearnedControlLayer,
    examples: list[InternalAdditionExample],
    config: InternalDecoderTrainConfig,
) -> InternalDecoderTrainResult:
    set_seed(config.seed)
    for parameter in bundle.model.parameters():
        parameter.requires_grad_(False)
    for parameter in wrapper.adapter.parameters():
        parameter.requires_grad_(True)
    parameters = [
        parameter
        for parameter in wrapper.adapter.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
    loss_function = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(config.seed)
    initial_loss = float("nan")
    final_loss = float("nan")
    started = time.perf_counter()
    for step in range(config.steps):
        indices = torch.randint(
            len(examples),
            (config.batch_size,),
            generator=generator,
        ).tolist()
        batch = make_internal_training_batch(
            bundle,
            [examples[index] for index in indices],
        )
        wrapper.set_context(
            LearnedControlContext(
                output_positions=batch.context.output_positions,
                output_mask=batch.target_mask,
            )
        )
        outputs = bundle.model.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            use_cache=False,
        )
        logits = bundle.model.lm_head(outputs.last_hidden_state)
        batch_grid = torch.arange(
            batch.input_ids.shape[0],
            device=bundle.device,
        )[:, None].expand_as(batch.target_ids)
        selected_logits = logits[
            batch_grid[batch.target_mask],
            batch.context.output_positions[batch.target_mask],
        ]
        selected_targets = batch.target_ids[batch.target_mask]
        loss = loss_function(selected_logits.float(), selected_targets)
        if step == 0:
            initial_loss = loss.item()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
    wrapper.set_context(None)
    if bundle.device.type == "mps":
        torch.mps.synchronize()
    return InternalDecoderTrainResult(
        config=asdict(config),
        trainable_parameters=sum(parameter.numel() for parameter in parameters),
        initial_loss=initial_loss,
        final_loss=final_loss,
        wall_time_seconds=time.perf_counter() - started,
    )


@torch.inference_mode()
def generate_internal_learned_control(
    bundle: ModelBundle,
    wrapper: InternalLearnedControlLayer,
    example: InternalAdditionExample,
    *,
    enabled: bool,
) -> GenerationResult:
    encoded = encode_internal_prompt(bundle.tokenizer, example.prompt)
    input_ids = torch.tensor(
        [encoded.input_ids],
        dtype=torch.long,
        device=bundle.device,
    )
    attention_mask = torch.ones_like(input_ids)
    context = LearnedControlContext(generation=True, enabled=enabled)
    wrapper.set_context(context if enabled else None)
    started = time.perf_counter()
    outputs = bundle.model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    hidden = outputs.last_hidden_state[:, -1, :]
    generated: list[int] = []
    maximum_tokens = len(example.answer) + 3
    for _ in range(maximum_tokens):
        logits = bundle.model.lm_head(hidden)
        next_token = logits.argmax(dim=-1)
        token_id = next_token.item()
        generated.append(token_id)
        if token_id == bundle.tokenizer.eos_token_id:
            break
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (1, 1),
                    dtype=attention_mask.dtype,
                    device=bundle.device,
                ),
            ],
            dim=1,
        )
        outputs = bundle.model.model(
            input_ids=next_token.unsqueeze(0),
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        hidden = outputs.last_hidden_state[:, -1, :]
    if bundle.device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    wrapper.set_context(None)
    text = bundle.tokenizer.decode(generated, skip_special_tokens=True).strip()
    return GenerationResult(
        prompt=example.prompt,
        expected=example.answer,
        generated_text=text,
        generated_token_ids=generated,
        route_probabilities=[],
        exact=text == example.answer,
        latency_seconds=elapsed,
    )
