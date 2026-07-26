from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from neural_firmware.pretrained_data import answer_token_ids, chat_prompt_ids
from neural_firmware.pretrained_evaluation import generate_one
from neural_firmware.pretrained_training import ModelBundle
from neural_firmware.semantic_data import (
    SemanticPromptExample,
    encode_semantic_prompt,
    exact_format_correct,
    mathematical_correct,
)
from neural_firmware.semantic_firmware import (
    RouteMode,
    SemanticControlContext,
    SemanticFirmwareContext,
    SemanticInternalFirmwareLayer,
    SemanticLearnedControlLayer,
    SemanticRouter,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _padded_batch(
    sequences: list[tuple[int, ...] | list[int]],
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
def collect_route_features(
    bundle: ModelBundle,
    examples: list[SemanticPromptExample],
    *,
    depth_after_blocks: int,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    started = time.perf_counter()
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        sequences = [chat_prompt_ids(bundle.tokenizer, row.prompt) for row in batch]
        input_ids, attention_mask = _padded_batch(
            sequences,
            pad_token_id=bundle.tokenizer.pad_token_id,
            device=bundle.device,
        )
        outputs = bundle.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
        final_positions = attention_mask.sum(dim=1) - 1
        row_indices = torch.arange(len(batch), device=bundle.device)
        features.append(
            outputs.hidden_states[depth_after_blocks][
                row_indices,
                final_positions,
            ]
            .float()
            .cpu()
        )
        labels.append(
            torch.tensor([row.route_label for row in batch], dtype=torch.float32)
        )
    return {
        "features": torch.cat(features),
        "labels": torch.cat(labels),
        "examples": torch.tensor(len(examples)),
        "extraction_seconds": torch.tensor(time.perf_counter() - started),
    }


def train_semantic_router(
    feature_set: dict[str, torch.Tensor],
    *,
    hidden_size: int,
    device: torch.device,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    hidden_width: int = 16,
) -> tuple[SemanticRouter, dict[str, float | int]]:
    set_seed(seed)
    router = SemanticRouter(hidden_size, hidden_width).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()
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
        logits = router(features[indices].to(device))
        loss = loss_function(logits, labels[indices].to(device))
        if step == 0:
            initial_loss = loss.item()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if device.type == "mps":
        torch.mps.synchronize()
    return router, {
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "hidden_width": hidden_width,
        "trainable_parameters": sum(
            parameter.numel() for parameter in router.parameters()
        ),
        "initial_loss": initial_loss,
        "final_loss": loss.item(),
        "wall_time_seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def evaluate_semantic_router(
    router: SemanticRouter,
    feature_set: dict[str, torch.Tensor],
    examples: list[SemanticPromptExample],
    *,
    device: torch.device,
    threshold: float,
) -> dict[str, object]:
    probabilities = torch.sigmoid(router(feature_set["features"].to(device))).cpu()
    predictions = probabilities >= threshold
    labels = feature_set["labels"].bool()
    correct = predictions == labels
    true_positive = int((predictions & labels).sum().item())
    false_positive = int((predictions & ~labels).sum().item())
    true_negative = int((~predictions & ~labels).sum().item())
    false_negative = int((~predictions & labels).sum().item())
    rows = [
        {
            "prompt": example.prompt,
            "family": example.family,
            "split": example.split,
            "label": bool(label),
            "probability": float(probability),
            "predicted": bool(predicted),
            "correct": bool(is_correct),
        }
        for example, label, probability, predicted, is_correct in zip(
            examples,
            labels.tolist(),
            probabilities.tolist(),
            predictions.tolist(),
            correct.tolist(),
            strict=True,
        )
    ]
    return {
        "examples": len(examples),
        "correct": int(correct.sum().item()),
        "accuracy": float(correct.float().mean().item()),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "true_positive_rate": (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else float("nan")
        ),
        "false_positive_rate": (
            false_positive / (false_positive + true_negative)
            if false_positive + true_negative
            else float("nan")
        ),
        "rows": rows,
    }


@dataclass(frozen=True)
class SemanticTrainConfig:
    seed: int
    steps: int
    batch_size: int
    learning_rate: float


@dataclass(frozen=True)
class SemanticTrainResult:
    config: dict[str, int | float]
    trainable_parameters: int
    initial_loss: float
    final_loss: float
    wall_time_seconds: float


@dataclass
class SemanticTrainingBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_ids: torch.Tensor
    target_mask: torch.Tensor
    internal_context: SemanticFirmwareContext


def make_semantic_training_batch(
    bundle: ModelBundle,
    examples: list[SemanticPromptExample],
) -> SemanticTrainingBatch:
    if any(example.answer is None for example in examples):
        raise ValueError("decoder training requires positive addition examples")
    encoded_rows = [
        (example, encode_semantic_prompt(bundle.tokenizer, example.prompt))
        for example in examples
    ]
    targets = [
        answer_token_ids(bundle.tokenizer, example.answer or "")
        for example, _ in encoded_rows
    ]
    sequences = [
        list(encoded.input_ids) + target[:-1]
        for (_, encoded), target in zip(encoded_rows, targets, strict=True)
    ]
    input_ids, attention_mask = _padded_batch(
        sequences,
        pad_token_id=bundle.tokenizer.pad_token_id,
        device=bundle.device,
    )
    maximum_a = max(len(encoded.a_digits) for _, encoded in encoded_rows)
    maximum_b = max(len(encoded.b_digits) for _, encoded in encoded_rows)
    maximum_symbols = max(
        max(len(encoded.a_digits), len(encoded.b_digits)) + 2
        for _, encoded in encoded_rows
    )
    a_digits = torch.zeros(
        (len(examples), maximum_a),
        dtype=torch.long,
        device=bundle.device,
    )
    b_digits = torch.zeros(
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
    for row, ((_, encoded), target) in enumerate(
        zip(encoded_rows, targets, strict=True)
    ):
        a_digits[row, : len(encoded.a_digits)] = torch.tensor(
            encoded.a_digits,
            dtype=torch.long,
            device=bundle.device,
        )
        b_digits[row, : len(encoded.b_digits)] = torch.tensor(
            encoded.b_digits,
            dtype=torch.long,
            device=bundle.device,
        )
        a_lengths[row] = len(encoded.a_digits)
        b_lengths[row] = len(encoded.b_digits)
        prompt_length = len(encoded.input_ids)
        positions = torch.arange(
            prompt_length - 1,
            prompt_length - 1 + len(target),
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
    return SemanticTrainingBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_ids=target_ids,
        target_mask=target_mask,
        internal_context=SemanticFirmwareContext(
            a_digits=a_digits,
            a_lengths=a_lengths,
            b_digits=b_digits,
            b_lengths=b_lengths,
            output_positions=output_positions,
            route_mode="force_on",
        ),
    )


def initialize_semantic_decoder_from_output_head(
    bundle: ModelBundle,
    wrapper: SemanticInternalFirmwareLayer,
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


def train_semantic_decoder(
    bundle: ModelBundle,
    wrapper: SemanticInternalFirmwareLayer,
    examples: list[SemanticPromptExample],
    config: SemanticTrainConfig,
) -> SemanticTrainResult:
    set_seed(config.seed)
    for parameter in bundle.model.parameters():
        parameter.requires_grad_(False)
    for parameter in wrapper.unit.router.parameters():
        parameter.requires_grad_(False)
    for parameter in wrapper.unit.symbol_decoder.parameters():
        parameter.requires_grad_(True)
    initialize_semantic_decoder_from_output_head(bundle, wrapper)
    parameters = list(wrapper.unit.symbol_decoder.parameters())
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
        batch = make_semantic_training_batch(
            bundle,
            [examples[index] for index in indices],
        )
        wrapper.set_context(batch.internal_context)
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
            batch.internal_context.output_positions[batch.target_mask],
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
    return SemanticTrainResult(
        config=asdict(config),
        trainable_parameters=sum(parameter.numel() for parameter in parameters),
        initial_loss=initial_loss,
        final_loss=final_loss,
        wall_time_seconds=time.perf_counter() - started,
    )


def train_semantic_control(
    bundle: ModelBundle,
    wrapper: SemanticLearnedControlLayer,
    examples: list[SemanticPromptExample],
    config: SemanticTrainConfig,
) -> SemanticTrainResult:
    set_seed(config.seed)
    for parameter in bundle.model.parameters():
        parameter.requires_grad_(False)
    for parameter in wrapper.router.parameters():
        parameter.requires_grad_(False)
    for parameter in wrapper.adapter.parameters():
        parameter.requires_grad_(True)
    parameters = list(wrapper.adapter.parameters())
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
        batch = make_semantic_training_batch(
            bundle,
            [examples[index] for index in indices],
        )
        wrapper.set_context(
            SemanticControlContext(
                output_positions=batch.internal_context.output_positions,
                output_mask=batch.target_mask,
                route_mode="force_on",
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
            batch.internal_context.output_positions[batch.target_mask],
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
    return SemanticTrainResult(
        config=asdict(config),
        trainable_parameters=sum(parameter.numel() for parameter in parameters),
        initial_loss=initial_loss,
        final_loss=final_loss,
        wall_time_seconds=time.perf_counter() - started,
    )


def _semantic_result(
    *,
    example: SemanticPromptExample,
    generated_text: str,
    generated_token_ids: list[int],
    latency_seconds: float,
    route_probability: float | None,
    route_active: bool | None,
) -> dict[str, object]:
    expected = example.answer
    return {
        "prompt": example.prompt,
        "family": example.family,
        "family_index": example.family_index,
        "split": example.split,
        "a": example.a,
        "b": example.b,
        "expected": expected,
        "generated_text": generated_text,
        "generated_token_ids": generated_token_ids,
        "mathematical_correct": (
            mathematical_correct(generated_text, expected)
            if expected is not None
            else None
        ),
        "exact_format_correct": (
            exact_format_correct(generated_text, expected)
            if expected is not None
            else None
        ),
        "route_label": example.route_label,
        "route_probability": route_probability,
        "route_active": route_active,
        "latency_seconds": latency_seconds,
    }


@torch.inference_mode()
def generate_base_semantic(
    bundle: ModelBundle,
    example: SemanticPromptExample,
    *,
    max_new_tokens: int,
) -> dict[str, object]:
    result = generate_one(
        bundle,
        example.prompt,
        mode="base",
        max_new_tokens=max_new_tokens,
    )
    return _semantic_result(
        example=example,
        generated_text=result.generated_text,
        generated_token_ids=result.generated_token_ids,
        latency_seconds=result.latency_seconds,
        route_probability=None,
        route_active=None,
    )


def _generation_context(
    bundle: ModelBundle,
    example: SemanticPromptExample,
    *,
    route_mode: RouteMode,
    route_threshold: float,
    symbol_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, SemanticFirmwareContext]:
    encoded = encode_semantic_prompt(bundle.tokenizer, example.prompt)
    input_ids = torch.tensor(
        [encoded.input_ids],
        dtype=torch.long,
        device=bundle.device,
    )
    attention_mask = torch.ones_like(input_ids)
    a_digits = torch.tensor(
        [encoded.a_digits],
        dtype=torch.long,
        device=bundle.device,
    )
    b_digits = torch.tensor(
        [encoded.b_digits],
        dtype=torch.long,
        device=bundle.device,
    )
    context = SemanticFirmwareContext(
        a_digits=a_digits,
        a_lengths=torch.tensor([len(encoded.a_digits)], device=bundle.device),
        b_digits=b_digits,
        b_lengths=torch.tensor([len(encoded.b_digits)], device=bundle.device),
        generation_index=0,
        route_mode=route_mode,
        route_threshold=route_threshold,
        symbol_override=symbol_override,
    )
    return input_ids, attention_mask, context


@torch.inference_mode()
def generate_semantic_internal(
    bundle: ModelBundle,
    wrapper: SemanticInternalFirmwareLayer,
    example: SemanticPromptExample,
    *,
    route_mode: RouteMode,
    max_new_tokens: int,
    route_threshold: float = 0.5,
    symbol_override: torch.Tensor | None = None,
) -> dict[str, object]:
    input_ids, attention_mask, context = _generation_context(
        bundle,
        example,
        route_mode=route_mode,
        route_threshold=route_threshold,
        symbol_override=symbol_override,
    )
    wrapper.set_context(context)
    started = time.perf_counter()
    outputs = bundle.model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    hidden = outputs.last_hidden_state[:, -1, :]
    generated: list[int] = []
    for step in range(max_new_tokens):
        logits = bundle.model.lm_head(hidden)
        next_token = logits.argmax(dim=-1)
        token_id = int(next_token.item())
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
    probability = (
        float(context.route_probabilities[0].item())
        if context.route_probabilities is not None
        else None
    )
    active = (
        bool(context.route_active[0].item())
        if context.route_active is not None
        else None
    )
    return _semantic_result(
        example=example,
        generated_text=text,
        generated_token_ids=generated,
        latency_seconds=elapsed,
        route_probability=probability,
        route_active=active,
    )


@torch.inference_mode()
def generate_semantic_control(
    bundle: ModelBundle,
    wrapper: SemanticLearnedControlLayer,
    example: SemanticPromptExample,
    *,
    route_mode: RouteMode,
    max_new_tokens: int,
    route_threshold: float = 0.5,
) -> dict[str, object]:
    input_ids = torch.tensor(
        [chat_prompt_ids(bundle.tokenizer, example.prompt)],
        dtype=torch.long,
        device=bundle.device,
    )
    attention_mask = torch.ones_like(input_ids)
    context = SemanticControlContext(
        generation=True,
        route_mode=route_mode,
        route_threshold=route_threshold,
    )
    wrapper.set_context(context)
    started = time.perf_counter()
    outputs = bundle.model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    hidden = outputs.last_hidden_state[:, -1, :]
    generated: list[int] = []
    for _ in range(max_new_tokens):
        logits = bundle.model.lm_head(hidden)
        next_token = logits.argmax(dim=-1)
        token_id = int(next_token.item())
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
    probability = (
        float(context.route_probabilities[0].item())
        if context.route_probabilities is not None
        else None
    )
    active = (
        bool(context.route_active[0].item())
        if context.route_active is not None
        else None
    )
    return _semantic_result(
        example=example,
        generated_text=text,
        generated_token_ids=generated,
        latency_seconds=elapsed,
        route_probability=probability,
        route_active=active,
    )
