from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from neural_firmware.phase6_data import Phase6Example
from neural_firmware.phase6_firmware import (
    PAD_DIGIT,
    NeuralCallController,
    NeuralFirmwareContext,
    NeuralFirmwareInstallation,
    NeuralRegisterMapper,
    Phase6RouteMode,
)
from neural_firmware.pretrained_data import answer_token_ids, chat_prompt_ids
from neural_firmware.pretrained_training import ModelBundle
from neural_firmware.semantic_data import (
    exact_format_correct,
    mathematical_correct,
)


def set_phase6_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _padded_sequences(
    sequences: list[list[int] | tuple[int, ...]],
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


def register_targets(
    examples: list[Phase6Example],
    *,
    max_digits: int,
) -> torch.Tensor:
    targets = torch.full(
        (len(examples), 3, max_digits),
        PAD_DIGIT,
        dtype=torch.long,
    )
    for row, example in enumerate(examples):
        for operand_index, operand in enumerate(example.operands):
            if len(operand) > max_digits:
                raise ValueError("operand exceeds configured register width")
            targets[row, operand_index, : len(operand)] = torch.tensor(
                [int(character) for character in operand],
            )
    return targets


@dataclass
class Phase6FeatureSet:
    early_hidden: torch.Tensor
    attention_mask: torch.Tensor
    anchor_positions: torch.Tensor
    late_hidden: torch.Tensor
    register_targets: torch.Tensor
    call_targets: torch.Tensor

    @property
    def examples(self) -> int:
        return self.early_hidden.shape[0]

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "early_hidden": self.early_hidden,
            "attention_mask": self.attention_mask,
            "anchor_positions": self.anchor_positions,
            "late_hidden": self.late_hidden,
            "register_targets": self.register_targets,
            "call_targets": self.call_targets,
        }


@torch.inference_mode()
def collect_phase6_features(
    bundle: ModelBundle,
    examples: list[Phase6Example],
    *,
    input_depth_after_blocks: int,
    output_depth_after_blocks: int,
    max_digits: int,
    batch_size: int,
) -> Phase6FeatureSet:
    encoded = [
        chat_prompt_ids(bundle.tokenizer, example.prompt)
        for example in examples
    ]
    maximum = max(map(len, encoded))
    hidden_size = bundle.model.config.hidden_size
    early_cache = torch.zeros(
        (len(examples), maximum, hidden_size),
        dtype=torch.float16,
    )
    mask_cache = torch.zeros((len(examples), maximum), dtype=torch.long)
    anchors = torch.zeros(len(examples), dtype=torch.long)
    late_cache = torch.zeros((len(examples), hidden_size), dtype=torch.float32)
    late_layer = bundle.model.model.layers[output_depth_after_blocks - 1]
    captured: list[torch.Tensor] = []

    def capture_output(
        module: nn.Module,
        inputs: tuple[object, ...],
        output: torch.Tensor,
    ) -> None:
        del module, inputs
        captured.append(output.detach())

    handle = late_layer.register_forward_hook(capture_output)
    try:
        for start in range(0, len(examples), batch_size):
            sequences = encoded[start : start + batch_size]
            input_ids, attention_mask = _padded_sequences(
                sequences,
                pad_token_id=bundle.tokenizer.pad_token_id,
                device=bundle.device,
            )
            captured.clear()
            outputs = bundle.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
            )
            if len(captured) != 1:
                raise RuntimeError("late feature hook did not fire exactly once")
            early = outputs.hidden_states[input_depth_after_blocks]
            width = early.shape[1]
            stop = start + len(sequences)
            early_cache[start:stop, :width] = early.float().cpu().half()
            mask_cache[start:stop, :width] = attention_mask.cpu()
            batch_anchors = attention_mask.sum(dim=1) - 1
            anchors[start:stop] = batch_anchors.cpu()
            rows = torch.arange(len(sequences), device=bundle.device)
            late_cache[start:stop] = (
                captured[0][rows, batch_anchors].float().cpu()
            )
    finally:
        handle.remove()
    return Phase6FeatureSet(
        early_hidden=early_cache,
        attention_mask=mask_cache,
        anchor_positions=anchors,
        late_hidden=late_cache,
        register_targets=register_targets(examples, max_digits=max_digits),
        call_targets=torch.tensor(
            [example.call_count for example in examples],
            dtype=torch.long,
        ),
    )


@dataclass(frozen=True)
class MapperTrainConfig:
    seed: int
    steps: int
    batch_size: int
    learning_rate: float
    pad_loss_weight: float = 0.25
    weight_decay: float = 0.01


def train_register_mapper(
    mapper: NeuralRegisterMapper,
    features: Phase6FeatureSet,
    config: MapperTrainConfig,
    *,
    device: torch.device,
) -> dict[str, object]:
    set_phase6_seed(config.seed)
    mapper.to(device)
    mapper.train()
    parameters = list(mapper.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    class_weights = torch.ones(11, device=device)
    class_weights[PAD_DIGIT] = config.pad_loss_weight
    generator = torch.Generator().manual_seed(config.seed)
    initial_loss = float("nan")
    final_loss = float("nan")
    started = time.perf_counter()
    for step in range(config.steps):
        indices = torch.randint(
            features.examples,
            (config.batch_size,),
            generator=generator,
        )
        logits = mapper(
            features.early_hidden[indices].to(device).float(),
            features.attention_mask[indices].to(device),
            features.anchor_positions[indices].to(device),
        )
        targets = features.register_targets[indices].to(device)
        loss = nn.functional.cross_entropy(
            logits.flatten(0, 2),
            targets.flatten(),
            weight=class_weights,
        )
        if step == 0:
            initial_loss = float(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        optimizer.step()
        final_loss = float(loss.item())
    if device.type == "mps":
        torch.mps.synchronize()
    mapper.eval()
    return {
        "config": asdict(config),
        "trainable_parameters": sum(
            parameter.numel() for parameter in parameters
        ),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "wall_time_seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def evaluate_register_mapper(
    mapper: NeuralRegisterMapper,
    features: Phase6FeatureSet,
    examples: list[Phase6Example],
    *,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, object]:
    predictions = []
    for start in range(0, features.examples, batch_size):
        stop = min(start + batch_size, features.examples)
        logits = mapper(
            features.early_hidden[start:stop].to(device).float(),
            features.attention_mask[start:stop].to(device),
            features.anchor_positions[start:stop].to(device),
        )
        predictions.append(logits.argmax(dim=-1).cpu())
    predicted = torch.cat(predictions)
    exact_by_register = (predicted == features.register_targets).all(dim=2)
    rows = []
    for index, example in enumerate(examples):
        used = len(example.operands)
        used_exact = bool(exact_by_register[index, :used].all())
        rows.append(
            {
                "prompt": example.prompt,
                "split": example.split,
                "call_count": example.call_count,
                "used_registers": used,
                "used_registers_exact": used_exact,
                "register_exact": exact_by_register[index].tolist(),
                "predictions": predicted[index].tolist(),
            }
        )
    positive_rows = [row for row in rows if row["call_count"] > 0]
    single_rows = [row for row in rows if row["call_count"] == 1]
    chain_rows = [row for row in rows if row["call_count"] == 2]
    return {
        "examples": len(rows),
        "used_registers_exact": sum(
            row["used_registers_exact"] for row in rows
        ),
        "used_register_accuracy": sum(
            row["used_registers_exact"] for row in rows
        )
        / len(rows),
        "positive_used_register_accuracy": sum(
            row["used_registers_exact"] for row in positive_rows
        )
        / len(positive_rows),
        "single_used_register_accuracy": sum(
            row["used_registers_exact"] for row in single_rows
        )
        / len(single_rows),
        "chain_used_register_accuracy": sum(
            row["used_registers_exact"] for row in chain_rows
        )
        / len(chain_rows),
        "digit_slot_accuracy": float(
            (predicted == features.register_targets).float().mean().item()
        ),
        "rows": rows,
    }


@dataclass(frozen=True)
class ControllerTrainConfig:
    seed: int
    steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float = 0.01


def train_call_controller(
    controller: NeuralCallController,
    features: Phase6FeatureSet,
    config: ControllerTrainConfig,
    *,
    device: torch.device,
) -> dict[str, object]:
    set_phase6_seed(config.seed)
    controller.to(device)
    controller.train()
    parameters = list(controller.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    counts = torch.bincount(features.call_targets, minlength=3).float()
    class_weights = counts.sum() / (3 * counts.clamp_min(1))
    class_weights = class_weights.to(device)
    generator = torch.Generator().manual_seed(config.seed)
    initial_loss = float("nan")
    final_loss = float("nan")
    started = time.perf_counter()
    for step in range(config.steps):
        indices = torch.randint(
            features.examples,
            (config.batch_size,),
            generator=generator,
        )
        logits = controller(features.late_hidden[indices].to(device))
        targets = features.call_targets[indices].to(device)
        loss = nn.functional.cross_entropy(
            logits,
            targets,
            weight=class_weights,
        )
        if step == 0:
            initial_loss = float(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
    if device.type == "mps":
        torch.mps.synchronize()
    controller.eval()
    return {
        "config": asdict(config),
        "trainable_parameters": sum(
            parameter.numel() for parameter in parameters
        ),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "wall_time_seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def evaluate_call_controller(
    controller: NeuralCallController,
    features: Phase6FeatureSet,
    examples: list[Phase6Example],
    *,
    device: torch.device,
    threshold: float,
) -> dict[str, object]:
    logits = controller(features.late_hidden.to(device)).cpu()
    probabilities = torch.softmax(logits, dim=-1)
    route_probabilities = 1 - probabilities[:, 0]
    positive_counts = probabilities[:, 1:].argmax(dim=-1) + 1
    predictions = torch.where(
        route_probabilities >= threshold,
        positive_counts,
        torch.zeros_like(positive_counts),
    )
    rows = [
        {
            "prompt": example.prompt,
            "family": example.family,
            "split": example.split,
            "target_call_count": example.call_count,
            "predicted_call_count": int(predictions[index].item()),
            "route_probability": float(route_probabilities[index].item()),
            "call_count_exact": (
                int(predictions[index].item()) == example.call_count
            ),
        }
        for index, example in enumerate(examples)
    ]
    positives = [row for row in rows if row["target_call_count"] > 0]
    negatives = [row for row in rows if row["target_call_count"] == 0]
    return {
        "examples": len(rows),
        "call_count_exact": sum(row["call_count_exact"] for row in rows),
        "call_count_accuracy": sum(row["call_count_exact"] for row in rows)
        / len(rows),
        "positive_call_count_accuracy": sum(
            row["call_count_exact"] for row in positives
        )
        / len(positives),
        "false_calls": sum(row["predicted_call_count"] > 0 for row in negatives),
        "false_call_rate": sum(
            row["predicted_call_count"] > 0 for row in negatives
        )
        / len(negatives),
        "rows": rows,
    }


def select_call_threshold(
    controller: NeuralCallController,
    features: Phase6FeatureSet,
    examples: list[Phase6Example],
    *,
    device: torch.device,
    maximum_false_call_rate: float = 0.01,
) -> dict[str, float | int]:
    candidates = []
    for integer in range(50, 100):
        threshold = integer / 100
        evaluation = evaluate_call_controller(
            controller,
            features,
            examples,
            device=device,
            threshold=threshold,
        )
        candidates.append(
            {
                "threshold": threshold,
                "call_count_accuracy": evaluation["call_count_accuracy"],
                "positive_call_count_accuracy": evaluation[
                    "positive_call_count_accuracy"
                ],
                "false_calls": evaluation["false_calls"],
                "false_call_rate": evaluation["false_call_rate"],
            }
        )
    eligible = [
        row
        for row in candidates
        if row["false_call_rate"] <= maximum_false_call_rate
    ]
    pool = eligible or candidates
    selected = max(
        pool,
        key=lambda row: (
            row["positive_call_count_accuracy"],
            row["call_count_accuracy"],
            -row["false_call_rate"],
            -row["threshold"],
        ),
    )
    return selected


@dataclass(frozen=True)
class OutputTrainConfig:
    seed: int
    steps: int
    batch_size: int
    learning_rate: float


def initialize_output_decoder(
    bundle: ModelBundle,
    installation: NeuralFirmwareInstallation,
) -> None:
    token_ids = [
        bundle.tokenizer.encode(str(digit), add_special_tokens=False)[0]
        for digit in range(10)
    ] + [bundle.tokenizer.eos_token_id]
    with torch.no_grad():
        vectors = bundle.model.lm_head.weight[token_ids].detach().float()
        installation.final.output_decoder.codebook.weight.copy_(
            nn.functional.normalize(vectors, dim=-1)
        )


def _make_output_batch(
    bundle: ModelBundle,
    examples: list[Phase6Example],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, NeuralFirmwareContext]:
    if any(example.answer is None for example in examples):
        raise ValueError("output training requires positive examples")
    prompt_ids = [
        chat_prompt_ids(bundle.tokenizer, example.prompt)
        for example in examples
    ]
    targets = [
        answer_token_ids(bundle.tokenizer, example.answer or "")
        for example in examples
    ]
    sequences = [
        prompt + target[:-1]
        for prompt, target in zip(prompt_ids, targets, strict=True)
    ]
    input_ids, attention_mask = _padded_sequences(
        sequences,
        pad_token_id=bundle.tokenizer.pad_token_id,
        device=bundle.device,
    )
    maximum = max(map(len, targets))
    output_positions = torch.zeros(
        (len(examples), maximum),
        dtype=torch.long,
        device=bundle.device,
    )
    symbols = torch.full(
        (len(examples), maximum),
        -1,
        dtype=torch.long,
        device=bundle.device,
    )
    symbol_mask = torch.zeros_like(symbols, dtype=torch.bool)
    target_ids = torch.full(
        (len(examples), maximum),
        bundle.tokenizer.eos_token_id,
        dtype=torch.long,
        device=bundle.device,
    )
    for row, (example, prompt, target) in enumerate(
        zip(examples, prompt_ids, targets, strict=True)
    ):
        answer = example.answer or ""
        row_symbols = [int(character) for character in answer] + [10]
        positions = torch.arange(
            len(prompt) - 1,
            len(prompt) - 1 + len(target),
            device=bundle.device,
        )
        output_positions[row, : len(target)] = positions
        symbols[row, : len(row_symbols)] = torch.tensor(
            row_symbols,
            device=bundle.device,
        )
        symbol_mask[row, : len(row_symbols)] = True
        target_ids[row, : len(target)] = torch.tensor(
            target,
            device=bundle.device,
        )
    context = NeuralFirmwareContext(
        attention_mask=attention_mask,
        anchor_positions=torch.tensor(
            [len(prompt) - 1 for prompt in prompt_ids],
            device=bundle.device,
        ),
        output_positions=output_positions,
        teacher_symbols=symbols,
        teacher_symbol_mask=symbol_mask,
    )
    return input_ids, attention_mask, target_ids, context


def train_output_decoder(
    bundle: ModelBundle,
    installation: NeuralFirmwareInstallation,
    examples: list[Phase6Example],
    config: OutputTrainConfig,
) -> dict[str, object]:
    set_phase6_seed(config.seed)
    for parameter in bundle.model.parameters():
        parameter.requires_grad_(False)
    for parameter in installation.final.output_decoder.parameters():
        parameter.requires_grad_(True)
    initialize_output_decoder(bundle, installation)
    parameters = list(installation.final.output_decoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
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
        batch_examples = [examples[index] for index in indices]
        input_ids, attention_mask, target_ids, context = _make_output_batch(
            bundle,
            batch_examples,
        )
        installation.set_context(context)
        outputs = bundle.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = bundle.model.lm_head(outputs.last_hidden_state)
        mask = context.teacher_symbol_mask
        if mask is None or context.output_positions is None:
            raise RuntimeError("output training context is incomplete")
        batch_grid = torch.arange(
            len(batch_examples),
            device=bundle.device,
        )[:, None].expand_as(target_ids)
        selected_logits = logits[
            batch_grid[mask],
            context.output_positions[mask],
        ]
        loss = nn.functional.cross_entropy(
            selected_logits.float(),
            target_ids[mask],
        )
        if step == 0:
            initial_loss = float(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
    installation.set_context(None)
    if bundle.device.type == "mps":
        torch.mps.synchronize()
    return {
        "config": asdict(config),
        "trainable_parameters": sum(
            parameter.numel() for parameter in parameters
        ),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "wall_time_seconds": time.perf_counter() - started,
    }


def _targets_for_example(
    example: Phase6Example,
    *,
    max_digits: int,
) -> torch.Tensor:
    return register_targets([example], max_digits=max_digits)[0]


@torch.inference_mode()
def generate_neural_firmware(
    bundle: ModelBundle,
    installation: NeuralFirmwareInstallation,
    example: Phase6Example,
    *,
    route_mode: Phase6RouteMode,
    route_threshold: float,
    max_new_tokens: int,
) -> dict[str, object]:
    prompt = chat_prompt_ids(bundle.tokenizer, example.prompt)
    input_ids = torch.tensor([prompt], dtype=torch.long, device=bundle.device)
    attention_mask = torch.ones_like(input_ids)
    context = NeuralFirmwareContext(
        attention_mask=attention_mask,
        anchor_positions=torch.tensor(
            [len(prompt) - 1],
            device=bundle.device,
        ),
        generation_index=0,
        route_mode=route_mode,
        route_threshold=route_threshold,
    )
    installation.set_context(context)
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
            (
                attention_mask,
                torch.ones(
                    (1, 1),
                    dtype=torch.long,
                    device=bundle.device,
                ),
            ),
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
    latency = time.perf_counter() - started
    installation.set_context(None)
    text = bundle.tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()
    predictions = context.diagnostics.get("register_predictions")
    targets = _targets_for_example(
        example,
        max_digits=installation.capture.mapper.max_digits,
    )
    register_exact: list[bool] | None = None
    used_registers_exact: bool | None = None
    if isinstance(predictions, torch.Tensor):
        register_exact = [
            bool(torch.equal(predictions[0, index], targets[index]))
            for index in range(3)
        ]
        used_registers_exact = all(
            register_exact[: len(example.operands)]
        )
    predicted_call_count = (
        int(context.call_counts[0].item())
        if context.call_counts is not None
        else None
    )
    return {
        "prompt": example.prompt,
        "family": example.family,
        "family_index": example.family_index,
        "split": example.split,
        "operands": list(example.operands),
        "target_call_count": example.call_count,
        "predicted_call_count": predicted_call_count,
        "call_count_exact": predicted_call_count == example.call_count,
        "expected": example.answer,
        "intermediate_answers": list(example.intermediate_answers),
        "generated_text": text,
        "generated_token_ids": generated,
        "mathematical_correct": (
            mathematical_correct(text, example.answer)
            if example.answer is not None
            else None
        ),
        "exact_format_correct": (
            exact_format_correct(text, example.answer)
            if example.answer is not None
            else None
        ),
        "route_probability": (
            float(context.route_probabilities[0].item())
            if context.route_probabilities is not None
            else None
        ),
        "route_active": (
            predicted_call_count is not None and predicted_call_count > 0
        ),
        "register_exact": register_exact,
        "used_registers_exact": used_registers_exact,
        "program_call_symbols": (
            context.program_call_symbols[0].cpu().tolist()
            if context.program_call_symbols is not None
            else None
        ),
        "program_call_masks": (
            context.program_call_masks[0].cpu().tolist()
            if context.program_call_masks is not None
            else None
        ),
        "latency_seconds": latency,
    }
