from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from neural_firmware.pretrained_data import (
    AdditionExample,
    answer_token_ids,
    chat_prompt_ids,
)
from neural_firmware.pretrained_model import (
    BridgeConfig,
    FirmwareBridge,
    install_all_layer_lora,
    install_last_layer_lora,
)


@dataclass(frozen=True)
class ModelBundle:
    model: nn.Module
    tokenizer: object
    device: torch.device
    model_id: str


@dataclass(frozen=True)
class HiddenCache:
    arithmetic_hidden: torch.Tensor
    symbols: torch.Tensor
    target_token_ids: torch.Tensor
    negative_hidden: torch.Tensor

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "arithmetic_hidden": self.arithmetic_hidden,
                "symbols": self.symbols,
                "target_token_ids": self.target_token_ids,
                "negative_hidden": self.negative_hidden,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> HiddenCache:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return cls(**payload)


@dataclass(frozen=True)
class BridgeTrainConfig:
    seed: int = 17
    steps: int = 120
    batch_size: int = 32
    learning_rate: float = 0.01
    strength: float = 32.0
    router_loss_weight: float = 0.25


@dataclass(frozen=True)
class BridgeTrainResult:
    config: dict[str, int | float]
    trainable_parameters: int
    initial_loss: float
    final_loss: float
    final_token_loss: float
    final_router_loss: float
    wall_time_seconds: float


@dataclass(frozen=True)
class AdapterTrainConfig:
    seed: int = 17
    steps: int = 240
    batch_size: int = 2
    learning_rate: float = 0.001
    rank: int = 8
    alpha: float = 16.0
    all_layers: bool = False


@dataclass(frozen=True)
class AdapterTrainResult:
    config: dict[str, int | float]
    trainable_parameters: int
    initial_loss: float
    final_loss: float
    wall_time_seconds: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model_bundle(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    *,
    revision: str | None = None,
    device_name: str | None = None,
) -> ModelBundle:
    if device_name is None:
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float32,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    model.to(device)
    return ModelBundle(model=model, tokenizer=tokenizer, device=device, model_id=model_id)


def _padded_batch(
    sequences: list[list[int]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), max_length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[row, :length] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[row, :length] = 1
    return input_ids, attention_mask


@torch.inference_mode()
def collect_hidden_cache(
    bundle: ModelBundle,
    arithmetic_examples: list[AdditionExample],
    negative_prompts: list[str],
    *,
    batch_size: int = 8,
) -> HiddenCache:
    model = bundle.model
    tokenizer = bundle.tokenizer
    device = bundle.device
    arithmetic_hidden: list[torch.Tensor] = []
    symbols: list[int] = []
    target_ids: list[int] = []
    zero_token_id = tokenizer.encode("0", add_special_tokens=False)[0]

    for start in range(0, len(arithmetic_examples), batch_size):
        batch = arithmetic_examples[start : start + batch_size]
        sequences: list[list[int]] = []
        positions_by_row: list[list[int]] = []
        batch_answers: list[list[int]] = []
        for example in batch:
            prompt_ids = chat_prompt_ids(tokenizer, example.prompt)
            targets = answer_token_ids(tokenizer, example.answer)
            sequences.append(prompt_ids + targets[:-1])
            positions_by_row.append(
                list(range(len(prompt_ids) - 1, len(prompt_ids) - 1 + len(targets)))
            )
            batch_answers.append(targets)
        input_ids, attention_mask = _padded_batch(
            sequences,
            pad_token_id=tokenizer.pad_token_id,
            device=device,
        )
        hidden = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        for row, (positions, targets) in enumerate(
            zip(positions_by_row, batch_answers, strict=True)
        ):
            arithmetic_hidden.append(hidden[row, positions].float().cpu())
            symbols.extend(
                [token_id - zero_token_id for token_id in targets[:-1]]
                + [10]
            )
            target_ids.extend(targets)

    negative_hidden: list[torch.Tensor] = []
    for start in range(0, len(negative_prompts), batch_size):
        batch_prompts = negative_prompts[start : start + batch_size]
        sequences = [chat_prompt_ids(tokenizer, prompt) for prompt in batch_prompts]
        input_ids, attention_mask = _padded_batch(
            sequences,
            pad_token_id=tokenizer.pad_token_id,
            device=device,
        )
        hidden = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        final_positions = attention_mask.sum(dim=1) - 1
        row_indices = torch.arange(len(batch_prompts), device=device)
        negative_hidden.append(
            hidden[row_indices, final_positions].float().cpu()
        )

    return HiddenCache(
        arithmetic_hidden=torch.cat(arithmetic_hidden, dim=0),
        symbols=torch.tensor(symbols, dtype=torch.long),
        target_token_ids=torch.tensor(target_ids, dtype=torch.long),
        negative_hidden=torch.cat(negative_hidden, dim=0),
    )


def initialize_bridge_from_output_head(
    bridge: FirmwareBridge,
    bundle: ModelBundle,
) -> None:
    tokenizer = bundle.tokenizer
    token_ids = [
        tokenizer.encode(str(digit), add_special_tokens=False)[0] for digit in range(10)
    ] + [tokenizer.eos_token_id]
    with torch.no_grad():
        vectors = bundle.model.lm_head.weight[token_ids].detach().float()
        bridge.symbol_vectors.weight.copy_(nn.functional.normalize(vectors, dim=-1))


def train_bridge(
    bundle: ModelBundle,
    cache: HiddenCache,
    config: BridgeTrainConfig,
) -> tuple[FirmwareBridge, BridgeTrainResult]:
    set_seed(config.seed)
    bridge = FirmwareBridge(
        BridgeConfig(
            hidden_size=bundle.model.config.hidden_size,
            strength=config.strength,
        )
    ).to(bundle.device)
    initialize_bridge_from_output_head(bridge, bundle)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=config.learning_rate)
    token_loss_function = nn.CrossEntropyLoss()
    router_loss_function = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(config.seed)
    initial_loss = float("nan")
    final_token_loss = float("nan")
    final_router_loss = float("nan")
    started = time.perf_counter()

    for step in range(config.steps):
        positive_indices = torch.randint(
            len(cache.symbols),
            (config.batch_size,),
            generator=generator,
        )
        negative_indices = torch.randint(
            len(cache.negative_hidden),
            (config.batch_size,),
            generator=generator,
        )
        hidden = cache.arithmetic_hidden[positive_indices].to(bundle.device)
        symbols = cache.symbols[positive_indices].to(bundle.device)
        targets = cache.target_token_ids[positive_indices].to(bundle.device)
        negative_hidden = cache.negative_hidden[negative_indices].to(bundle.device)

        steered, positive_router_logits = bridge.routed_hidden(
            hidden,
            symbols,
            hard_route=False,
        )
        token_logits = bundle.model.lm_head(steered)
        token_loss = token_loss_function(token_logits.float(), targets)
        negative_router_logits = bridge.router_logits(negative_hidden)
        router_logits = torch.cat([positive_router_logits, negative_router_logits])
        router_targets = torch.cat(
            [
                torch.ones_like(positive_router_logits),
                torch.zeros_like(negative_router_logits),
            ]
        )
        router_loss = router_loss_function(router_logits, router_targets)
        loss = token_loss + config.router_loss_weight * router_loss
        if step == 0:
            initial_loss = loss.item()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_token_loss = token_loss.item()
        final_router_loss = router_loss.item()

    result = BridgeTrainResult(
        config=asdict(config),
        trainable_parameters=bridge.trainable_parameter_count,
        initial_loss=initial_loss,
        final_loss=final_token_loss + config.router_loss_weight * final_router_loss,
        final_token_loss=final_token_loss,
        final_router_loss=final_router_loss,
        wall_time_seconds=time.perf_counter() - started,
    )
    return bridge, result


def save_bridge(
    bridge: FirmwareBridge,
    result: BridgeTrainResult,
    output_directory: Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    torch.save(bridge.state_dict(), output_directory / "bridge.pt")
    (output_directory / "train_result.json").write_text(
        json.dumps(asdict(result), indent=2) + "\n"
    )


def load_bridge(
    path: Path,
    *,
    hidden_size: int,
    strength: float,
    device: torch.device,
) -> FirmwareBridge:
    bridge = FirmwareBridge(
        BridgeConfig(hidden_size=hidden_size, strength=strength)
    ).to(device)
    bridge.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    bridge.eval()
    return bridge


def _arithmetic_training_batch(
    tokenizer: object,
    examples: list[AdditionExample],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[list[int]], list[list[int]]]:
    sequences: list[list[int]] = []
    positions: list[list[int]] = []
    targets_by_row: list[list[int]] = []
    for example in examples:
        prompt_ids = chat_prompt_ids(tokenizer, example.prompt)
        targets = answer_token_ids(tokenizer, example.answer)
        sequences.append(prompt_ids + targets[:-1])
        positions.append(
            list(range(len(prompt_ids) - 1, len(prompt_ids) - 1 + len(targets)))
        )
        targets_by_row.append(targets)
    input_ids, attention_mask = _padded_batch(
        sequences,
        pad_token_id=tokenizer.pad_token_id,
        device=device,
    )
    return input_ids, attention_mask, positions, targets_by_row


def train_learned_adapter(
    bundle: ModelBundle,
    examples: list[AdditionExample],
    config: AdapterTrainConfig,
) -> AdapterTrainResult:
    set_seed(config.seed)
    installer = install_all_layer_lora if config.all_layers else install_last_layer_lora
    trainable_parameters = installer(
        bundle.model,
        rank=config.rank,
        alpha=config.alpha,
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
    )
    loss_function = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(config.seed)
    initial_loss = float("nan")
    final_loss = float("nan")
    started = time.perf_counter()
    bundle.model.train()

    for step in range(config.steps):
        indices = torch.randint(
            len(examples),
            (config.batch_size,),
            generator=generator,
        ).tolist()
        batch = [examples[index] for index in indices]
        input_ids, attention_mask, positions, targets_by_row = (
            _arithmetic_training_batch(
                bundle.tokenizer,
                batch,
                device=bundle.device,
            )
        )
        hidden = bundle.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state
        selected_hidden: list[torch.Tensor] = []
        selected_targets: list[int] = []
        for row, (row_positions, row_targets) in enumerate(
            zip(positions, targets_by_row, strict=True)
        ):
            selected_hidden.append(hidden[row, row_positions])
            selected_targets.extend(row_targets)
        answer_hidden = torch.cat(selected_hidden, dim=0)
        targets = torch.tensor(
            selected_targets,
            dtype=torch.long,
            device=bundle.device,
        )
        logits = bundle.model.lm_head(answer_hidden)
        loss = loss_function(logits.float(), targets)
        if step == 0:
            initial_loss = loss.item()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    bundle.model.eval()
    return AdapterTrainResult(
        config=asdict(config),
        trainable_parameters=sum(
            parameter.numel() for parameter in trainable_parameters
        ),
        initial_loss=initial_loss,
        final_loss=final_loss,
        wall_time_seconds=time.perf_counter() - started,
    )


def save_learned_adapter(
    bundle: ModelBundle,
    result: AdapterTrainResult,
    output_directory: Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    state = {
        name: parameter.detach().cpu()
        for name, parameter in bundle.model.state_dict().items()
        if "lora_" in name
    }
    torch.save(state, output_directory / "adapter.pt")
    (output_directory / "train_result.json").write_text(
        json.dumps(asdict(result), indent=2) + "\n"
    )


def load_learned_adapter(
    bundle: ModelBundle,
    path: Path,
    config: AdapterTrainConfig,
) -> None:
    installer = install_all_layer_lora if config.all_layers else install_last_layer_lora
    installer(
        bundle.model,
        rank=config.rank,
        alpha=config.alpha,
    )
    state = torch.load(path, map_location=bundle.device, weights_only=True)
    missing, unexpected = bundle.model.load_state_dict(state, strict=False)
    unexpected_non_lora = [name for name in unexpected if "lora_" not in name]
    if unexpected_non_lora:
        raise ValueError(f"unexpected adapter keys: {unexpected_non_lora}")
    if not any("lora_" in name for name in state):
        raise ValueError("adapter checkpoint contains no low-rank weights")
    bundle.model.eval()
