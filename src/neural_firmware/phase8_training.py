from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass

import torch
from torch import nn

from neural_firmware.phase7_training import padded_batch, set_phase7_seed
from neural_firmware.phase8_adapter import (
    MatchedResidualAdapterMLP,
    adapter_enabled,
)
from neural_firmware.pretrained_data import answer_token_ids, chat_prompt_ids
from neural_firmware.pretrained_training import ModelBundle
from neural_firmware.semantic_data import SemanticPromptExample


@dataclass(frozen=True)
class MatchedAdapterTrainConfig:
    seed: int
    steps: int = 800
    batch_size: int = 4
    learning_rate: float = 0.002


@torch.inference_mode()
def collect_base_first_tokens(
    bundle: ModelBundle,
    adapter: MatchedResidualAdapterMLP,
    examples: list[SemanticPromptExample],
    *,
    batch_size: int = 8,
) -> dict[str, int]:
    result: dict[str, int] = {}
    with adapter_enabled(adapter, False):
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            prompts = [
                chat_prompt_ids(bundle.tokenizer, example.prompt)
                for example in batch
            ]
            input_ids, attention_mask = padded_batch(
                prompts,
                pad_token_id=bundle.tokenizer.pad_token_id,
                device=bundle.device,
            )
            hidden = bundle.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).last_hidden_state
            positions = attention_mask.sum(dim=1) - 1
            rows = torch.arange(len(batch), device=bundle.device)
            tokens = bundle.model.lm_head(hidden[rows, positions]).argmax(dim=-1)
            result.update(
                {
                    example.prompt: int(token)
                    for example, token in zip(batch, tokens.tolist(), strict=True)
                }
            )
            del input_ids, attention_mask, hidden, tokens
            if bundle.device.type == "mps" and (start // batch_size) % 16 == 15:
                torch.mps.empty_cache()
    return result


def train_matched_adapter(
    bundle: ModelBundle,
    adapter: MatchedResidualAdapterMLP,
    examples: list[SemanticPromptExample],
    *,
    config: MatchedAdapterTrainConfig,
    base_first_tokens: dict[str, int] | None = None,
) -> dict[str, object]:
    positives = [example for example in examples if example.route_label]
    negatives = [example for example in examples if not example.route_label]
    if not positives or not negatives:
        raise ValueError("matched-adapter training requires both prompt classes")
    set_phase7_seed(config.seed)
    if base_first_tokens is None:
        base_first_tokens = collect_base_first_tokens(
            bundle,
            adapter,
            negatives,
        )
    missing = {
        example.prompt
        for example in negatives
        if example.prompt not in base_first_tokens
    }
    if missing:
        raise ValueError("base-token cache is missing negative training prompts")
    parameters = [adapter.down.weight, adapter.up.weight]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
    rng = random.Random(config.seed)
    initial_loss = float("nan")
    final_loss = float("nan")
    started = time.perf_counter()
    adapter.enabled = True
    bundle.model.train()
    for step in range(config.steps):
        positive_count = max(1, config.batch_size // 2)
        batch = [
            rng.choice(positives) for _ in range(positive_count)
        ] + [
            rng.choice(negatives)
            for _ in range(config.batch_size - positive_count)
        ]
        sequences: list[list[int]] = []
        positions_by_row: list[list[int]] = []
        targets_by_row: list[list[int]] = []
        for example in batch:
            prompt = chat_prompt_ids(bundle.tokenizer, example.prompt)
            if example.route_label:
                targets = answer_token_ids(
                    bundle.tokenizer,
                    example.answer or "",
                )
                sequence = prompt + targets[:-1]
                positions = list(
                    range(len(prompt) - 1, len(prompt) - 1 + len(targets))
                )
            else:
                targets = [base_first_tokens[example.prompt]]
                sequence = prompt
                positions = [len(prompt) - 1]
            sequences.append(sequence)
            positions_by_row.append(positions)
            targets_by_row.append(targets)
        input_ids, attention_mask = padded_batch(
            sequences,
            pad_token_id=bundle.tokenizer.pad_token_id,
            device=bundle.device,
        )
        hidden = bundle.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state
        selected: list[torch.Tensor] = []
        target_ids: list[int] = []
        for row, (positions, targets) in enumerate(
            zip(positions_by_row, targets_by_row, strict=True)
        ):
            selected.append(hidden[row, positions])
            target_ids.extend(targets)
        logits = bundle.model.lm_head(torch.cat(selected))
        loss = nn.functional.cross_entropy(
            logits.float(),
            torch.tensor(target_ids, dtype=torch.long, device=bundle.device),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0:
            initial_loss = float(loss.detach().cpu())
        final_loss = float(loss.detach().cpu())
        del input_ids, attention_mask, hidden, selected, logits, loss
        if bundle.device.type == "mps" and step % 25 == 24:
            torch.mps.empty_cache()
    bundle.model.eval()
    return {
        "config": asdict(config),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "trainable_parameters": adapter.trainable_parameter_count,
        "wall_time_seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def generate_matched_adapter(
    bundle: ModelBundle,
    adapter: MatchedResidualAdapterMLP,
    prompt: str,
    *,
    max_new_tokens: int,
    enabled: bool = True,
) -> dict[str, object]:
    full_ids = chat_prompt_ids(bundle.tokenizer, prompt)
    generated: list[int] = []
    with adapter_enabled(adapter, enabled):
        for _ in range(max_new_tokens):
            input_ids = torch.tensor(
                [full_ids],
                dtype=torch.long,
                device=bundle.device,
            )
            hidden = bundle.model.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
            ).last_hidden_state[:, -1]
            token = int(bundle.model.lm_head(hidden).argmax(dim=-1).item())
            generated.append(token)
            full_ids.append(token)
            del input_ids, hidden
            if bundle.device.type == "mps":
                torch.mps.empty_cache()
            if token == bundle.tokenizer.eos_token_id:
                break
    return {
        "generated_token_ids": generated,
        "generated_text": bundle.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip(),
    }
