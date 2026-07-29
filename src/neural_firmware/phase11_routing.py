from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch
from torch import nn

from neural_firmware.phase7_sequence_implant import SequenceNeuronImplantMLP
from neural_firmware.phase7_sequence_training import balanced_sample
from neural_firmware.phase7_training import (
    padded_batch,
    select_route_threshold,
    set_phase7_seed,
)
from neural_firmware.pretrained_data import chat_prompt_ids_and_content_mask
from neural_firmware.pretrained_training import ModelBundle
from neural_firmware.semantic_data import SemanticPromptExample

REQUEST_ROUTER_KINDS = (
    "last",
    "sequence_mean",
    "user_mean",
    "user_tail_mean",
)


@dataclass(frozen=True)
class RequestRouteFeatureSet:
    hidden: torch.Tensor
    targets: torch.Tensor

    @property
    def rows(self) -> int:
        return int(self.hidden.shape[0])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"hidden": self.hidden, "targets": self.targets}

    @classmethod
    def load_state_dict(
        cls,
        state: dict[str, torch.Tensor],
    ) -> RequestRouteFeatureSet:
        return cls(hidden=state["hidden"], targets=state["targets"])


@dataclass(frozen=True)
class RequestRouterTrainConfig:
    seed: int
    steps: int = 2_500
    batch_size: int = 256
    learning_rate: float = 0.0005
    route_temperature: float = 2.0
    maximum_calibration_false_positive_rate: float = 0.01


@torch.inference_mode()
def collect_request_route_feature_bank(
    bundle: ModelBundle,
    examples: list[SemanticPromptExample],
    *,
    layer_index: int,
    representation_checkpoints: dict[int, dict[str, object]],
    batch_size: int = 8,
    tail_tokens: int = 8,
) -> dict[int, dict[str, RequestRouteFeatureSet]]:
    mlp = bundle.model.model.layers[layer_index].mlp
    if isinstance(mlp, SequenceNeuronImplantMLP):
        raise ValueError("feature collection requires an unmodified MLP")
    captured: list[torch.Tensor] = []
    feature_rows: dict[int, dict[str, list[torch.Tensor]]] = {
        seed: {kind: [] for kind in REQUEST_ROUTER_KINDS}
        for seed in representation_checkpoints
    }
    adapter_tensors: dict[int, tuple[torch.Tensor, torch.Tensor, int]] = {}
    for seed, checkpoint in representation_checkpoints.items():
        rank = int(checkpoint["representation_rank"])
        if rank < 1:
            raise ValueError("request feature collection requires an adapter")
        adapter_tensors[seed] = (
            checkpoint["representation_down"].to(bundle.device),
            checkpoint["representation_up"].to(bundle.device),
            rank,
        )

    def capture_input(
        _module: nn.Module,
        arguments: tuple[torch.Tensor, ...],
    ) -> None:
        captured.append(arguments[0].detach())

    handle = mlp.register_forward_pre_hook(capture_input)
    try:
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            encoded = [
                chat_prompt_ids_and_content_mask(
                    bundle.tokenizer,
                    example.prompt,
                )
                for example in batch
            ]
            prompt_ids = [item[0] for item in encoded]
            content_masks = [item[1] for item in encoded]
            input_ids, attention_mask = padded_batch(
                prompt_ids,
                pad_token_id=bundle.tokenizer.pad_token_id,
                device=bundle.device,
            )
            user_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
            for row, mask in enumerate(content_masks):
                user_mask[row, : len(mask)] = torch.tensor(
                    mask,
                    dtype=torch.bool,
                    device=bundle.device,
                )
            captured.clear()
            bundle.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            if len(captured) != 1:
                raise RuntimeError("expected one request feature capture")
            hidden = captured[0]
            sequence_mask = attention_mask.to(torch.bool)
            for seed, (down, up, rank) in adapter_tensors.items():
                adapted = adapt_request_hidden(
                    hidden,
                    down=down,
                    up=up,
                    rank=rank,
                )
                for kind in REQUEST_ROUTER_KINDS:
                    feature_rows[seed][kind].append(
                        pool_request_hidden(
                            adapted,
                            kind=kind,
                            sequence_mask=sequence_mask,
                            user_mask=user_mask,
                            tail_tokens=tail_tokens,
                        ).cpu()
                    )
                del adapted
            captured.clear()
            del input_ids, attention_mask, user_mask, hidden
            if bundle.device.type == "mps" and (start // batch_size) % 16 == 15:
                torch.mps.empty_cache()
    finally:
        handle.remove()
    targets = torch.tensor(
        [int(example.route_label) for example in examples],
        dtype=torch.long,
    )
    return {
        seed: {
            kind: RequestRouteFeatureSet(
                hidden=torch.cat(rows),
                targets=targets.clone(),
            )
            for kind, rows in per_kind.items()
        }
        for seed, per_kind in feature_rows.items()
    }


def pool_request_hidden(
    hidden: torch.Tensor,
    *,
    kind: str,
    sequence_mask: torch.Tensor,
    user_mask: torch.Tensor,
    tail_tokens: int = 8,
) -> torch.Tensor:
    if kind not in REQUEST_ROUTER_KINDS:
        raise ValueError(f"unknown request router kind: {kind}")
    if hidden.shape[:-1] != sequence_mask.shape:
        raise ValueError("sequence mask does not match request hidden states")
    if user_mask.shape != sequence_mask.shape:
        raise ValueError("user mask does not match sequence mask")
    if tail_tokens < 1:
        raise ValueError("tail token count must be positive")
    if kind == "last":
        positions = sequence_mask.sum(dim=-1).clamp_min(1) - 1
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[rows, positions].float()
    pool_mask = sequence_mask if kind == "sequence_mean" else user_mask
    if kind == "user_tail_mean":
        selected = torch.zeros_like(pool_mask)
        for row in range(pool_mask.shape[0]):
            positions = torch.where(pool_mask[row])[0]
            selected[row, positions[-tail_tokens:]] = True
        pool_mask = selected
    if not bool(pool_mask.any(dim=-1).all()):
        raise ValueError("every request needs at least one pooled token")
    expanded = pool_mask.unsqueeze(-1)
    return (
        (hidden.float() * expanded).sum(dim=1)
        / pool_mask.sum(dim=1, keepdim=True).clamp_min(1)
    )


def adapt_request_hidden(
    hidden: torch.Tensor,
    *,
    down: torch.Tensor,
    up: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    if rank < 1:
        raise ValueError("representation rank must be positive")
    residual = nn.functional.linear(hidden.float(), down.float())
    residual = nn.functional.silu(residual)
    residual = nn.functional.linear(residual, up.float())
    return hidden.float() + residual / rank


def request_route_probabilities(
    rows: torch.Tensor,
    features: RequestRouteFeatureSet,
    *,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("route temperature must be positive")
    with torch.inference_mode():
        logits = nn.functional.linear(
            features.hidden.float(),
            rows.float(),
        )
        return (logits / temperature).softmax(dim=-1)[..., 1].cpu()


def evaluate_request_router(
    rows: torch.Tensor,
    features: RequestRouteFeatureSet,
    *,
    threshold: float,
    temperature: float,
) -> dict[str, object]:
    probabilities = request_route_probabilities(
        rows,
        features,
        temperature=temperature,
    )
    targets = features.targets.to(torch.bool)
    predictions = probabilities >= threshold
    positives = targets
    negatives = ~targets
    return {
        "rows": features.rows,
        "positive_rows": int(positives.sum()),
        "negative_rows": int(negatives.sum()),
        "threshold": float(threshold),
        "true_positive_rate": float(
            predictions[positives].float().mean()
        ),
        "false_positive_rate": float(
            predictions[negatives].float().mean()
        ),
        "accuracy": float((predictions == targets).float().mean()),
        "positive_probability_min": float(probabilities[positives].min()),
        "positive_probability_median": float(
            probabilities[positives].median()
        ),
        "negative_probability_median": float(
            probabilities[negatives].median()
        ),
        "negative_probability_max": float(probabilities[negatives].max()),
        "predictions": predictions,
        "probabilities": probabilities,
    }


def train_request_router(
    initial_rows: torch.Tensor,
    train: RequestRouteFeatureSet,
    calibration: RequestRouteFeatureSet,
    *,
    device: torch.device,
    config: RequestRouterTrainConfig,
) -> tuple[torch.Tensor, dict[str, object], dict[str, object]]:
    if initial_rows.shape != (2, train.hidden.shape[1]):
        raise ValueError("request route rows do not match feature width")
    if calibration.hidden.shape[1] != train.hidden.shape[1]:
        raise ValueError("training and calibration widths differ")
    set_phase7_seed(config.seed)
    rows = nn.Parameter(initial_rows.detach().clone().to(device).float())
    optimizer = torch.optim.AdamW(
        [rows],
        lr=config.learning_rate,
        weight_decay=0.0,
    )
    generator = torch.Generator().manual_seed(config.seed)
    positives = torch.where(train.targets == 1)[0]
    negatives = torch.where(train.targets == 0)[0]
    initial_loss = float("nan")
    final_loss = float("nan")
    started = time.perf_counter()
    for step in range(config.steps):
        indices = balanced_sample(
            positives,
            negatives,
            count=config.batch_size,
            generator=generator,
        )
        logits = nn.functional.linear(
            train.hidden[indices].to(device).float(),
            rows,
        )
        loss = nn.functional.cross_entropy(
            logits,
            train.targets[indices].to(device),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0:
            initial_loss = float(loss.detach().cpu())
        final_loss = float(loss.detach().cpu())
    trained = rows.detach().cpu()
    calibration_probabilities = request_route_probabilities(
        trained,
        calibration,
        temperature=config.route_temperature,
    )
    threshold = select_route_threshold(
        calibration_probabilities,
        calibration.targets,
        maximum_false_positive_rate=(
            config.maximum_calibration_false_positive_rate
        ),
    )
    calibration_metrics = evaluate_request_router(
        trained,
        calibration,
        threshold=float(threshold["threshold"]),
        temperature=config.route_temperature,
    )
    compact_calibration = {
        key: value
        for key, value in calibration_metrics.items()
        if key not in {"predictions", "probabilities"}
    }
    return (
        trained,
        {
            "config": asdict(config),
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "wall_time_seconds": time.perf_counter() - started,
            "trainable_parameters": trained.numel(),
        },
        compact_calibration,
    )


def phase11_checkpoint_state(
    source: dict[str, object],
    *,
    router_kind: str,
    request_route_rows: torch.Tensor,
    request_route_threshold: float,
    request_route_temperature: float,
    request_tail_tokens: int = 8,
) -> dict[str, object]:
    if router_kind not in REQUEST_ROUTER_KINDS:
        raise ValueError(f"unknown request router kind: {router_kind}")
    if request_route_rows.shape != (2, source["input_rows"].shape[1]):
        raise ValueError("request route rows do not match source hidden width")
    return {
        **source,
        "request_router_kind": router_kind,
        "request_route_rows": request_route_rows.detach().cpu(),
        "request_route_threshold": float(request_route_threshold),
        "request_route_temperature": float(request_route_temperature),
        "request_tail_tokens": int(request_tail_tokens),
    }
