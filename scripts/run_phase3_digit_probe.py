from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from neural_firmware.internal_data import (
    InternalAdditionExample,
    encode_internal_prompt,
    make_internal_addition_examples,
)
from neural_firmware.internal_firmware import ResidualDigitEncoder
from neural_firmware.pretrained_training import load_model_bundle


def stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def padded_batch(
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
def collect_features(
    bundle: object,
    examples: list[InternalAdditionExample],
    *,
    depths: list[int],
    batch_size: int,
) -> dict[int, dict[str, torch.Tensor]]:
    encoded = [
        (example, encode_internal_prompt(bundle.tokenizer, example.prompt))
        for example in examples
    ]
    feature_chunks: dict[int, list[torch.Tensor]] = {depth: [] for depth in depths}
    label_chunks: list[torch.Tensor] = []
    example_id_chunks: list[torch.Tensor] = []
    started = time.perf_counter()
    for start in range(0, len(encoded), batch_size):
        batch = encoded[start : start + batch_size]
        input_ids, attention_mask = padded_batch(
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
            digit_labels = [int(char) for char in example.a + example.b]
            positions.extend((local_row, position) for position in digit_positions)
            labels.extend(digit_labels)
            example_ids.extend([start + local_row] * len(digit_labels))
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
        for depth in depths:
            feature_chunks[depth].append(
                outputs.hidden_states[depth][batch_indices, token_indices]
                .float()
                .cpu()
            )
        label_chunks.append(torch.tensor(labels, dtype=torch.long))
        example_id_chunks.append(torch.tensor(example_ids, dtype=torch.long))
    labels = torch.cat(label_chunks)
    example_ids = torch.cat(example_id_chunks)
    return {
        depth: {
            "features": torch.cat(feature_chunks[depth]),
            "labels": labels,
            "example_ids": example_ids,
            "examples": torch.tensor(len(examples)),
            "extraction_seconds": torch.tensor(time.perf_counter() - started),
        }
        for depth in depths
    }


def train_probe(
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
    probe = ResidualDigitEncoder(hidden_size).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate)
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
        logits = probe(features[indices].to(device))
        loss = loss_function(logits, labels[indices].to(device))
        if step == 0:
            initial_loss = loss.item()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if device.type == "mps":
        torch.mps.synchronize()
    return probe, {
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "trainable_parameters": sum(
            parameter.numel() for parameter in probe.parameters()
        ),
        "initial_loss": initial_loss,
        "final_loss": loss.item(),
        "wall_time_seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def evaluate_probe(
    probe: ResidualDigitEncoder,
    feature_set: dict[str, torch.Tensor],
    *,
    device: torch.device,
    batch_size: int = 2_048,
) -> dict[str, float | int]:
    predictions: list[torch.Tensor] = []
    features = feature_set["features"]
    for start in range(0, len(features), batch_size):
        logits = probe(features[start : start + batch_size].to(device))
        predictions.append(logits.argmax(dim=-1).cpu())
    predicted = torch.cat(predictions)
    labels = feature_set["labels"]
    correct = predicted == labels
    example_ids = feature_set["example_ids"]
    example_count = int(feature_set["examples"].item())
    exact_examples = 0
    for example_id in range(example_count):
        exact_examples += int(bool(correct[example_ids == example_id].all()))
    return {
        "digits": len(labels),
        "correct_digits": int(correct.sum().item()),
        "digit_accuracy": float(correct.float().mean().item()),
        "examples": example_count,
        "exact_registers": exact_examples,
        "exact_register_accuracy": exact_examples / example_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument(
        "--revision",
        default="7ae557604adf67be50417f59c2c2f167def9a775",
    )
    parser.add_argument("--depths", type=int, nargs="+", default=[6, 12, 18, 22])
    parser.add_argument("--train-examples", type=int, default=1_000)
    parser.add_argument("--eval-examples", type=int, default=300)
    parser.add_argument("--train-seed", type=int, default=43_211)
    parser.add_argument("--eval-seed", type=int, default=97_013)
    parser.add_argument("--probe-seed", type=int, default=701)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--probe-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=Path("phase3_artifacts/digit_probe_pilot_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("phase3_results/digit_probe_pilot_v1.json"),
    )
    args = parser.parse_args()
    for depth in args.depths:
        if depth < 1 or depth > 24:
            raise ValueError("depths count completed blocks and must be in 1..24")

    datasets = {
        "train_1_4": make_internal_addition_examples(
            count=args.train_examples,
            min_digits=1,
            max_digits=4,
            seed=args.train_seed,
            split="train_1_4",
        ),
        "id_1_4": make_internal_addition_examples(
            count=args.eval_examples,
            min_digits=1,
            max_digits=4,
            seed=args.eval_seed,
            split="id_1_4",
        ),
        "ood_5_8": make_internal_addition_examples(
            count=args.eval_examples,
            min_digits=5,
            max_digits=8,
            seed=args.eval_seed + 1,
            split="ood_5_8",
        ),
        "ood_9_12": make_internal_addition_examples(
            count=args.eval_examples,
            min_digits=9,
            max_digits=12,
            seed=args.eval_seed + 2,
            split="ood_9_12",
        ),
    }
    dataset_hash = stable_hash(
        {
            split: [example.to_dict() for example in examples]
            for split, examples in datasets.items()
        }
    )
    bundle = load_model_bundle(args.model_id, revision=args.revision)
    args.artifact_directory.mkdir(parents=True, exist_ok=True)
    feature_sets: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    extraction_times: dict[str, float] = {}
    for split, examples in datasets.items():
        cache_path = args.artifact_directory / f"{split}_features.pt"
        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        else:
            cached = collect_features(
                bundle,
                examples,
                depths=args.depths,
                batch_size=args.batch_size,
            )
            torch.save(cached, cache_path)
        feature_sets[split] = cached
        extraction_times[split] = float(
            cached[args.depths[0]]["extraction_seconds"].item()
        )

    depth_results = []
    for depth in args.depths:
        probe, training = train_probe(
            feature_sets["train_1_4"][depth],
            hidden_size=bundle.model.config.hidden_size,
            device=bundle.device,
            seed=args.probe_seed + depth,
            steps=args.steps,
            batch_size=args.probe_batch_size,
            learning_rate=args.learning_rate,
        )
        checkpoint_path = args.artifact_directory / f"depth_{depth}_probe.pt"
        torch.save(probe.state_dict(), checkpoint_path)
        evaluations = {
            split: evaluate_probe(
                probe,
                split_features[depth],
                device=bundle.device,
            )
            for split, split_features in feature_sets.items()
        }
        depth_results.append(
            {
                "depth_after_blocks": depth,
                "training": training,
                "evaluations": evaluations,
            }
        )
    result = {
        "status": "pilot",
        "model_id": args.model_id,
        "revision": args.revision,
        "dataset_sha256": dataset_hash,
        "configuration": {
            "depths": args.depths,
            "train_examples": args.train_examples,
            "eval_examples": args.eval_examples,
            "train_seed": args.train_seed,
            "eval_seed": args.eval_seed,
            "probe_seed": args.probe_seed,
            "steps": args.steps,
            "feature_batch_size": args.batch_size,
            "probe_batch_size": args.probe_batch_size,
            "learning_rate": args.learning_rate,
        },
        "feature_extraction_seconds": extraction_times,
        "depth_results": depth_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
