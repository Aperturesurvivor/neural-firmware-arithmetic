from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from neural_firmware.data import sample_training_batch
from neural_firmware.model import CausalArithmeticTransformer, ModelConfig
from neural_firmware.tokenizer import ArithmeticTokenizer


@dataclass
class TrainingResult:
    model: str
    seed: int
    steps: int
    final_loss: float
    wall_time_seconds: float
    trainable_parameters: int
    device: str
    checkpoint: str
    history: list[dict[str, float | int]]


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment_record() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        commit = "uncommitted"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "git_commit": commit,
        "pid": os.getpid(),
    }


def train_one(
    experiment_config: dict[str, Any],
    mode: str,
    seed: int,
    output_root: Path,
) -> TrainingResult:
    seed_everything(seed)
    tokenizer = ArithmeticTokenizer()
    model_config = ModelConfig(**experiment_config["model"])
    device = resolve_device(experiment_config.get("device", "auto"))
    model = CausalArithmeticTransformer(tokenizer, model_config, mode).to(device)
    training = experiment_config["training"]
    data = experiment_config["data"]
    steps = training.get("steps_by_model", {}).get(mode, training["steps"])
    if steps < 1:
        raise ValueError("Every run must perform at least one optimization step")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    loss_function = nn.CrossEntropyLoss(ignore_index=-100)
    rng = np.random.default_rng(seed + 10_000)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    model.train()
    for step in range(1, steps + 1):
        _, batch = sample_training_batch(
            rng=rng,
            batch_size=training["batch_size"],
            min_digits=data["train_min_digits"],
            max_digits=data["train_max_digits"],
            tokenizer=tokenizer,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = loss_function(logits.reshape(-1, tokenizer.vocab_size), batch["labels"].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % training["log_every"] == 0 or step == steps:
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(record)
            print(
                f"[{mode} seed={seed}] step={step}/{steps} "
                f"loss={record['loss']:.6f} elapsed={record['elapsed_seconds']:.1f}s",
                flush=True,
            )

    wall_time = time.perf_counter() - started
    run_dir = output_root / f"{mode}-seed-{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "model.pt"
    torch.save(
        {
            "mode": mode,
            "seed": seed,
            "model_config": model_config.to_dict(),
            "model_state": model.state_dict(),
            "tokenizer_tokens": tokenizer.tokens,
            "experiment_config": experiment_config,
            "history": history,
            "environment": environment_record(),
        },
        checkpoint_path,
    )
    result = TrainingResult(
        model=mode,
        seed=seed,
        steps=steps,
        final_loss=history[-1]["loss"],
        wall_time_seconds=wall_time,
        trainable_parameters=model.trainable_parameter_count(),
        device=str(device),
        checkpoint=str(checkpoint_path),
        history=history,
    )
    (run_dir / "training.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
    return result


def load_model(checkpoint_path: Path, device: torch.device) -> CausalArithmeticTransformer:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tokenizer = ArithmeticTokenizer()
    model = CausalArithmeticTransformer(
        tokenizer=tokenizer,
        config=ModelConfig(**payload["model_config"]),
        mode=payload["mode"],
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
