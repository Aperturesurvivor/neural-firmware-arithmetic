from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch

from neural_firmware.pretrained_data import AdditionExample, chat_prompt_ids
from neural_firmware.pretrained_firmware import FrozenDecimalFirmware
from neural_firmware.pretrained_model import FirmwareBridge
from neural_firmware.pretrained_training import ModelBundle


@dataclass(frozen=True)
class GenerationResult:
    prompt: str
    expected: str | None
    generated_text: str
    generated_token_ids: list[int]
    route_probabilities: list[float]
    exact: bool | None
    latency_seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _token_id_for_symbol(bundle: ModelBundle, symbol: int) -> int:
    if symbol == 10:
        return bundle.tokenizer.eos_token_id
    ids = bundle.tokenizer.encode(str(symbol), add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"firmware symbol {symbol} does not map to one token")
    return ids[0]


@torch.inference_mode()
def generate_one(
    bundle: ModelBundle,
    prompt: str,
    *,
    mode: str,
    bridge: FirmwareBridge | None = None,
    max_new_tokens: int | None = None,
) -> GenerationResult:
    if mode not in {
        "base",
        "learned_adapter",
        "latent",
        "direct",
        "firmware_off",
    }:
        raise ValueError(f"unknown generation mode: {mode}")
    if mode == "latent" and bridge is None:
        raise ValueError("latent generation requires a bridge")

    firmware = FrozenDecimalFirmware()
    plan = firmware.parse(prompt)
    planned_symbols = firmware.symbols(prompt)
    expected = plan.answer if plan is not None else None
    if max_new_tokens is None:
        max_new_tokens = (len(expected) + 3) if expected is not None else 16

    input_ids = torch.tensor(
        [chat_prompt_ids(bundle.tokenizer, prompt)],
        dtype=torch.long,
        device=bundle.device,
    )
    attention_mask = torch.ones_like(input_ids)
    started = time.perf_counter()
    outputs = bundle.model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    hidden = outputs.last_hidden_state[:, -1, :]
    generated: list[int] = []
    route_probabilities: list[float] = []

    for step in range(max_new_tokens):
        symbol = (
            planned_symbols[step]
            if planned_symbols is not None and step < len(planned_symbols)
            else 10
        )
        logits_hidden = hidden
        route_probability = 0.0
        if bridge is not None:
            route_probability = torch.sigmoid(bridge.router_logits(hidden)).item()
        route_probabilities.append(route_probability)

        if mode == "latent" and planned_symbols is not None:
            symbols = torch.tensor([symbol], dtype=torch.long, device=bundle.device)
            logits_hidden, _ = bridge.routed_hidden(
                hidden,
                symbols,
                hard_route=True,
                eligible=torch.ones(1, device=bundle.device),
            )
        logits = bundle.model.lm_head(logits_hidden)
        if mode == "direct" and planned_symbols is not None:
            logits[:, _token_id_for_symbol(bundle, symbol)] += 1_000_000.0

        next_token = logits.argmax(dim=-1)
        token_id = next_token.item()
        generated.append(token_id)
        if token_id == bundle.tokenizer.eos_token_id:
            break

        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((1, 1), dtype=attention_mask.dtype, device=bundle.device),
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
    text = bundle.tokenizer.decode(generated, skip_special_tokens=True).strip()
    exact = None if expected is None else text == expected
    return GenerationResult(
        prompt=prompt,
        expected=expected,
        generated_text=text,
        generated_token_ids=generated,
        route_probabilities=route_probabilities,
        exact=exact,
        latency_seconds=elapsed,
    )


def evaluate_additions(
    bundle: ModelBundle,
    examples: list[AdditionExample],
    *,
    mode: str,
    bridge: FirmwareBridge | None = None,
) -> tuple[dict[str, float | int | str], list[GenerationResult]]:
    results = [
        generate_one(bundle, example.prompt, mode=mode, bridge=bridge)
        for example in examples
    ]
    correct = sum(result.exact is True for result in results)
    total = len(results)
    summary: dict[str, float | int | str] = {
        "mode": mode,
        "split": examples[0].split if examples else "empty",
        "examples": total,
        "correct": correct,
        "exact_match_accuracy": correct / total if total else float("nan"),
        "mean_latency_seconds": (
            sum(result.latency_seconds for result in results) / total
            if total
            else float("nan")
        ),
    }
    return summary, results


def evaluate_preservation(
    bundle: ModelBundle,
    prompts: list[str],
    *,
    bridge: FirmwareBridge,
) -> tuple[dict[str, float | int], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for prompt in prompts:
        base = generate_one(bundle, prompt, mode="base", max_new_tokens=16)
        latent = generate_one(
            bundle,
            prompt,
            mode="latent",
            bridge=bridge,
            max_new_tokens=16,
        )
        rows.append(
            {
                "prompt": prompt,
                "base_text": base.generated_text,
                "latent_text": latent.generated_text,
                "token_exact_preserved": (
                    base.generated_token_ids == latent.generated_token_ids
                ),
                "initial_route_probability": latent.route_probabilities[0],
            }
        )
    preserved = sum(bool(row["token_exact_preserved"]) for row in rows)
    false_routes = sum(float(row["initial_route_probability"]) >= 0.5 for row in rows)
    return (
        {
            "prompts": len(rows),
            "token_exact_preserved": preserved,
            "preservation_rate": preserved / len(rows) if rows else float("nan"),
            "initial_false_routes": false_routes,
            "initial_false_route_rate": (
                false_routes / len(rows) if rows else float("nan")
            ),
        },
        rows,
    )
