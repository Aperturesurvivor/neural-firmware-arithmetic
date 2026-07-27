from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from neural_firmware.model import CausalArithmeticTransformer, ModelConfig
from neural_firmware.tokenizer import ArithmeticTokenizer
from neural_firmware.training import load_model


def test_model_loader_requests_weights_only(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = ArithmeticTokenizer()
    config = ModelConfig(
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        max_sequence_length=24,
    )
    source = CausalArithmeticTransformer(tokenizer, config, "baseline")
    observed: dict[str, object] = {}

    def fake_load(
        path: Path,
        *,
        map_location: str,
        weights_only: bool,
    ) -> dict[str, object]:
        observed.update(
            path=path,
            map_location=map_location,
            weights_only=weights_only,
        )
        return {
            "model_config": config.to_dict(),
            "mode": "baseline",
            "model_state": source.state_dict(),
        }

    monkeypatch.setattr(torch, "load", fake_load)
    checkpoint = Path("shared-checkpoint.pt")
    loaded = load_model(checkpoint, torch.device("cpu"))

    assert observed == {
        "path": checkpoint,
        "map_location": "cpu",
        "weights_only": True,
    }
    assert loaded.mode == "baseline"


def test_publication_inputs_reject_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = Path(__file__).parents[1] / "publication" / "build_packages.py"
    spec = importlib.util.spec_from_file_location("publication_build_packages", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "ROOT", tmp_path)
    source = tmp_path / "publication" / "main.tex"
    source.parent.mkdir()
    source.write_text("safe source\n")
    assert module.contained_file(source) == source.resolve()

    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("private marker\n")
    symlink = source.with_name("linked.tex")
    symlink.symlink_to(outside)
    try:
        with pytest.raises(ValueError, match="symlinked publication input"):
            module.contained_file(symlink)
    finally:
        outside.unlink()
