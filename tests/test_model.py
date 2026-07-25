import torch

from neural_firmware.model import CausalArithmeticTransformer, ModelConfig
from neural_firmware.tokenizer import ArithmeticTokenizer


def _small_config() -> ModelConfig:
    return ModelConfig(
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        max_sequence_length=48,
        firmware_strength=8.0,
    )


def test_modes_have_equal_trainable_parameter_counts() -> None:
    tokenizer = ArithmeticTokenizer()
    counts = {
        mode: CausalArithmeticTransformer(tokenizer, _small_config(), mode)
        .trainable_parameter_count()
        for mode in CausalArithmeticTransformer.VALID_MODES
    }
    assert len(set(counts.values())) == 1


def test_direct_firmware_overrides_untrained_logits() -> None:
    tokenizer = ArithmeticTokenizer()
    model = CausalArithmeticTransformer(tokenizer, _small_config(), "direct_firmware")
    prompt = tokenizer.encode_expression(98765, 12345, include_answer=False)
    input_ids = torch.tensor([prompt], dtype=torch.long)
    generated: list[int] = []
    for _ in range(8):
        logits = model(input_ids)
        next_id = int(logits[0, -1].argmax())
        generated.append(next_id)
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=1)
        if next_id == tokenizer.eos_id:
            break
    assert tokenizer.decode_answer(generated) == str(98765 + 12345)

