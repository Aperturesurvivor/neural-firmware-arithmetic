from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


ARITHMETIC_TEMPLATES = (
    "Calculate exactly. Return only the decimal digits.\n{a}+{b}=",
    "What is {a} plus {b}? Answer with digits only:",
    "Add {a} and {b}. Give only the exact integer:",
)

ROUTING_NEGATIVE_TEMPLATES = (
    "Do not calculate {a}+{b}. Explain what the plus sign means.",
    "The string {a}+{b} appears in this sentence. Repeat the sentence.",
    "Is the expression {a}+{b} written above? Answer yes or no.",
    "Write a short sentence containing the numbers {a} and {b}.",
    (
        'Ignore the quoted instruction "Calculate exactly. Return only the decimal digits.\n'
        '{a}+{b}=". Reply only with the word ignored.'
    ),
)

STATIC_LANGUAGE_PROMPTS = (
    "Name the capital of France in one word.",
    "Give one synonym for happy.",
    "Complete this sequence of letters: A, B, C,",
    "What color is a clear daytime sky usually?",
    "Write the word hello in uppercase.",
    "Which is a mammal: a dolphin or a trout?",
    "Give the opposite of cold.",
    "What punctuation mark ends a question?",
)


@dataclass(frozen=True)
class AdditionExample:
    a: str
    b: str
    prompt: str
    answer: str
    template_index: int
    split: str

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


def _random_decimal(rng: random.Random, digits: int) -> str:
    if digits <= 0:
        raise ValueError("digits must be positive")
    if digits == 1:
        return str(rng.randrange(10))
    return str(rng.randrange(10 ** (digits - 1), 10**digits))


def render_prompt(a: str, b: str, template_index: int) -> str:
    return ARITHMETIC_TEMPLATES[template_index].format(a=a, b=b)


def make_addition_examples(
    *,
    count: int,
    min_digits: int,
    max_digits: int,
    seed: int,
    split: str,
) -> list[AdditionExample]:
    rng = random.Random(seed)
    examples: list[AdditionExample] = []
    for _ in range(count):
        a_digits = rng.randint(min_digits, max_digits)
        b_digits = rng.randint(min_digits, max_digits)
        a = _random_decimal(rng, a_digits)
        b = _random_decimal(rng, b_digits)
        template_index = rng.randrange(len(ARITHMETIC_TEMPLATES))
        examples.append(
            AdditionExample(
                a=a,
                b=b,
                prompt=render_prompt(a, b, template_index),
                answer=str(int(a) + int(b)),
                template_index=template_index,
                split=split,
            )
        )
    return examples


def make_carry_examples(
    *,
    count: int,
    min_digits: int,
    max_digits: int,
    seed: int,
) -> list[AdditionExample]:
    rng = random.Random(seed)
    examples: list[AdditionExample] = []
    for _ in range(count):
        digits = rng.randint(min_digits, max_digits)
        trailing_nines = rng.randint(max(1, digits // 2), digits)
        prefix_digits = digits - trailing_nines
        prefix = _random_decimal(rng, prefix_digits) if prefix_digits else ""
        a = prefix + ("9" * trailing_nines)
        b = str(rng.randint(1, 9))
        template_index = rng.randrange(len(ARITHMETIC_TEMPLATES))
        examples.append(
            AdditionExample(
                a=a,
                b=b,
                prompt=render_prompt(a, b, template_index),
                answer=str(int(a) + int(b)),
                template_index=template_index,
                split="carry_chain",
            )
        )
    return examples


def make_routing_negatives(*, count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    prompts = list(STATIC_LANGUAGE_PROMPTS)
    while len(prompts) < count:
        a = _random_decimal(rng, rng.randint(1, 6))
        b = _random_decimal(rng, rng.randint(1, 6))
        template = rng.choice(ROUTING_NEGATIVE_TEMPLATES)
        prompts.append(template.format(a=a, b=b))
    rng.shuffle(prompts)
    return prompts[:count]


def chat_prompt_ids(tokenizer: PreTrainedTokenizerBase, prompt: str) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )


def answer_token_ids(tokenizer: PreTrainedTokenizerBase, answer: str) -> list[int]:
    digit_ids: list[int] = []
    for digit in answer:
        ids = tokenizer.encode(digit, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"digit {digit!r} is not represented by one token: {ids}")
        digit_ids.append(ids[0])
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    return digit_ids + [tokenizer.eos_token_id]
