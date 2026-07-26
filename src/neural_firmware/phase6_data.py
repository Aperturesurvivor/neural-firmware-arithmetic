from __future__ import annotations

import random
from dataclasses import asdict, dataclass

PHASE6_TRAIN_SINGLE_FAMILIES = (
    "Add {a} and {b}; answer with only the resulting integer.",
    "What is {a} plus {b}? Return digits only.",
    "Calculate the sum of {a} with {b}. Give one number.",
    "Increase {a} by {b} and print only the result.",
    "Combine {a} and {b} by addition. No explanation.",
    "Find the additive total for {a} and {b}; output the integer alone.",
    "Apply integer addition to {a} and {b}. Reply solely with digits.",
    "Starting at {a}, advance by {b}. What value is reached?",
    (
        "A store holds {a} items and receives {b} more. "
        "How many items are there now? Number only."
    ),
    (
        "One counter recorded {a} events and another recorded {b}. "
        "Give their combined count using digits only."
    ),
    (
        "A fund contains {a} dollars and gains {b} dollars. "
        "What is its new balance? Return only the integer."
    ),
    (
        "A route covers {a} meters, followed by another {b} meters. "
        "State the total distance as one number."
    ),
)

PHASE6_TRAIN_CHAIN_FAMILIES = (
    "Add {a} and {b}, then add {c} to that result. Return only the final integer.",
    "Compute ({a} + {b}) + {c}. Give only the final number.",
    "Start with {a}, increase it by {b}, then increase it by {c}. Number only.",
    "First total {a} with {b}; next add {c}. Output the eventual result alone.",
    "Evaluate the two additions {a} plus {b}, followed by plus {c}. Digits only.",
    "Take {a}, add {b}, and add {c} to the intermediate total. No words.",
    (
        "A warehouse starts with {a} parts, receives {b}, and later receives "
        "{c} more. How many parts are present? Answer with digits only."
    ),
    (
        "A player has {a} points, earns {b}, then earns another {c}. "
        "What is the final score? Return one integer."
    ),
    (
        "Three batches contain {a}, {b}, and {c} samples. "
        "How many samples are there altogether? Give only the total."
    ),
    (
        "A project imported {a} records in one stage, {b} in the next, and "
        "{c} in the last. State the complete record count."
    ),
)

# These hard negatives deliberately preserve the imperative and output shape
# of positives while changing operation identity or suppressing evaluation.
PHASE6_TRAIN_NEGATIVE_FAMILIES = (
    "Compute the product of {a} and {b}; provide digits only.",
    "Multiply {a} by {b}. Return only the resulting integer.",
    "Subtract {b} from {a}; output one signed integer.",
    "Find the arithmetic mean of {a} and {b}. Give digits only.",
    "Return the larger of {a} and {b}; answer with one number.",
    "Concatenate {a} and {b}. Output only the joined digits.",
    'Quote exactly "{a} plus {b}" without evaluating it.',
    "Do not add {a} and {b}; reply only with withheld.",
    "Explain how to add {a} and {b}, but do not state their total.",
    "Check whether {a} + {b} is valid syntax. Answer yes or no.",
    "Multiply {a} and {b}, then add {c}. Return only the final result.",
    "Add {a} and {b}, then multiply that result by {c}. Digits only.",
    "Subtract {b} from {a}, then add {c}. Output one integer.",
    "List the identifiers {a}, {b}, and {c} without calculating.",
    "Return the largest of {a}, {b}, and {c}. Number only.",
    "Describe adding {a}, {b}, and {c} without performing either addition.",
)

PHASE6_TRAIN_UNSUPPORTED_ONE_FAMILIES = frozenset(
    PHASE6_TRAIN_NEGATIVE_FAMILIES[:5]
)
PHASE6_TRAIN_UNSUPPORTED_TWO_FAMILIES = frozenset(
    (
        PHASE6_TRAIN_NEGATIVE_FAMILIES[10],
        PHASE6_TRAIN_NEGATIVE_FAMILIES[11],
        PHASE6_TRAIN_NEGATIVE_FAMILIES[12],
        PHASE6_TRAIN_NEGATIVE_FAMILIES[14],
    )
)

PHASE6_DEVELOPMENT_SINGLE_FAMILIES = (
    "Supply the exact additive result for {a} together with {b}. Integer only.",
    "By adding {b} onto {a}, what number is obtained? No explanation.",
    "Resolve the sum whose two operands are {a} and {b}; print digits alone.",
    (
        "A cabinet has {a} files and receives {b} additional files. "
        "What is the resulting file count? Give only the number."
    ),
)

PHASE6_DEVELOPMENT_CHAIN_FAMILIES = (
    "Find {a} + {b}, then increase that intermediate result by {c}. Final number only.",
    "Perform two additions in order: {a} with {b}, then the result with {c}.",
    (
        "A depot logged {a} parcels, then {b} parcels, then {c} more. "
        "How many parcels were logged in all? Digits only."
    ),
    (
        "An account begins at {a}, gains {b}, and later gains {c}. "
        "Return its final value as an integer."
    ),
)

PHASE6_DEVELOPMENT_NEGATIVE_FAMILIES = (
    "Calculate the multiplication of {a} and {b}; return digits only.",
    "Take the difference between {a} and {b}. Output one integer.",
    "Choose the smaller of {a} and {b}; number only.",
    "Print {a} immediately followed by {b}; do not calculate.",
    'Repeat the question "What is {a} plus {b}?" without answering it.',
    "Assess whether {a} plus {b} needs carrying, but leave it unsolved.",
    "Multiply {a} by {b}, and afterward add {c}. State the final value.",
    "Add {a} to {b}, then divide the result by {c}. Give one number.",
    "Sort the values {a}, {b}, and {c}; perform no arithmetic.",
    "Discuss a two-step sum of {a}, {b}, and {c} without computing it.",
)

PHASE6_DEVELOPMENT_UNSUPPORTED_ONE_FAMILIES = frozenset(
    PHASE6_DEVELOPMENT_NEGATIVE_FAMILIES[:3]
)
PHASE6_DEVELOPMENT_UNSUPPORTED_TWO_FAMILIES = frozenset(
    (
        PHASE6_DEVELOPMENT_NEGATIVE_FAMILIES[6],
        PHASE6_DEVELOPMENT_NEGATIVE_FAMILIES[7],
    )
)

# Frozen-confirmation candidates. Pilot code must not import these constants.
PHASE6_CONFIRMATORY_SINGLE_FAMILIES = (
    "Produce only the integer formed by adding {a} to {b}.",
    "What total results when {a} is augmented by {b}? Digits alone.",
    "Execute addition on the pair {a}, {b}; omit all explanatory text.",
    (
        "A terminal processed {a} jobs and then {b} more. "
        "How many jobs did it process? Return only the count."
    ),
)

PHASE6_CONFIRMATORY_CHAIN_FAMILIES = (
    "Add {a} to {b}; add {c} to the intermediate answer; report the final integer.",
    "Evaluate the left-associated total of {a}, {b}, and {c}. Number only.",
    (
        "A lab collected {a} samples, then {b}, then another {c}. "
        "State the complete sample count using only digits."
    ),
    (
        "A balance starts at {a}, rises by {b}, and rises again by {c}. "
        "What final balance is reached? Give one integer."
    ),
)

PHASE6_CONFIRMATORY_NEGATIVE_FAMILIES = (
    "Return the product of {a} and {b}; digits only.",
    "Compute {a} minus {b}. Give one signed integer.",
    "Select the maximum of {a} and {b}. Number only.",
    "Join the decimal strings {a} and {b} without arithmetic.",
    'Quote "{a} plus {b}" exactly and do not solve it.',
    "Explain the sum of {a} and {b} without giving its result.",
    "Multiply {a} and {b}, then add {c}; output the eventual integer.",
    "Add {a} and {b}, then subtract {c}. Return only the final value.",
    "List {a}, {b}, and {c} in descending order.",
    "Say whether {a} + {b} + {c} is positive without stating the total.",
)


@dataclass(frozen=True)
class Phase6Example:
    prompt: str
    operands: tuple[str, ...]
    call_count: int
    controller_target: int
    answer: str | None
    intermediate_answers: tuple[str, ...]
    family: str
    family_index: int
    split: str

    @property
    def route_label(self) -> bool:
        return self.call_count > 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _random_decimal(rng: random.Random, digits: int) -> str:
    if digits < 1:
        raise ValueError("digits must be positive")
    if digits == 1:
        return str(rng.randrange(10))
    return str(rng.randrange(10 ** (digits - 1), 10**digits))


def _make_examples(
    *,
    count: int,
    min_digits: int,
    max_digits: int,
    seed: int,
    split: str,
    families: tuple[str, ...],
    call_count: int,
    unsupported_one_families: frozenset[str] = frozenset(),
    unsupported_two_families: frozenset[str] = frozenset(),
) -> list[Phase6Example]:
    if call_count not in (0, 1, 2):
        raise ValueError("call_count must be zero, one, or two")
    rng = random.Random(seed)
    examples = []
    for index in range(count):
        family_index = index % len(families)
        family = families[family_index]
        operand_count = 3 if "{c}" in family or call_count == 2 else 2
        named_operands = {
            name: _random_decimal(rng, rng.randint(min_digits, max_digits))
            for name in ("a", "b", "c")[:operand_count]
        }
        textual_names = sorted(
            named_operands,
            key=lambda name: family.index("{" + name + "}"),
        )
        operands = tuple(named_operands[name] for name in textual_names)
        values = [int(named_operands[name]) for name in ("a", "b", "c")[:operand_count]]
        intermediates: tuple[str, ...] = ()
        answer: str | None = None
        if call_count == 1:
            answer = str(values[0] + values[1])
            intermediates = (answer,)
        elif call_count == 2:
            first = values[0] + values[1]
            answer = str(first + values[2])
            intermediates = (str(first), answer)
        prompt = family.format(
            a=named_operands["a"],
            b=named_operands["b"],
            c=named_operands.get("c", ""),
        )
        controller_target = call_count
        if family in unsupported_one_families:
            controller_target = 3
        elif family in unsupported_two_families:
            controller_target = 4
        examples.append(
            Phase6Example(
                prompt=prompt,
                operands=operands,
                call_count=call_count,
                controller_target=controller_target,
                answer=answer,
                intermediate_answers=intermediates,
                family=family,
                family_index=family_index,
                split=split,
            )
        )
    return examples


def build_phase6_training_examples(
    *,
    single_count: int = 2400,
    chain_count: int = 2400,
    negative_count: int = 3200,
) -> list[Phase6Example]:
    return (
        _make_examples(
            count=single_count,
            min_digits=1,
            max_digits=8,
            seed=11_501,
            split="phase6_train_single",
            families=PHASE6_TRAIN_SINGLE_FAMILIES,
            call_count=1,
        )
        + _make_examples(
            count=chain_count,
            min_digits=1,
            max_digits=8,
            seed=11_502,
            split="phase6_train_chain",
            families=PHASE6_TRAIN_CHAIN_FAMILIES,
            call_count=2,
        )
        + _make_examples(
            count=negative_count,
            min_digits=1,
            max_digits=8,
            seed=11_503,
            split="phase6_train_negative",
            families=PHASE6_TRAIN_NEGATIVE_FAMILIES,
            call_count=0,
            unsupported_one_families=PHASE6_TRAIN_UNSUPPORTED_ONE_FAMILIES,
            unsupported_two_families=PHASE6_TRAIN_UNSUPPORTED_TWO_FAMILIES,
        )
    )


def build_phase6_development_examples(
    *,
    single_count: int = 400,
    chain_count: int = 400,
    negative_count: int = 600,
) -> list[Phase6Example]:
    return (
        _make_examples(
            count=single_count,
            min_digits=1,
            max_digits=8,
            seed=11_504,
            split="phase6_development_single",
            families=PHASE6_DEVELOPMENT_SINGLE_FAMILIES,
            call_count=1,
        )
        + _make_examples(
            count=chain_count,
            min_digits=1,
            max_digits=8,
            seed=11_505,
            split="phase6_development_chain",
            families=PHASE6_DEVELOPMENT_CHAIN_FAMILIES,
            call_count=2,
        )
        + _make_examples(
            count=negative_count,
            min_digits=1,
            max_digits=8,
            seed=11_506,
            split="phase6_development_negative",
            families=PHASE6_DEVELOPMENT_NEGATIVE_FAMILIES,
            call_count=0,
            unsupported_one_families=(
                PHASE6_DEVELOPMENT_UNSUPPORTED_ONE_FAMILIES
            ),
            unsupported_two_families=(
                PHASE6_DEVELOPMENT_UNSUPPORTED_TWO_FAMILIES
            ),
        )
    )


def build_phase6_output_training_examples(
    *,
    single_count: int = 800,
    chain_count: int = 800,
) -> list[Phase6Example]:
    return (
        _make_examples(
            count=single_count,
            min_digits=1,
            max_digits=8,
            seed=11_507,
            split="phase6_output_single",
            families=PHASE6_TRAIN_SINGLE_FAMILIES,
            call_count=1,
        )
        + _make_examples(
            count=chain_count,
            min_digits=1,
            max_digits=8,
            seed=11_508,
            split="phase6_output_chain",
            families=PHASE6_TRAIN_CHAIN_FAMILIES,
            call_count=2,
        )
    )
