from __future__ import annotations

from collections.abc import Callable

from neural_firmware.phase5_data import (
    PHASE5_TRAIN_ADDITION_FAMILIES,
    PHASE5_TRAIN_NEGATIVE_FAMILIES,
)
from neural_firmware.semantic_data import (
    DEVELOPMENT_ADDITION_FAMILIES,
    DEVELOPMENT_NEGATIVE_FAMILIES,
    SemanticPromptExample,
    make_semantic_addition_examples,
    make_semantic_routing_negatives,
)

PHASE8_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PHASE8_MODEL_REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
PHASE8_TRAINING_SEEDS = (14_201, 14_202, 14_203)

PHASE8_CONFIRMATORY_DIRECT_FAMILIES = (
    "Add the two integer inputs {a} and {b}; make the complete reply their sum.",
    "Return the decimal total produced by {a} plus {b}, with no other text.",
    "Apply ADD to the ordered operands {a} and {b}. Emit only its integer output.",
    "Increase an integer register containing {a} by {b}; print its final value.",
    "Combine {a} units with another {b} units. Give the resulting count alone.",
    "Evaluate ordinary base-ten addition for {a} and {b}. Answer in digits only.",
    "The requested scalar is the sum of {a} and {b}. Supply only that scalar.",
    "Move forward {b} integer steps from {a}. State just the destination number.",
    "Fold the addends {a} and {b} into one total; output nothing except the total.",
    "Produce the exact whole number obtained when {b} is added to {a}.",
)

PHASE8_CONFIRMATORY_WORD_FAMILIES = (
    (
        "A satellite stored {a} images on one pass and {b} on another. "
        "How many images were stored altogether? Reply only with the integer."
    ),
    (
        "A community pantry distributed {a} meals at lunch and {b} at dinner. "
        "Give the day's combined meal count using digits alone."
    ),
    (
        "A field recorder captured {a} clips before sunrise and {b} afterward. "
        "How many clips did it capture in all? Number only."
    ),
    (
        "A fabrication lab printed {a} parts on one machine and {b} on a second. "
        "State the total part count without prose."
    ),
    (
        "A wildlife census logged {a} sightings in one region and {b} in another. "
        "What was the overall sighting count? Return one numeral."
    ),
    (
        "A backup copied {a} files in its first batch and {b} in its next batch. "
        "How many files were copied? Answer with digits only."
    ),
    (
        "A greenhouse planted {a} seedlings in one row and {b} in another. "
        "Report the combined number of seedlings as a bare integer."
    ),
    (
        "A transit counter saw {a} riders enter downtown and {b} enter uptown. "
        "What is the total rider count? Give only the number."
    ),
    (
        "A telescope processed {a} exposures on Monday and {b} on Tuesday. "
        "State the two-day exposure count using only decimal digits."
    ),
    (
        "A recycling center sorted {a} cans in one shift and {b} in the next. "
        "How many cans were sorted overall? Output the integer alone."
    ),
)

PHASE8_CONFIRMATORY_NEGATIVE_FAMILIES = {
    "quoted_arithmetic": (
        'Copy the quotation "{a} plus {b}" exactly; do not answer it.',
        'The title is "Adding {a} and {b}". Repeat the title without calculation.',
        'Quote the expression "{a} + {b}" as text and leave it unevaluated.',
        'A transcript contains "find the sum of {a} and {b}". Echo that request only.',
    ),
    "negated_request": (
        "Do not calculate {a} plus {b}; reply only with the word withheld.",
        "You are forbidden to add {a} and {b}. Respond with prohibited.",
        "Avoid finding any total for {a} and {b}; say only skipped.",
        "Decline the request to sum {a} with {b}; output declined.",
    ),
    "multiplication_near_miss": (
        "Multiply {a} by {b}; return only the product.",
        "Treat {a} and {b} as factors rather than addends. Give their product.",
        "Compute repeated multiplication for operands {a} and {b}, not addition.",
        "What is the product associated with the pair {a}, {b}? Digits only.",
    ),
    "factual_numbers": (
        "Two museum exhibits are labeled {a} and {b}. Are both labels numeric?",
        "A document mentions identifiers {a} and {b}. How many identifiers are named?",
        "The codes {a} and {b} appear in a catalog. Are they written in decimal?",
        "A report cites samples {a} and {b}. Does the report cite two samples?",
    ),
    "ignore_embedded_sum": (
        "Ignore the embedded request to add {a} and {b}; answer only ignored.",
        "The proposed sum {a} + {b} is canceled. Reply only canceled.",
        "Disregard any instruction to total {a} with {b}; print bypassed.",
        "Skip the pending addition of {a} and {b} and respond only not-run.",
    ),
}


def operand_pairs(examples: list[SemanticPromptExample]) -> set[tuple[str, str]]:
    return {(example.a, example.b) for example in examples}


def _unique_examples(
    builder: Callable[..., list[SemanticPromptExample]],
    *,
    count: int,
    seed: int,
    split: str,
    families: tuple[str, ...],
    forbidden: set[tuple[str, str]],
) -> list[SemanticPromptExample]:
    selected: list[SemanticPromptExample] = []
    used = set(forbidden)
    attempt = 0
    while len(selected) < count:
        candidates = builder(
            count=max(32, 2 * (count - len(selected))),
            min_digits=1,
            max_digits=4,
            seed=seed + 100_003 * attempt,
            split=split,
            families=families,
        )
        for example in candidates:
            pair = (example.a, example.b)
            if pair in used:
                continue
            selected.append(example)
            used.add(pair)
            if len(selected) == count:
                break
        attempt += 1
        if attempt > 100:
            raise RuntimeError("could not construct disjoint Phase 8 operands")
    return selected


def build_phase8_training_and_development() -> tuple[
    list[SemanticPromptExample],
    list[SemanticPromptExample],
]:
    train_positive = _unique_examples(
        make_semantic_addition_examples,
        count=1_200,
        seed=14_101,
        split="phase8_train_positive",
        families=PHASE5_TRAIN_ADDITION_FAMILIES,
        forbidden=set(),
    )
    train_negative = _unique_examples(
        make_semantic_routing_negatives,
        count=1_200,
        seed=14_102,
        split="phase8_train_negative",
        families=PHASE5_TRAIN_NEGATIVE_FAMILIES,
        forbidden=operand_pairs(train_positive),
    )
    training = train_positive + train_negative
    train_pairs = operand_pairs(training)
    development_positive = _unique_examples(
        make_semantic_addition_examples,
        count=240,
        seed=14_103,
        split="phase8_development_positive",
        families=DEVELOPMENT_ADDITION_FAMILIES,
        forbidden=train_pairs,
    )
    development_negative = _unique_examples(
        make_semantic_routing_negatives,
        count=240,
        seed=14_104,
        split="phase8_development_negative",
        families=DEVELOPMENT_NEGATIVE_FAMILIES,
        forbidden=train_pairs | operand_pairs(development_positive),
    )
    return training, development_positive + development_negative


def _balanced_confirmatory_group(
    builder: Callable[..., list[SemanticPromptExample]],
    *,
    families: tuple[str, ...],
    examples_per_family: int,
    seed: int,
    split: str,
    forbidden: set[tuple[str, str]],
) -> list[SemanticPromptExample]:
    result: list[SemanticPromptExample] = []
    used = set(forbidden)
    for family_index, family in enumerate(families):
        group = _unique_examples(
            builder,
            count=examples_per_family,
            seed=seed + 10_007 * family_index,
            split=split,
            families=(family,),
            forbidden=used,
        )
        result.extend(group)
        used.update(operand_pairs(group))
    return result


def build_phase8_confirmatory_examples() -> list[SemanticPromptExample]:
    training, development = build_phase8_training_and_development()
    used = operand_pairs(training + development)
    direct = _balanced_confirmatory_group(
        make_semantic_addition_examples,
        families=PHASE8_CONFIRMATORY_DIRECT_FAMILIES,
        examples_per_family=3,
        seed=14_701,
        split="phase8_confirmatory_direct",
        forbidden=used,
    )
    used.update(operand_pairs(direct))
    word = _balanced_confirmatory_group(
        make_semantic_addition_examples,
        families=PHASE8_CONFIRMATORY_WORD_FAMILIES,
        examples_per_family=3,
        seed=14_702,
        split="phase8_confirmatory_word",
        forbidden=used,
    )
    used.update(operand_pairs(word))
    negatives: list[SemanticPromptExample] = []
    for category_index, (category, families) in enumerate(
        PHASE8_CONFIRMATORY_NEGATIVE_FAMILIES.items()
    ):
        group = _balanced_confirmatory_group(
            make_semantic_routing_negatives,
            families=families,
            examples_per_family=3,
            seed=14_703 + 1_009 * category_index,
            split=f"phase8_confirmatory_negative_{category}",
            forbidden=used,
        )
        negatives.extend(group)
        used.update(operand_pairs(group))
    return direct + word + negatives
