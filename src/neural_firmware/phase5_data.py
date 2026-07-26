from __future__ import annotations

from neural_firmware.semantic_data import (
    AUGMENTED_TRAIN_ADDITION_FAMILIES,
    AUGMENTED_TRAIN_NEGATIVE_FAMILIES,
    DEVELOPMENT_ADDITION_FAMILIES,
    DEVELOPMENT_NEGATIVE_FAMILIES,
    HELDOUT_ADDITION_FAMILIES,
    HELDOUT_NEGATIVE_FAMILIES,
    TRAIN_ADDITION_FAMILIES,
    TRAIN_NEGATIVE_FAMILIES,
    WORD_PROBLEM_FAMILIES,
    SemanticPromptExample,
    make_semantic_addition_examples,
    make_semantic_routing_negatives,
)

PHASE5_TRAIN_ADDITION_FAMILIES = (
    TRAIN_ADDITION_FAMILIES
    + HELDOUT_ADDITION_FAMILIES
    + WORD_PROBLEM_FAMILIES
    + AUGMENTED_TRAIN_ADDITION_FAMILIES
)
PHASE5_TRAIN_NEGATIVE_FAMILIES = (
    TRAIN_NEGATIVE_FAMILIES
    + HELDOUT_NEGATIVE_FAMILIES
    + AUGMENTED_TRAIN_NEGATIVE_FAMILIES
)

PHASE5_CONFIRMATORY_ADDITION_FAMILIES = (
    "Produce the exact total for {a} together with {b}; return only digits.",
    "What integer results from adding {b} to {a}? Give that integer alone.",
    "Resolve the addition of {a} and {b}. Include no explanatory text.",
    "Take the sum of the pair {a}, {b}, and print only its decimal form.",
    "Apply integer addition to operands {a} and {b}; state just the result.",
    "How many is {a} after an increase of {b}? Answer with one number.",
    "Supply only the arithmetic total obtained from {a} plus {b}.",
    "Combine {a} with {b} additively and output nothing except the answer.",
)

PHASE5_CONFIRMATORY_WORD_FAMILIES = (
    (
        "A lab logged {a} samples in one run and {b} in a second run. "
        "How many samples were logged altogether? Reply with digits only."
    ),
    (
        "A depot stored {a} parcels before receiving {b} additional parcels. "
        "What is the parcel count now? Return only the integer."
    ),
    (
        "One server handled {a} requests and another handled {b}. "
        "How many requests did they handle in total? Give just the number."
    ),
    (
        "A route covered {a} meters in the morning and {b} meters later. "
        "What full distance was covered? State only the result."
    ),
    (
        "A project had {a} records and imported a further {b} records. "
        "How many records does it have now? Use digits only."
    ),
    (
        "Two bins contain {a} and {b} components respectively. "
        "How many components are in both bins? Output only the total."
    ),
)

PHASE5_CONFIRMATORY_NEGATIVE_FAMILIES = (
    "Subtract {b} from {a} and output only the signed result.",
    "Compute the product of {a} and {b}; provide digits only.",
    "Divide {a} by {b} and report only the quotient.",
    "Return the remainder after dividing {a} by {b}.",
    "State which value is smaller, {a} or {b}.",
    "Is {a} greater than {b}? Answer only yes or no.",
    "Print the strings {a} and {b} separated by a vertical bar.",
    "Append the decimal text {b} directly after {a}; do not calculate.",
    "Repeat these two identifiers unchanged: {a}, {b}.",
    'Quote exactly "{a} plus {b}" without answering the quoted expression.',
    "Decline to find {a} + {b}; respond with declined.",
    "Explain the addition symbol using {a} and {b}, but leave the sum unsolved.",
    "Say whether {a} + {b} would exceed 100 without giving the total.",
    "Check the grammar of the expression {a} + {b}; do not evaluate it.",
    "Treat {a} and {b} as serial numbers and list them, not their sum.",
    "Find the arithmetic mean of {a} and {b}, not their sum.",
)


def build_phase5_training_examples(
    *,
    positive_count: int = 2400,
    negative_count: int = 2400,
) -> list[SemanticPromptExample]:
    return (
        make_semantic_addition_examples(
            count=positive_count,
            min_digits=1,
            max_digits=12,
            seed=10_501,
            split="phase5_train_positive",
            families=PHASE5_TRAIN_ADDITION_FAMILIES,
        )
        + make_semantic_routing_negatives(
            count=negative_count,
            min_digits=1,
            max_digits=12,
            seed=10_502,
            split="phase5_train_negative",
            families=PHASE5_TRAIN_NEGATIVE_FAMILIES,
        )
    )


def build_phase5_output_training_examples(
    *,
    count: int = 1600,
) -> list[SemanticPromptExample]:
    return make_semantic_addition_examples(
        count=count,
        min_digits=1,
        max_digits=8,
        seed=10_503,
        split="phase5_output_train",
        families=PHASE5_TRAIN_ADDITION_FAMILIES,
    )


def build_phase5_development_examples(
    *,
    positive_count: int = 400,
    negative_count: int = 400,
) -> list[SemanticPromptExample]:
    return (
        make_semantic_addition_examples(
            count=positive_count,
            min_digits=1,
            max_digits=12,
            seed=10_504,
            split="phase5_development_positive",
            families=DEVELOPMENT_ADDITION_FAMILIES,
        )
        + make_semantic_routing_negatives(
            count=negative_count,
            min_digits=1,
            max_digits=12,
            seed=10_505,
            split="phase5_development_negative",
            families=DEVELOPMENT_NEGATIVE_FAMILIES,
        )
    )


def build_phase5_confirmatory_positive_sets(
    *,
    count_per_split: int = 100,
) -> dict[str, list[SemanticPromptExample]]:
    return {
        "id_1_4": make_semantic_addition_examples(
            count=count_per_split,
            min_digits=1,
            max_digits=4,
            seed=10_551,
            split="phase5_confirmatory_id_1_4",
            families=PHASE5_CONFIRMATORY_ADDITION_FAMILIES,
        ),
        "ood_5_8": make_semantic_addition_examples(
            count=count_per_split,
            min_digits=5,
            max_digits=8,
            seed=10_552,
            split="phase5_confirmatory_ood_5_8",
            families=PHASE5_CONFIRMATORY_ADDITION_FAMILIES,
        ),
        "long_9_12": make_semantic_addition_examples(
            count=count_per_split,
            min_digits=9,
            max_digits=12,
            seed=10_553,
            split="phase5_confirmatory_long_9_12",
            families=PHASE5_CONFIRMATORY_ADDITION_FAMILIES,
        ),
        "word_5_8": make_semantic_addition_examples(
            count=count_per_split,
            min_digits=5,
            max_digits=8,
            seed=10_554,
            split="phase5_confirmatory_word_5_8",
            families=PHASE5_CONFIRMATORY_WORD_FAMILIES,
        ),
    }


def build_phase5_confirmatory_negatives(
    *,
    count: int = 160,
) -> list[SemanticPromptExample]:
    return make_semantic_routing_negatives(
        count=count,
        min_digits=1,
        max_digits=12,
        seed=10_555,
        split="phase5_confirmatory_negative",
        families=PHASE5_CONFIRMATORY_NEGATIVE_FAMILIES,
    )
