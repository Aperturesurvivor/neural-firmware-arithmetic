from __future__ import annotations

from neural_firmware.phase5_data import (
    PHASE5_CONFIRMATORY_ADDITION_FAMILIES,
    PHASE5_CONFIRMATORY_NEGATIVE_FAMILIES,
    PHASE5_CONFIRMATORY_WORD_FAMILIES,
    PHASE5_TRAIN_ADDITION_FAMILIES,
    PHASE5_TRAIN_NEGATIVE_FAMILIES,
)
from neural_firmware.semantic_data import (
    CONFIRMATORY_ADDITION_FAMILIES,
    CONFIRMATORY_NEGATIVE_FAMILIES,
    CONFIRMATORY_WORD_PROBLEM_FAMILIES,
    DEVELOPMENT_ADDITION_FAMILIES,
    DEVELOPMENT_NEGATIVE_FAMILIES,
    SemanticPromptExample,
    make_semantic_addition_examples,
    make_semantic_routing_negatives,
)

# Frozen before the Phase 7 held-out audit. These wordings are disjoint from
# every family used to train or tune the v2 sequence interface and from the
# earlier Phase 4/5 confirmation families.
PHASE7_AUDIT_ADDITION_FAMILIES = (
    (
        "Return the base-ten total after combining {a} with {b}; "
        "write one numeral only."
    ),
    (
        "Beginning with {a}, count forward another {b}. "
        "Where do you land? Digits only."
    ),
    (
        "The requested operation is addition: operands {a} and {b}. "
        "Emit the result alone."
    ),
    "Give the exact integer at {a} + {b}, without a sentence.",
    (
        "Aggregate {a} units and {b} units. "
        "Respond solely with the aggregate."
    ),
    (
        "Put {b} on top of {a} by ordinary integer addition. "
        "Output only the endpoint."
    ),
    (
        "What total has {a} as its first addend and {b} as its second? "
        "Just the number."
    ),
    "Evaluate only this sum, {a} + {b}, and return its decimal digits.",
)

PHASE7_AUDIT_WORD_FAMILIES = (
    (
        "A telescope catalog already contains {a} objects and the next survey "
        "adds {b}. How many objects are cataloged afterward? Give only digits."
    ),
    (
        "A battery delivered {a} watt-hours before noon and {b} watt-hours "
        "afterward. State the day's combined watt-hours as one integer."
    ),
    (
        "A museum counted {a} visitors through one entrance and {b} through "
        "another. What was the combined attendance? Number only."
    ),
    (
        "A robot placed {a} tiles, then placed {b} additional tiles. "
        "How many tiles did it place altogether? Reply with the numeral."
    ),
    (
        "A clinic scheduled {a} appointments and later booked {b} more. "
        "What is the resulting appointment count? Output only that count."
    ),
    (
        "A spacecraft transmitted {a} packets on its first pass and {b} on "
        "its next pass. How many packets were transmitted in all? Digits only."
    ),
    (
        "A grower harvested {a} crates from one field and {b} from a second. "
        "Report the whole harvest in crates, using only a number."
    ),
    (
        "A map database had {a} entries before an update inserted {b} new "
        "entries. How many entries does it contain now? Return the integer."
    ),
)

PHASE7_AUDIT_NEGATIVE_FAMILIES = (
    "Take {b} away from {a}; print only what remains.",
    "Raise {a} to the second power and disregard {b}. Give the numeral.",
    "Form the product whose factors are {a} and {b}. Answer with digits.",
    "Which operand comes first numerically, {a} or {b}? Name that operand only.",
    "Test whether {a} is at most {b}. Respond yes or no.",
    "Render {a} and {b} with a colon between them; do no arithmetic.",
    "Glue the decimal characters of {a} to those of {b}. Return that text.",
    "Copy both reference codes, {a} and {b}, without treating them as amounts.",
    'Echo the inscription "{a} + {b}" exactly; leave it unsolved.',
    "A student proposed adding {a} to {b}. Do not solve it; answer postponed.",
    (
        "Describe in words how one could total {a} and {b}, "
        "but omit the resulting total."
    ),
    "Is the unseen sum of {a} and {b} odd? Do not state the sum.",
    "Does the notation {a} + {b} use a binary operator? Reply yes or no.",
    "Treat {a} and {b} as page numbers and list both, not a calculation.",
    "Compute the midpoint of {a} and {b} rather than adding them.",
    "Give the absolute distance from {a} to {b}; output only that distance.",
)


def build_phase7_audit_examples(
    *,
    symbolic_count: int = 20,
    word_count: int = 20,
    negative_count: int = 40,
) -> list[SemanticPromptExample]:
    return (
        make_semantic_addition_examples(
            count=symbolic_count,
            min_digits=1,
            max_digits=4,
            seed=12_901,
            split="phase7_audit_symbolic",
            families=PHASE7_AUDIT_ADDITION_FAMILIES,
        )
        + make_semantic_addition_examples(
            count=word_count,
            min_digits=1,
            max_digits=4,
            seed=12_902,
            split="phase7_audit_word",
            families=PHASE7_AUDIT_WORD_FAMILIES,
        )
        + make_semantic_routing_negatives(
            count=negative_count,
            min_digits=1,
            max_digits=4,
            seed=12_903,
            split="phase7_audit_negative",
            families=PHASE7_AUDIT_NEGATIVE_FAMILIES,
        )
    )


def phase7_audit_family_sets() -> tuple[set[str], set[str]]:
    prior_positive = set(
        PHASE5_TRAIN_ADDITION_FAMILIES
        + DEVELOPMENT_ADDITION_FAMILIES
        + CONFIRMATORY_ADDITION_FAMILIES
        + CONFIRMATORY_WORD_PROBLEM_FAMILIES
        + PHASE5_CONFIRMATORY_ADDITION_FAMILIES
        + PHASE5_CONFIRMATORY_WORD_FAMILIES
    )
    prior_negative = set(
        PHASE5_TRAIN_NEGATIVE_FAMILIES
        + DEVELOPMENT_NEGATIVE_FAMILIES
        + CONFIRMATORY_NEGATIVE_FAMILIES
        + PHASE5_CONFIRMATORY_NEGATIVE_FAMILIES
    )
    return prior_positive, prior_negative
