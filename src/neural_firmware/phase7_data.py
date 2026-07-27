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

# Frozen only after audit v1 had been consumed for layer and handshake
# engineering. Audit v2 must remain disjoint from both prior project families
# and every audit-v1 construction.
PHASE7_AUDIT2_ADDITION_FAMILIES = (
    (
        "Using ordinary integer arithmetic, combine {a} with {b}. "
        "Your entire reply must be the resulting numeral."
    ),
    (
        "Advance the value {a} by exactly {b}. "
        "At what integer do you arrive? Print digits alone."
    ),
    (
        "The two terms of a sum are {a} and {b}. "
        "Supply only the sum's decimal representation."
    ),
    "Perform ADD on inputs {a} and {b}; emit one bare integer.",
    (
        "How many units does {a} become after gaining {b} units? "
        "Give just the final count."
    ),
    (
        "For this prompt, the operation joining {a} and {b} is addition. "
        "Output its value and no other text."
    ),
    (
        "Accumulate {b} on top of an initial {a}. "
        "Respond exclusively with the new integer."
    ),
    (
        "Start at {b} and move forward by {a} on the number line. "
        "Where do you finish? Number only."
    ),
    (
        "An addition operator receives left input {a} and right input {b}. "
        "Return only its output."
    ),
    (
        "Determine the combined count represented by {a} together with {b}. "
        "No explanation; use digits."
    ),
)

PHASE7_AUDIT2_WORD_FAMILIES = (
    (
        "A weather station recorded {a} pulses during daylight and {b} after "
        "dark. How many pulses were recorded altogether? Reply only in digits."
    ),
    (
        "An orchard packed {a} boxes on Monday and {b} boxes on Tuesday. "
        "Report the two-day box total as a bare integer."
    ),
    (
        "A train carried {a} riders before a stop and boarded {b} more there. "
        "How many riders are aboard now? Output the number alone."
    ),
    (
        "A rendering job finished {a} frames, then completed another {b}. "
        "State the complete frame count using digits only."
    ),
    (
        "A research freezer held {a} vials and received a shipment of {b} "
        "vials. What is the new vial count? Give only the integer."
    ),
    (
        "A radio sent {a} bytes in one burst and {b} bytes in the next. "
        "How many bytes did both bursts send? Return just the total."
    ),
    (
        "A stadium admitted {a} guests through the north gate and {b} through "
        "the south gate. Give the overall guest count, number only."
    ),
    (
        "A workshop assembled {a} devices before lunch and {b} afterward. "
        "How many devices were assembled that day? Respond with one numeral."
    ),
    (
        "A reservoir contained {a} liters before another {b} liters flowed in. "
        "What volume does it contain now? Print digits and nothing else."
    ),
    (
        "A compiler reported {a} warnings in one module and {b} in another. "
        "State the aggregate warning count as an integer alone."
    ),
)

PHASE7_AUDIT2_NEGATIVE_FAMILIES = (
    "Remove {b} units from {a}; return only the signed remainder.",
    "Find {a} times {b}. Your reply should contain only the product.",
    "Compute the integer quotient when {a} is divided by {b}.",
    "Return {a} modulo {b}, not any additive result.",
    "Select the greater identifier from {a} and {b}; echo that identifier.",
    "Is {a} no smaller than {b}? Reply true or false.",
    "Write the pair {a} and {b} inside parentheses without calculating.",
    "Place the character sequence for {b} immediately before {a}.",
    "Copy these catalog labels verbatim and separately: {a}; {b}.",
    'Reproduce the text "{a} plus {b}" but do not answer the expression.',
    "Politely refuse the request to add {a} and {b}; say only unable.",
    "Teach the procedure for adding {a} to {b} while withholding its answer.",
    "Would adding {a} and {b} require a carry? Do not reveal the total.",
    "Classify {a} + {b} as an arithmetic expression without evaluating it.",
    "Interpret {a} and {b} as room numbers and list both room numbers.",
    "Calculate the geometric mean of {a} and {b}, rather than their sum.",
    "State how many integers lie strictly between {a} and {b}.",
    "Reverse the decimal digits of {a}; leave {b} unchanged after it.",
    "Choose whichever of {a} or {b} is closer to zero.",
    "Say whether {a} and {b} have the same number of decimal digits.",
)

# Shared multi-seed holdout, frozen only after all three checkpoints existed.
PHASE7_AUDIT3_ADDITION_FAMILIES = (
    (
        "Addend alpha is {a}; addend beta is {b}. "
        "Write their resulting integer and nothing else."
    ),
    (
        "Treat {a} and {b} as quantities to be totaled. "
        "Reply with only the total's digits."
    ),
    (
        "Increment a register holding {a} by the amount {b}. "
        "Return its final integer value alone."
    ),
    (
        "Apply base-ten addition between {a} and {b}. "
        "The response must consist solely of the answer."
    ),
    (
        "Give the scalar obtained by summing {a} with {b}. "
        "Use one bare decimal numeral."
    ),
    (
        "Move {b} steps upward from {a} on an integer scale. "
        "State only the destination."
    ),
    (
        "The requested total combines a first amount of {a} and a second "
        "amount of {b}. Output digits only."
    ),
    (
        "Evaluate the additive operator whose arguments are {a} and {b}. "
        "Do not include prose."
    ),
    (
        "What is the whole-number accumulation of {a} followed by {b}? "
        "Print only that whole number."
    ),
    (
        "Produce the decimal integer for the total of {b} alongside {a}; "
        "nothing except the numeral."
    ),
)

PHASE7_AUDIT3_WORD_FAMILIES = (
    (
        "A seismograph detected {a} tremors in one interval and {b} in the "
        "next. How many tremors did it detect overall? Digits only."
    ),
    (
        "A bakery made {a} loaves before opening and {b} more later. "
        "Give the day's loaf count as one bare integer."
    ),
    (
        "An observatory captured {a} exposures on one night and {b} on "
        "another. State the combined exposure count, number only."
    ),
    (
        "A ferry transported {a} vehicles eastbound and {b} westbound. "
        "How many vehicles did it transport in total? Return only digits."
    ),
    (
        "A pharmacy prepared {a} doses in the morning and {b} in the evening. "
        "Report the full dose count with no words."
    ),
    (
        "A drone saved {a} photographs on its first flight and {b} on its "
        "second. How many photographs were saved? Output the integer alone."
    ),
    (
        "A theater sold {a} balcony tickets and {b} floor tickets. "
        "Give the total tickets sold, using only a numeral."
    ),
    (
        "A laboratory incubated {a} cultures, then started {b} additional "
        "cultures. What is the culture count now? Number only."
    ),
    (
        "A canal moved {a} boats before noon and {b} after noon. "
        "State the complete boat count as digits and nothing else."
    ),
    (
        "A backup copied {a} files from one drive and {b} from another. "
        "How many files were copied altogether? Reply with one integer."
    ),
)

PHASE7_AUDIT3_NEGATIVE_FAMILIES = (
    "Decrease {a} by {b}; print the resulting signed integer.",
    "Square {a} and then append the unchanged label {b}.",
    "Multiply {a} with {b}; answer solely with their product.",
    "Find the greatest common divisor of {a} and {b}.",
    "Which value is nearer to one hundred, {a} or {b}? Return that value.",
    "Does {a} divide {b} evenly? Respond yes or no.",
    "Display {a} and {b} separated by a forward slash, without arithmetic.",
    "Concatenate {a} after {b} as text, with no numeric operation.",
    "Repeat the two inventory codes {a} and {b} in their original order.",
    'Copy the literal sentence "{a} added to {b}" without solving it.',
    "Reject the proposal to total {a} and {b}; respond only rejected.",
    "Explain column addition using {a} and {b}, but stop before the answer.",
    "Could the sum of {a} and {b} overflow one byte? Do not calculate it.",
    "Identify the operator in {a} + {b}, leaving the expression untouched.",
    "Interpret {a} and {b} as highway route labels and repeat both labels.",
    "Calculate the harmonic mean of {a} and {b}, not their total.",
    "How many decimal places appear in {a} and {b}? Do not combine them.",
    "Rotate the digits of {a} left once, then print {b} separately.",
    "Choose the operand with fewer digits: {a} or {b}.",
    "State whether {a} and {b} are congruent modulo two.",
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


def build_phase7_audit2_examples(
    *,
    symbolic_count: int = 30,
    word_count: int = 30,
    negative_count: int = 60,
) -> list[SemanticPromptExample]:
    return (
        make_semantic_addition_examples(
            count=symbolic_count,
            min_digits=1,
            max_digits=4,
            seed=13_301,
            split="phase7_audit2_symbolic",
            families=PHASE7_AUDIT2_ADDITION_FAMILIES,
        )
        + make_semantic_addition_examples(
            count=word_count,
            min_digits=1,
            max_digits=4,
            seed=13_302,
            split="phase7_audit2_word",
            families=PHASE7_AUDIT2_WORD_FAMILIES,
        )
        + make_semantic_routing_negatives(
            count=negative_count,
            min_digits=1,
            max_digits=4,
            seed=13_303,
            split="phase7_audit2_negative",
            families=PHASE7_AUDIT2_NEGATIVE_FAMILIES,
        )
    )


def phase7_audit2_prior_family_sets() -> tuple[set[str], set[str]]:
    prior_positive, prior_negative = phase7_audit_family_sets()
    prior_positive.update(
        PHASE7_AUDIT_ADDITION_FAMILIES + PHASE7_AUDIT_WORD_FAMILIES
    )
    prior_negative.update(PHASE7_AUDIT_NEGATIVE_FAMILIES)
    return prior_positive, prior_negative


def build_phase7_audit3_examples(
    *,
    symbolic_count: int = 30,
    word_count: int = 30,
    negative_count: int = 60,
) -> list[SemanticPromptExample]:
    return (
        make_semantic_addition_examples(
            count=symbolic_count,
            min_digits=1,
            max_digits=4,
            seed=13_401,
            split="phase7_audit3_symbolic",
            families=PHASE7_AUDIT3_ADDITION_FAMILIES,
        )
        + make_semantic_addition_examples(
            count=word_count,
            min_digits=1,
            max_digits=4,
            seed=13_402,
            split="phase7_audit3_word",
            families=PHASE7_AUDIT3_WORD_FAMILIES,
        )
        + make_semantic_routing_negatives(
            count=negative_count,
            min_digits=1,
            max_digits=4,
            seed=13_403,
            split="phase7_audit3_negative",
            families=PHASE7_AUDIT3_NEGATIVE_FAMILIES,
        )
    )


def phase7_audit3_prior_family_sets() -> tuple[set[str], set[str]]:
    prior_positive, prior_negative = phase7_audit2_prior_family_sets()
    prior_positive.update(
        PHASE7_AUDIT2_ADDITION_FAMILIES + PHASE7_AUDIT2_WORD_FAMILIES
    )
    prior_negative.update(PHASE7_AUDIT2_NEGATIVE_FAMILIES)
    return prior_positive, prior_negative
