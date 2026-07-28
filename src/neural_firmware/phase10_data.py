from __future__ import annotations

from neural_firmware.phase8_data import operand_pairs
from neural_firmware.phase9_data import (
    PHASE9_SOURCE_SEEDS,
    Phase9Family,
    _balanced_custom_examples,
    _phase8_used_pairs,
    build_phase9_confirmatory_examples,
    build_phase9_development,
    build_phase9_generic_training,
    build_phase9_hard_training,
)
from neural_firmware.semantic_data import SemanticPromptExample

PHASE10_TRAINING_SEEDS = (16_201, 16_202, 16_203)
PHASE10_SOURCE_SEEDS = {
    seed: PHASE9_SOURCE_SEEDS[phase9_seed]
    for seed, phase9_seed in zip(
        PHASE10_TRAINING_SEEDS,
        (15_201, 15_202, 15_203),
        strict=True,
    )
}
PHASE10_DEVELOPMENT_SEED = 16_199


PHASE10_CONFIRMATORY_POSITIVE_FAMILIES = (
    Phase9Family(
        "Apply integer addition to {x} and {y}; print the resulting scalar alone.",
        "direct",
    ),
    Phase9Family(
        "Raise the counter value {x} by exactly {y}. Return its final reading.",
        "direct",
    ),
    Phase9Family(
        "Combine operands {x} and {y} under ADD and emit only the decimal output.",
        "direct",
    ),
    Phase9Family(
        "Starting with {x}, take {y} unit steps upward. Give the endpoint in digits.",
        "direct",
    ),
    Phase9Family(
        "Evaluate the additive composition whose inputs are {x} and {y}. Number only.",
        "direct",
    ),
    Phase9Family(
        "Load {x} into a register and accumulate {y}. Report the register afterward.",
        "direct",
    ),
    Phase9Family(
        "Map the ordered values {x}, {y} to their exact sum. Omit all prose.",
        "direct",
    ),
    Phase9Family(
        "Increase the integer state {x} using increment {y}; output the new integer.",
        "direct",
    ),
    Phase9Family(
        "Consolidate the addends {x} and {y} into one base-ten value.",
        "direct",
    ),
    Phase9Family(
        "Produce only the numeral obtained when {y} is added to {x}.",
        "direct",
    ),
    Phase9Family(
        "A seismograph captured {x} traces before midnight and {y} afterward. "
        "How many traces were captured altogether? Digits only.",
        "word",
    ),
    Phase9Family(
        "A laboratory processed {x} samples on one bench and {y} on another. "
        "State the complete sample count.",
        "word",
    ),
    Phase9Family(
        "A telescope stored {x} exposures in the first session and {y} in the "
        "second. Return the total exposures.",
        "word",
    ),
    Phase9Family(
        "A depot shipped {x} crates in the morning and {y} later that day. "
        "Give the full-day crate count.",
        "word",
    ),
    Phase9Family(
        "A simulation completed {x} trials on one node and {y} on another. "
        "How many trials completed in all?",
        "word",
    ),
    Phase9Family(
        "Reference tag {c} is metadata. Add only measurements {x} and {y}.",
        "distractor",
        True,
    ),
    Phase9Family(
        "Experiment ID {c} is not an operand; its runs yielded {x} and {y} events. "
        "Return the combined yield.",
        "distractor",
        True,
    ),
    Phase9Family(
        "Ignore timestamp {c}. The quantities requiring addition are {x} and {y}.",
        "distractor",
        True,
    ),
    Phase9Family(
        "Record key {c} labels the row. Total the actual counts {x} and {y}.",
        "distractor",
        True,
    ),
    Phase9Family(
        "Device code {c} provides context only; channels reported {x} and {y} hits. "
        "Give the aggregate hit count.",
        "distractor",
        True,
    ),
)


PHASE10_CONFIRMATORY_NEGATIVE_FAMILIES = (
    Phase9Family("Compute the multiplicative product of {x} and {y}.", "multiplication"),
    Phase9Family("Run a multiplication instruction on operands {x}, {y}.", "multiplication"),
    Phase9Family("Treat {x} and {y} as factors and return their product.", "multiplication"),
    Phase9Family("Find {x} multiplied by {y}; do not add them.", "multiplication"),
    Phase9Family("Use MUL to combine {x} with {y}.", "multiplication"),
    Phase9Family(
        "Cabinets numbered {x} and {y} are locked. Are both numbers displayed?",
        "factual",
    ),
    Phase9Family(
        "The index lists entries {x} and {y}. Confirm that two entries are listed.",
        "factual",
    ),
    Phase9Family(
        "Labels {x} and {y} mark two cables. Identify them as cable labels.",
        "factual",
    ),
    Phase9Family(
        "Seats {x} and {y} appear on the tickets. Are two seats named?",
        "factual",
    ),
    Phase9Family(
        "Specimens carry codes {x} and {y}. State whether both codes are numeric.",
        "factual",
    ),
    Phase9Family('Transcribe the phrase "{x} added to {y}" without solving it.', "quoted"),
    Phase9Family('Copy the displayed text "{x} + {y}" exactly as text.', "quoted"),
    Phase9Family('Repeat the heading "Addition of {x} and {y}".', "quoted"),
    Phase9Family('Echo the literal instruction "sum {x} with {y}".', "quoted"),
    Phase9Family(
        "A note says, “combine {x} and {y} by addition.” Quote the note only.",
        "quoted",
    ),
    Phase9Family("Do not total {x} and {y}; answer rejected.", "negated"),
    Phase9Family("Calculating {x} plus {y} is forbidden. Reply forbidden.", "negated"),
    Phase9Family("Avoid producing the sum for {x} with {y}.", "negated"),
    Phase9Family("Decline the request to add {x} and {y}.", "negated"),
    Phase9Family("No additive answer for {x} and {y} may be returned.", "negated"),
    Phase9Family("The instruction to add {x} and {y} was rescinded.", "cancelled"),
    Phase9Family("Cancel the queued ADD operation on {x}, {y}.", "cancelled"),
    Phase9Family("The proposed total of {x} with {y} is annulled.", "cancelled"),
    Phase9Family("Abort addition for operands {x} and {y}; say aborted.", "cancelled"),
    Phase9Family("The sum request involving {x} and {y} is no longer active.", "cancelled"),
    Phase9Family("Return the result of {x} minus {y}.", "subtraction"),
    Phase9Family("Subtract operand {y} from operand {x}.", "subtraction"),
    Phase9Family("Measure the absolute distance between {x} and {y}.", "subtraction"),
    Phase9Family("Execute SUB with inputs {x}, {y}.", "subtraction"),
    Phase9Family("Find the signed difference separating {x} and {y}.", "subtraction"),
    Phase9Family("Say whether {x} is greater than {y}.", "comparison"),
    Phase9Family("Choose the minimum of {x} and {y}.", "comparison"),
    Phase9Family("Order {x} and {y} from greatest to least.", "comparison"),
    Phase9Family("Test whether the two values {x} and {y} match.", "comparison"),
    Phase9Family("Compare {x} against {y} without combining them.", "comparison"),
    Phase9Family("Concatenate decimal strings {x} and {y}.", "concatenation"),
    Phase9Family("Print {x}, then a colon, then {y}.", "concatenation"),
    Phase9Family("Append the characters of {y} after those of {x}.", "concatenation"),
    Phase9Family("Display the pair {x}; {y} without arithmetic.", "concatenation"),
    Phase9Family("Repeat both identifiers as {x} / {y}.", "concatenation"),
    Phase9Family(
        "If asked tomorrow to add {x} and {y}, name the operation now without running it.",
        "hypothetical",
    ),
    Phase9Family(
        "Would combining {x} and {y} by addition yield an integer? Do not calculate.",
        "hypothetical",
    ),
    Phase9Family(
        "Explain what the plus sign would mean between {x} and {y}; omit the value.",
        "hypothetical",
    ),
    Phase9Family(
        "Can values {x} and {y} serve as addends? Answer yes or no only.",
        "hypothetical",
    ),
    Phase9Family(
        "Describe a possible carry when adding {x} and {y}, but find no answer.",
        "hypothetical",
    ),
    Phase9Family(
        "Registry {c} contains item labels {x} and {y}. How many labels are mentioned?",
        "distractor",
        True,
    ),
    Phase9Family(
        "Document {c} displays figures {x} and {y}; confirm that three numerals appear.",
        "distractor",
        True,
    ),
    Phase9Family(
        "Unit {c} has ports labeled {x} and {y}. Repeat only the word ports.",
        "distractor",
        True,
    ),
    Phase9Family(
        "During year {c}, stations {x} and {y} opened. Are two stations named?",
        "distractor",
        True,
    ),
    Phase9Family(
        "Table {c} includes columns {x} and {y}. State whether two columns are named.",
        "distractor",
        True,
    ),
)


def _all_prior_pairs() -> set[tuple[str, str]]:
    phase9 = (
        build_phase9_generic_training()
        + build_phase9_hard_training()
        + build_phase9_development()
        + build_phase9_confirmatory_examples()
    )
    return _phase8_used_pairs() | operand_pairs(phase9)


def build_phase10_confirmatory_examples() -> list[SemanticPromptExample]:
    used = _all_prior_pairs()
    positives = _balanced_custom_examples(
        PHASE10_CONFIRMATORY_POSITIVE_FAMILIES,
        examples_per_family=5,
        seed=16_701,
        split_prefix="phase10_confirmatory_positive",
        route_label=True,
        forbidden=used,
    )
    used.update(operand_pairs(positives))
    negatives = _balanced_custom_examples(
        PHASE10_CONFIRMATORY_NEGATIVE_FAMILIES,
        examples_per_family=4,
        seed=16_702,
        split_prefix="phase10_confirmatory_negative",
        route_label=False,
        forbidden=used,
    )
    return positives + negatives


def phase10_family_set() -> set[str]:
    return {
        family.template
        for family in (
            PHASE10_CONFIRMATORY_POSITIVE_FAMILIES
            + PHASE10_CONFIRMATORY_NEGATIVE_FAMILIES
        )
    }
