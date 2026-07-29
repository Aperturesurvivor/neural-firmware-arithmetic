from __future__ import annotations

from neural_firmware.phase8_data import operand_pairs
from neural_firmware.phase9_data import (
    Phase9Family,
    _balanced_custom_examples,
)
from neural_firmware.phase10_data import (
    PHASE10_TRAINING_SEEDS,
    _all_prior_pairs,
    build_phase10_confirmatory_examples,
)
from neural_firmware.semantic_data import SemanticPromptExample

PHASE11_SOURCE_SEEDS = PHASE10_TRAINING_SEEDS
PHASE11_ROUTER_SEEDS = {
    16_201: 17_201,
    16_202: 17_202,
    16_203: 17_203,
}
PHASE11_POSITIVE_DATA_SEED = 17_701
PHASE11_NEGATIVE_DATA_SEED = 17_702


PHASE11_CONFIRMATORY_POSITIVE_FAMILIES = (
    Phase9Family(
        "Resolve ADD for decimal inputs {x} and {y}; emit the scalar result alone.",
        "direct",
    ),
    Phase9Family(
        "Transform the pair {x}, {y} by integer addition. Return one numeral.",
        "direct",
    ),
    Phase9Family(
        "Begin at {x} and advance the count by {y}. Print the attained count.",
        "direct",
    ),
    Phase9Family(
        "Fold addend {y} into addend {x}; give only their base-ten total.",
        "direct",
    ),
    Phase9Family(
        "The requested binary operation is SUM on {x} and {y}. Output its value.",
        "direct",
    ),
    Phase9Family(
        "Set a counter to {x}, increment it {y} times, and state the final integer.",
        "direct",
    ),
    Phase9Family(
        "Reduce the additive expression with operands {x} and {y} to digits only.",
        "direct",
    ),
    Phase9Family(
        "Return the exact integer reached by increasing {x} through amount {y}.",
        "direct",
    ),
    Phase9Family(
        "Compute one combined quantity from {x} and {y} using plus, not times.",
        "direct",
    ),
    Phase9Family(
        "Evaluate {x} ADD {y}; the response must contain only the answer.",
        "direct",
    ),
    Phase9Family(
        "A radio array detected {x} pulses before a retune and {y} after it. "
        "How many pulses did it detect in total?",
        "word",
    ),
    Phase9Family(
        "A bioreactor logged {x} cells in one sample and {y} in another. "
        "Give the combined cell count as an integer.",
        "word",
    ),
    Phase9Family(
        "A satellite downlinked {x} packets on one orbit and {y} on the next. "
        "Return the two-orbit packet total.",
        "word",
    ),
    Phase9Family(
        "An imaging run produced {x} slices in its first block and {y} in its "
        "second. State the complete slice count.",
        "word",
    ),
    Phase9Family(
        "A greenhouse counted {x} seedlings in one bay and {y} in a second bay. "
        "How many seedlings were counted altogether?",
        "word",
    ),
    Phase9Family(
        "Tracking number {c} is not data. Add the observed values {x} and {y}.",
        "distractor",
        True,
    ),
    Phase9Family(
        "Discard file code {c}; counters A and B read {x} and {y}. Return their sum.",
        "distractor",
        True,
    ),
    Phase9Family(
        "Run label {c} identifies the trial only. Total measurements {x} and {y}.",
        "distractor",
        True,
    ),
    Phase9Family(
        "Metadata field {c} must be excluded. The addends are {x} and {y}.",
        "distractor",
        True,
    ),
    Phase9Family(
        "Ignore specimen ID {c}; batches contained {x} and {y} units. "
        "Give the aggregate units.",
        "distractor",
        True,
    ),
)


PHASE11_CONFIRMATORY_NEGATIVE_FAMILIES = (
    Phase9Family("Evaluate the product operation on {x} and {y}.", "multiplication"),
    Phase9Family("Compute {x} times {y}; addition is not requested.", "multiplication"),
    Phase9Family("Send operands {x}, {y} through a MULTIPLY instruction.", "multiplication"),
    Phase9Family("Return the value of {x} repeated {y} times multiplicatively.", "multiplication"),
    Phase9Family("Use factors {x} and {y} to produce a product only.", "multiplication"),
    Phase9Family(
        "Lockers bear numbers {x} and {y}. Confirm that two lockers are referenced.",
        "factual",
    ),
    Phase9Family(
        "The manifest names lots {x} and {y}. Are both lot names numerical?",
        "factual",
    ),
    Phase9Family(
        "Pins {x} and {y} are marked on the board. Describe them as pin labels.",
        "factual",
    ),
    Phase9Family(
        "Two pages, {x} and {y}, are flagged. State how many pages are flagged.",
        "factual",
    ),
    Phase9Family(
        "The form displays reference numbers {x} and {y}. Say whether both appear.",
        "factual",
    ),
    Phase9Family('Copy the literal string "add {x} and {y}" without acting on it.', "quoted"),
    Phase9Family('Reproduce the title "{x} Plus {y}: A History".', "quoted"),
    Phase9Family('Echo the unevaluated notation "SUM({x}, {y})".', "quoted"),
    Phase9Family('Quote the sentence "{x} combined with {y} gives a total."', "quoted"),
    Phase9Family(
        "A transcript contains “increase {x} by {y}.” Repeat those words only.",
        "quoted",
    ),
    Phase9Family("Withhold the additive result for {x} and {y}; reply withheld.", "negated"),
    Phase9Family("You must not add {x} to {y}. State only prohibited.", "negated"),
    Phase9Family("Leave {x} + {y} unresolved and answer unresolved.", "negated"),
    Phase9Family("Reject any attempt to total {x} and {y}; say rejected.", "negated"),
    Phase9Family("Never output the sum involving {x} and {y}.", "negated"),
    Phase9Family("A prior ADD request for {x} and {y} has expired. Say expired.", "cancelled"),
    Phase9Family("Remove the scheduled summation of {x}, {y}; respond removed.", "cancelled"),
    Phase9Family("The instruction totaling {x} with {y} was revoked.", "cancelled"),
    Phase9Family("Do not execute the abandoned addition of {x} and {y}.", "cancelled"),
    Phase9Family("Mark the pending sum for {x} and {y} as canceled.", "cancelled"),
    Phase9Family("Calculate {x} after decreasing it by {y}.", "subtraction"),
    Phase9Family("Apply a MINUS operation to ordered inputs {x}, {y}.", "subtraction"),
    Phase9Family("Return the magnitude of the gap from {x} to {y}.", "subtraction"),
    Phase9Family("Find what remains when {y} is removed from {x}.", "subtraction"),
    Phase9Family("Obtain the difference, rather than the sum, of {x} and {y}.", "subtraction"),
    Phase9Family("Report whether {x} comes before {y} numerically.", "comparison"),
    Phase9Family("Select the greater member of the pair {x}, {y}.", "comparison"),
    Phase9Family("Rank {x} and {y} in ascending order.", "comparison"),
    Phase9Family("Test the equality relation between {x} and {y}.", "comparison"),
    Phase9Family("Choose the lower value from {x} and {y}; do no addition.", "comparison"),
    Phase9Family("Form a text token by placing {y} directly after {x}.", "concatenation"),
    Phase9Family("Render the pair as {x}::{y} with no arithmetic.", "concatenation"),
    Phase9Family("Write identifier {x} followed by identifier {y}.", "concatenation"),
    Phase9Family("Output both strings {x} and {y}, separated by a dash.", "concatenation"),
    Phase9Family("Join the decimal characters in {x} with those in {y}.", "concatenation"),
    Phase9Family(
        "Imagine a later instruction to total {x} and {y}; identify its operation "
        "without evaluating it.",
        "hypothetical",
    ),
    Phase9Family(
        "Could {x} and {y} be operands of addition? Answer yes or no, with no result.",
        "hypothetical",
    ),
    Phase9Family(
        "Discuss whether adding {x} to {y} might require a carry; calculate nothing.",
        "hypothetical",
    ),
    Phase9Family(
        "Name the operator in the expression {x} + {y} but leave it unsolved.",
        "hypothetical",
    ),
    Phase9Family(
        "If {x} and {y} were summed, would the output be an integer? Do not sum them.",
        "hypothetical",
    ),
    Phase9Family(
        "Asset record {c} lists serials {x} and {y}. State how many serials are listed.",
        "distractor",
        True,
    ),
    Phase9Family(
        "Log {c} mentions channels {x} and {y}. Confirm that three numbers occur.",
        "distractor",
        True,
    ),
    Phase9Family(
        "Building {c} contains rooms labeled {x} and {y}. Repeat only room labels.",
        "distractor",
        True,
    ),
    Phase9Family(
        "Catalog {c} references entries {x} and {y}. Are two entries referenced?",
        "distractor",
        True,
    ),
    Phase9Family(
        "Survey {c} names sites {x} and {y}. Say whether both site names are numeric.",
        "distractor",
        True,
    ),
)


def _all_pairs_before_phase11() -> set[tuple[str, str]]:
    return _all_prior_pairs() | operand_pairs(build_phase10_confirmatory_examples())


def build_phase11_confirmatory_examples() -> list[SemanticPromptExample]:
    used = _all_pairs_before_phase11()
    positives = _balanced_custom_examples(
        PHASE11_CONFIRMATORY_POSITIVE_FAMILIES,
        examples_per_family=5,
        seed=PHASE11_POSITIVE_DATA_SEED,
        split_prefix="phase11_confirmatory_positive",
        route_label=True,
        forbidden=used,
    )
    used.update(operand_pairs(positives))
    negatives = _balanced_custom_examples(
        PHASE11_CONFIRMATORY_NEGATIVE_FAMILIES,
        examples_per_family=4,
        seed=PHASE11_NEGATIVE_DATA_SEED,
        split_prefix="phase11_confirmatory_negative",
        route_label=False,
        forbidden=used,
    )
    return positives + negatives


def phase11_family_set() -> set[str]:
    return {
        family.template
        for family in (
            PHASE11_CONFIRMATORY_POSITIVE_FAMILIES
            + PHASE11_CONFIRMATORY_NEGATIVE_FAMILIES
        )
    }
