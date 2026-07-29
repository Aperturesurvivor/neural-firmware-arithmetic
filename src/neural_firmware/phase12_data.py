from __future__ import annotations

from neural_firmware.phase8_data import operand_pairs
from neural_firmware.phase9_data import Phase9Family, _balanced_custom_examples
from neural_firmware.phase11_data import (
    _all_pairs_before_phase11,
    build_phase11_confirmatory_examples,
)
from neural_firmware.semantic_data import SemanticPromptExample

PHASE12_POSITIVE_DATA_SEED = 22_701
PHASE12_NEGATIVE_DATA_SEED = 22_702


PHASE12_CONFIRMATORY_POSITIVE_FAMILIES = (
    Phase9Family(
        "Add the decimal quantities {x} and {y}. Supply only their integer total.",
        "direct",
    ),
    Phase9Family(
        "Take whole number {x} together with whole number {y}; return the sum alone.",
        "direct",
    ),
    Phase9Family(
        "Update a tally of {x} by adding {y}. Print the updated tally as digits.",
        "direct",
    ),
    Phase9Family(
        "Execute integer SUM with left input {x} and right input {y}. Bare numeral only.",
        "direct",
    ),
    Phase9Family(
        "What number is {y} units beyond {x}? Respond with that number and nothing else.",
        "direct",
    ),
    Phase9Family(
        "Deposit {y} into an accumulator containing {x}; report its new contents.",
        "direct",
    ),
    Phase9Family(
        "Map operands {x} and {y} through the plus operation. Emit one base-ten value.",
        "direct",
    ),
    Phase9Family(
        "Determine the exact result of increasing {x} by the quantity {y}. Digits only.",
        "direct",
    ),
    Phase9Family(
        "Join the amounts {x} and {y} arithmetically by addition, not as text.",
        "direct",
    ),
    Phase9Family(
        "Please furnish the single integer answer to {x} added with {y}.",
        "direct",
    ),
    Phase9Family(
        "A particle counter saw {x} events in the first exposure and {y} in the "
        "second. How many events did it see altogether?",
        "word",
    ),
    Phase9Family(
        "A genome sequencer read {x} fragments before maintenance and {y} after. "
        "Return the complete fragment count.",
        "word",
    ),
    Phase9Family(
        "One battery test logged {x} cycles and a second logged {y}. State their "
        "combined cycles as an integer.",
        "word",
    ),
    Phase9Family(
        "A hydrophone recorded {x} clicks near shore and {y} offshore. Give the "
        "overall number of clicks.",
        "word",
    ),
    Phase9Family(
        "Two compute queues completed {x} and {y} jobs respectively. How many jobs "
        "completed across both queues?",
        "word",
    ),
    Phase9Family(
        "Audit code {c} is not an operand. Add the actual readings {x} and {y}.",
        "distractor",
        True,
    ),
    Phase9Family(
        "Ignore batch marker {c}; bins hold {x} and {y} pieces. Return the combined "
        "piece count.",
        "distractor",
        True,
    ),
    Phase9Family(
        "Sample identifier {c} supplies metadata only. Total observations {x} and {y}.",
        "distractor",
        True,
    ),
    Phase9Family(
        "Exclude tracking value {c}. The two addends are {x} and {y}; give their result.",
        "distractor",
        True,
    ),
    Phase9Family(
        "Run number {c} labels the record. Sensors counted {x} and {y} signals; "
        "report the aggregate.",
        "distractor",
        True,
    ),
)


PHASE12_CONFIRMATORY_NEGATIVE_FAMILIES = (
    Phase9Family(
        "Do not use addition: multiply {x} by {y} and return the product.",
        "multiplication",
    ),
    Phase9Family(
        "The required operation for {x} and {y} is PRODUCT, not SUM.",
        "multiplication",
    ),
    Phase9Family(
        "Process factors {x}, {y} with integer multiplication.",
        "multiplication",
    ),
    Phase9Family(
        "Find how much {x} times {y} equals; no additive answer is wanted.",
        "multiplication",
    ),
    Phase9Family(
        "Issue a MUL command for operands {x} and {y}.",
        "multiplication",
    ),
    Phase9Family(
        "Badges numbered {x} and {y} were issued. Confirm that two badges exist.",
        "factual",
    ),
    Phase9Family(
        "The ledger references accounts {x} and {y}. Are both account labels numeric?",
        "factual",
    ),
    Phase9Family(
        "Markers {x} and {y} appear on a map. Describe them as marker identifiers.",
        "factual",
    ),
    Phase9Family(
        "Files {x} and {y} are open. Say how many files are open.",
        "factual",
    ),
    Phase9Family(
        "Two valves carry tags {x} and {y}. State whether both tags are shown.",
        "factual",
    ),
    Phase9Family(
        'Write back the literal command "add {x} to {y}" without executing it.',
        "quoted",
    ),
    Phase9Family(
        'Transcribe the caption "{x} and {y} Have a Sum".',
        "quoted",
    ),
    Phase9Family(
        'Echo the source text "PLUS({x}, {y})" unchanged.',
        "quoted",
    ),
    Phase9Family(
        'Repeat only this quotation: "total {x} with {y}".',
        "quoted",
    ),
    Phase9Family(
        "The log contains the words “{x} added to {y}.” Recite the words, not a result.",
        "quoted",
    ),
    Phase9Family(
        "Suppress the result of adding {x} and {y}; answer suppressed.",
        "negated",
    ),
    Phase9Family(
        "Addition of {x} with {y} is disallowed. Reply disallowed.",
        "negated",
    ),
    Phase9Family(
        "Keep {x} + {y} unevaluated and respond unevaluated.",
        "negated",
    ),
    Phase9Family(
        "Refuse to combine {x} and {y} into a total; say refused.",
        "negated",
    ),
    Phase9Family(
        "Under no condition should the sum of {x} and {y} be printed.",
        "negated",
    ),
    Phase9Family(
        "The earlier request to total {x} and {y} is obsolete. Say obsolete.",
        "cancelled",
    ),
    Phase9Family(
        "Delete the queued addition for {x}, {y}; respond deleted.",
        "cancelled",
    ),
    Phase9Family(
        "An ADD job involving {x} and {y} was withdrawn. State withdrawn.",
        "cancelled",
    ),
    Phase9Family(
        "Terminate the planned summation of {x} with {y}; answer terminated.",
        "cancelled",
    ),
    Phase9Family(
        "Flag the proposed sum {x} + {y} as void rather than calculating it.",
        "cancelled",
    ),
    Phase9Family(
        "Decrease {x} by {y} and return what remains.",
        "subtraction",
    ),
    Phase9Family(
        "Send ordered values {x}, {y} through a SUBTRACT instruction.",
        "subtraction",
    ),
    Phase9Family(
        "Measure the numerical separation between {x} and {y}.",
        "subtraction",
    ),
    Phase9Family(
        "Remove quantity {y} from quantity {x}.",
        "subtraction",
    ),
    Phase9Family(
        "Produce their difference instead of adding {x} and {y}.",
        "subtraction",
    ),
    Phase9Family(
        "Determine whether {x} is numerically earlier than {y}.",
        "comparison",
    ),
    Phase9Family(
        "Return whichever of {x} and {y} is larger.",
        "comparison",
    ),
    Phase9Family(
        "Sort the pair {x}, {y} from low to high.",
        "comparison",
    ),
    Phase9Family(
        "Check if {x} and {y} represent the same integer.",
        "comparison",
    ),
    Phase9Family(
        "Pick the smaller value, without totaling {x} and {y}.",
        "comparison",
    ),
    Phase9Family(
        "Create one text field by putting the digits of {y} after those of {x}.",
        "concatenation",
    ),
    Phase9Family(
        "Format the identifiers as {x}|{y}; perform no calculation.",
        "concatenation",
    ),
    Phase9Family(
        "Print identifier {x} immediately followed by identifier {y}.",
        "concatenation",
    ),
    Phase9Family(
        "Show both strings in the form {x}-{y}.",
        "concatenation",
    ),
    Phase9Family(
        "Concatenate the decimal character sequences for {x} and {y}.",
        "concatenation",
    ),
    Phase9Family(
        "Consider a hypothetical request to add {x} and {y}; name it but do not run it.",
        "hypothetical",
    ),
    Phase9Family(
        "Would {x} and {y} be valid inputs to addition? Answer yes or no only.",
        "hypothetical",
    ),
    Phase9Family(
        "Explain whether totaling {x} with {y} could carry, without finding the total.",
        "hypothetical",
    ),
    Phase9Family(
        "Identify the symbol between {x} + {y}; leave the expression unresolved.",
        "hypothetical",
    ),
    Phase9Family(
        "If one added {x} and {y}, would a number result? Do not perform the addition.",
        "hypothetical",
    ),
    Phase9Family(
        "Inventory code {c} lists components {x} and {y}. State how many components "
        "are listed.",
        "distractor",
        True,
    ),
    Phase9Family(
        "Report {c} contains references {x} and {y}. Confirm that three numbers appear.",
        "distractor",
        True,
    ),
    Phase9Family(
        "Facility {c} has doors labeled {x} and {y}. Repeat only the door labels.",
        "distractor",
        True,
    ),
    Phase9Family(
        "Dataset {c} names records {x} and {y}. Are two records named?",
        "distractor",
        True,
    ),
    Phase9Family(
        "Route {c} includes stops {x} and {y}. Say whether both stop names are numeric.",
        "distractor",
        True,
    ),
)


def _all_pairs_before_phase12() -> set[tuple[str, str]]:
    return _all_pairs_before_phase11() | operand_pairs(
        build_phase11_confirmatory_examples()
    )


def build_phase12_confirmatory_examples() -> list[SemanticPromptExample]:
    used = _all_pairs_before_phase12()
    positives = _balanced_custom_examples(
        PHASE12_CONFIRMATORY_POSITIVE_FAMILIES,
        examples_per_family=5,
        seed=PHASE12_POSITIVE_DATA_SEED,
        split_prefix="phase12_confirmatory_positive",
        route_label=True,
        forbidden=used,
    )
    used.update(operand_pairs(positives))
    negatives = _balanced_custom_examples(
        PHASE12_CONFIRMATORY_NEGATIVE_FAMILIES,
        examples_per_family=4,
        seed=PHASE12_NEGATIVE_DATA_SEED,
        split_prefix="phase12_confirmatory_negative",
        route_label=False,
        forbidden=used,
    )
    return positives + negatives


def phase12_family_set() -> set[str]:
    return {
        family.template
        for family in (
            PHASE12_CONFIRMATORY_POSITIVE_FAMILIES
            + PHASE12_CONFIRMATORY_NEGATIVE_FAMILIES
        )
    }
