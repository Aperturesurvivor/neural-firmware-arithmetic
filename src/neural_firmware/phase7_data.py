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

# Router-hardening confirmation families. These were written only after the
# three hardened checkpoint hashes had been frozen.
PHASE7_AUDIT4_ADDITION_FAMILIES = (
    (
        "Collapse the two integer amounts {a} and {b} into their sum. "
        "The sum alone should be your response."
    ),
    (
        "A numeric accumulator currently contains {a}; deposit {b} more. "
        "Print its resulting contents as digits only."
    ),
    (
        "Use the operation SUM with arguments {a} and {b}. "
        "Return the integer output without commentary."
    ),
    (
        "Take an initial magnitude of {a} and enlarge it by {b}. "
        "Give the ending magnitude as one numeral."
    ),
    (
        "Resolve the additive combination whose left value is {a} and whose "
        "right value is {b}. Emit only the answer."
    ),
    (
        "Merge {a} units with a further {b} units by arithmetic addition. "
        "Respond with nothing except the combined integer."
    ),
    (
        "Starting at {a}, advance exactly {b} integer positions. "
        "Which integer is reached? Supply digits only."
    ),
    (
        "Find the base-ten numeral representing {a} together with {b} as "
        "addends. No prose is permitted."
    ),
    (
        "The desired scalar is the additive total of {b} and {a}. "
        "Write that scalar and nothing else."
    ),
    (
        "Perform ordinary whole-number summation on {a} and {b}; "
        "make the complete reply a single integer."
    ),
)

PHASE7_AUDIT4_WORD_FAMILIES = (
    (
        "A sonar station logged {a} echoes before dawn and {b} after dawn. "
        "How many echoes were logged altogether? Answer with digits only."
    ),
    (
        "A ceramic studio fired {a} pieces in one kiln and {b} in another. "
        "State the total pieces fired as a bare integer."
    ),
    (
        "A wildlife camera stored {a} images on one card and {b} on a second "
        "card. How many images were stored in all? Number only."
    ),
    (
        "A relief team delivered {a} meals at one site and {b} at another. "
        "Report the combined meal count without words."
    ),
    (
        "A satellite received {a} commands during one orbit and {b} during "
        "the next. Give the complete command count as one numeral."
    ),
    (
        "A print shop bound {a} manuals yesterday and {b} today. "
        "How many manuals did it bind across both days? Return only digits."
    ),
    (
        "A greenhouse planted {a} seedlings in its east wing and {b} in its "
        "west wing. What is the overall seedling count? Integer only."
    ),
    (
        "A network monitor saw {a} alerts on one link and {b} on another. "
        "State how many alerts it saw altogether, using no prose."
    ),
    (
        "An excavation cataloged {a} fragments in one trench, then {b} more "
        "in another. What is the catalog total now? Give just the number."
    ),
    (
        "A recycling center processed {a} cans in the morning and {b} in the "
        "afternoon. Output the day's full can count as digits alone."
    ),
)

PHASE7_AUDIT4_NEGATIVE_FAMILIES = (
    "Reduce {a} by {b}; return the signed difference only.",
    "Take the product of {a} and {b}; provide one integer.",
    "Divide {a} by {b} and report the quotient rather than a total.",
    "Raise {a} to a power while treating {b} only as an unused label.",
    "Select the lesser value from {a} and {b}; print the selected value.",
    "Is {a} strictly greater than {b}? Answer yes or no.",
    "Determine whether {a} and {b} have matching parity.",
    "Give the distance between {a} and {b} on the number line.",
    "Write {a} followed by a hyphen and then {b}; do not calculate.",
    "Join the text of {b} directly onto the text of {a}.",
    "Treat {a} and {b} as specimen identifiers and repeat both identifiers.",
    'Transcribe the quoted request "add {a} to {b}" without carrying it out.',
    "Decline the instruction to sum {a} with {b}; reply only declined.",
    "Explain why addition is commutative using {a} and {b}, but omit their sum.",
    "Say whether adding {a} to {b} would increase the digit count; do not add.",
    "Check that the expression {a} + {b} is syntactically valid, not its value.",
    "A note says that {a} was added to {b}; copy that note without solving it.",
    "Although the words mention totaling {a} and {b}, perform multiplication.",
    "Do not combine {a} and {b}; state which one has more decimal digits.",
    "Find the remainder when the larger of {a} and {b} is divided by the smaller.",
    "Return the arithmetic average of {a} and {b}, not their combined amount.",
    "Sort the labels {a} and {b} in ascending order without summing them.",
    "Count how many digit characters occur across {a} and {b}; do no addition.",
    "Would the unseen total of {a} and {b} be even? Do not reveal the total.",
    "Name the first operand in {a} + {b} while leaving the expression unsolved.",
    "Replace {a} with {b} in a register; this is assignment, not arithmetic.",
    "Reverse {a} and then display {b} unchanged beside it.",
    "Compute the greatest common factor shared by {a} and {b}.",
    "Test whether {a} is a multiple of {b}; respond with a Boolean.",
    "A user previously asked to add {a} and {b}; answer only that the request expired.",
)

# Operand-register confirmation families. These were written only after the
# register implementation and its consumed audit-4 diagnostic were complete.
PHASE7_AUDIT5_ADDITION_FAMILIES = (
    (
        "Add the cardinalities {a} and {b}, then return their cardinality sum "
        "as a decimal integer alone."
    ),
    (
        "An integer buffer holds {a} and receives an increment of {b}. "
        "Output only the buffer's updated value."
    ),
    (
        "Compute the additive result of the ordered pair ({a}, {b}). "
        "Your reply must contain just the resulting numeral."
    ),
    (
        "Let the starting count be {a}; include {b} additional units. "
        "State the final count with digits and no words."
    ),
    (
        "Use standard integer addition to fold {a} and {b} into one value. "
        "Emit that value alone."
    ),
    (
        "Determine where a counter at {a} lands after moving ahead by {b}. "
        "Give only the destination integer."
    ),
    (
        "The required arithmetic combines {a} as one addend with {b} as the "
        "other. Return the exact sum without prose."
    ),
    (
        "Produce one base-ten number equal to the total contributed by {a} "
        "and {b}. Nothing else should appear."
    ),
    (
        "Apply an additive update of {b} to the value {a}. "
        "Print the updated whole number only."
    ),
    (
        "Evaluate the sum operation on inputs {a} and {b}; "
        "make your full answer its decimal digits."
    ),
)

PHASE7_AUDIT5_WORD_FAMILIES = (
    (
        "A weather balloon collected {a} readings while rising and {b} while "
        "descending. How many readings were collected overall? Digits only."
    ),
    (
        "A food bank packed {a} parcels on one shift and {b} on the next. "
        "Give the combined parcel count as a bare integer."
    ),
    (
        "A microscope saved {a} frames from one sample and {b} from another. "
        "How many frames did it save in all? Return only the number."
    ),
    (
        "A bus depot dispatched {a} vehicles before noon and {b} afterward. "
        "State the full vehicle count without any words."
    ),
    (
        "A seed vault stored {a} packets in one chamber and {b} in another. "
        "What is the total packet count? Answer with one numeral."
    ),
    (
        "A climbing gym recorded {a} visits on Saturday and {b} on Sunday. "
        "Output the weekend visit count using digits alone."
    ),
    (
        "A data logger wrote {a} records to one file and {b} to a second. "
        "How many records were written altogether? Number only."
    ),
    (
        "A repair shop completed {a} jobs in its first queue and {b} in its "
        "second. Give the complete job count as an integer alone."
    ),
    (
        "A marine survey marked {a} locations in one zone and {b} in another. "
        "Report the overall marked-location count with no prose."
    ),
    (
        "A concert venue scanned {a} tickets at one door and {b} at another. "
        "How many tickets were scanned in total? Reply only with digits."
    ),
)

PHASE7_AUDIT5_NEGATIVE_FAMILIES = (
    "Subtract {b} from {a}; answer with only the difference.",
    "Multiply the quantities {a} and {b}; emit their product.",
    "Find the integer division result for {a} over {b}.",
    "Compute {a} modulo {b}, with no additive calculation.",
    "Choose the numerically larger member of {a} and {b}.",
    "Are {a} and {b} relatively prime? Reply yes or no.",
    "Determine whether {a} has fewer digits than {b}.",
    "Place {a} and {b} on separate lines without combining them.",
    "Concatenate the decimal spelling of {a} before that of {b}.",
    "Use {a} and {b} as database keys and echo both keys unchanged.",
    'Proofread the sentence "the sum of {a} and {b}" without evaluating it.',
    "Refuse to calculate {a} plus {b}; respond only unavailable.",
    "Describe a mental-addition strategy for {a} and {b} but never give its answer.",
    "Could adding {a} and {b} create a carry? Answer without stating the sum.",
    "Label the operands in {a} + {b} as left and right; leave it unsolved.",
    "A transcript claims that {a} plus {b} was requested; summarize that claim only.",
    "Ignore the proposed sum of {a} and {b}; instead report that it was canceled.",
    "Despite mentioning addition, compare {a} against {b} and select the first if larger.",
    "Return the midpoint between {a} and {b}, rather than their total.",
    "Give the absolute difference separating {a} from {b}.",
    "Order the codes {a} and {b} lexicographically, not arithmetically.",
    "Count the total number of decimal digits written in {a} and {b}.",
    "Would the sum of {a} and {b} be divisible by three? Do not compute the sum.",
    "State which operator appears between {a} and {b} in a hypothetical subtraction.",
    "Overwrite a slot containing {a} with {b}; print only the replacement value.",
    "Rotate the decimal representation of {a}, then show the untouched tag {b}.",
    "Find the least common multiple of {a} and {b}.",
    "Test whether {a} divides {b}; return true or false.",
    "The pending request to total {a} and {b} is on hold; reply only pending.",
    "Quote the phrase that {a} was added to {b}, but do not produce a number.",
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


def build_phase7_audit4_examples(
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
            seed=13_601,
            split="phase7_audit4_symbolic",
            families=PHASE7_AUDIT4_ADDITION_FAMILIES,
        )
        + make_semantic_addition_examples(
            count=word_count,
            min_digits=1,
            max_digits=4,
            seed=13_602,
            split="phase7_audit4_word",
            families=PHASE7_AUDIT4_WORD_FAMILIES,
        )
        + make_semantic_routing_negatives(
            count=negative_count,
            min_digits=1,
            max_digits=4,
            seed=13_603,
            split="phase7_audit4_negative",
            families=PHASE7_AUDIT4_NEGATIVE_FAMILIES,
        )
    )


def phase7_audit4_prior_family_sets() -> tuple[set[str], set[str]]:
    prior_positive, prior_negative = phase7_audit3_prior_family_sets()
    prior_positive.update(
        PHASE7_AUDIT3_ADDITION_FAMILIES + PHASE7_AUDIT3_WORD_FAMILIES
    )
    prior_negative.update(PHASE7_AUDIT3_NEGATIVE_FAMILIES)
    return prior_positive, prior_negative


def build_phase7_audit5_examples(
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
            seed=13_701,
            split="phase7_audit5_symbolic",
            families=PHASE7_AUDIT5_ADDITION_FAMILIES,
        )
        + make_semantic_addition_examples(
            count=word_count,
            min_digits=1,
            max_digits=4,
            seed=13_702,
            split="phase7_audit5_word",
            families=PHASE7_AUDIT5_WORD_FAMILIES,
        )
        + make_semantic_routing_negatives(
            count=negative_count,
            min_digits=1,
            max_digits=4,
            seed=13_703,
            split="phase7_audit5_negative",
            families=PHASE7_AUDIT5_NEGATIVE_FAMILIES,
        )
    )


def phase7_audit5_prior_family_sets() -> tuple[set[str], set[str]]:
    prior_positive, prior_negative = phase7_audit4_prior_family_sets()
    prior_positive.update(
        PHASE7_AUDIT4_ADDITION_FAMILIES + PHASE7_AUDIT4_WORD_FAMILIES
    )
    prior_negative.update(PHASE7_AUDIT4_NEGATIVE_FAMILIES)
    return prior_positive, prior_negative
