from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass

from neural_firmware.pretrained_data import chat_prompt_ids

TRAIN_ADDITION_FAMILIES = (
    "Add {a} and {b}. Reply with only the number.",
    "What is {a} plus {b}? Give only the numeric answer.",
    "Calculate {a} + {b}. Respond with digits only.",
    "Find the sum of {a} and {b}. Return just the result.",
    "Please add {a} to {b}. Answer only with the integer.",
    "{a} plus {b} equals what? Output only the number.",
    "Compute the total of {a} and {b}. Use digits only.",
    "Give the exact result of adding {a} and {b}. Number only.",
)

HELDOUT_ADDITION_FAMILIES = (
    "How much is {a} added to {b}? State only the answer.",
    "Determine {a} + {b}, writing nothing except the result.",
    "Combine the values {a} and {b} by addition. Give the integer alone.",
    "Sum these two integers: {a} and {b}. Reply solely with the total.",
    "Evaluate the addition expression {a} + {b}. Only digits in your response.",
    "Increase {a} by {b}. What number results? Return only that number.",
    "The addends are {a} and {b}. Supply their sum and nothing else.",
    "Perform integer addition on {a} with {b}. Print just the answer.",
)

WORD_PROBLEM_FAMILIES = (
    (
        "A warehouse has {a} parts and receives {b} more. "
        "How many parts are there now? Answer with only the number."
    ),
    (
        "A library owns {a} books and acquires another {b}. "
        "What is the new total? Give only the number."
    ),
    (
        "A game account has {a} coins and earns {b} additional coins. "
        "How many coins does it have? Respond only with the total."
    ),
    (
        "There are {a} people in one hall and {b} in another. "
        "How many people are there altogether? Return just the integer."
    ),
    (
        "A sensor counted {a} events before noon and {b} after noon. "
        "What was the full-day count? Use digits only."
    ),
    (
        "A factory completed {a} units, then completed {b} more. "
        "How many units were completed in total? State only the result."
    ),
)

TRAIN_NEGATIVE_FAMILIES = (
    "Do not add {a} and {b}. Reply only with the word skipped.",
    'The phrase "{a} plus {b}" appears in a document. Reply only with quoted.',
    "Which is larger, {a} or {b}? Reply only with first or second.",
    "Subtract {b} from {a}. Reply only with the resulting integer.",
    "Write {a} followed by {b}. Do not perform arithmetic.",
    "Are {a} and {b} both integers? Reply only with yes or no.",
    "Multiply {a} by {b}. Return only the product.",
    "Explain what the plus sign between {a} and {b} means without solving it.",
)

HELDOUT_NEGATIVE_FAMILIES = (
    "Without calculating anything, repeat these values: {a} and {b}.",
    'Someone asked "What is {a} plus {b}?" Do not answer it; reply ignored.',
    "Find the difference between {a} and {b}. Give only the answer.",
    "Compare {a} with {b}. State only whether the first is larger.",
    "Concatenate the decimal strings {a} and {b}; do not add them.",
    "Describe a method for adding {a} and {b}, but do not compute the result.",
    "Is {a} + {b} syntactically an addition expression? Answer yes or no.",
    "The values {a} and {b} are labels, not quantities. Reply only labels.",
)

# These development families were added after semantic pilot v1. The original
# held-out families above were consumed by that pilot and may now participate
# in router development, but they remain separate for exact reproduction.
AUGMENTED_TRAIN_ADDITION_FAMILIES = (
    "Obtain the total when {a} is incremented by {b}. Return only that total.",
    "What single number do {a} and {b} make when summed?",
    "Take the integer {a}, add {b}, and print the resulting integer alone.",
    "Starting from {a}, move upward by {b}. Give just the endpoint.",
    "Produce only the decimal representation of {a} added to {b}.",
    "Apply the addition operation to {a} and {b}; output its result only.",
    "The first addend is {a}; the second is {b}. What is their total?",
    "Please calculate the combined quantity of {a} together with {b}.",
    (
        "A counter reads {a} and advances another {b} steps. "
        "What does it read now? Number only."
    ),
    (
        "A shipment contains {a} boxes, with {b} more arriving. "
        "State the final box count only."
    ),
    (
        "One register records {a} events and another records {b}. "
        "Give their aggregate count."
    ),
    (
        "A collection begins with {a} items and gains {b} items. "
        "Respond only with its new size."
    ),
)

AUGMENTED_TRAIN_NEGATIVE_FAMILIES = (
    "Calculate {a} less {b}. Give only the answer.",
    "What remains after taking {b} away from {a}? Output only the number.",
    "Compute {a} minus {b}. Return the result alone.",
    "Return the product of {a} and {b}, not their sum.",
    "Divide {a} by {b} and give the quotient if it is integral.",
    "Select the smaller of {a} and {b}. Reply with that value only.",
    "Select the maximum of {a} and {b}. Give only the chosen value.",
    "Repeat {a} and {b} exactly as written without calculating.",
    "List the two numbers {a} and {b}; do not combine them.",
    'Quote the question "What is {a} plus {b}?" without answering it.',
    "Do not solve {a} + {b}; answer only with the word withheld.",
    "Say whether adding {a} and {b} would be legal, but do not add them.",
    "Explain how carrying would work for {a} and {b} without finding the sum.",
    "Join the text {a} to the text {b} as one decimal string.",
    "Are the integers {a} and {b} equal? Answer yes or no.",
    "How far apart are {a} and {b}? Give their absolute difference.",
)

DEVELOPMENT_ADDITION_FAMILIES = (
    "Report the sum obtained from the pair {a} and {b}. Output the answer alone.",
    "By adding {b} onto {a}, what value is reached? Give only that value.",
    "Resolve this integer sum: {a} plus {b}. No words in the response.",
    "What is the additive combination of {a} with {b}? Return digits only.",
    "Put {a} and {b} together arithmetically. State their total.",
    "Supply the numeric outcome after increasing {a} by {b}.",
    (
        "A tank holds {a} liters and is filled with {b} more liters. "
        "What is its new volume? Number only."
    ),
    (
        "A team scored {a} points early and {b} points later. "
        "How many points did it score overall? Give just the total."
    ),
)

DEVELOPMENT_NEGATIVE_FAMILIES = (
    "Deduct {b} from {a} and print the answer.",
    "Measure the numerical gap separating {a} and {b}.",
    "Square {a}, then ignore {b}. Return only the square.",
    "Place a slash between {a} and {b}; perform no calculation.",
    "Tell me if {a} exceeds {b}. Answer yes or no.",
    "Restate the expression {a} + {b}, but leave it unevaluated.",
    "Discuss whether {a} plus {b} requires carrying. Do not give the total.",
    'The title of a book is "{a} Plus {b}". Repeat only the title.',
)

# Frozen-confirmation candidates. Pilot scripts must never import these names.
CONFIRMATORY_ADDITION_FAMILIES = (
    "Add together the integers {a} and {b}; provide nothing but the outcome.",
    "What does the sum operation yield for {a} alongside {b}? Digits only.",
    "Give the number produced by augmenting {a} with {b}.",
    "Find their combined arithmetic value: {a}, {b}. Answer only with it.",
    "Compute the additive result for the pair {a} and {b}.",
    "When {b} is added onto {a}, where do you end up? Number only.",
    "Return only the integer equal to the total of {a} plus {b}.",
    "Evaluate the sum whose operands are {a} and {b}; omit all explanation.",
)

CONFIRMATORY_WORD_PROBLEM_FAMILIES = (
    (
        "A ledger shows {a} incoming units and later records {b} more. "
        "How many incoming units are recorded altogether? Answer only with digits."
    ),
    (
        "A trail is {a} meters long and receives a {b}-meter extension. "
        "What is its new length? Return only the number."
    ),
    (
        "A study enrolled {a} people in one cohort and {b} in another. "
        "How many people were enrolled in all? Give just the total."
    ),
    (
        "A device processed {a} jobs, then processed {b} additional jobs. "
        "State the complete job count only."
    ),
    (
        "A fund contains {a} dollars and gets a further {b} dollars. "
        "What is the balance now? Respond with the integer alone."
    ),
    (
        "Two archives hold {a} and {b} records respectively. "
        "How many records do they hold together? Use digits only."
    ),
)

CONFIRMATORY_NEGATIVE_FAMILIES = (
    "Calculate the result of subtracting {b} from {a}. Number only.",
    "How much greater is {a} than {b}? Return the signed difference.",
    "Multiply the two operands {a} and {b}; output only their product.",
    "Give the remainder when {a} is divided by {b}.",
    "Choose whichever number is larger: {a} or {b}.",
    "Determine whether {a} is less than {b}. Reply true or false.",
    "Write {a}, then a comma, then {b}. Do not calculate.",
    "Merge the character sequences {a} and {b} without arithmetic.",
    "Echo these inputs unchanged: {a} and {b}.",
    'Repeat the quoted expression "{a} + {b}" and do not evaluate it.',
    "Refuse to calculate {a} plus {b}; reply only refused.",
    "Define addition using {a} and {b} as examples without solving the example.",
    "Would the sum of {a} and {b} be even? Answer yes or no without stating it.",
    "Check whether {a} + {b} is well formed, but do not evaluate it.",
    "Treat {a} and {b} as identification codes and list both codes.",
    "Return the average of {a} and {b}, not the total.",
)

DECIMAL_PATTERN = re.compile(r"(?<![0-9])[0-9]+(?![0-9])")


@dataclass(frozen=True)
class SemanticPromptExample:
    prompt: str
    a: str
    b: str
    answer: str | None
    route_label: bool
    family: str
    family_index: int
    split: str

    def to_dict(self) -> dict[str, str | int | bool | None]:
        return asdict(self)


@dataclass(frozen=True)
class EncodedSemanticPrompt:
    input_ids: tuple[int, ...]
    a_digits: tuple[int, ...]
    b_digits: tuple[int, ...]


def _random_decimal(rng: random.Random, digits: int) -> str:
    if digits < 1:
        raise ValueError("digits must be positive")
    if digits == 1:
        return str(rng.randrange(10))
    return str(rng.randrange(10 ** (digits - 1), 10**digits))


def locate_two_decimal_operands(prompt: str) -> tuple[str, str] | None:
    """Return two candidate decimal spans without deciding the requested operation."""

    matches = DECIMAL_PATTERN.findall(prompt)
    if len(matches) != 2:
        return None
    return matches[0], matches[1]


def encode_semantic_prompt(tokenizer: object, prompt: str) -> EncodedSemanticPrompt:
    operands = locate_two_decimal_operands(prompt)
    if operands is None:
        raise ValueError("semantic prompt must contain exactly two decimal spans")
    a, b = operands
    return EncodedSemanticPrompt(
        input_ids=tuple(chat_prompt_ids(tokenizer, prompt)),
        a_digits=tuple(int(character) for character in a),
        b_digits=tuple(int(character) for character in b),
    )


def render_family(
    families: tuple[str, ...],
    family_index: int,
    a: str,
    b: str,
) -> str:
    return families[family_index].format(a=a, b=b)


def make_semantic_addition_examples(
    *,
    count: int,
    min_digits: int,
    max_digits: int,
    seed: int,
    split: str,
    families: tuple[str, ...],
) -> list[SemanticPromptExample]:
    rng = random.Random(seed)
    examples: list[SemanticPromptExample] = []
    for _ in range(count):
        a = _random_decimal(rng, rng.randint(min_digits, max_digits))
        b = _random_decimal(rng, rng.randint(min_digits, max_digits))
        family_index = rng.randrange(len(families))
        prompt = render_family(families, family_index, a, b)
        located = locate_two_decimal_operands(prompt)
        if located is None:
            raise ValueError("positive family did not render exactly two operands")
        examples.append(
            SemanticPromptExample(
                prompt=prompt,
                a=located[0],
                b=located[1],
                answer=str(int(a) + int(b)),
                route_label=True,
                family=families[family_index],
                family_index=family_index,
                split=split,
            )
        )
    return examples


def make_semantic_routing_negatives(
    *,
    count: int,
    min_digits: int,
    max_digits: int,
    seed: int,
    split: str,
    families: tuple[str, ...],
) -> list[SemanticPromptExample]:
    rng = random.Random(seed)
    examples: list[SemanticPromptExample] = []
    for _ in range(count):
        a = _random_decimal(rng, rng.randint(min_digits, max_digits))
        b = _random_decimal(rng, rng.randint(min_digits, max_digits))
        family_index = rng.randrange(len(families))
        prompt = render_family(families, family_index, a, b)
        located = locate_two_decimal_operands(prompt)
        if located is None:
            raise ValueError("negative family did not render exactly two operands")
        examples.append(
            SemanticPromptExample(
                prompt=prompt,
                a=located[0],
                b=located[1],
                answer=None,
                route_label=False,
                family=families[family_index],
                family_index=family_index,
                split=split,
            )
        )
    return examples


def last_integer(text: str) -> str | None:
    """Return the final decimal integer in generated text, allowing commas."""

    matches = re.findall(r"(?<![0-9])[0-9][0-9,]*(?![0-9])", text)
    if not matches:
        return None
    candidate = matches[-1].replace(",", "")
    return candidate if candidate.isdigit() else None


def mathematical_correct(text: str, expected: str) -> bool:
    return last_integer(text) == expected


def exact_format_correct(text: str, expected: str) -> bool:
    return text.strip() == expected
