from neural_firmware.pretrained_firmware import FrozenDecimalFirmware, add_decimal_strings


def test_ripple_carry_matches_python_on_large_examples() -> None:
    pairs = [
        ("0", "0"),
        ("9", "1"),
        ("999999999999", "1"),
        ("123456789012", "987654321098"),
    ]
    for a, b in pairs:
        assert add_decimal_strings(a, b) == str(int(a) + int(b))


def test_parser_accepts_registered_templates_only() -> None:
    firmware = FrozenDecimalFirmware()
    parsed = firmware.parse("Calculate exactly. Return only the decimal digits.\n12+30=")
    assert parsed is not None and parsed.answer == "42"
    assert firmware.parse("What is 12 plus 30? Answer with digits only:").answer == "42"
    assert firmware.parse("Add 12 and 30. Give only the exact integer:").answer == "42"
    assert firmware.parse("Do not calculate 12+30. Explain the plus sign.") is None
