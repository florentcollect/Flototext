"""Normalize spoken French numbers to digits."""

import re
from typing import Optional


UNITS = {
    "zero": 0,
    "zéro": 0,
    "un": 1,
    "une": 1,
    "deux": 2,
    "trois": 3,
    "quatre": 4,
    "cinq": 5,
    "six": 6,
    "sept": 7,
    "huit": 8,
    "neuf": 9,
    "dix": 10,
    "onze": 11,
    "douze": 12,
    "treize": 13,
    "quatorze": 14,
    "quinze": 15,
    "seize": 16,
}

TENS = {
    "vingt": 20,
    "vingts": 20,
    "trente": 30,
    "quarante": 40,
    "cinquante": 50,
    "soixante": 60,
}

NUMBER_WORDS = sorted(
    set(UNITS) | set(TENS) | {"et", "cent", "cents", "mille"},
    key=len,
    reverse=True,
)
FIRST_NUMBER_WORDS = [word for word in NUMBER_WORDS if word != "et"]

NUMBER_PATTERN = re.compile(
    r"(?<!\w)("
    + "|".join(re.escape(word) for word in FIRST_NUMBER_WORDS)
    + r")(?:[\s-]+("
    + "|".join(re.escape(word) for word in NUMBER_WORDS)
    + r"))*(?!\w)",
    re.IGNORECASE,
)


def normalize_french_numbers(text: str) -> str:
    """Replace spoken French cardinal numbers with digits.

    This intentionally handles common dictation forms rather than the full
    grammar of French numbers. It covers the cases users dictate most often:
    units, tens, "quatre-vingt", "cent", and "mille" combinations.
    """
    if not text:
        return text

    def replace_match(match: re.Match) -> str:
        value = parse_french_number(match.group(0))
        return str(value) if value is not None else match.group(0)

    return NUMBER_PATTERN.sub(replace_match, text)


def parse_french_number(phrase: str) -> Optional[int]:
    tokens = _tokenize(phrase)
    if not tokens:
        return None
    return _parse_tokens(tokens)


def _tokenize(phrase: str) -> list[str]:
    normalized = phrase.lower().replace("-", " ")
    return [token for token in normalized.split() if token != "et"]


def _parse_tokens(tokens: list[str]) -> Optional[int]:
    if not tokens:
        return None

    if "mille" in tokens:
        index = tokens.index("mille")
        left = tokens[:index]
        right = tokens[index + 1:]
        multiplier = _parse_tokens(left) if left else 1
        remainder = _parse_tokens(right) if right else 0
        if multiplier is None or remainder is None:
            return None
        return multiplier * 1000 + remainder

    if "cent" in tokens or "cents" in tokens:
        index = tokens.index("cent") if "cent" in tokens else tokens.index("cents")
        left = tokens[:index]
        right = tokens[index + 1:]
        multiplier = _parse_tokens(left) if left else 1
        remainder = _parse_tokens(right) if right else 0
        if multiplier is None or remainder is None:
            return None
        return multiplier * 100 + remainder

    return _parse_below_hundred(tokens)


def _parse_below_hundred(tokens: list[str]) -> Optional[int]:
    if not tokens:
        return 0

    if len(tokens) == 1:
        token = tokens[0]
        if token in UNITS:
            return UNITS[token]
        if token in TENS:
            return TENS[token]
        return None

    if tokens[0] == "dix":
        unit = _parse_below_hundred(tokens[1:])
        return 10 + unit if unit is not None and 1 <= unit <= 9 else None

    if tokens[0] == "soixante":
        rest = _parse_below_hundred(tokens[1:])
        return 60 + rest if rest is not None and 1 <= rest <= 19 else None

    if tokens[:2] == ["quatre", "vingt"] or tokens[:2] == ["quatre", "vingts"]:
        if len(tokens) == 2:
            return 80
        rest = _parse_below_hundred(tokens[2:])
        return 80 + rest if rest is not None and 1 <= rest <= 19 else None

    if tokens[0] in TENS:
        rest = _parse_below_hundred(tokens[1:])
        return TENS[tokens[0]] + rest if rest is not None and 1 <= rest <= 9 else None

    return None
