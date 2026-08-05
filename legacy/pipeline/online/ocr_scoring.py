from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

GENERIC_TEXT_TOKENS = {
    "de",
    "du",
    "des",
    "la",
    "le",
    "les",
    "quai",
    "quais",
    "repere",
    "reperes",
    "voie",
    "voies",
    "track",
    "tracks",
    "platform",
    "platforms",
    "gate",
    "gates",
    "marker",
    "markers",
    "sign",
    "signe",
    "panneau",
    "hall",
    "level",
    "niveau",
}


@dataclass(frozen=True)
class OcrIdentityQuery:
    numbers: frozenset[str]
    letters: frozenset[str]
    alphanumerics: frozenset[str]

    @property
    def has_identity(self) -> bool:
        return bool(self.numbers or self.letters or self.alphanumerics)


@dataclass(frozen=True)
class OcrCandidate:
    numbers: frozenset[str]
    letters: frozenset[str]
    tokens: frozenset[str]
    key_numbers: frozenset[str]
    key_letters: frozenset[str]


def normalize_ocr_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = text.replace("÷", " ").replace("+", " ").replace(":", " ").replace("|", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_ocr_query(text: str) -> OcrIdentityQuery:
    normalized = normalize_ocr_text(text)
    tokens = tuple(normalized.split())
    numbers = set(re.findall(r"\d+", normalized))
    letters: set[str] = set()
    alphanumerics: set[str] = set()

    for token in tokens:
        has_alpha = bool(re.search(r"[a-z]", token))
        has_digit = bool(re.search(r"\d", token))
        if has_alpha and has_digit:
            alphanumerics.add(token)
            alpha = "".join(re.findall(r"[a-z]+", token))
            digit = "".join(re.findall(r"\d+", token))
            if alpha and alpha not in GENERIC_TEXT_TOKENS:
                letters.add(alpha)
            if digit:
                numbers.add(digit)
            continue

        if has_alpha:
            alpha = "".join(re.findall(r"[a-z]+", token))
            if alpha and alpha not in GENERIC_TEXT_TOKENS and len(alpha) <= 2:
                letters.add(alpha)

    return OcrIdentityQuery(
        numbers=frozenset(numbers),
        letters=frozenset(letters),
        alphanumerics=frozenset(alphanumerics),
    )


def _parse_ocr_key(key: str) -> tuple[set[str], set[str]]:
    numbers: set[str] = set()
    letters: set[str] = set()
    for chunk in str(key).split(";"):
        name, sep, raw_values = chunk.partition("=")
        if not sep:
            continue
        values = {normalize_ocr_text(value) for value in raw_values.split(",")}
        values = {value for value in values if value}
        name = name.strip()
        if name == "numbers":
            numbers.update(values)
        elif name == "letters":
            letters.update(values)
    return numbers, letters


def _candidate_from_ocr(
    *, ocr_key: str = "", ocr_tokens: str = "", ocr_text: str = ""
) -> OcrCandidate:
    key_numbers, key_letters = _parse_ocr_key(ocr_key)
    normalized_payload = normalize_ocr_text(" ".join([str(ocr_tokens), str(ocr_text)]))
    tokens = set(normalized_payload.split())
    numbers = set(key_numbers)
    letters = set(key_letters)

    if normalized_payload:
        numbers.update(re.findall(r"\d+", normalized_payload))
        for token in tokens:
            alpha = "".join(re.findall(r"[a-z]+", token))
            if alpha and alpha not in GENERIC_TEXT_TOKENS and len(alpha) <= 2:
                letters.add(alpha)

    return OcrCandidate(
        numbers=frozenset(numbers),
        letters=frozenset(letters),
        tokens=frozenset(tokens),
        key_numbers=frozenset(key_numbers),
        key_letters=frozenset(key_letters),
    )


def score_ocr_candidate(
    query: OcrIdentityQuery,
    *,
    ocr_key: str = "",
    ocr_tokens: str = "",
    ocr_text: str = "",
) -> float:
    if not query.has_identity:
        return 0.0

    candidate = _candidate_from_ocr(
        ocr_key=ocr_key,
        ocr_tokens=ocr_tokens,
        ocr_text=ocr_text,
    )
    if not candidate.numbers and not candidate.letters and not candidate.tokens:
        return 0.0

    if (
        query.numbers
        and candidate.numbers
        and query.numbers.isdisjoint(candidate.numbers)
    ):
        return 0.0

    number_match = bool(
        query.numbers and not query.numbers.isdisjoint(candidate.key_numbers)
    )
    any_number_match = bool(
        query.numbers and not query.numbers.isdisjoint(candidate.numbers)
    )
    number_token_match = bool(
        query.numbers and not query.numbers.isdisjoint(candidate.tokens)
    )
    letter_match = bool(
        query.letters and not query.letters.isdisjoint(candidate.key_letters)
    )
    any_letter_match = bool(
        query.letters and not query.letters.isdisjoint(candidate.letters)
    )
    alnum_token_match = bool(
        query.alphanumerics and not query.alphanumerics.isdisjoint(candidate.tokens)
    )

    if alnum_token_match:
        return 1.0
    if query.numbers and query.letters:
        if number_match and letter_match:
            return 1.0
        if any_number_match and any_letter_match:
            return 0.90
        if number_match or any_number_match:
            return 0.85
        if number_token_match:
            return 0.70
        return 0.0
    if number_match:
        return 1.0
    if any_number_match or number_token_match:
        return 0.80
    if query.letters and letter_match:
        return 0.70
    if query.letters and any_letter_match:
        return 0.60
    return 0.0


def best_ocr_score(
    query: OcrIdentityQuery,
    candidates: Iterable[tuple[str, str, str]],
) -> float:
    best = 0.0
    for ocr_key, ocr_tokens, ocr_text in candidates:
        best = max(
            best,
            score_ocr_candidate(
                query,
                ocr_key=ocr_key,
                ocr_tokens=ocr_tokens,
                ocr_text=ocr_text,
            ),
        )
        if best >= 1.0:
            break
    return float(best)
