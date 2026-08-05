"""OCR identity and lightweight read result types (shared by refinement and
Paddle recognizers)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class OcrIdentity:
    tokens: tuple[str, ...]
    numbers: tuple[str, ...]
    letters: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        return not self.numbers and not self.letters

    @property
    def exact_key(self) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
        if self.is_empty:
            return None
        return (self.numbers, self.letters)


@dataclass(frozen=True)
class LightweightOcrRead:
    text: str
    score: float
    identity: OcrIdentity
    accepted: bool
    text_region: Image.Image | None = None


# Token filters for `ocr_identity` (no language-specific stems; generic venue words);
# re-exported for tests and advanced callers.
GENERIC_TEXT_TOKENS = {
    "voie",
    "voies",
    "vois",
    "vote",
    "track",
    "tracks",
    "platform",
    "platforms",
    "repere",
    "marker",
    "gate",
    "hall",
    "level",
}


def normalize_ocr_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = text.replace("÷", " ").replace("+", " ").replace(":", " ").replace("|", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def ocr_identity(text: str) -> OcrIdentity:
    normalized = normalize_ocr_text(text)
    tokens = tuple(normalized.split())
    numbers = tuple(re.findall(r"\d+", normalized))
    letters: list[str] = []
    for token in tokens:
        alpha = "".join(re.findall(r"[a-z]+", token))
        if not alpha or alpha in GENERIC_TEXT_TOKENS:
            continue
        letters.append(alpha)
    return OcrIdentity(
        tokens=tokens,
        numbers=tuple(dict.fromkeys(numbers)),
        letters=tuple(dict.fromkeys(letters)),
    )


def ocr_key_string(identity: OcrIdentity) -> str:
    if identity.is_empty:
        return ""
    chunks = []
    if identity.letters:
        chunks.append("letters=" + ",".join(identity.letters))
    if identity.numbers:
        chunks.append("numbers=" + ",".join(identity.numbers))
    return ";".join(chunks)


def ocr_token_string(identity: OcrIdentity) -> str:
    return " ".join(identity.tokens)


def ocr_compatible(left: OcrIdentity, right: OcrIdentity) -> bool:
    if left.is_empty or right.is_empty:
        return False
    if left.numbers and right.numbers and left.numbers != right.numbers:
        return False
    if left.letters and right.letters and left.letters != right.letters:
        return False
    return bool(
        (left.numbers and right.numbers)
        or (left.letters and right.letters)
        or left.numbers
        or right.numbers
    )
