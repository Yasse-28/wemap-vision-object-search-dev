"""Tests for hybrid detector prompt and vocabulary presets."""

from pipeline.offline.detect.prompts import (
    BROAD_VOCAB,
    SAFETY_SECURITY_VOCAB,
    TRAIN_STATION_OPERATIONS_VOCAB,
    TRAIN_STATION_PUBLIC_INFRA_VOCAB,
    TRAIN_STATION_SPECIFIC_VOCAB,
    VENUE_PROMPTS,
    VENUE_YOLO_VOCABS,
    _dedupe_normalized,
)


def _prompt_categories(prompt: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in prompt.split(".") if part.strip())


def test_train_station_yolo_vocab_counts_and_terms() -> None:
    assert len(BROAD_VOCAB) == 39
    assert len(TRAIN_STATION_OPERATIONS_VOCAB) == 30
    assert len(TRAIN_STATION_PUBLIC_INFRA_VOCAB) == 25
    assert len(SAFETY_SECURITY_VOCAB) == 13
    assert len(TRAIN_STATION_SPECIFIC_VOCAB) == 66
    assert VENUE_YOLO_VOCABS["train_station"] == TRAIN_STATION_SPECIFIC_VOCAB
    assert len(_dedupe_normalized(BROAD_VOCAB, TRAIN_STATION_SPECIFIC_VOCAB)) == 98

    for category in (
        "ticket machine",
        "ticket gate",
        "platform screen door",
        "platform bench",
        "luggage locker",
        "security camera",
        "fire extinguisher",
    ):
        assert category in TRAIN_STATION_SPECIFIC_VOCAB


def test_train_station_gdino_prompt_is_curated_second_check() -> None:
    categories = _prompt_categories(VENUE_PROMPTS["train_station"])

    assert len(categories) == 29
    assert len(set(categories)) == 29

    for category in (
        "security camera",
        "ticket validator",
        "platform letter sign",
        "tactile guidance strip",
        "service disruption board",
        "evacuation map",
        "carriage number sign",
    ):
        assert category in categories
