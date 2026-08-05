from pipeline.online.ocr_scoring import (
    extract_ocr_query,
    score_ocr_candidate,
)


def test_extract_ocr_query_from_natural_french_prompt():
    query = extract_ocr_query("repère de quai 16")

    assert query.numbers == frozenset({"16"})
    assert query.letters == frozenset()
    assert query.alphanumerics == frozenset()


def test_extract_ocr_query_keeps_short_letter_marker():
    query = extract_ocr_query("repere s voie 14")

    assert query.numbers == frozenset({"14"})
    assert query.letters == frozenset({"s"})


def test_extract_ocr_query_keeps_alphanumeric_identifier():
    query = extract_ocr_query("platform B12")

    assert query.numbers == frozenset({"12"})
    assert query.letters == frozenset({"b"})
    assert query.alphanumerics == frozenset({"b12"})


def test_score_ocr_candidate_prefers_matching_number_key():
    query = extract_ocr_query("repère de quai 16")

    assert score_ocr_candidate(query, ocr_key="numbers=16") == 1.0
    assert score_ocr_candidate(query, ocr_key="numbers=17") == 0.0


def test_score_ocr_candidate_uses_token_fallback():
    query = extract_ocr_query("repère de quai 16")

    assert score_ocr_candidate(query, ocr_tokens="repere quai 16") == 0.80


def test_score_ocr_candidate_matches_alphanumeric_token():
    query = extract_ocr_query("platform B12")

    assert score_ocr_candidate(query, ocr_tokens="b12") == 1.0
    assert score_ocr_candidate(query, ocr_key="letters=b;numbers=12") == 1.0
