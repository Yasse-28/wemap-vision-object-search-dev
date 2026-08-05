"""Byte-level tests for the binary COPY encoding.

This is the highest-consequence, least-observable code in the port: the only thing
that validates it in normal use is the Postgres server accepting the stream, and a
subtly wrong width produces a runtime error far from the cause (or, worse, silently
shifted column values). So the bytes are asserted directly.

Field encoding under `COPY ... WITH BINARY`: a big-endian int32 length, then the
value. Length -1 means NULL and is followed by nothing.
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np
import pytest

from toolbox.bricks.ingest import (
    _EWKB_POINTZ_PREFIX,
    _N_FIELDS,
    _PGCOPY_HEADER,
    _PGCOPY_TRAILER,
    encode_copy_stream,
)

DIM = 4  # a real index uses 1024; 4 keeps the assertions readable


def _minimal_kwargs(n: int = 1, **overrides: Any) -> dict[str, Any]:
    kwargs = dict(
        geo_ref_id=7,
        geokeyframe_ids=np.arange(100, 100 + n, dtype=np.int64),
        theta_center=np.full(n, 0.25),
        phi_center=np.full(n, -0.5),
        angular_width=np.full(n, 1.0),
        angular_height=np.full(n, 0.75),
        embeddings=np.zeros((n, DIM), dtype=np.float16),
    )
    kwargs.update(overrides)
    return kwargs


def _rows(**kwargs: Any) -> list[bytes]:
    """The per-row blobs, with the envelope stripped and checked."""
    chunks = list(encode_copy_stream(**kwargs))
    assert chunks[0] == _PGCOPY_HEADER
    assert chunks[-1] == _PGCOPY_TRAILER
    return chunks[1:-1]


class _Field:
    """Cursor over one row's fields."""

    def __init__(self, row: bytes) -> None:
        self.row = row
        self.pos = 0
        n_fields = struct.unpack_from(">h", row, 0)[0]
        assert n_fields == _N_FIELDS, f"expected {_N_FIELDS} fields, got {n_fields}"
        self.pos = 2

    def next(self) -> bytes | None:
        """Next field's payload, or None for a NULL field."""
        (length,) = struct.unpack_from(">i", self.row, self.pos)
        self.pos += 4
        if length == -1:
            return None
        payload = self.row[self.pos : self.pos + length]
        assert len(payload) == length, "row truncated"
        self.pos += length
        return payload

    def require(self) -> bytes:
        """Next field, asserting it is not NULL — for fields under test."""
        payload = self.next()
        assert payload is not None, "expected a value, got NULL"
        return payload

    def skip(self, count: int) -> None:
        for _ in range(count):
            self.next()

    def assert_exhausted(self) -> None:
        assert self.pos == len(
            self.row
        ), f"{len(self.row) - self.pos} trailing byte(s) — a field width is wrong"


def test_envelope_and_row_count() -> None:
    rows = _rows(**_minimal_kwargs(n=3))
    assert len(rows) == 3


def test_header_is_the_documented_19_bytes() -> None:
    # PGCOPY\n\377\r\n\0 + int32 flags + int32 header extension length
    assert _PGCOPY_HEADER == b"PGCOPY\n\xff\r\n\x00" + struct.pack(">II", 0, 0)
    assert len(_PGCOPY_HEADER) == 19


def test_scalar_fields_are_bigint_and_float8() -> None:
    (row,) = _rows(**_minimal_kwargs())
    fields = _Field(row)
    assert struct.unpack(">q", fields.require())[0] == 7  # geo_ref_id
    assert struct.unpack(">q", fields.require())[0] == 100  # geokeyframe_id
    assert struct.unpack(">d", fields.require())[0] == pytest.approx(0.25)
    assert struct.unpack(">d", fields.require())[0] == pytest.approx(-0.5)
    assert struct.unpack(">d", fields.require())[0] == pytest.approx(1.0)
    assert struct.unpack(">d", fields.require())[0] == pytest.approx(0.75)


def test_halfvec_is_dim_header_plus_big_endian_float16() -> None:
    embeddings = np.array([[1.0, -2.0, 0.5, 0.0]], dtype=np.float16)
    (row,) = _rows(**_minimal_kwargs(embeddings=embeddings))
    fields = _Field(row)
    fields.skip(6)
    payload = fields.next()
    assert payload is not None
    assert len(payload) == 4 + DIM * 2, "halfvec = uint16 dim + uint16 pad + dim×f16"
    dim, unused = struct.unpack_from(">HH", payload, 0)
    assert (dim, unused) == (DIM, 0)
    # Big-endian on the wire regardless of host byte order.
    decoded = np.frombuffer(payload[4:], dtype=">f2")
    np.testing.assert_array_equal(decoded, embeddings[0])


def test_object_position_is_a_33_byte_ewkb_pointz_with_srid_zero() -> None:
    positions = np.array([[1.5, 2.5, -3.5]], dtype=np.float64)
    (row,) = _rows(**_minimal_kwargs(object_positions=positions))
    fields = _Field(row)
    fields.skip(9)  # through embedding, thumbnail, depth
    ewkb = fields.next()
    assert ewkb is not None
    assert len(ewkb) == 33
    assert ewkb[:9] == _EWKB_POINTZ_PREFIX
    assert ewkb[0] == 1, "little-endian byte-order marker"
    assert struct.unpack_from("<I", ewkb, 1)[0] == 0xA0000001, "Point|HasZ|HasSRID"
    # SRID 0, NOT 4326 — declaring the column as 4326 rejects every row.
    assert struct.unpack_from("<i", ewkb, 5)[0] == 0
    np.testing.assert_allclose(struct.unpack_from("<ddd", ewkb, 9), positions[0])


# 1-based positions in _COLUMNS, so the parametrize below reads like the schema.
_FIELD = {
    "thumbnail": 8,
    "depth": 9,
    "object_position": 10,
    "detector_source": 11,
    "label": 12,
    "detection_score": 13,
}


@pytest.mark.parametrize(
    "kwargs, field, label",
    [
        ({}, "thumbnail", "thumbnail absent"),
        ({"thumbnail_keys": np.array([""])}, "thumbnail", "thumbnail empty string"),
        ({}, "depth", "depth absent"),
        ({"depths": np.array([np.nan])}, "depth", "depth NaN"),
        ({}, "object_position", "object_position absent"),
        (
            {"object_positions": np.array([[1.0, np.nan, 3.0]])},
            "object_position",
            "one NaN coordinate",
        ),
        ({}, "detector_source", "detector_source absent"),
        (
            {"detector_sources": np.array([np.nan], dtype=object)},
            "detector_source",
            "pandas NaN string",
        ),
        ({}, "detection_score", "detection_score absent"),
        (
            {"detection_scores": np.array([np.nan])},
            "detection_score",
            "detection_score NaN",
        ),
    ],
)
def test_null_encoding(kwargs: dict[str, Any], field: str, label: str) -> None:
    """NULL is a bare -1 length. NaN and empty string must reach the DB as NULL.

    The NaN paths matter: `_compute_object_positions` returns NaN when depth is
    unavailable, and pandas materialises parquet string nulls as float NaN.
    """
    (row,) = _rows(**_minimal_kwargs(**kwargs))
    fields = _Field(row)
    fields.skip(_FIELD[field] - 1)
    assert fields.next() is None, f"{label} should encode as NULL"


def test_text_fields_are_utf8() -> None:
    (row,) = _rows(
        **_minimal_kwargs(
            thumbnail_keys=np.array(["thumbs/café.jpg"]),
            detector_sources=np.array(["gdino"]),
            labels=np.array(["bâtiment"]),
        )
    )
    fields = _Field(row)
    fields.skip(6)
    fields.next()  # embedding
    assert fields.next() == "thumbs/café.jpg".encode()
    fields.skip(2)  # depth, object_position
    assert fields.next() == b"gdino"
    assert fields.next() == "bâtiment".encode()


def test_row_is_fully_consumed_with_every_column_present() -> None:
    """Catches a wrong field width anywhere: leftover bytes mean a bad encoding."""
    (row,) = _rows(
        **_minimal_kwargs(
            thumbnail_keys=np.array(["t.jpg"]),
            depths=np.array([4.25]),
            object_positions=np.array([[1.0, 2.0, 3.0]]),
            detector_sources=np.array(["yolo"]),
            labels=np.array(["bench"]),
            detection_scores=np.array([0.75]),
        )
    )
    fields = _Field(row)
    fields.skip(_N_FIELDS)
    fields.assert_exhausted()


def test_length_mismatch_is_rejected() -> None:
    from toolbox.bricks.ingest import bulk_copy

    with pytest.raises(ValueError, match="same length"):
        bulk_copy(
            None,
            **_minimal_kwargs(n=2, theta_center=np.full(3, 0.1)),
        )


def test_empty_batch_writes_nothing() -> None:
    from toolbox.bricks.ingest import bulk_copy

    # conn is never touched for an empty batch, so None is safe here.
    assert bulk_copy(None, **_minimal_kwargs(n=0)) == 0
