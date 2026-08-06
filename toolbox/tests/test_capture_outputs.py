"""Tests for reading one capture's prepare outputs before ingest.

`embeddings.npy` is not a `.npy`: `prepare.writer.StreamingWriter` streams each
batch with `tofile`, so the file is a headerless raw float16 dump. Production reads
it as such (`np.frombuffer` on the S3 body); an earlier port here used `np.load`,
which rejects the file outright and takes the whole ingest with it. These tests pin
the on-disk format so the reader cannot drift back.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from toolbox.bricks.ingest_cli import EMBEDDING_DIM, _read_capture_outputs


def _write_capture(
    capture_dir: Path, n_rows: int, *, n_embeddings: int | None = None
) -> np.ndarray:
    """A capture directory in the exact format `prepare` writes.

    `n_embeddings` defaults to `n_rows`; pass it to simulate a half-written run.
    """
    capture_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_index": range(n_rows)}).to_parquet(
        capture_dir / "metadata.parquet"
    )
    rows = n_rows if n_embeddings is None else n_embeddings
    embeddings = np.arange(rows * EMBEDDING_DIM, dtype=np.float16)
    # tofile, not np.save — this is the whole point of the test.
    embeddings.tofile(capture_dir / "embeddings.npy")
    return embeddings.reshape(-1, EMBEDDING_DIM)


def test_reads_headerless_float16_dump(tmp_path: Path) -> None:
    expected = _write_capture(tmp_path / "capture", 4)

    outputs = _read_capture_outputs(tmp_path / "capture")

    assert outputs is not None
    metadata, embeddings = outputs
    assert len(metadata) == 4
    assert embeddings.shape == (4, EMBEDDING_DIM)
    assert embeddings.dtype == np.float16
    np.testing.assert_array_equal(embeddings, expected)


def test_missing_outputs_are_skipped(tmp_path: Path) -> None:
    empty = tmp_path / "capture"
    empty.mkdir()

    assert _read_capture_outputs(empty) is None


def test_length_mismatch_raises(tmp_path: Path) -> None:
    _write_capture(tmp_path / "capture", 4, n_embeddings=3)

    with pytest.raises(ValueError, match="length mismatch"):
        _read_capture_outputs(tmp_path / "capture")
