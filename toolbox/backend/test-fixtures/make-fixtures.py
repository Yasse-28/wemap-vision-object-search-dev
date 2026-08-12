"""Regenerate the parquet fixtures for `object-search-metadata.test.ts`.

    python toolbox/backend/test-fixtures/make-fixtures.py

The fixtures are committed because the reader test has to run under `tsx --test`
with no Python around, and hand-writing parquet in TypeScript would test the
hand-written writer rather than the real thing. They are a few kB each.

Schemas mirror `prepare/writer.py` (+ the two columns
`toolbox/bricks/prepare_postprocess.py` adds), including the dtypes that trip the
reader up: INT64 ids that arrive as `BigInt`, float16 angles, and a nullable
string `label`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).parent


def _base(video_keyframe_ids: list[int]) -> dict[str, object]:
    n = len(video_keyframe_ids)
    return {
        "row_index": list(range(n)),
        "video_keyframe_id": video_keyframe_ids,
        "theta_center": np.linspace(-3.0, 3.0, n).astype(np.float16),
        "phi_center": np.linspace(-0.5, 0.5, n).astype(np.float16),
        "angular_width": np.full(n, 0.2, dtype=np.float16),
        "angular_height": np.full(n, 0.1, dtype=np.float16),
        "detector_source": ["yolo" if i % 2 == 0 else "gdino" for i in range(n)],
        "label": pd.array(
            [None if i % 2 else f"class-{i}" for i in range(n)], dtype="string"
        ),
        "detection_score": np.linspace(0.3, 0.9, n).astype(np.float32),
        "vk_image_path": [f"images/keyframe-{i % 2}.jpg" for i in range(n)],
    }


def write_postprocessed() -> None:
    """The normal case: 7 rows over keyframes 0, 3, 7, with thumbnails and depth."""
    frame = pd.DataFrame(_base([0, 0, 0, 3, 3, 7, 7]))
    frame["thumbnail_key"] = [
        *[f"object-search/thumbnails/{i:06d}.jpg" for i in range(3)],
        *[f"capture-b/thumbnails/{i:06d}.jpg" for i in range(3, 6)],
        "",  # last row has no stored crop
    ]
    table = pa.Table.from_pandas(frame)
    depth = pa.array([1.5, 2.0, np.nan, 3.0, 4.0, None, 5.0], type=pa.float64())
    table = table.append_column("depth", depth)
    pq.write_table(table, HERE / "metadata-postprocessed.parquet")


def write_raw() -> None:
    """Straight out of `prepare`: `thumbnail_file`, no `depth`."""
    frame = pd.DataFrame(_base([0, 0, 5]))
    frame["thumbnail_file"] = ["000000.jpg", "000001.jpg", "000002.jpg"]
    pq.write_table(pa.Table.from_pandas(frame), HERE / "metadata-raw.parquet")


def write_noncontiguous() -> None:
    """Keyframe 0 resumes after keyframe 1 — not something this pipeline emits."""
    frame = pd.DataFrame(_base([0, 1, 0]))
    frame["thumbnail_key"] = ["a.jpg", "b.jpg", "c.jpg"]
    frame["depth"] = np.array([1.0, 2.0, 3.0])
    pq.write_table(pa.Table.from_pandas(frame), HERE / "metadata-noncontiguous.parquet")


def write_thumbnail_fallback() -> None:
    """One legacy thumbnail forces correct plain-string storage for the column."""
    frame = pd.DataFrame(_base([0, 0, 0]))
    frame["thumbnail_key"] = [
        "capture-a/thumbnails/000042.jpg",
        "legacy-thumb.jpg",
        "capture-b/thumbnails/000007.jpg",
    ]
    frame["depth"] = np.array([1.0, 2.0, 3.0])
    pq.write_table(
        pa.Table.from_pandas(frame), HERE / "metadata-thumbnail-fallback.parquet"
    )


def write_int32_overflow() -> None:
    """An INT64 id that cannot be represented by the columnar Int32Array."""
    frame = pd.DataFrame(_base([0]))
    frame["row_index"] = np.array([2**31], dtype=np.int64)
    pq.write_table(
        pa.Table.from_pandas(frame), HERE / "metadata-int32-overflow.parquet"
    )


if __name__ == "__main__":
    write_postprocessed()
    write_raw()
    write_noncontiguous()
    write_thumbnail_fallback()
    write_int32_overflow()
    print(f"wrote fixtures to {HERE}")
