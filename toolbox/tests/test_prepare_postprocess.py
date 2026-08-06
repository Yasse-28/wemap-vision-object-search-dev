"""Tests for the depth bridge between `prepare` and `ingest`.

`prepare` emits no `depth` column. Without this step `bulk_copy` writes NULL, every
`object_position` comes out NULL, enrichment filters the rows away and `localize`
returns `[]` — indistinguishable from "the model found nothing". Nothing else in the
pipeline notices, which is why the resolution rule is pinned here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from toolbox.bricks.prepare_postprocess import sample_depths
from toolbox.bricks.vendored.depth_decode import decode_uint16_meters

DEPTH_HEIGHT = 16
DEPTH_WIDTH = 32


def _encode_meters(depth_m: float) -> int:
    """Inverse of `decode_uint16_meters`, for building a fixture at a known depth."""
    sqrt_lo, sqrt_hi = float(np.sqrt(0.1)), float(np.sqrt(200.0))
    normed = (np.sqrt(depth_m) - sqrt_lo) / (sqrt_hi - sqrt_lo)
    return (int(round(normed * 1022.0)) + 1) << 6


def _make_map(tmp_path: Path, *, depth_filename: str, code: int) -> Path:
    """A map whose single keyframe has a depth map filled with one uint16 code."""
    map_path = tmp_path / "map"
    (map_path / "depths").mkdir(parents=True)

    depth = np.full((DEPTH_HEIGHT, DEPTH_WIDTH), code, dtype=np.uint16)
    tifffile.imwrite(map_path / "depths" / depth_filename, depth)

    (map_path / "m_1_20260101_000000.json").write_text(
        json.dumps(
            {
                "local_origin": [2.35, 48.85, 0.0],
                "map": {"name": "m", "uuid": "u", "venue_type": "rail"},
                "geo_levels": [
                    {
                        "value": 0,
                        "min_altitude": -2.0,
                        "max_altitude": 4.0,
                        "geo_ref": 1,
                        "geometry": None,
                    }
                ],
                "geo_keyframes": [
                    {
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0,
                        "orientation": [1.0, 0.0, 0.0, 0.0],
                        "image_url": "https://example/images/a.jpg?X-Amz-Signature=x",
                        "depth_url": f"https://example/depths/{depth_filename}"
                        "?X-Amz-Signature=x",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return map_path


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "video_keyframe_id": [0],
            "theta_center": [0.0],
            "phi_center": [0.0],
        }
    )


def test_depth_is_read_from_the_manifest_filename(tmp_path: Path) -> None:
    code = _encode_meters(12.0)
    map_path = _make_map(tmp_path, depth_filename="a.tif", code=code)

    depths = sample_depths(_metadata(), map_path)

    assert depths.shape == (1,)
    np.testing.assert_allclose(depths[0], float(decode_uint16_meters(code)), rtol=1e-6)


def test_zero_sentinel_becomes_nan(tmp_path: Path) -> None:
    """0 means "no depth here"; persisting it as 0 m would place the object on the
    camera."""
    map_path = _make_map(tmp_path, depth_filename="a.tif", code=0)

    depths = sample_depths(_metadata(), map_path)

    assert np.isnan(depths[0])


def test_depth_named_after_the_keyframe_id_is_not_used(tmp_path: Path) -> None:
    """Keyframe ids are `geo_keyframes` indices, so `0.tif` names nothing.

    Honouring it would silently pair an unrelated depth map with a real pose —
    the reason the v1 fallbacks were removed rather than kept "just in case".
    """
    map_path = _make_map(tmp_path, depth_filename="a.tif", code=_encode_meters(12.0))
    (map_path / "depths" / "a.tif").rename(map_path / "depths" / "0.tif")

    depths = sample_depths(_metadata(), map_path)

    assert np.isnan(depths[0])


def test_empty_metadata_returns_an_empty_array(tmp_path: Path) -> None:
    map_path = _make_map(tmp_path, depth_filename="a.tif", code=_encode_meters(5.0))
    empty = _metadata().iloc[0:0]

    depths = sample_depths(empty, map_path)

    assert depths.shape == (0,)


@pytest.mark.parametrize("depth_m", [0.5, 5.0, 50.0, 150.0])
def test_decode_round_trips_across_the_range(depth_m: float) -> None:
    """The quantisation is coarse (10 bits over a sqrt curve) but unbiased."""
    decoded = float(decode_uint16_meters(_encode_meters(depth_m)))
    assert decoded == pytest.approx(depth_m, rel=0.02)
