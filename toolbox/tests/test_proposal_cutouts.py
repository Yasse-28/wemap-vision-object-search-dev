"""Tests for the memory-bounded cutout override.

The override exists only to bound peak GPU memory (see
`toolbox/bricks/vendored/proposal_cutouts.py`), so the property that matters is that
it changes **nothing** about the output. That is asserted against the mirrored
implementation itself, on CPU, so it runs without a GPU.

Peak-memory behaviour is not asserted here — it needs a CUDA device and a
full-resolution ERP. The measurements are recorded in the module docstring and in
ADR 0002.
"""

from __future__ import annotations

import numpy as np
import pytest
from prepare.proposal_cutouts import create_proposal_cutouts as mirrored

from toolbox.bricks.vendored.proposal_cutouts import (
    DEFAULT_CUTOUT_BATCH,
    create_proposal_cutouts,
    install,
)

OUT_SIZE = 16


def _erp(height: int = 64, width: int = 128) -> np.ndarray:
    """A deterministic, non-uniform ERP, so a misplaced sample changes the output."""
    rng = np.random.default_rng(20260805)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _boxes(count: int) -> list[tuple[int, int, int, int]]:
    """Boxes of varied size and aspect ratio, including wide, tall and square."""
    rng = np.random.default_rng(11)
    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(count):
        x1 = int(rng.integers(0, 100))
        y1 = int(rng.integers(0, 40))
        boxes.append(
            (x1, y1, x1 + int(rng.integers(2, 25)), y1 + int(rng.integers(2, 20)))
        )
    return boxes


def _stack(cutouts: list) -> np.ndarray:
    return np.stack([cutout.image for cutout in cutouts])


def test_default_batch_matches_production() -> None:
    """The default must stay production's literal, so behaviour is unchanged."""
    assert DEFAULT_CUTOUT_BATCH == 10


@pytest.mark.parametrize("n_boxes", [1, 9, 10, 11, 25])
@pytest.mark.parametrize("batch", [1, 2, 3, 10, 64])
def test_output_is_bitwise_identical_to_the_mirror(n_boxes: int, batch: int) -> None:
    """Any batch size gives byte-for-byte what the mirrored function gives.

    Batching only decides how many proposals share one `grid_sample` call. Every
    per-proposal quantity is computed row-wise, so the split cannot affect a result —
    this pins that, including across the batch boundary (`n_boxes` 9/10/11 against
    `batch` 10).
    """
    erp = _erp()
    boxes = _boxes(n_boxes)

    expected = mirrored(erp, boxes, "cpu", OUT_SIZE)
    actual = create_proposal_cutouts(erp, boxes, "cpu", OUT_SIZE, batch=batch)

    assert len(actual) == len(expected) == n_boxes
    assert np.array_equal(_stack(actual), _stack(expected))
    for got, want in zip(actual, expected):
        assert got.theta_center == want.theta_center
        assert got.phi_center == want.phi_center
        assert got.angular_width == want.angular_width
        assert got.angular_height == want.angular_height


def test_no_boxes_returns_empty() -> None:
    assert create_proposal_cutouts(_erp(), [], "cpu", OUT_SIZE) == []


@pytest.mark.parametrize("batch", [0, -1])
def test_non_positive_batch_is_rejected(batch: int) -> None:
    """Silently clamping would hide a bad `--cutout-batch` behind a slow run."""
    with pytest.raises(ValueError, match="batch must be >= 1"):
        create_proposal_cutouts(_erp(), _boxes(3), "cpu", OUT_SIZE, batch=batch)
    with pytest.raises(ValueError, match="batch must be >= 1"):
        install(batch)


def test_install_rebinds_the_pipeline_name() -> None:
    """`run_prepare` resolves the name from its own module globals.

    If this ever stops being true — the mirror importing the function differently, say
    — the override would silently stop applying and the OOM would come back.
    """
    import prepare.pipeline

    original = prepare.pipeline.create_proposal_cutouts
    try:
        install(2)
        assert prepare.pipeline.create_proposal_cutouts is not original

        erp = _erp()
        boxes = _boxes(7)
        patched = prepare.pipeline.create_proposal_cutouts(erp, boxes, "cpu", OUT_SIZE)
        assert np.array_equal(
            _stack(patched), _stack(mirrored(erp, boxes, "cpu", OUT_SIZE))
        )
    finally:
        prepare.pipeline.create_proposal_cutouts = original
