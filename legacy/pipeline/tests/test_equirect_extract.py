"""Cutout id encoding and cubemap-style id scheme."""

from __future__ import annotations

from PIL import Image

from pipeline.offline.schema.cutout_schema import (
    CutoutRecord,
    decode_keyframe_id,
    decode_local_index,
    encode_cutout_id,
)

# Same order as pipeline.offline.shared.geometry.CUBEMAP_FACE_ORDER
_CUBEMAP_FACE_ORDER = ("front", "back", "left", "right", "top", "bottom")


def test_encode_decode_roundtrip():
    stride = 1024
    for kf in (0, 1, 99):
        for local in range(6):
            cid = encode_cutout_id(kf, local, stride)
            assert decode_keyframe_id(cid, stride) == kf
            assert decode_local_index(cid, stride) == local


def test_cubemap_style_six_unique_cutout_ids():
    """Mimics one keyframe expanded to six cubemap faces."""
    stride = 1024
    kf = 42
    recs: list[CutoutRecord] = []
    for local_idx, face_name in enumerate(_CUBEMAP_FACE_ORDER):
        cid = encode_cutout_id(kf, local_idx, stride)
        img = Image.new("RGB", (8, 8), color=(local_idx * 40, 0, 0))
        recs.append(
            CutoutRecord(cutout_id=cid, keyframe_id=kf, face_label=face_name, image=img)
        )
    assert len(recs) == 6
    assert len({r.cutout_id for r in recs}) == 6
    assert all(r.keyframe_id == kf for r in recs)
    assert [r.face_label for r in recs] == list(_CUBEMAP_FACE_ORDER)


def test_local_index_monotonic_per_keyframe():
    stride = 10000
    ids = [encode_cutout_id(5, i, stride) for i in range(6)]
    assert ids == sorted(ids)
    assert decode_local_index(ids[-1], stride) == 5
