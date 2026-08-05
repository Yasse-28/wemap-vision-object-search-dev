from __future__ import annotations

import numpy as np
from PIL import Image

from pipeline.core.models.ocr.identity import LightweightOcrRead
from pipeline.core.models.ocr.paddle_utils import is_lightweight_ocr_accepted
from pipeline.offline.refine import ocr_refine


class _FakeClip:
    def get_text_features(self, prompts):
        # First four prompts are positive and the last four are negative.
        features = np.zeros((len(prompts), 2), dtype=np.float32)
        features[:4, 0] = 1.0
        features[4:, 1] = 1.0

        class _Tensor:
            def __init__(self, arr):
                self.arr = arr

            def detach(self):
                return self

            def float(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.arr

        return _Tensor(features)


class _FakeRecognizer:
    texts: dict[int, str] = {}

    def __init__(self, **_kwargs):
        pass

    @classmethod
    def destroy_instance(cls) -> bool:
        return True

    def read_text(self, image):
        return self.texts[int(image.getpixel((0, 0))[0])]


def _crop(idx: int) -> Image.Image:
    return Image.new("RGB", (4, 4), (idx, idx, idx))


class _FakeLightweightRecognizer:
    reads: dict[int, tuple[str, float]] = {}

    def __init__(self, **kwargs):
        self.min_score = float(kwargs.get("min_score", 0.7))
        self.single_letter_min_score = float(
            kwargs.get("single_letter_min_score", 0.85)
        )

    @classmethod
    def destroy_instance(cls) -> bool:
        return True

    def read(self, image):
        idx = int(image.getpixel((0, 0))[0])
        text, score = self.reads[idx]
        identity = ocr_refine.ocr_identity(text)
        return LightweightOcrRead(
            text=text,
            score=score,
            identity=identity,
            accepted=is_lightweight_ocr_accepted(
                identity,
                score=score,
                min_score=self.min_score,
                single_letter_min_score=self.single_letter_min_score,
            ),
            text_region=image if text else None,
        )


class _FailingVlRecognizer:
    @classmethod
    def destroy_instance(cls) -> bool:
        return True

    def __init__(self, **_kwargs):
        raise AssertionError(
            "PaddleOCR-VL should not be loaded when lightweight OCR resolves"
            " all candidates"
        )


def test_ocr_refinement_splits_mixed_text_cluster_and_preserves_non_text_cluster(
    monkeypatch,
):
    monkeypatch.setattr(ocr_refine, "PaddleOcrVlRecognizer", _FakeRecognizer)
    _FakeRecognizer.texts = {
        0: "16",
        1: "16",
        2: "18",
        3: "18",
        4: "",
    }

    cluster_ids = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int32)
    valid_mask = np.ones(6, dtype=bool)
    positions_local = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.2, 0.2, 0.0],
            [10.0, 0.0, 0.0],
            [10.2, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    keyframe_ids = np.asarray([10, 11, 12, 13, 20, 21], dtype=np.int64)
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    result = ocr_refine.refine_clusters_with_ocr(
        cluster_ids=cluster_ids,
        valid_mask=valid_mask,
        positions_world=positions_local,
        object_keyframe_ids=keyframe_ids,
        object_embeddings=embeddings,
        object_crops=[_crop(i) for i in range(5)] + [_crop(4)],
        clip_model=_FakeClip(),
        device="cpu",
        textness_threshold=0.5,
        spatial_radius_m=1.0,
        min_anchor_observations=2,
        min_anchor_keyframes=2,
    )

    assert result.ocr_keys[:4].tolist() == [
        "numbers=16",
        "numbers=16",
        "numbers=18",
        "numbers=18",
    ]
    assert result.ocr_tokens[:4].tolist() == ["16", "16", "18", "18"]
    assert result.cluster_ids[0] == result.cluster_ids[1]
    assert result.cluster_ids[2] == result.cluster_ids[3]
    assert result.cluster_ids[0] != result.cluster_ids[2]
    assert result.cluster_ids[4] == result.cluster_ids[5]


def test_ocr_refinement_merges_unambiguous_partial_anchor(monkeypatch):
    monkeypatch.setattr(ocr_refine, "PaddleOcrVlRecognizer", _FakeRecognizer)
    _FakeRecognizer.texts = {
        0: "repere s voie 14",
        1: "repere s voie 14",
        2: "voie 14",
        3: "voie 14",
    }

    result = ocr_refine.refine_clusters_with_ocr(
        cluster_ids=np.asarray([0, 0, 1, 1], dtype=np.int32),
        valid_mask=np.ones(4, dtype=bool),
        positions_world=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.3, 0.0, 0.0],
                [0.4, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        object_keyframe_ids=np.asarray([10, 11, 12, 13], dtype=np.int64),
        object_embeddings=np.asarray([[1.0, 0.0]] * 4, dtype=np.float32),
        object_crops=[_crop(i) for i in range(4)],
        clip_model=_FakeClip(),
        device="cpu",
        textness_threshold=0.5,
        spatial_radius_m=1.0,
        min_anchor_observations=2,
        min_anchor_keyframes=2,
    )

    assert result.cluster_ids.tolist() == [0, 0, 0, 0]


def test_ocr_refinement_does_not_use_partial_anchor_to_bridge_conflicts(monkeypatch):
    monkeypatch.setattr(ocr_refine, "PaddleOcrVlRecognizer", _FakeRecognizer)
    _FakeRecognizer.texts = {
        0: "repere s voie 14",
        1: "repere s voie 14",
        2: "voie 14",
        3: "voie 14",
        4: "repere t voie 14",
        5: "repere t voie 14",
    }

    result = ocr_refine.refine_clusters_with_ocr(
        cluster_ids=np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int32),
        valid_mask=np.ones(6, dtype=bool),
        positions_world=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.3, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.6, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        object_keyframe_ids=np.asarray([10, 11, 12, 13, 14, 15], dtype=np.int64),
        object_embeddings=np.asarray([[1.0, 0.0]] * 6, dtype=np.float32),
        object_crops=[_crop(i) for i in range(6)],
        clip_model=_FakeClip(),
        device="cpu",
        textness_threshold=0.5,
        spatial_radius_m=1.0,
        min_anchor_observations=2,
        min_anchor_keyframes=2,
    )

    assert result.cluster_ids[0] == result.cluster_ids[1]
    assert result.cluster_ids[4] == result.cluster_ids[5]
    assert result.cluster_ids[0] != result.cluster_ids[4]
    assert result.cluster_ids[2] != result.cluster_ids[0]
    assert result.cluster_ids[2] != result.cluster_ids[4]


def test_ocr_refinement_skips_clusters_that_cannot_form_anchor(monkeypatch):
    monkeypatch.setattr(ocr_refine, "PaddleOcrVlRecognizer", _FakeRecognizer)
    _FakeRecognizer.texts = {
        1: "18",
        2: "18",
    }

    result = ocr_refine.refine_clusters_with_ocr(
        cluster_ids=np.asarray([0, 1, 1], dtype=np.int32),
        valid_mask=np.ones(3, dtype=bool),
        positions_world=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.2, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        object_keyframe_ids=np.asarray([10, 20, 21], dtype=np.int64),
        object_embeddings=np.asarray([[1.0, 0.0]] * 3, dtype=np.float32),
        object_crops=[_crop(i) for i in range(3)],
        textness_scores=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        device="cpu",
        textness_threshold=0.5,
        spatial_radius_m=1.0,
        min_anchor_observations=2,
        min_anchor_keyframes=2,
    )

    assert result.ocr_candidate_mask.tolist() == [False, True, True]
    assert result.ocr_keys.tolist() == ["", "numbers=18", "numbers=18"]


def test_ocr_refinement_limits_candidates_per_cluster(monkeypatch):
    monkeypatch.setattr(ocr_refine, "PaddleOcrVlRecognizer", _FakeRecognizer)
    _FakeRecognizer.texts = {
        0: "16",
        1: "16",
    }

    result = ocr_refine.refine_clusters_with_ocr(
        cluster_ids=np.asarray([0, 0, 0, 0], dtype=np.int32),
        valid_mask=np.ones(4, dtype=bool),
        positions_world=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [0.6, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        object_keyframe_ids=np.asarray([10, 11, 12, 13], dtype=np.int64),
        object_embeddings=np.asarray([[1.0, 0.0]] * 4, dtype=np.float32),
        object_crops=[_crop(i) for i in range(4)],
        textness_scores=np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
        device="cpu",
        textness_threshold=0.5,
        spatial_radius_m=1.0,
        min_anchor_observations=2,
        min_anchor_keyframes=2,
        max_ocr_candidates_per_cluster=2,
    )

    assert result.ocr_candidate_mask.tolist() == [True, True, False, False]
    assert result.ocr_keys[:2].tolist() == ["numbers=16", "numbers=16"]


def test_ocr_refinement_uses_lightweight_first_without_vl_fallback(monkeypatch):
    monkeypatch.setattr(
        ocr_refine, "LightweightPaddleOcrRecognizer", _FakeLightweightRecognizer
    )
    monkeypatch.setattr(ocr_refine, "PaddleOcrVlRecognizer", _FailingVlRecognizer)
    _FakeLightweightRecognizer.reads = {
        0: ("repere s voie 14", 0.96),
        1: ("S 14", 0.95),
    }

    result = ocr_refine.refine_clusters_with_ocr(
        cluster_ids=np.asarray([0, 0], dtype=np.int32),
        valid_mask=np.ones(2, dtype=bool),
        positions_world=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        object_keyframe_ids=np.asarray([10, 11], dtype=np.int64),
        object_embeddings=np.asarray([[1.0, 0.0]] * 2, dtype=np.float32),
        object_crops=[_crop(0), _crop(1)],
        textness_scores=np.asarray([1.0, 1.0], dtype=np.float32),
        device="cpu",
        textness_threshold=0.5,
        spatial_radius_m=1.0,
        min_anchor_observations=2,
        min_anchor_keyframes=2,
        lightweight_first=True,
    )

    assert result.ocr_keys.tolist() == ["letters=s;numbers=14", "letters=s;numbers=14"]
    assert result.ocr_tokens.tolist() == ["repere s voie 14", "s 14"]
    assert result.ocr_source.tolist() == [
        ocr_refine.OCR_SOURCE_LIGHTWEIGHT_PPOCR,
        ocr_refine.OCR_SOURCE_LIGHTWEIGHT_PPOCR,
    ]
    assert result.cluster_ids.tolist() == [0, 0]


def test_ocr_refinement_sends_ambiguous_lightweight_reads_to_vl(monkeypatch):
    monkeypatch.setattr(
        ocr_refine, "LightweightPaddleOcrRecognizer", _FakeLightweightRecognizer
    )
    monkeypatch.setattr(ocr_refine, "PaddleOcrVlRecognizer", _FakeRecognizer)
    _FakeLightweightRecognizer.reads = {
        0: ("m419", 0.95),
        1: ("m419", 0.95),
    }
    _FakeRecognizer.texts = {
        0: "2",
        1: "2",
    }

    result = ocr_refine.refine_clusters_with_ocr(
        cluster_ids=np.asarray([0, 0], dtype=np.int32),
        valid_mask=np.ones(2, dtype=bool),
        positions_world=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        object_keyframe_ids=np.asarray([10, 11], dtype=np.int64),
        object_embeddings=np.asarray([[1.0, 0.0]] * 2, dtype=np.float32),
        object_crops=[_crop(0), _crop(1)],
        textness_scores=np.asarray([1.0, 1.0], dtype=np.float32),
        device="cpu",
        textness_threshold=0.5,
        spatial_radius_m=1.0,
        min_anchor_observations=2,
        min_anchor_keyframes=2,
        lightweight_first=True,
    )

    assert result.ocr_keys.tolist() == ["numbers=2", "numbers=2"]
    assert result.ocr_source.tolist() == [
        ocr_refine.OCR_SOURCE_PADDLEOCR_VL,
        ocr_refine.OCR_SOURCE_PADDLEOCR_VL,
    ]


def test_ocr_refinement_skips_vl_when_lightweight_cluster_has_consensus(monkeypatch):
    monkeypatch.setattr(
        ocr_refine, "LightweightPaddleOcrRecognizer", _FakeLightweightRecognizer
    )
    monkeypatch.setattr(ocr_refine, "PaddleOcrVlRecognizer", _FailingVlRecognizer)
    _FakeLightweightRecognizer.reads = {
        0: ("Voie 16", 0.96),
        1: ("16", 0.96),
        2: ("m419", 0.95),
    }

    result = ocr_refine.refine_clusters_with_ocr(
        cluster_ids=np.asarray([0, 0, 0], dtype=np.int32),
        valid_mask=np.ones(3, dtype=bool),
        positions_world=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.4, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        object_keyframe_ids=np.asarray([10, 11, 12], dtype=np.int64),
        object_embeddings=np.asarray([[1.0, 0.0]] * 3, dtype=np.float32),
        object_crops=[_crop(0), _crop(1), _crop(2)],
        textness_scores=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        device="cpu",
        textness_threshold=0.5,
        spatial_radius_m=1.0,
        min_anchor_observations=2,
        min_anchor_keyframes=2,
        lightweight_first=True,
    )

    assert result.ocr_keys.tolist() == ["numbers=16", "numbers=16", ""]
    assert result.ocr_source.tolist() == [
        ocr_refine.OCR_SOURCE_LIGHTWEIGHT_PPOCR,
        ocr_refine.OCR_SOURCE_LIGHTWEIGHT_PPOCR,
        ocr_refine.OCR_SOURCE_NONE,
    ]


def test_ocr_refinement_uses_vl_when_lightweight_cluster_has_no_consensus(monkeypatch):
    monkeypatch.setattr(
        ocr_refine, "LightweightPaddleOcrRecognizer", _FakeLightweightRecognizer
    )
    monkeypatch.setattr(ocr_refine, "PaddleOcrVlRecognizer", _FakeRecognizer)
    _FakeLightweightRecognizer.reads = {
        0: ("16", 0.96),
        1: ("18", 0.96),
    }
    _FakeRecognizer.texts = {
        0: "16",
        1: "16",
    }

    result = ocr_refine.refine_clusters_with_ocr(
        cluster_ids=np.asarray([0, 0], dtype=np.int32),
        valid_mask=np.ones(2, dtype=bool),
        positions_world=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        object_keyframe_ids=np.asarray([10, 11], dtype=np.int64),
        object_embeddings=np.asarray([[1.0, 0.0]] * 2, dtype=np.float32),
        object_crops=[_crop(0), _crop(1)],
        textness_scores=np.asarray([1.0, 1.0], dtype=np.float32),
        device="cpu",
        textness_threshold=0.5,
        spatial_radius_m=1.0,
        min_anchor_observations=2,
        min_anchor_keyframes=2,
        lightweight_first=True,
    )

    assert result.ocr_keys.tolist() == ["numbers=16", "numbers=16"]
    assert result.ocr_source.tolist() == [
        ocr_refine.OCR_SOURCE_PADDLEOCR_VL,
        ocr_refine.OCR_SOURCE_PADDLEOCR_VL,
    ]


def test_ocr_refinement_skips_vl_when_lightweight_detects_no_text(monkeypatch):
    monkeypatch.setattr(
        ocr_refine, "LightweightPaddleOcrRecognizer", _FakeLightweightRecognizer
    )
    monkeypatch.setattr(ocr_refine, "PaddleOcrVlRecognizer", _FailingVlRecognizer)
    _FakeLightweightRecognizer.reads = {
        0: ("", np.nan),
        1: ("", np.nan),
    }

    result = ocr_refine.refine_clusters_with_ocr(
        cluster_ids=np.asarray([0, 0], dtype=np.int32),
        valid_mask=np.ones(2, dtype=bool),
        positions_world=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        object_keyframe_ids=np.asarray([10, 11], dtype=np.int64),
        object_embeddings=np.asarray([[1.0, 0.0]] * 2, dtype=np.float32),
        object_crops=[_crop(0), _crop(1)],
        textness_scores=np.asarray([1.0, 1.0], dtype=np.float32),
        device="cpu",
        textness_threshold=0.5,
        spatial_radius_m=1.0,
        min_anchor_observations=2,
        min_anchor_keyframes=2,
        lightweight_first=True,
    )

    assert result.ocr_keys.tolist() == ["", ""]
    assert result.ocr_source.tolist() == [
        ocr_refine.OCR_SOURCE_NONE,
        ocr_refine.OCR_SOURCE_NONE,
    ]
