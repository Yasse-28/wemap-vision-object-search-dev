"""Tests for pipeline.offline.build_index (hybrid pipeline)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stubs for heavy optional dependencies (torch, transformers, cv2, pyproj, zarr)
# ---------------------------------------------------------------------------

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def empty_cache() -> None:
            return None

        @staticmethod
        def ipc_collect() -> None:
            return None

    def _no_grad(func=None):
        if func is None:
            return lambda inner: inner
        return func

    class _Tensor:
        def __init__(self, data):
            self._data = np.asarray(data, dtype=np.float32)

        def __iter__(self):
            for row in self._data:
                yield _Tensor(row)

        def __array__(self, dtype=None):
            return np.asarray(self._data, dtype=dtype)

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(self._data)

    def _tensor(data, dtype=None):
        return _Tensor(
            np.asarray(data, dtype=np.float32 if dtype is not None else None)
        )

    def _stack(rows, dim=0):
        return _Tensor(np.stack([np.asarray(row) for row in rows], axis=dim))

    def _normalize(value, p=2, dim=-1):
        arr = np.asarray(value, dtype=np.float32)
        norm = np.linalg.norm(arr, axis=dim, keepdims=True)
        return arr / np.clip(norm, 1e-12, None)

    torch_stub.cuda = _Cuda()
    torch_stub.no_grad = _no_grad
    torch_stub.inference_mode = _no_grad
    torch_stub.tensor = _tensor
    torch_stub.stack = _stack
    torch_stub.Tensor = _Tensor
    torch_stub.float32 = np.float32
    torch_stub.float16 = np.float16
    sys.modules["torch"] = torch_stub
    torch_nn_stub = types.ModuleType("torch.nn")

    class _Module:
        def __init__(self, *args, **kwargs):
            return None

    torch_nn_stub.Module = _Module
    torch_nn_functional_stub = types.ModuleType("torch.nn.functional")
    torch_nn_functional_stub.normalize = _normalize
    torch_nn_stub.functional = torch_nn_functional_stub
    sys.modules["torch.nn"] = torch_nn_stub
    sys.modules["torch.nn.functional"] = torch_nn_functional_stub
    torch_inductor_stub = types.ModuleType("torch._inductor")
    torch_inductor_config_stub = types.ModuleType("torch._inductor.config")
    torch_inductor_config_stub.fx_graph_cache = False
    torch_inductor_stub.config = torch_inductor_config_stub
    sys.modules["torch._inductor"] = torch_inductor_stub
    sys.modules["torch._inductor.config"] = torch_inductor_config_stub

if "transformers" not in sys.modules:
    transformers_stub = types.ModuleType("transformers")

    class _UnusedTransformersFactory:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError("transformers is stubbed in tests")

    transformers_stub.AutoModelForZeroShotObjectDetection = _UnusedTransformersFactory
    transformers_stub.AutoProcessor = _UnusedTransformersFactory
    transformers_stub.AutoTokenizer = _UnusedTransformersFactory
    sys.modules["transformers"] = transformers_stub
    transformers_models_stub = types.ModuleType("transformers.models")
    metaclip_pkg_stub = types.ModuleType("transformers.models.metaclip_2")
    metaclip_modeling_stub = types.ModuleType(
        "transformers.models.metaclip_2.modeling_metaclip_2"
    )
    metaclip_modeling_stub.MetaClip2Model = _UnusedTransformersFactory
    sys.modules["transformers.models"] = transformers_models_stub
    sys.modules["transformers.models.metaclip_2"] = metaclip_pkg_stub
    sys.modules["transformers.models.metaclip_2.modeling_metaclip_2"] = (
        metaclip_modeling_stub
    )

if "cv2" not in sys.modules:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.INTER_CUBIC = 0
    cv2_stub.BORDER_WRAP = 0
    cv2_stub.remap = lambda image, map_x, map_y, interpolation, borderMode=None: image
    sys.modules["cv2"] = cv2_stub

if "pyproj" not in sys.modules:
    pyproj_stub = types.ModuleType("pyproj")

    class _Transformer:
        @classmethod
        def from_crs(cls, *args, **kwargs):
            return cls()

        def transform(self, xx, yy, zz=None, radians=False):
            return (xx, yy, 0.0 if zz is None else zz)

    pyproj_stub.Transformer = _Transformer
    sys.modules["pyproj"] = pyproj_stub

if "zarr" not in sys.modules:
    zarr_stub = types.ModuleType("zarr")
    zarr_stub.open = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("zarr is stubbed in tests")
    )
    sys.modules["zarr"] = zarr_stub

from pipeline.offline.build_index import (  # noqa: E402
    _as_normalized_tuple,
    _load_config,
    _resolve_ocr_config,
    _sample_image_paths_by_min_dist,
)
from pipeline.offline.localize.cluster_cutout_membership import (  # noqa: E402
    build_cluster_cutout_membership as _build_cluster_cutout_membership,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _camera_pose_for_position(x: float, y: float, z: float = 0.0) -> np.ndarray:
    """Return a world-to-camera pose with no rotation, camera at (x, y, z)."""
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [-x, -y, -z]
    return pose


# ---------------------------------------------------------------------------
# _sample_image_paths_by_min_dist
# ---------------------------------------------------------------------------


def test_sample_image_paths_zero_min_dist_returns_all() -> None:
    paths = [Path("1.jpg"), Path("2.jpg"), Path("3.jpg")]
    poses = {
        1: _camera_pose_for_position(0, 0),
        2: _camera_pose_for_position(0.1, 0),
        3: _camera_pose_for_position(0.2, 0),
    }
    result = _sample_image_paths_by_min_dist(
        paths,
        image_filename_to_keyframe_id=None,
        keyframe_poses=poses,
        min_dist_m=0.0,
    )
    assert result is paths


def test_sample_image_paths_keeps_all_when_far_enough() -> None:
    paths = [Path("1.jpg"), Path("2.jpg"), Path("3.jpg")]
    poses = {
        1: _camera_pose_for_position(0, 0),
        2: _camera_pose_for_position(2, 0),
        3: _camera_pose_for_position(4, 0),
    }
    result = _sample_image_paths_by_min_dist(
        paths,
        image_filename_to_keyframe_id=None,
        keyframe_poses=poses,
        min_dist_m=1.5,
    )
    assert [p.name for p in result] == ["1.jpg", "2.jpg", "3.jpg"]


def test_sample_image_paths_drops_close_keyframes() -> None:
    # All three are within 1.5 m of each other; sorted desc by id: 3 is kept first,
    # then 2 (0.5 m away) and 1 (1.0 m away) are suppressed.
    paths = [Path("1.jpg"), Path("2.jpg"), Path("3.jpg")]
    poses = {
        1: _camera_pose_for_position(0, 0),
        2: _camera_pose_for_position(0.5, 0),
        3: _camera_pose_for_position(1.0, 0),
    }
    result = _sample_image_paths_by_min_dist(
        paths,
        image_filename_to_keyframe_id=None,
        keyframe_poses=poses,
        min_dist_m=1.5,
    )
    assert [p.name for p in result] == ["3.jpg"]


def test_sample_image_paths_result_sorted_ascending_by_id() -> None:
    # Input order does not match id order; output must be ascending by id.
    paths = [Path("3.jpg"), Path("1.jpg"), Path("2.jpg")]
    poses = {
        1: _camera_pose_for_position(0, 0),
        2: _camera_pose_for_position(5, 0),
        3: _camera_pose_for_position(10, 0),
    }
    result = _sample_image_paths_by_min_dist(
        paths,
        image_filename_to_keyframe_id=None,
        keyframe_poses=poses,
        min_dist_m=1.5,
    )
    assert [p.name for p in result] == ["1.jpg", "2.jpg", "3.jpg"]


def test_sample_image_paths_drops_images_with_no_pose() -> None:
    paths = [Path("1.jpg"), Path("2.jpg")]
    poses = {1: _camera_pose_for_position(0, 0)}  # id 2 has no pose
    result = _sample_image_paths_by_min_dist(
        paths,
        image_filename_to_keyframe_id=None,
        keyframe_poses=poses,
        min_dist_m=1.5,
    )
    assert [p.name for p in result] == ["1.jpg"]


def test_sample_image_paths_greedy_uses_descending_id_order() -> None:
    # Positions: id=1 → 0 m, id=2 → 2 m, id=3 → 3.5 m  (min_dist=1.5 m)
    # Greedy pass sorted desc: seed=3 (3.5 m) kept;
    # id=2 (|3.5-2|=1.5 → not < 1.5 → kept);
    # id=1 (|3.5-0|=3.5 → not < 1.5, kept too).
    # All three are ≥ 1.5 m apart from the seed → all kept.
    paths = [Path("1.jpg"), Path("2.jpg"), Path("3.jpg")]
    poses = {
        1: _camera_pose_for_position(0, 0),
        2: _camera_pose_for_position(2, 0),
        3: _camera_pose_for_position(3.5, 0),
    }
    result = _sample_image_paths_by_min_dist(
        paths,
        image_filename_to_keyframe_id=None,
        keyframe_poses=poses,
        min_dist_m=1.5,
    )
    assert [p.name for p in result] == ["1.jpg", "2.jpg", "3.jpg"]


def test_sample_image_paths_uses_image_filename_to_keyframe_id() -> None:
    # Non-integer filenames resolved via the lookup dict.
    paths = [Path("abc.jpg"), Path("def.jpg")]
    id_map = {"abc.jpg": 10, "def.jpg": 20}
    poses = {
        10: _camera_pose_for_position(0, 0),
        20: _camera_pose_for_position(10, 0),
    }
    result = _sample_image_paths_by_min_dist(
        paths,
        image_filename_to_keyframe_id=id_map,
        keyframe_poses=poses,
        min_dist_m=1.5,
    )
    assert [p.name for p in result] == ["abc.jpg", "def.jpg"]


def test_sample_image_paths_skips_paths_missing_from_id_map() -> None:
    paths = [Path("abc.jpg"), Path("unknown.jpg")]
    id_map = {"abc.jpg": 10}
    poses = {10: _camera_pose_for_position(0, 0)}
    result = _sample_image_paths_by_min_dist(
        paths,
        image_filename_to_keyframe_id=id_map,
        keyframe_poses=poses,
        min_dist_m=1.5,
    )
    assert [p.name for p in result] == ["abc.jpg"]


def test_sample_image_paths_empty_input_returns_empty() -> None:
    result = _sample_image_paths_by_min_dist(
        [],
        image_filename_to_keyframe_id=None,
        keyframe_poses={},
        min_dist_m=1.5,
    )
    assert result == []


# ---------------------------------------------------------------------------
# _load_config
# ---------------------------------------------------------------------------


def test_load_config_missing_file_returns_empty_dict(tmp_path: Path) -> None:
    cfg = _load_config(tmp_path / "nonexistent.yaml")
    assert cfg == {}


def test_load_config_reads_yaml_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("offline:\n  mini_batch_size: 8\n")
    cfg = _load_config(config_path)
    assert cfg == {"offline": {"mini_batch_size": 8}}


def test_load_config_empty_file_returns_empty_dict(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("")
    cfg = _load_config(config_path)
    assert cfg == {}


# ---------------------------------------------------------------------------
# _resolve_ocr_config
# ---------------------------------------------------------------------------


def test_resolve_ocr_config_prefers_top_level_ocr() -> None:
    cfg = {"ocr": {"enabled": True, "batch_size": 32}}
    offline_cfg = {"ocr": {"enabled": False}}
    result = _resolve_ocr_config(cfg, offline_cfg)
    assert result["enabled"] is True
    assert result["batch_size"] == 32


def test_resolve_ocr_config_falls_back_to_offline_ocr() -> None:
    cfg: dict = {}
    offline_cfg = {"ocr": {"enabled": True}}
    result = _resolve_ocr_config(cfg, offline_cfg)
    assert result == {"enabled": True}


def test_resolve_ocr_config_returns_empty_when_absent() -> None:
    result = _resolve_ocr_config({}, {})
    assert result == {}


def test_resolve_ocr_config_top_level_none_falls_through_to_offline() -> None:
    cfg: dict = {}
    offline_cfg = {"ocr": {"lightweight_lang": "en"}}
    result = _resolve_ocr_config(cfg, offline_cfg)
    assert result == {"lightweight_lang": "en"}


# ---------------------------------------------------------------------------
# _as_normalized_tuple
# ---------------------------------------------------------------------------


def test_as_normalized_tuple_none_returns_empty() -> None:
    assert _as_normalized_tuple(None) == ()


def test_as_normalized_tuple_empty_string_returns_empty() -> None:
    assert _as_normalized_tuple("") == ()


def test_as_normalized_tuple_splits_comma_separated_string() -> None:
    assert _as_normalized_tuple("door,window") == ("door", "window")


def test_as_normalized_tuple_normalizes_case_and_whitespace() -> None:
    assert _as_normalized_tuple("  Door , WINDOW  ") == ("door", "window")


def test_as_normalized_tuple_deduplicates_preserving_order() -> None:
    assert _as_normalized_tuple("door,window,door") == ("door", "window")


def test_as_normalized_tuple_from_list() -> None:
    assert _as_normalized_tuple(["Door", "Window"]) == ("door", "window")


def test_as_normalized_tuple_single_value() -> None:
    assert _as_normalized_tuple("exit") == ("exit",)


# ---------------------------------------------------------------------------
# cluster_cutout_membership (separate module — kept from prior test suite)
# ---------------------------------------------------------------------------


def test_build_cluster_cutout_membership_deduplicates_and_keeps_level() -> None:
    membership = _build_cluster_cutout_membership(
        object_cluster_ids=np.array([0, 0, 0, 1, -1], dtype=np.int32),
        object_cutout_ids=np.array([1001, 1001, 1002, 2001, 9999], dtype=np.int64),
        object_keyframe_ids=np.array([10, 10, 10, 20, 99], dtype=np.int64),
        detection_levels=np.array([3, 3, 3, 5, 7], dtype=np.int32),
    )

    assert membership["cluster_cutout_cluster_ids"].tolist() == [0, 0, 1]
    assert membership["cluster_cutout_ids"].tolist() == [1001, 1002, 2001]
    assert membership["cluster_cutout_keyframe_ids"].tolist() == [10, 10, 20]
    assert membership["cluster_cutout_levels"].tolist() == [3, 3, 5]
    assert membership["cluster_cutout_observation_counts"].tolist() == [2, 1, 1]


def test_build_cluster_cutout_membership_uses_most_frequent_level() -> None:
    membership = _build_cluster_cutout_membership(
        object_cluster_ids=np.array([0, 0, 0], dtype=np.int32),
        object_cutout_ids=np.array([1001, 1001, 1001], dtype=np.int64),
        object_keyframe_ids=np.array([10, 10, 10], dtype=np.int64),
        detection_levels=np.array([2, 2, 4], dtype=np.int32),
    )

    assert membership["cluster_cutout_levels"].tolist() == [2]
    assert membership["cluster_cutout_observation_counts"].tolist() == [3]


def test_build_cluster_cutout_membership_rejects_mixed_keyframes_per_cutout() -> None:
    with pytest.raises(ValueError, match="spans multiple keyframes"):
        _build_cluster_cutout_membership(
            object_cluster_ids=np.array([0, 0], dtype=np.int32),
            object_cutout_ids=np.array([1001, 1001], dtype=np.int64),
            object_keyframe_ids=np.array([10, 11], dtype=np.int64),
            detection_levels=np.array([2, 2], dtype=np.int32),
        )
