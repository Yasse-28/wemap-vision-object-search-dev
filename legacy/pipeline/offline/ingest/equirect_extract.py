"""
Extract pinhole cutouts from equirectangular (360) keyframe images.

Default: 6-face cubemap via the local standalone geometry helpers.
Optional: horizon ring via ``EquirectangularProjector.create_cutouts_from_keyframe``.

Equirect images are loaded with PIL (RGB). Projectors use OpenCV internally
for remapping.

Cutout IDs: ``cutout_id = keyframe_id * id_stride + local_index`` (see
``cutout_schema``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

from pipeline.core.logging import logger
from pipeline.core.types import DEFAULT_ID_STRIDE, GeometryMode
from pipeline.offline.ingest.keyframe_id import keyframe_id_from_image_path
from pipeline.offline.schema.cutout_schema import CutoutRecord, encode_cutout_id
from pipeline.offline.shared.geometry import (
    CUBEMAP_FACE_ORDER,
    CutoutParams,
    EquirectangularProjector,
    lonlat2XY,
    relative_pose_cubemap_face,
    retrieve_cubemap_face,
    xyz2lonlat,
)

# Some map exports contain JPEGs missing only trailing bytes; PIL can still decode them.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _cutout_to_pil_rgb(cutout_arr: np.ndarray) -> Image.Image:
    if cutout_arr.ndim == 2:
        rgb = np.stack([cutout_arr, cutout_arr, cutout_arr], axis=-1)
    else:
        rgb = np.asarray(cutout_arr)
    return Image.fromarray(rgb.astype(np.uint8))


def _pose_center_xy(
    relative_pose: np.ndarray, image_shape: tuple[int, ...]
) -> tuple[float, float]:
    """Return the cutout center in equirectangular pixel coordinates."""
    forward = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    rot = np.asarray(relative_pose[:3, :3], dtype=np.float32)
    xyz = forward @ rot.T
    lonlat = xyz2lonlat(xyz)
    xy = lonlat2XY(lonlat, shape=image_shape).astype(np.float32)[0]
    return float(xy[0]), float(xy[1])


def decode_keyframe_id(cutout_id: int, id_stride: int) -> int:
    """Decode keyframe_id from cutout_id."""
    return cutout_id // id_stride


def decode_local_index(cutout_id: int, id_stride: int) -> int:
    """Decode local_index from cutout_id."""
    return cutout_id % id_stride


def load_equirect_rgb(path: Path) -> np.ndarray | None:
    try:
        pil = Image.open(path).convert("RGB")
        return np.asarray(pil)
    except OSError as e:
        logger.warning("Could not read equirect image %s: %s", path, e)
        return None


def extract_cutouts_cubemap_for_keyframe(
    equirect_rgb: np.ndarray,
    keyframe_id: int,
    *,
    face_size: int,
    fov_deg: float,
    id_stride: int,
) -> List[CutoutRecord]:
    out: List[CutoutRecord] = []
    for local_idx, face_name in enumerate(CUBEMAP_FACE_ORDER):
        pose = relative_pose_cubemap_face(face_name)
        face_arr = retrieve_cubemap_face(
            equirect_rgb,
            face_name,
            face_size=face_size,
            fov_deg=fov_deg,
        )
        pil = _cutout_to_pil_rgb(face_arr)
        cid = encode_cutout_id(keyframe_id, local_idx, id_stride)
        out.append(
            CutoutRecord(
                cutout_id=cid,
                keyframe_id=keyframe_id,
                face_label=face_name,
                image=pil,
                center_xy=_pose_center_xy(pose, equirect_rgb.shape),
                rotation_cutout_to_equirect=np.asarray(pose, dtype=np.float32),
            )
        )
    return out


def extract_cutouts_horizon_for_keyframe(
    equirect_rgb: np.ndarray,
    keyframe_id: int,
    *,
    params: CutoutParams,
    id_stride: int,
) -> List[CutoutRecord]:
    cutouts = EquirectangularProjector.create_cutouts_from_keyframe(
        equirect_rgb, params
    )
    out: List[CutoutRecord] = []
    for local_idx, (cutout_image, _r) in enumerate(cutouts):
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = np.asarray(_r, dtype=np.float32)
        pil = _cutout_to_pil_rgb(cutout_image)
        label = f"cut_{local_idx}"
        cid = encode_cutout_id(keyframe_id, local_idx, id_stride)
        if local_idx >= id_stride:
            raise ValueError(
                f"local_index {local_idx} >= id_stride {id_stride}; increase id_stride"
            )
        out.append(
            CutoutRecord(
                cutout_id=cid,
                keyframe_id=keyframe_id,
                face_label=label,
                image=pil,
                center_xy=_pose_center_xy(pose, equirect_rgb.shape),
                rotation_cutout_to_equirect=pose,
            )
        )
    return out


def iter_cutouts_by_keyframe(
    image_paths: Sequence[Path],
    geometry: GeometryMode,
    *,
    id_stride: int = DEFAULT_ID_STRIDE,
    cubemap_face_size: int = 512,
    cubemap_fov_deg: float = 90.0,
    horizon_params: CutoutParams | None = None,
    image_filename_to_keyframe_id: dict[str, int] | None = None,
) -> Iterator[List[CutoutRecord]]:
    """Yield cutouts for one keyframe at a time to minimize memory usage."""
    if geometry == "horizon" and horizon_params is None:
        horizon_params = CutoutParams()

    for path in tqdm(image_paths, desc=f"Generating {geometry} cutouts"):
        kf_id = keyframe_id_from_image_path(
            path,
            image_filename_to_keyframe_id=image_filename_to_keyframe_id,
        )
        if kf_id is None:
            logger.warning("Skip equirect (no keyframe id for image): %s", path)
            continue
        rgb = load_equirect_rgb(path)
        if rgb is None:
            continue
        if geometry == "cubemap":
            yield extract_cutouts_cubemap_for_keyframe(
                rgb,
                kf_id,
                face_size=cubemap_face_size,
                fov_deg=cubemap_fov_deg,
                id_stride=id_stride,
            )
        else:
            assert horizon_params is not None
            n = horizon_params.nb_of_cuts
            if n > id_stride:
                raise ValueError(f"nb_of_cuts {n} > id_stride {id_stride}")
            yield extract_cutouts_horizon_for_keyframe(
                rgb,
                kf_id,
                params=horizon_params,
                id_stride=id_stride,
            )


def collect_cutouts_from_equirect_dir(
    image_paths: Sequence[Path],
    geometry: GeometryMode,
    *,
    id_stride: int = DEFAULT_ID_STRIDE,
    cubemap_face_size: int = 512,
    cubemap_fov_deg: float = 90.0,
    horizon_params: CutoutParams | None = None,
    image_filename_to_keyframe_id: dict[str, int] | None = None,
) -> List[CutoutRecord]:
    """Backward-compatible wrapper: collects all cutouts into a list."""
    return list(
        record
        for kf_records in iter_cutouts_by_keyframe(
            image_paths,
            geometry,
            id_stride=id_stride,
            cubemap_face_size=cubemap_face_size,
            cubemap_fov_deg=cubemap_fov_deg,
            horizon_params=horizon_params,
            image_filename_to_keyframe_id=image_filename_to_keyframe_id,
        )
        for record in kf_records
    )


class LazyCutoutProvider:
    """Re-loads equirect images on demand, caching the last keyframe's cutouts.

    Memory: at most one equirect image + its cutouts in RAM at any time.
    """

    def __init__(
        self,
        image_paths: Sequence[Path],
        geometry: GeometryMode,
        *,
        id_stride: int = DEFAULT_ID_STRIDE,
        cubemap_face_size: int = 512,
        cubemap_fov_deg: float = 90.0,
        horizon_params: CutoutParams | None = None,
        image_filename_to_keyframe_id: dict[str, int] | None = None,
    ) -> None:
        # map kf_id -> Path
        self._paths: dict[int, Path] = {}
        for p in image_paths:
            kf_id = keyframe_id_from_image_path(
                p,
                image_filename_to_keyframe_id=image_filename_to_keyframe_id,
            )
            if kf_id is not None:
                self._paths[kf_id] = p
        self._geometry = geometry
        self._id_stride = id_stride
        self._cubemap_face_size = cubemap_face_size
        self._cubemap_fov_deg = cubemap_fov_deg
        self._horizon_params = horizon_params
        self._cached_kf_id: int | None = None
        self._cached_cutouts: dict[int, Image.Image] = {}  # local_index -> image

    def get_image(self, cutout_id: int) -> Image.Image | None:
        """Get a cutout image by its cutout_id."""
        kf_id = decode_keyframe_id(cutout_id, self._id_stride)
        local_idx = decode_local_index(cutout_id, self._id_stride)
        if kf_id != self._cached_kf_id:
            self._load_keyframe(kf_id)
        return self._cached_cutouts.get(local_idx)

    def _load_keyframe(self, kf_id: int) -> None:
        """Load cutouts for a specific keyframe, caching them."""
        self._cached_cutouts = {}
        self._cached_kf_id = kf_id
        path = self._paths.get(kf_id)
        if path is None:
            return
        rgb = load_equirect_rgb(path)
        if rgb is None:
            return
        if self._geometry == "cubemap":
            records = extract_cutouts_cubemap_for_keyframe(
                rgb,
                kf_id,
                face_size=self._cubemap_face_size,
                fov_deg=self._cubemap_fov_deg,
                id_stride=self._id_stride,
            )
        else:
            params = self._horizon_params or CutoutParams()
            records = extract_cutouts_horizon_for_keyframe(
                rgb, kf_id, params=params, id_stride=self._id_stride
            )
        for r in records:
            local = decode_local_index(r.cutout_id, self._id_stride)
            self._cached_cutouts[local] = r.image
