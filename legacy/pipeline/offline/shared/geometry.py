from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CutoutParams:
    use_clahe: bool = True
    nb_of_cuts: int = 12
    fov: float = 60
    inclination: float = 0
    width: int = 720
    height: int = 720


CUBEMAP_FACE_ORDER: tuple[str, ...] = (
    "front",
    "back",
    "left",
    "right",
    "top",
    "bottom",
)

_FACE_ANGLES_DEG: dict[str, tuple[float, float]] = {
    "front": (0.0, 0.0),
    "back": (180.0, 0.0),
    "left": (270.0, 0.0),
    "right": (90.0, 0.0),
    "top": (0.0, 90.0),
    "bottom": (0.0, -90.0),
}


def _rodrigues(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta <= 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = np.asarray(rotvec, dtype=np.float64) / theta
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.asarray(
        np.eye(3, dtype=np.float64)
        + np.sin(theta) * k
        + (1.0 - np.cos(theta)) * (k @ k)
    )


def _remap(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return np.asarray(
        cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
    )


def xyz2lonlat(xyz: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]
    return np.concatenate([np.arctan2(x, z), np.arcsin(y)], axis=-1)


def lonlat2XY(lonlat: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    x = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    y = (lonlat[..., 1:] / np.pi + 0.5) * (shape[0] - 1)
    return np.concatenate([x, y], axis=-1)


def relative_pose_cubemap_face(face_name: str) -> np.ndarray:
    if face_name not in _FACE_ANGLES_DEG:
        raise ValueError(
            f"Unknown cubemap face {face_name!r}; expected one of {CUBEMAP_FACE_ORDER}"
        )
    theta_deg, inc_deg = _FACE_ANGLES_DEG[face_name]
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    r1 = _rodrigues(y_axis * np.radians(theta_deg))
    r2 = _rodrigues(np.dot(r1, x_axis) * np.radians(inc_deg))
    r = r2 @ r1
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = r.T.astype(np.float64)
    return pose


class EquirectangularProjector:
    _remap_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _compute_remap_coordinates(
        shape: tuple[int, ...], relative_pose: np.ndarray, params: CutoutParams
    ) -> tuple[np.ndarray, np.ndarray]:
        f = 0.5 * params.width / np.tan(0.5 * params.fov / 180.0 * np.pi)
        cx = (params.width - 1) / 2.0
        cy = (params.height - 1) / 2.0
        k = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], np.float32)
        k_inv = np.linalg.inv(k)

        x = np.arange(params.width)
        y = np.arange(params.height)
        x, y = np.meshgrid(x, y)
        z = np.ones_like(x)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        xyz = xyz @ k_inv.T

        r = relative_pose[:3, :3].T
        xyz = xyz @ r.T
        xy = lonlat2XY(xyz2lonlat(xyz), shape=shape).astype(np.float32)
        return xy[..., 0], xy[..., 1]

    @classmethod
    def retrieve_image(
        cls,
        equirectangular_image: np.ndarray,
        relative_pose: np.ndarray,
        params: CutoutParams,
    ) -> np.ndarray:
        cache_key = (
            equirectangular_image.shape,
            hash(relative_pose.tobytes()),
            params.width,
            params.height,
            params.fov,
        )
        if cache_key not in cls._remap_cache:
            cls._remap_cache[cache_key] = cls._compute_remap_coordinates(
                equirectangular_image.shape,
                relative_pose,
                params,
            )
            if len(cls._remap_cache) > 1000:
                cls._remap_cache.pop(next(iter(cls._remap_cache)))
        map_x, map_y = cls._remap_cache[cache_key]
        return _remap(equirectangular_image, map_x, map_y)

    @staticmethod
    def create_cutouts_from_keyframe(
        equirectangular_image: np.ndarray, params: CutoutParams
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        cutouts: list[tuple[np.ndarray, np.ndarray]] = []
        for ith_projection in range(params.nb_of_cuts):
            theta = 360 / params.nb_of_cuts * ith_projection
            f = 0.5 * params.width / np.tan(0.5 * params.fov / 180.0 * np.pi)
            cx = (params.width - 1) / 2.0
            cy = (params.height - 1) / 2.0
            k = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], np.float32)
            k_inv = np.linalg.inv(k)
            x = np.arange(params.width)
            y = np.arange(params.height)
            x, y = np.meshgrid(x, y)
            z = np.ones_like(x)
            xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
            xyz = xyz @ k_inv.T

            y_axis = np.array([0.0, 1.0, 0.0], np.float32)
            x_axis = np.array([1.0, 0.0, 0.0], np.float32)
            r1 = _rodrigues(y_axis * np.radians(theta))
            r2 = _rodrigues(np.dot(r1, x_axis) * np.radians(params.inclination))
            r = r2 @ r1
            xyz = xyz @ r.T
            xy = lonlat2XY(xyz2lonlat(xyz), shape=equirectangular_image.shape).astype(
                np.float32
            )
            cutout_image = _remap(equirectangular_image, xy[..., 0], xy[..., 1])
            cutouts.append((cutout_image, r.T))
        return cutouts


def retrieve_cubemap_face(
    equirect_rgb: np.ndarray,
    face_name: str,
    *,
    face_size: int,
    fov_deg: float = 90.0,
) -> np.ndarray:
    params = CutoutParams(
        nb_of_cuts=1,
        fov=fov_deg,
        inclination=0.0,
        width=face_size,
        height=face_size,
        use_clahe=False,
    )
    return EquirectangularProjector.retrieve_image(
        equirect_rgb,
        relative_pose_cubemap_face(face_name),
        params,
    )
