"""Resolve GeoRefKeyframe ids from on-disk equirect image paths."""

from __future__ import annotations

from pathlib import Path


def keyframe_id_from_image_path(
    path: Path,
    *,
    image_filename_to_keyframe_id: dict[str, int] | None = None,
) -> int | None:
    """
    Map an image path to ``GeoRefKeyframe.id``.

    When ``image_filename_to_keyframe_id`` is provided (from georef.db), lookup uses
    ``path.name`` (e.g. ``uuid.jpg``). Otherwise falls back to ``int(path.stem)``.
    """
    if image_filename_to_keyframe_id is not None:
        keyframe_id = image_filename_to_keyframe_id.get(path.name)
        if keyframe_id is not None:
            return int(keyframe_id)
        return None
    try:
        return int(path.stem)
    except ValueError:
        return None
