from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from toolbox.georef.georef import (
    load_image_filename_to_keyframe_id,
    lookup_georef_keyframe_image_filename,
    resolve_keyframe_equirect_image_path,
)
from toolbox.georef.keyframe_id import keyframe_id_from_image_path


def _write_georef_with_image_filenames(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE GeoRefKeyframe (
            id INTEGER PRIMARY KEY,
            pose BLOB NOT NULL,
            frame_id INTEGER NOT NULL,
            video_path TEXT NOT NULL,
            time REAL NOT NULL,
            image_filename TEXT NOT NULL
        )
        """
    )
    pose = np.eye(4, dtype=np.float64).T.tobytes()
    conn.execute(
        "INSERT INTO GeoRefKeyframe "
        "(id, pose, frame_id, video_path, time, image_filename) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (42, pose, 0, "images/uuid-a.jpg", 0.0, "uuid-a.jpg"),
    )
    conn.commit()
    conn.close()


def test_keyframe_id_from_integer_stem_without_mapping(tmp_path: Path) -> None:
    assert keyframe_id_from_image_path(tmp_path / "7.jpg") == 7
    assert keyframe_id_from_image_path(tmp_path / "not-a-number.jpg") is None


def test_keyframe_id_from_image_filename_mapping(tmp_path: Path) -> None:
    mapping = {"uuid-a.jpg": 42, "uuid-b.jpg": 99}
    assert (
        keyframe_id_from_image_path(
            tmp_path / "uuid-a.jpg", image_filename_to_keyframe_id=mapping
        )
        == 42
    )
    assert (
        keyframe_id_from_image_path(
            tmp_path / "missing.jpg", image_filename_to_keyframe_id=mapping
        )
        is None
    )
    assert (
        keyframe_id_from_image_path(
            tmp_path / "7.jpg", image_filename_to_keyframe_id=mapping
        )
        is None
    )


def test_load_image_filename_to_keyframe_id(tmp_path: Path) -> None:
    db_path = tmp_path / "georef.db"
    _write_georef_with_image_filenames(db_path)
    mapping = load_image_filename_to_keyframe_id(db_path)
    assert mapping == {"uuid-a.jpg": 42}


def test_load_image_filename_to_keyframe_id_with_filename_column(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "georef.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE GeoRefKeyframe (
            id INTEGER PRIMARY KEY,
            pose BLOB NOT NULL,
            filename TEXT NOT NULL
        )
        """
    )
    pose = np.eye(4, dtype=np.float64).T.tobytes()
    conn.execute(
        "INSERT INTO GeoRefKeyframe (id, pose, filename) VALUES (?, ?, ?)",
        (2780, pose, "a1b2c3d4-e5f6-7890-abcd-ef1234567890.jpg"),
    )
    conn.commit()
    conn.close()
    assert load_image_filename_to_keyframe_id(db_path) == {
        "a1b2c3d4-e5f6-7890-abcd-ef1234567890.jpg": 2780,
    }
    assert (
        lookup_georef_keyframe_image_filename(db_path, "2780")
        == "a1b2c3d4-e5f6-7890-abcd-ef1234567890.jpg"
    )


def test_resolve_keyframe_equirect_image_path_uses_images_360_uuid(
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "map"
    images_360 = map_path / "images_360"
    images_360.mkdir(parents=True)
    image_name = "002d85bf-3450-40b8-804d-3dc79faed186.jpg"
    image_path = images_360 / image_name
    image_path.write_bytes(b"fake")

    db_path = map_path / "georef.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE GeoRefKeyframe (
            id INTEGER PRIMARY KEY,
            pose BLOB NOT NULL,
            image_filename TEXT NOT NULL
        )
        """
    )
    pose = np.eye(4, dtype=np.float64).T.tobytes()
    conn.execute(
        "INSERT INTO GeoRefKeyframe (id, pose, image_filename) VALUES (?, ?, ?)",
        (2780, pose, image_name),
    )
    conn.commit()
    conn.close()

    resolved, tried = resolve_keyframe_equirect_image_path(map_path, "2780")
    assert resolved == image_path.resolve()
    assert tried == [str(image_path.resolve())]


def test_resolve_keyframe_equirect_image_path_falls_back_when_uuid_missing(
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "map"
    images_360 = map_path / "images_360"
    images_360.mkdir(parents=True)
    fallback_path = images_360 / "2780.jpg"
    fallback_path.write_bytes(b"fake")

    db_path = map_path / "georef.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE GeoRefKeyframe (
            id INTEGER PRIMARY KEY,
            pose BLOB NOT NULL,
            image_filename TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO GeoRefKeyframe (id, pose, image_filename) VALUES (?, ?, ?)",
        (2780, np.eye(4, dtype=np.float64).T.tobytes(), "missing-uuid.jpg"),
    )
    conn.commit()
    conn.close()

    resolved, tried = resolve_keyframe_equirect_image_path(map_path, "2780")
    assert resolved == fallback_path.resolve()
    assert str((images_360 / "missing-uuid.jpg").resolve()) in tried
    assert str(fallback_path.resolve()) in tried


def test_resolve_keyframe_equirect_image_path_falls_back_to_keyframe_id(
    tmp_path: Path,
) -> None:
    map_path = tmp_path / "map"
    images_360 = map_path / "images_360"
    images_360.mkdir(parents=True)
    fallback_path = images_360 / "42.jpg"
    fallback_path.write_bytes(b"fake")

    db_path = map_path / "georef.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE GeoRefKeyframe (id INTEGER PRIMARY KEY, pose BLOB NOT NULL)"
    )
    conn.execute(
        "INSERT INTO GeoRefKeyframe (id, pose) VALUES (?, ?)",
        (42, np.eye(4, dtype=np.float64).T.tobytes()),
    )
    conn.commit()
    conn.close()

    resolved, tried = resolve_keyframe_equirect_image_path(map_path, "42")
    assert resolved == fallback_path.resolve()
    assert str(fallback_path.resolve()) in tried


def test_load_image_filename_to_keyframe_id_returns_none_without_column(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "georef.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE GeoRefKeyframe (id INTEGER PRIMARY KEY, pose BLOB NOT NULL)"
    )
    conn.execute(
        "INSERT INTO GeoRefKeyframe (id, pose) VALUES (1, ?)", (np.eye(4).T.tobytes(),)
    )
    conn.commit()
    conn.close()
    assert load_image_filename_to_keyframe_id(db_path) is None
