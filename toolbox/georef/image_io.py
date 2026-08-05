from pathlib import Path
from typing import List


def load_image_paths(images_dir: Path) -> List[Path]:
    image_extensions = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"}
    paths = [p for p in images_dir.iterdir() if p.suffix in image_extensions]
    paths.sort()
    return paths
