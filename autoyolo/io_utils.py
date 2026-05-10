from __future__ import annotations

from pathlib import Path
from typing import Iterable
import re

from autoyolo.models import BoxLabel


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(
        [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    )


def read_classes(classes_file: Path) -> list[str]:
    if not classes_file.exists():
        return []
    raw = classes_file.read_text(encoding="utf-8").splitlines()
    classes = [line.strip() for line in raw if line.strip()]
    return classes


def ensure_dirs(*dirs: Path) -> None:
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def yolo_label_path(image_path: Path, labels_dir: Path, images_dir: Path) -> Path:
    rel = image_path.relative_to(images_dir)
    return (labels_dir / rel).with_suffix(".txt")


def next_sequential_label_index(labels_dir: Path) -> int:
    max_idx = 0
    if not labels_dir.exists():
        return 1
    for p in labels_dir.rglob("*.txt"):
        stem = p.stem
        if re.fullmatch(r"\d+", stem):
            max_idx = max(max_idx, int(stem))
    return max_idx + 1


def next_sequential_subdir_index(parent_dir: Path) -> int:
    max_idx = 0
    if not parent_dir.exists():
        return 1
    for p in parent_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name.strip()
        if re.fullmatch(r"\d+", name):
            max_idx = max(max_idx, int(name))
    return max_idx + 1


def write_yolo_labels(file_path: Path, labels: Iterable[BoxLabel]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{l.class_id} {l.x_center:.6f} {l.y_center:.6f} {l.width:.6f} {l.height:.6f}"
        for l in labels
    ]
    file_path.write_text("\n".join(lines), encoding="utf-8")
