from __future__ import annotations

import json
from pathlib import Path

from autoyolo.io_utils import yolo_label_path
from autoyolo.models import RunConfig


def _parse_yolo_line(line: str) -> tuple[int, float, float, float, float] | None:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        return int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    except ValueError:
        return None


def run_qc(
    *,
    config: RunConfig,
    images: list[Path],
    classes: list[str],
    report_file: Path,
    label_map: dict[str, str] | None = None,
) -> dict:
    issues: list[dict] = []
    labels_dir = config.abs_path(config.labels_dir)
    images_dir = config.abs_path(config.images_dir)

    for image in images:
        if label_map and str(image) in label_map:
            label_file = Path(label_map[str(image)])
        else:
            label_file = yolo_label_path(image, labels_dir, images_dir)
        if not label_file.exists():
            issues.append({"type": "missing_label", "image": str(image)})
            continue

        lines = label_file.read_text(encoding="utf-8").splitlines()
        for line_no, line in enumerate(lines, start=1):
            parsed = _parse_yolo_line(line)
            if parsed is None:
                issues.append(
                    {
                        "type": "bad_line_format",
                        "image": str(image),
                        "label_file": str(label_file),
                        "line": line_no,
                        "content": line,
                    }
                )
                continue
            class_id, x_center, y_center, width, height = parsed
            if class_id < 0 or class_id >= len(classes):
                issues.append({"type": "invalid_class_id", "image": str(image), "line": line_no})
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
                issues.append({"type": "invalid_bbox_range", "image": str(image), "line": line_no})

    report = {
        "images_checked": len(images),
        "issues_total": len(issues),
        "issues": issues,
    }
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
