from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from autoyolo.io_utils import ensure_dirs, list_images, read_classes
from autoyolo.models import RunConfig
from autoyolo.services.plan import build_plan
from autoyolo.services.preannotate import run_preannotation
from autoyolo.services.qc import run_qc


def run_pipeline(config: RunConfig, console: Console) -> dict:
    images_dir = config.abs_path(config.images_dir)
    labels_dir = config.abs_path(config.labels_dir)
    reports_dir = config.abs_path(config.reports_dir)
    classes_file = config.abs_path(config.classes_file)

    if images_dir.is_file():
        raise RuntimeError(
            f"images_dir points to a file, not a folder: {images_dir}. "
            "Please set images_dir to an image directory."
        )
    if classes_file.is_dir():
        raise RuntimeError(
            f"classes_file points to a folder, not a file: {classes_file}. "
            "Please set classes_file to classes.txt."
        )

    ensure_dirs(labels_dir, reports_dir)

    images = list_images(images_dir)
    classes = read_classes(classes_file)
    if not images:
        raise RuntimeError(f"No images found in: {images_dir}")
    if not classes:
        raise RuntimeError(f"No classes found in: {classes_file}")

    console.print(f"[cyan]Images:[/cyan] {len(images)}")
    console.print(f"[cyan]Classes:[/cyan] {len(classes)} -> {classes}")

    plan = build_plan(config, classes)
    plan_file = reports_dir / "annotation_plan.json"
    plan_file.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    console.print(f"[green]Plan saved:[/green] {plan_file}")
    if plan.get("plan_provider") in {"mock_fallback", "opencode_fallback"}:
        warning = plan.get("plan_warning", "unknown reason")
        console.print(f"[yellow]Plan fallback activated:[/yellow] {warning}")

    preannotate_report = run_preannotation(config=config, images=images, classes=classes, console=console)

    qc_file = reports_dir / "qc_report.json"
    qc_report = run_qc(
        config=config,
        images=images,
        classes=classes,
        report_file=qc_file,
        label_map=preannotate_report.get("label_map"),
    )
    console.print(f"[green]QC saved:[/green] {qc_file}")

    result = {
        "images": len(images),
        "classes": len(classes),
        "plan_file": str(plan_file),
        "qc_file": str(qc_file),
        "preannotate": preannotate_report,
        "qc": {"issues_total": qc_report["issues_total"]},
    }
    return result
