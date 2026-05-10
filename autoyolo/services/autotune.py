from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from openai import OpenAI
from rich.console import Console

from autoyolo.config import save_config
from autoyolo.io_utils import list_images, yolo_label_path
from autoyolo.models import RunConfig
from autoyolo.pipeline import run_pipeline
from autoyolo.services.plan import build_plan
from autoyolo.services.preannotate import run_preannotation
from autoyolo.services.qc import run_qc


def _extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found")
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : idx + 1])
    raise ValueError("Unclosed JSON object")


def _build_default_objective(profile: str) -> dict[str, Any]:
    profile_lower = profile.lower()
    if any(k in profile_lower for k in ["digit", "number", "symbol", "数字", "字符", "符号"]):
        avg_min, avg_max = 0.85, 1.4
        empty_max, multi_max = 0.10, 0.20
        area_min, area_max = 0.001, 0.12
    elif any(k in profile_lower for k in ["single", "每图一个", "one object", "单目标"]):
        avg_min, avg_max = 0.85, 1.2
        empty_max, multi_max = 0.12, 0.18
        area_min, area_max = 0.0003, 0.18
    elif any(k in profile_lower for k in ["dense", "密集", "multi object", "多目标"]):
        avg_min, avg_max = 2.0, 12.0
        empty_max, multi_max = 0.25, 0.75
        area_min, area_max = 0.00005, 0.3
    else:
        avg_min, avg_max = 0.8, 3.0
        empty_max, multi_max = 0.3, 0.6
        area_min, area_max = 0.0001, 0.25

    return {
        "objective_source": "default",
        "profile": profile,
        "metrics": [
            {"name": "avg_boxes_per_image", "target_min": avg_min, "target_max": avg_max, "weight": 3.0},
            {"name": "empty_rate", "target_min": 0.0, "target_max": empty_max, "weight": 2.0},
            {"name": "multi_rate", "target_min": 0.0, "target_max": multi_max, "weight": 2.0},
            {"name": "avg_box_area_norm", "target_min": area_min, "target_max": area_max, "weight": 0.9},
        ],
    }


def _normalize_objective(raw: dict[str, Any], fallback_profile: str) -> dict[str, Any]:
    metrics = raw.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        return _build_default_objective(fallback_profile)

    valid_metrics: list[dict[str, Any]] = []
    allowed = {"avg_boxes_per_image", "empty_rate", "multi_rate", "avg_box_area_norm"}
    for item in metrics:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if name not in allowed:
            continue
        try:
            target_min = float(item.get("target_min", 0.0))
            target_max = float(item.get("target_max", 1.0))
            if target_min > target_max:
                target_min, target_max = target_max, target_min
            weight = float(item.get("weight", 1.0))
        except (TypeError, ValueError):
            continue
        valid_metrics.append(
            {"name": name, "target_min": target_min, "target_max": target_max, "weight": max(weight, 0.1)}
        )

    if not valid_metrics:
        return _build_default_objective(fallback_profile)

    normalized = {
        "objective_source": raw.get("objective_source", "llm"),
        "profile": raw.get("profile", fallback_profile),
        "metrics": valid_metrics,
    }
    return normalized


def _build_objective_via_openai(config: RunConfig, profile: str, classes: list[str]) -> dict[str, Any]:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    client = OpenAI(api_key=api_key or None, base_url=config.openai_base_url or os.getenv("OPENAI_BASE_URL"))
    prompt = (
        "Given dataset profile and class names, build tuning objective for detector pre-annotation. "
        "Return strict JSON with key 'metrics' (list). Each metric item: "
        "name(one of avg_boxes_per_image, empty_rate, multi_rate, avg_box_area_norm), "
        "target_min(number), target_max(number), weight(number). "
        "Keep objective generic for this profile, not hardcoded for a specific test set. "
        "Infer plausible size constraints from profile (tiny/medium/large objects) and avoid unrealistic ranges.\n"
        f"Profile: {profile}\n"
        f"Classes: {classes}"
    )
    response = client.chat.completions.create(
        model=config.llm_model,
        messages=[
            {"role": "system", "content": "You output JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        timeout=60,
    )
    content = response.choices[0].message.content or "{}"
    data = json.loads(content)
    data["objective_source"] = "openai"
    return data


def _build_objective_via_opencode(config: RunConfig, profile: str, classes: list[str]) -> dict[str, Any]:
    prompt = (
        "Given dataset profile and class names, build tuning objective for detector pre-annotation. "
        "Return strict JSON with key 'metrics' (list). Each metric item: "
        "name(one of avg_boxes_per_image, empty_rate, multi_rate, avg_box_area_norm), "
        "target_min(number), target_max(number), weight(number). "
        "Keep objective generic for this profile, not hardcoded for a specific test set. "
        "Infer plausible size constraints from profile (tiny/medium/large objects) and avoid unrealistic ranges.\n"
        f"Profile: {profile}\n"
        f"Classes: {classes}"
    )
    cmd = [config.opencode_executable, *config.opencode_runner_args.split(), "--model", config.llm_model, prompt]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=max(60, int(config.opencode_timeout_sec)),
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "OpenCode failed")
    data = _extract_json_object(result.stdout)
    data["objective_source"] = "opencode"
    return data


def _build_objective(config: RunConfig, profile: str, classes: list[str], console: Console) -> dict[str, Any]:
    if config.llm_provider == "opencode":
        console.print("[cyan]Building autotune objective via OpenCode...[/cyan]")
        try:
            return _normalize_objective(_build_objective_via_opencode(config, profile, classes), profile)
        except Exception as exc:
            console.print(f"[yellow]Objective fallback:[/yellow] OpenCode failed: {exc}")
            return _build_default_objective(profile)

    if config.llm_provider == "openai":
        console.print("[cyan]Building autotune objective via OpenAI...[/cyan]")
        try:
            return _normalize_objective(_build_objective_via_openai(config, profile, classes), profile)
        except Exception as openai_exc:
            if config.opencode_fallback_on_openai_error:
                console.print("[yellow]OpenAI objective failed, trying OpenCode fallback...[/yellow]")
                try:
                    data = _build_objective_via_opencode(config, profile, classes)
                    data["objective_source"] = "opencode_fallback"
                    data["objective_warning"] = f"OpenAI objective failed: {openai_exc}"
                    return _normalize_objective(data, profile)
                except Exception as opencode_exc:
                    console.print(
                        "[yellow]Objective fallback:[/yellow] "
                        f"OpenAI failed: {openai_exc}; OpenCode failed: {opencode_exc}"
                    )
                    return _build_default_objective(profile)
            console.print(f"[yellow]Objective fallback:[/yellow] OpenAI failed: {openai_exc}")
            return _build_default_objective(profile)

    return _build_default_objective(profile)


def _collect_label_metrics_for_images(config: RunConfig, images: list[Path]) -> dict[str, float]:
    images_dir = config.abs_path(config.images_dir)
    labels_dir = config.abs_path(config.labels_dir)
    if not images:
        return {
            "images": 0,
            "total_boxes": 0,
            "avg_boxes_per_image": 0.0,
            "empty_rate": 1.0,
            "multi_rate": 0.0,
            "avg_box_area_norm": 0.0,
        }

    total_boxes = 0
    empty_count = 0
    multi_count = 0
    area_sum = 0.0
    area_n = 0

    for image in images:
        label_file = yolo_label_path(image, labels_dir, images_dir)
        boxes_this_image = 0
        if label_file.exists():
            for line in label_file.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    width = float(parts[3])
                    height = float(parts[4])
                except ValueError:
                    continue
                boxes_this_image += 1
                area_sum += max(0.0, width) * max(0.0, height)
                area_n += 1

        total_boxes += boxes_this_image
        if boxes_this_image == 0:
            empty_count += 1
        if boxes_this_image > 1:
            multi_count += 1

    image_count = len(images)
    return {
        "images": float(image_count),
        "total_boxes": float(total_boxes),
        "avg_boxes_per_image": total_boxes / image_count,
        "empty_rate": empty_count / image_count,
        "multi_rate": multi_count / image_count,
        "avg_box_area_norm": (area_sum / area_n) if area_n else 0.0,
    }


def _distance_to_range(value: float, minimum: float, maximum: float) -> float:
    if minimum <= value <= maximum:
        return 0.0
    if value < minimum:
        return (minimum - value) / max(abs(minimum), 1e-6)
    return (value - maximum) / max(abs(maximum), 1e-6)


def _objective_loss(objective: dict[str, Any], metrics: dict[str, float]) -> float:
    loss = 0.0
    for item in objective.get("metrics", []):
        name = item["name"]
        value = float(metrics.get(name, 0.0))
        loss += float(item.get("weight", 1.0)) * _distance_to_range(
            value,
            float(item.get("target_min", 0.0)),
            float(item.get("target_max", 1.0)),
        )
    return loss


def _metric_range(objective: dict[str, Any], name: str, default_min: float, default_max: float) -> tuple[float, float]:
    for item in objective.get("metrics", []):
        if item.get("name") == name:
            return float(item.get("target_min", default_min)), float(item.get("target_max", default_max))
    return default_min, default_max


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _adjust_config(config: RunConfig, metrics: dict[str, float], objective: dict[str, Any]) -> RunConfig:
    avg_min, avg_max = _metric_range(objective, "avg_boxes_per_image", 0.8, 3.0)
    empty_min, empty_max = _metric_range(objective, "empty_rate", 0.0, 0.3)
    multi_min, multi_max = _metric_range(objective, "multi_rate", 0.0, 0.6)

    avg_boxes = float(metrics.get("avg_boxes_per_image", 0.0))
    empty_rate = float(metrics.get("empty_rate", 1.0))
    multi_rate = float(metrics.get("multi_rate", 0.0))

    delta_up = 0.0
    delta_down = 0.0
    if avg_boxes > avg_max:
        delta_up += min(0.08, 0.02 + (avg_boxes - avg_max) * 0.02)
    elif avg_boxes < avg_min:
        delta_down += min(0.08, 0.02 + (avg_min - avg_boxes) * 0.02)

    if empty_rate > empty_max:
        delta_down += min(0.08, 0.015 + (empty_rate - empty_max) * 0.08)
    if multi_rate > multi_max:
        delta_up += min(0.08, 0.015 + (multi_rate - multi_max) * 0.06)

    new_cfg = copy.deepcopy(config)
    net = delta_up - delta_down
    new_cfg.box_threshold = _clamp(config.box_threshold + net, 0.02, 0.8)
    new_cfg.text_threshold = _clamp(config.text_threshold + net * 0.75, 0.02, 0.8)

    area_scale = 1.0 + delta_up * 1.1 - delta_down * 0.9
    new_cfg.min_box_area_norm = _clamp(config.min_box_area_norm * area_scale, 1e-7, 0.02)

    if multi_rate > multi_max:
        new_cfg.nms_iou_threshold = _clamp(config.nms_iou_threshold - 0.03, 0.3, 0.9)
        new_cfg.max_detections_per_class = max(3, config.max_detections_per_class - 1)
    elif avg_boxes < avg_min and empty_rate > empty_max:
        new_cfg.nms_iou_threshold = _clamp(config.nms_iou_threshold + 0.02, 0.3, 0.9)
        new_cfg.max_detections_per_class = min(120, config.max_detections_per_class + 1)

    return new_cfg


def _round_sample(images: list[Path], probe_images: int) -> list[Path]:
    if probe_images <= 0 or probe_images >= len(images):
        return images
    return images[:probe_images]


def _run_probe_round(
    *,
    config: RunConfig,
    images: list[Path],
    classes: list[str],
    console: Console,
) -> dict[str, Any]:
    preannotate_report = run_preannotation(config=config, images=images, classes=classes, console=console)
    qc_file = config.abs_path(config.reports_dir) / "qc_report_probe.json"
    qc_report = run_qc(config=config, images=images, classes=classes, report_file=qc_file)
    return {
        "images": len(images),
        "classes": len(classes),
        "preannotate": preannotate_report,
        "qc": {"issues_total": qc_report["issues_total"]},
        "qc_file": str(qc_file),
    }


def run_autotune(
    *,
    config: RunConfig,
    config_path: Path,
    profile: str,
    max_rounds: int,
    target_loss: float,
    probe_images: int,
    full_eval_trigger_loss: float,
    console: Console,
) -> dict[str, Any]:
    images = list_images(config.abs_path(config.images_dir))
    classes_file = config.abs_path(config.classes_file)
    classes = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    console.print(
        f"[bold cyan]Autotune start[/bold cyan] | images={len(images)} classes={len(classes)} provider={config.llm_provider}"
    )
    console.print(f"[cyan]Profile:[/cyan] {profile}")

    objective = _build_objective(config, profile, classes, console)
    objective_file = config.abs_path(config.reports_dir) / "objective_spec.json"
    objective_file.parent.mkdir(parents=True, exist_ok=True)
    objective_file.write_text(json.dumps(objective, indent=2), encoding="utf-8")
    console.print(f"[green]Objective saved:[/green] {objective_file}")

    plan = build_plan(config, classes)
    plan_file = config.abs_path(config.reports_dir) / "annotation_plan.json"
    plan_file.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    console.print(f"[green]Plan saved:[/green] {plan_file}")
    if plan.get("plan_provider") in {"mock_fallback", "opencode_fallback"}:
        warning = plan.get("plan_warning", "unknown reason")
        console.print(f"[yellow]Plan fallback activated:[/yellow] {warning}")

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_round = 0
    best_config = config.model_dump(mode="python")

    current = copy.deepcopy(config)
    for round_idx in range(1, max_rounds + 1):
        console.print(f"[bold cyan]Autotune round {round_idx}/{max_rounds}[/bold cyan]")
        console.print(
            "[cyan]Round params:[/cyan] "
            f"box={current.box_threshold:.4f}, text={current.text_threshold:.4f}, "
            f"min_area={current.min_box_area_norm:.6f}, nms_iou={current.nms_iou_threshold:.3f}, "
            f"max_det_cls={current.max_detections_per_class}"
        )

        sample_images = _round_sample(images, probe_images)
        console.print(
            f"[cyan]Probe evaluation:[/cyan] using {len(sample_images)}/{len(images)} images for fast feedback"
        )
        pipeline_result = _run_probe_round(config=current, images=sample_images, classes=classes, console=console)
        metrics = _collect_label_metrics_for_images(current, sample_images)
        loss = _objective_loss(objective, metrics)

        full_metrics: dict[str, float] | None = None
        full_loss: float | None = None
        if loss <= full_eval_trigger_loss and len(sample_images) < len(images):
            console.print(
                f"[cyan]Probe looks promising (loss={loss:.4f}), running full dataset validation...[/cyan]"
            )
            pipeline_result = run_pipeline(current, console)
            full_metrics = _collect_label_metrics_for_images(current, images)
            full_loss = _objective_loss(objective, full_metrics)
            metrics = full_metrics
            loss = full_loss

        row = {
            "round": round_idx,
            "loss": loss,
            "pipeline": pipeline_result,
            "probe_images": len(sample_images),
            "full_eval_trigger_loss": full_eval_trigger_loss,
            "ran_full_eval": full_metrics is not None,
            "metrics": metrics,
            "params": {
                "box_threshold": current.box_threshold,
                "text_threshold": current.text_threshold,
                "min_box_area_norm": current.min_box_area_norm,
                "nms_iou_threshold": current.nms_iou_threshold,
                "max_detections_per_class": current.max_detections_per_class,
            },
        }
        history.append(row)
        console.print(
            "[magenta]Round metrics:[/magenta] "
            f"boxes={int(metrics['total_boxes'])}, avg={metrics['avg_boxes_per_image']:.3f}, "
            f"empty={metrics['empty_rate']:.3f}, multi={metrics['multi_rate']:.3f}, loss={loss:.4f}"
        )

        if loss < best_loss:
            best_loss = loss
            best_round = round_idx
            best_config = current.model_dump(mode="python")
            save_config(RunConfig.model_validate(best_config), config_path)
            console.print(f"[green]Best config updated and persisted at round {best_round}[/green]")

        if loss <= target_loss:
            console.print(f"[green]Converged:[/green] loss {loss:.4f} <= target {target_loss:.4f}")
            break

        anneal = max(0.35, 1.0 - (round_idx / max_rounds) * 0.55)
        adjusted_metrics = dict(metrics)
        adjusted_metrics["avg_boxes_per_image"] = float(adjusted_metrics.get("avg_boxes_per_image", 0.0))
        current = _adjust_config(current, adjusted_metrics, objective)
        current.box_threshold = _clamp(config.box_threshold + (current.box_threshold - config.box_threshold) * anneal, 0.02, 0.8)
        current.text_threshold = _clamp(
            config.text_threshold + (current.text_threshold - config.text_threshold) * anneal, 0.02, 0.8
        )

    history_file = config.abs_path(config.reports_dir) / "tune_history.json"
    history_file.write_text(json.dumps(history, indent=2), encoding="utf-8")

    final_cfg = RunConfig.model_validate(best_config)
    save_config(final_cfg, config_path)
    if best_round != len(history):
        console.print(f"[cyan]Replaying best round config (round {best_round})[/cyan]")
        try:
            run_pipeline(final_cfg, console)
        except Exception as replay_exc:
            console.print(f"[yellow]Replay skipped due to transient error:[/yellow] {replay_exc}")

    return {
        "images": len(images),
        "probe_images": min(max(1, probe_images), len(images)),
        "objective_file": str(objective_file),
        "history_file": str(history_file),
        "best_round": best_round,
        "best_loss": best_loss,
        "rounds_executed": len(history),
        "objective_source": objective.get("objective_source", "default"),
    }
