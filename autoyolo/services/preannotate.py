from __future__ import annotations

from dataclasses import dataclass
import base64
import inspect
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from openai import OpenAI

from autoyolo.io_utils import (
    next_sequential_label_index,
    next_sequential_subdir_index,
    write_yolo_labels,
    yolo_label_path,
)
from autoyolo.models import BoxLabel, RunConfig


@dataclass
class _RawDetection:
    class_id: int
    score: float
    x1: float
    y1: float
    x2: float
    y2: float


_GDINO_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}
_QWEN_VL_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}


def _resolve_device(config: RunConfig, torch_module: Any) -> str:
    if config.inference_device != "auto":
        return config.inference_device
    if torch_module.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch_module.backends, "mps", None)
    if mps_backend and mps_backend.is_available():
        return "mps"
    return "cpu"


def _iou(a: _RawDetection, b: _RawDetection) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _nms_per_class(detections: list[_RawDetection], iou_thr: float, max_keep: int) -> list[_RawDetection]:
    kept: list[_RawDetection] = []
    for class_id in sorted({d.class_id for d in detections}):
        pool = sorted([d for d in detections if d.class_id == class_id], key=lambda d: d.score, reverse=True)
        while pool and len([k for k in kept if k.class_id == class_id]) < max_keep:
            best = pool.pop(0)
            kept.append(best)
            pool = [cand for cand in pool if _iou(best, cand) < iou_thr]
    return kept


def _to_box_label(det: _RawDetection, image_w: int, image_h: int) -> BoxLabel | None:
    bw = max(0.0, det.x2 - det.x1)
    bh = max(0.0, det.y2 - det.y1)
    if bw <= 0 or bh <= 0:
        return None
    x_center = ((det.x1 + det.x2) / 2.0) / image_w
    y_center = ((det.y1 + det.y2) / 2.0) / image_h
    width = bw / image_w
    height = bh / image_h
    return BoxLabel(
        class_id=det.class_id,
        x_center=min(max(x_center, 0.0), 1.0),
        y_center=min(max(y_center, 0.0), 1.0),
        width=min(max(width, 1e-6), 1.0),
        height=min(max(height, 1e-6), 1.0),
        confidence=det.score,
    )


def _run_grounding_dino(
    *,
    config: RunConfig,
    images: list[Path],
    classes: list[str],
    console: Console,
    label_targets: dict[Path, Path] | None = None,
) -> dict:
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "GroundingDINO backend needs extra deps. Install with: pip install torch pillow transformers"
        ) from exc

    device = _resolve_device(config, torch)
    model_id = config.grounding_dino_model_id
    console.print(f"[cyan]Loading GroundingDINO:[/cyan] {model_id} on {device}")

    cache_key = (model_id, device)
    if cache_key in _GDINO_CACHE:
        processor, model = _GDINO_CACHE[cache_key]
        console.print("[cyan]Using cached GroundingDINO model in memory[/cyan]")
    else:
        try:
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        except Exception as exc:
            console.print(
                "[yellow]Online model metadata fetch failed, retrying from local HF cache...[/yellow]"
            )
            processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, local_files_only=True).to(device)
            console.print(f"[yellow]Recovered from local cache after error:[/yellow] {exc}")
        model.eval()
        _GDINO_CACHE[cache_key] = (processor, model)

    images_dir = config.abs_path(config.images_dir)
    labels_dir = config.abs_path(config.labels_dir)
    total_boxes = 0

    with torch.inference_mode(), Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task_id = progress.add_task("GroundingDINO pre-annotation", total=len(images))
        for idx, image_path in enumerate(images, start=1):
            image = Image.open(image_path).convert("RGB")
            image_w, image_h = image.size
            raw_dets: list[_RawDetection] = []

            for class_id, class_name in enumerate(classes):
                text_prompt = class_name.strip().lower()
                if not text_prompt:
                    continue

                inputs = processor(images=image, text=text_prompt, return_tensors="pt")
                inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
                outputs = model(**inputs)

                post_process = processor.post_process_grounded_object_detection
                sig = inspect.signature(post_process)
                kwargs: dict[str, Any] = {"target_sizes": [(image_h, image_w)]}
                if "box_threshold" in sig.parameters:
                    kwargs["box_threshold"] = config.box_threshold
                elif "threshold" in sig.parameters:
                    kwargs["threshold"] = config.box_threshold
                if "text_threshold" in sig.parameters:
                    kwargs["text_threshold"] = config.text_threshold

                results = post_process(outputs, inputs["input_ids"], **kwargs)

                for box_tensor, score_tensor in zip(results[0]["boxes"], results[0]["scores"]):
                    x1, y1, x2, y2 = [float(v) for v in box_tensor.tolist()]
                    x1 = min(max(x1, 0.0), image_w)
                    y1 = min(max(y1, 0.0), image_h)
                    x2 = min(max(x2, 0.0), image_w)
                    y2 = min(max(y2, 0.0), image_h)
                    det = _RawDetection(class_id=class_id, score=float(score_tensor), x1=x1, y1=y1, x2=x2, y2=y2)
                    area_norm = ((det.x2 - det.x1) * (det.y2 - det.y1)) / max(1.0, image_w * image_h)
                    if area_norm >= config.min_box_area_norm:
                        raw_dets.append(det)

            filtered = _nms_per_class(raw_dets, config.nms_iou_threshold, config.max_detections_per_class)
            labels: list[BoxLabel] = []
            for det in filtered:
                label = _to_box_label(det, image_w, image_h)
                if label is not None:
                    labels.append(label)

            label_file = label_targets.get(image_path) if label_targets else yolo_label_path(image_path, labels_dir, images_dir)
            write_yolo_labels(label_file, labels)
            total_boxes += len(labels)
            progress.update(
                task_id,
                advance=1,
                description=f"GroundingDINO pre-annotation | last={image_path.name} | boxes={len(labels)}",
            )

            if idx % 10 == 0 or idx == len(images):
                console.print(
                    f"[cyan]Checkpoint:[/cyan] {idx}/{len(images)} images, accumulated boxes={total_boxes}"
                )

    return {
        "label_files_written": len(images),
        "detector_backend": config.detector_backend,
        "total_boxes": total_boxes,
        "model_id": model_id,
        "device": device,
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in VLM output")
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : idx + 1])
    raise ValueError("Unclosed JSON object in VLM output")


def _read_image_as_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower().lstrip(".")
    media_type = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    raw = image_path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:image/{media_type};base64,{encoded}"


def _read_image_as_base64(image_path: Path) -> str:
    raw = image_path.read_bytes()
    return base64.b64encode(raw).decode("ascii")


def _normalize_vlm_detections(payload: dict[str, Any], classes: list[str], image_w: int, image_h: int) -> list[BoxLabel]:
    raw = payload.get("detections", payload)
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    class_map = {name: idx for idx, name in enumerate(classes)}
    labels: list[BoxLabel] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        cls_name = str(item.get("class", "")).strip()
        if cls_name not in class_map:
            continue
        try:
            x1 = float(item.get("x1"))
            y1 = float(item.get("y1"))
            x2 = float(item.get("x2"))
            y2 = float(item.get("y2"))
            conf = float(item.get("confidence", 0.6))
        except (TypeError, ValueError):
            continue

        x1 = min(max(x1, 0.0), float(image_w))
        y1 = min(max(y1, 0.0), float(image_h))
        x2 = min(max(x2, 0.0), float(image_w))
        y2 = min(max(y2, 0.0), float(image_h))
        if x2 <= x1 or y2 <= y1:
            continue

        label = BoxLabel(
            class_id=class_map[cls_name],
            x_center=((x1 + x2) / 2.0) / image_w,
            y_center=((y1 + y2) / 2.0) / image_h,
            width=(x2 - x1) / image_w,
            height=(y2 - y1) / image_h,
            confidence=min(max(conf, 0.0), 1.0),
        )
        labels.append(label)
    return labels


def _load_local_qwen_vl(model_path: str, device: str) -> tuple[Any, Any]:
    cache_key = (model_path, device)
    if cache_key in _QWEN_VL_CACHE:
        return _QWEN_VL_CACHE[cache_key]

    try:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Local Qwen-VL backend needs torch/transformers. Install: pip install torch torchvision transformers accelerate qwen-vl-utils"
        ) from exc

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    _QWEN_VL_CACHE[cache_key] = (model, processor)
    return model, processor


def _resize_for_qwen(image_path: Path, max_side: int) -> tuple[Path, tuple[int, int], tuple[int, int]]:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    if max_side <= 0:
        return image_path, (orig_w, orig_h), (orig_w, orig_h)
    longest = max(orig_w, orig_h)
    if longest <= max_side:
        return image_path, (orig_w, orig_h), (orig_w, orig_h)

    ratio = max_side / float(longest)
    new_w = max(1, int(orig_w * ratio))
    new_h = max(1, int(orig_h * ratio))
    resized = image.resize((new_w, new_h), Image.LANCZOS)

    fd, tmp_name = tempfile.mkstemp(suffix=".jpg", prefix="autoyolo_qwen_")
    os.close(fd)
    tmp_path = Path(tmp_name)
    resized.save(tmp_path, format="JPEG", quality=85)
    return tmp_path, (orig_w, orig_h), (new_w, new_h)


def _parse_labeled_boxes_from_text(content: str, classes: list[str]) -> list[tuple[str, float, float, float, float]]:
    class_set = set(classes)
    labeled: list[tuple[str, float, float, float, float]] = []

    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned).strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    try:
        parsed_json = json.loads(cleaned)
    except Exception:
        parsed_json = None

    if isinstance(parsed_json, dict):
        parsed_json = [parsed_json]
    if isinstance(parsed_json, list):
        for item in parsed_json:
            if isinstance(item, dict) and "bbox_2d" in item:
                bbox = item.get("bbox_2d")
                label = str(item.get("label", item.get("class", "")).strip())
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                try:
                    x1, y1, x2, y2 = [float(v) for v in bbox]
                except (TypeError, ValueError):
                    continue
                if not label:
                    if len(classes) == 1:
                        label = classes[0]
                    else:
                        continue
                if label in class_set:
                    labeled.append((label, x1, y1, x2, y2))
        if labeled:
            return labeled

    pattern = re.compile(
        r"\[\s*['\"]?([^\]\[,\"']+)['\"]?\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
    )
    for match in pattern.finditer(content):
        label = match.group(1).strip()
        if label not in class_set:
            continue
        x1, y1, x2, y2 = [float(match.group(i)) for i in range(2, 6)]
        labeled.append((label, x1, y1, x2, y2))

    if labeled:
        return labeled

    numeric_pattern = re.compile(
        r"\[\s*(\d{1,2})\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
    )
    for match in numeric_pattern.finditer(content):
        label_idx = int(match.group(1))
        if label_idx < 0 or label_idx >= len(classes):
            continue
        label = classes[label_idx]
        x1, y1, x2, y2 = [float(match.group(i)) for i in range(2, 6)]
        labeled.append((label, x1, y1, x2, y2))

    return labeled


def _map_candidate_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    orig_w: int,
    orig_h: int,
    run_w: int,
    run_h: int,
) -> tuple[float, float, float, float]:
    vals = [x1, y1, x2, y2]
    if max(abs(v) for v in vals) <= 1.0:
        ax1 = x1 * orig_w
        ay1 = y1 * orig_h
        ax2 = x2 * orig_w
        ay2 = y2 * orig_h
        return ax1, ay1, ax2, ay2

    sx = float(orig_w) / float(run_w)
    sy = float(orig_h) / float(run_h)
    return x1 * sx, y1 * sy, x2 * sx, y2 * sy


def _parse_single_box_from_text(content: str) -> tuple[float, float, float, float] | None:
    pattern = re.compile(
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
    )
    m = pattern.search(content)
    if not m:
        return None
    x1, y1, x2, y2 = [float(m.group(i)) for i in range(1, 5)]
    return x1, y1, x2, y2


def _run_local_qwen_vl(
    *,
    config: RunConfig,
    images: list[Path],
    classes: list[str],
    console: Console,
    label_targets: dict[Path, Path] | None = None,
) -> dict:
    try:
        from qwen_vl_utils import process_vision_info
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "local_qwen_vl backend needs qwen-vl-utils and torch. Install required dependencies first."
        ) from exc

    model_path = config.local_qwen_model_path
    model, processor = _load_local_qwen_vl(model_path, config.local_qwen_device)

    images_dir = config.abs_path(config.images_dir)
    labels_dir = config.abs_path(config.labels_dir)
    class_to_id = {name: idx for idx, name in enumerate(classes)}
    canonical_classes_file = labels_dir / "classes.txt"
    if canonical_classes_file.exists():
        canonical_classes = [
            line.strip()
            for line in canonical_classes_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        canonical_map = {name: idx for idx, name in enumerate(canonical_classes)}
        class_to_id = {name: canonical_map.get(name, class_to_id[name]) for name in class_to_id}

    query_classes = classes
    if config.local_qwen_query_classes.strip():
        requested = [s.strip() for s in config.local_qwen_query_classes.split(",") if s.strip()]
        query_classes = [c for c in requested if c in class_to_id]
        if not query_classes:
            raise RuntimeError(
                f"local_qwen_query_classes has no valid class in current classes: {config.local_qwen_query_classes}"
            )
    total_boxes = 0

    with Progress(
        TextColumn("[cyan]Working[/cyan]"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task_id = progress.add_task("Local Qwen-VL pre-annotation", total=len(images))
        for idx, image_path in enumerate(images, start=1):
            tmp_path = None
            try:
                resized_path, (orig_w, orig_h), (run_w, run_h) = _resize_for_qwen(
                    image_path,
                    max(256, int(config.local_qwen_max_image_side)),
                )
                if resized_path != image_path:
                    tmp_path = resized_path

                labels: list[BoxLabel] = []
                last_snippet = ""
                for target in query_classes:
                    prompts = [
                        (
                            f"Find the number {target}. Output ONLY its bounding box as [x1,y1,x2,y2]. "
                            "No other text. If not found, output []."
                        ),
                        (
                            f"Find class '{target}'. Return ONLY one box as [x1,y1,x2,y2]. "
                            "No words. If not found, return []."
                        ),
                    ]

                    content = ""
                    parsed_target: list[tuple[str, float, float, float, float]] = []
                    for prompt in prompts:
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": str(resized_path)},
                                    {"type": "text", "text": prompt},
                                ],
                            }
                        ]
                        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        image_inputs, _ = process_vision_info(messages)
                        inputs = processor(
                            text=[text],
                            images=image_inputs,
                            padding=True,
                            return_tensors="pt",
                        ).to(config.local_qwen_device)

                        with torch.inference_mode():
                            generated_ids = model.generate(
                                **inputs,
                                max_new_tokens=max(16, int(config.local_qwen_max_new_tokens)),
                                do_sample=False,
                            )
                        generated_trimmed = [
                            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        content = processor.batch_decode(
                            generated_trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )[0]
                        box = _parse_single_box_from_text(content)
                        parsed_target = []
                        if box is not None:
                            parsed_target = [(target, box[0], box[1], box[2], box[3])]
                        if parsed_target or content.strip() == "[]":
                            break

                    if not parsed_target and content.strip() not in {"[]", ""}:
                        snippet = content.strip().replace("\n", " ")
                        if len(snippet) > 240:
                            snippet = snippet[:240] + "..."
                        last_snippet = snippet

                    for cls_name, x1, y1, x2, y2 in parsed_target:
                        class_id = class_to_id[cls_name]
                        x1, y1, x2, y2 = _map_candidate_box(
                            x1,
                            y1,
                            x2,
                            y2,
                            orig_w=orig_w,
                            orig_h=orig_h,
                            run_w=run_w,
                            run_h=run_h,
                        )
                        x1 = min(max(x1, 0.0), float(orig_w))
                        y1 = min(max(y1, 0.0), float(orig_h))
                        x2 = min(max(x2, 0.0), float(orig_w))
                        y2 = min(max(y2, 0.0), float(orig_h))
                        if x2 <= x1:
                            x2 = min(float(orig_w), x1 + 1.0)
                        if y2 <= y1:
                            y2 = min(float(orig_h), y1 + 1.0)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        labels.append(
                            BoxLabel(
                                class_id=class_id,
                                x_center=((x1 + x2) / 2.0) / orig_w,
                                y_center=((y1 + y2) / 2.0) / orig_h,
                                width=(x2 - x1) / orig_w,
                                height=(y2 - y1) / orig_h,
                                confidence=0.65,
                            )
                        )

                if not labels:
                    label_file = label_targets.get(image_path) if label_targets else yolo_label_path(image_path, labels_dir, images_dir)
                    write_yolo_labels(label_file, [])
                    if last_snippet:
                        console.print(
                            f"[yellow]Local Qwen-VL parse warning:[/yellow] {image_path.name} -> {last_snippet}"
                        )
                    progress.update(
                        task_id,
                        advance=1,
                        description=f"Local Qwen-VL pre-annotation | last={image_path.name} | boxes=0",
                    )
                    if idx % 10 == 0 or idx == len(images):
                        console.print(
                            f"[cyan]Checkpoint:[/cyan] {idx}/{len(images)} images, accumulated boxes={total_boxes}"
                        )
                    continue

                label_file = label_targets.get(image_path) if label_targets else yolo_label_path(image_path, labels_dir, images_dir)
                write_yolo_labels(label_file, labels)
                total_boxes += len(labels)
                progress.update(
                    task_id,
                    advance=1,
                    description=f"Local Qwen-VL pre-annotation | last={image_path.name} | boxes={len(labels)}",
                )
                if idx % 10 == 0 or idx == len(images):
                    console.print(
                        f"[cyan]Checkpoint:[/cyan] {idx}/{len(images)} images, accumulated boxes={total_boxes}"
                    )
            finally:
                if tmp_path is not None and tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass

    return {
        "label_files_written": len(images),
        "detector_backend": config.detector_backend,
        "total_boxes": total_boxes,
        "model_id": model_path,
        "device": config.local_qwen_device,
    }


def _run_vlm_api(
    *,
    config: RunConfig,
    images: list[Path],
    classes: list[str],
    console: Console,
    label_targets: dict[Path, Path] | None = None,
) -> dict:
    api_key = os.getenv(config.vlm_api_key_env, "")
    if not api_key:
        raise RuntimeError(
            f"VLM API key env var not found: {config.vlm_api_key_env}. "
            "Set it in PowerShell or .env before running."
        )

    client = OpenAI(api_key=api_key, base_url=config.vlm_base_url)
    images_dir = config.abs_path(config.images_dir)
    labels_dir = config.abs_path(config.labels_dir)

    total_boxes = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task_id = progress.add_task("VLM API pre-annotation", total=len(images))
        for idx, image_path in enumerate(images, start=1):
            from PIL import Image

            image = Image.open(image_path)
            image_w, image_h = image.size
            data_url = _read_image_as_data_url(image_path)

            prompt = (
                "Detect objects from the class list in this image and return strict JSON only. "
                "Schema: {\"detections\":[{\"class\":string,\"x1\":number,\"y1\":number,\"x2\":number,\"y2\":number,\"confidence\":number}]}. "
                "Coordinates must be pixel values in the original image size. "
                "Do not include markdown or extra text. "
                "If nothing detected, return {\"detections\":[]}. "
                f"Class list: {classes}. Image size: {image_w}x{image_h}."
            )

            payload: dict[str, Any] = {"detections": []}
            last_error = ""
            for _ in range(max(1, config.vlm_max_retries + 1)):
                try:
                    response = client.chat.completions.create(
                        model=config.vlm_model,
                        messages=[
                            {"role": "system", "content": "You are a strict JSON object detection assistant."},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": data_url}},
                                ],
                            },
                        ],
                        temperature=0,
                        timeout=max(20, int(config.vlm_timeout_sec)),
                    )
                    content = response.choices[0].message.content or "{}"
                    payload = _extract_json_object(content)
                    break
                except Exception as exc:
                    last_error = str(exc)
                    continue

            labels = _normalize_vlm_detections(payload, classes, image_w, image_h)
            if not labels and last_error:
                console.print(f"[yellow]VLM parse/retry warning:[/yellow] {image_path.name} -> {last_error}")

            label_file = label_targets.get(image_path) if label_targets else yolo_label_path(image_path, labels_dir, images_dir)
            write_yolo_labels(label_file, labels)
            total_boxes += len(labels)

            progress.update(
                task_id,
                advance=1,
                description=f"VLM API pre-annotation | last={image_path.name} | boxes={len(labels)}",
            )
            if idx % 10 == 0 or idx == len(images):
                console.print(
                    f"[cyan]Checkpoint:[/cyan] {idx}/{len(images)} images, accumulated boxes={total_boxes}"
                )

    return {
        "label_files_written": len(images),
        "detector_backend": config.detector_backend,
        "total_boxes": total_boxes,
        "model_id": config.vlm_model,
        "device": "remote_api",
    }


def run_preannotation(
    *,
    config: RunConfig,
    images: list[Path],
    classes: list[str],
    console: Console,
) -> dict:
    labels_dir = config.abs_path(config.labels_dir)
    images_dir = config.abs_path(config.images_dir)

    if config.label_output_dir_mode == "sequential_subdir":
        next_folder_idx = next_sequential_subdir_index(labels_dir)
        labels_dir = labels_dir / str(next_folder_idx)
        labels_dir.mkdir(parents=True, exist_ok=True)
        config.labels_dir = labels_dir

    label_targets: dict[Path, Path] | None = None
    if config.label_naming_mode == "sequential":
        start_idx = next_sequential_label_index(labels_dir)
        padding = max(0, int(config.label_sequence_padding))
        label_targets = {}
        for offset, image in enumerate(images):
            idx = start_idx + offset
            stem = str(idx).zfill(padding) if padding > 0 else str(idx)
            label_targets[image] = labels_dir / f"{stem}.txt"

    if config.detector_backend == "local_qwen_vl":
        report = _run_local_qwen_vl(
            config=config,
            images=images,
            classes=classes,
            console=console,
            label_targets=label_targets,
        )
        if label_targets:
            report["label_map"] = {str(k): str(v) for k, v in label_targets.items()}
        console.print(
            f"[green]Pre-annotation complete.[/green] Wrote {report['label_files_written']} label files."
        )
        return report

    if config.detector_backend == "vlm_api":
        report = _run_vlm_api(
            config=config,
            images=images,
            classes=classes,
            console=console,
            label_targets=label_targets,
        )
        if label_targets:
            report["label_map"] = {str(k): str(v) for k, v in label_targets.items()}
        console.print(
            f"[green]Pre-annotation complete.[/green] Wrote {report['label_files_written']} label files."
        )
        return report

    if config.detector_backend == "grounding_dino":
        report = _run_grounding_dino(
            config=config,
            images=images,
            classes=classes,
            console=console,
            label_targets=label_targets,
        )
        if label_targets:
            report["label_map"] = {str(k): str(v) for k, v in label_targets.items()}
        console.print(
            f"[green]Pre-annotation complete.[/green] Wrote {report['label_files_written']} label files."
        )
        return report

    written = 0
    for image in images:
        label_file = label_targets.get(image) if label_targets else yolo_label_path(image, labels_dir, images_dir)
        labels: list[BoxLabel] = []
        if config.mock_add_center_box and classes:
            labels = [
                BoxLabel(
                    class_id=0,
                    x_center=0.5,
                    y_center=0.5,
                    width=0.3,
                    height=0.3,
                    confidence=0.5,
                )
            ]
        write_yolo_labels(label_file, labels)
        written += 1

    console.print(f"[green]Pre-annotation complete.[/green] Wrote {written} label files.")
    report = {"label_files_written": written, "detector_backend": config.detector_backend, "total_boxes": 0}
    if label_targets:
        report["label_map"] = {str(k): str(v) for k, v in label_targets.items()}
    return report
