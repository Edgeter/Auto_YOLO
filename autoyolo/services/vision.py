from __future__ import annotations

import os
import tempfile
from pathlib import Path

from PIL import Image
from rich.console import Console

from autoyolo.models import RunConfig


_VISION_QWEN_CACHE: dict[tuple[str, str], tuple[object, object, object]] = {}


def _load_qwen_runtime(model_path: str, device: str) -> tuple[object, object, object]:
    import torch

    effective_device = device
    if device.startswith("cuda") and not torch.cuda.is_available():
        effective_device = "cpu"

    cache_key = (model_path, effective_device)
    if cache_key in _VISION_QWEN_CACHE:
        return _VISION_QWEN_CACHE[cache_key]

    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=effective_device,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    _VISION_QWEN_CACHE[cache_key] = (model, processor, process_vision_info)
    return model, processor, process_vision_info


def _prepare_image(image_path: Path, max_side: int) -> tuple[Path, tuple[int, int], tuple[int, int]]:
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    if max_side <= 0 or max(orig_w, orig_h) <= max_side:
        return image_path, (orig_w, orig_h), (orig_w, orig_h)

    ratio = max_side / float(max(orig_w, orig_h))
    new_w = max(1, int(orig_w * ratio))
    new_h = max(1, int(orig_h * ratio))
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    fd, tmp_name = tempfile.mkstemp(suffix=".jpg", prefix="autoyolo_vision_")
    os.close(fd)
    tmp = Path(tmp_name)
    resized.save(tmp, format="JPEG", quality=90)
    return tmp, (orig_w, orig_h), (new_w, new_h)


def run_vision_query(*, config: RunConfig, image_path: Path, ask: str, console: Console) -> dict:
    if not image_path.exists() or not image_path.is_file():
        raise RuntimeError(f"Image not found: {image_path}")

    runtime_device = config.local_qwen_device
    try:
        import torch

        if runtime_device.startswith("cuda") and not torch.cuda.is_available():
            runtime_device = "cpu"
            console.print("[yellow]CUDA unavailable for vision query, fallback to CPU.[/yellow]")

        model, processor, process_vision_info = _load_qwen_runtime(
            config.local_qwen_model_path,
            runtime_device,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load local Qwen-VL runtime. Ensure torch/transformers/qwen-vl-utils are installed."
        ) from exc

    run_image = image_path
    tmp_path: Path | None = None
    try:
        run_image, orig_size, run_size = _prepare_image(image_path, int(config.local_qwen_max_image_side))
        if run_image != image_path:
            tmp_path = run_image

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(run_image)},
                    {"type": "text", "text": ask},
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
        ).to(runtime_device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max(32, int(config.local_qwen_max_new_tokens)),
                do_sample=False,
            )

        generated_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        answer = processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        console.print("[green]Vision query complete.[/green]")
        return {
            "image": str(image_path),
            "orig_size": {"width": orig_size[0], "height": orig_size[1]},
            "run_size": {"width": run_size[0], "height": run_size[1]},
            "device": runtime_device,
            "answer": answer,
        }
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
