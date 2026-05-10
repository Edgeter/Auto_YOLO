from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from datetime import datetime

import typer
import yaml
from rich.console import Console

from autoyolo.adapters import MockLLMAdapter, OpenAILLMAdapter, OpenCodeLLMAdapter
from autoyolo.models import RunConfig
from autoyolo.services.vision import run_vision_query


def _normalize_input_path(raw: str) -> Path:
    txt = raw.strip()
    if (txt.startswith('"') and txt.endswith('"')) or (txt.startswith("'") and txt.endswith("'")):
        txt = txt[1:-1].strip()
    return Path(txt).expanduser().resolve()


def _next_task_id(tasks_dir: Path) -> int:
    max_id = 0
    if not tasks_dir.exists():
        return 1
    for p in tasks_dir.glob("*.yaml"):
        if p.stem.isdigit():
            max_id = max(max_id, int(p.stem))
    return max_id + 1


def _extract_json_object(text: str) -> dict:
    start = text.find("{")
    if start < 0:
        return {}
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except Exception:
                    return {}
    return {}


def _local_generate_prompt_bundle(
    config: RunConfig,
    classes: list[str],
    images_dir: Path,
    user_prompt: str,
    console: Console,
) -> tuple[str, str]:
    sample_images = [p for p in sorted(images_dir.rglob("*")) if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    if not sample_images:
        raise RuntimeError(f"No sample image found in images dir: {images_dir}")

    ask = (
        "You are helping object-detection prompt engineering. "
        "Given user intent and class list, generate a high-quality English detection prompt and a Chinese explanation. "
        "Return strict JSON only: {\"optimized_prompt_en\": string, \"optimized_prompt_zh\": string}. "
        "English prompt should be concise and directly usable for model inference, and must include constraints: "
        "ignore tiny targets under 12px unless critical; prefer single class per object even with overlap; "
        "mark uncertain objects for human review. "
        f"Class list: {classes}. User intent: {user_prompt}"
    )
    result = run_vision_query(config=config, image_path=sample_images[0], ask=ask, console=console)
    obj = _extract_json_object(str(result.get("answer", "")))
    en = str(obj.get("optimized_prompt_en", "")).strip()
    zh = str(obj.get("optimized_prompt_zh", "")).strip()
    if en and zh:
        return en, zh

    class_text = ", ".join(classes)
    en = (
        "Detect objects from the given class list in this image set. "
        f"Class list: [{class_text}]. "
        f"Task intent: {user_prompt.strip()}. "
        "Output must be strict and detection-focused: ignore tiny targets under 12px unless critical, "
        "prefer a single class per object even in overlap, and mark uncertain objects for human review."
    ).strip()
    zh = (
        "从给定类别列表中检测目标。"
        f"任务意图：{user_prompt.strip()}。"
        "要求：除非关键否则忽略小于12像素的小目标；即使重叠也尽量每个对象只给一个类别；"
        "不确定目标标记为人工复核。"
    )
    return en, zh


def _parse_prompt_bundle_text(text: str) -> tuple[str, str] | None:
    obj = _extract_json_object(text)
    en = str(obj.get("optimized_prompt_en", "")).strip()
    zh = str(obj.get("optimized_prompt_zh", "")).strip()
    if en and zh:
        return en, zh

    m_en = re.search(r"optimized_prompt_en\s*[:=]\s*[\"'](.+?)[\"']", text, flags=re.DOTALL)
    m_zh = re.search(r"optimized_prompt_zh\s*[:=]\s*[\"'](.+?)[\"']", text, flags=re.DOTALL)
    if m_en and m_zh:
        return m_en.group(1).strip(), m_zh.group(1).strip()

    m_en2 = re.search(r"^EN\s*:\s*(.+)$", text, flags=re.MULTILINE)
    m_zh2 = re.search(r"^ZH\s*:\s*(.+)$", text, flags=re.MULTILINE)
    if m_en2 and m_zh2:
        return m_en2.group(1).strip(), m_zh2.group(1).strip()

    return None


def _remote_generate_prompt_bundle(
    config: RunConfig,
    classes: list[str],
    user_prompt: str,
    console: Console,
) -> tuple[str, str] | None:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    adapter = OpenAILLMAdapter(model=config.llm_model, base_url=config.openai_base_url)
    prompt = (
            "You are a prompt engineer for object detection pre-annotation. "
            "Given class list and user intent, output strict JSON only with keys: "
            "optimized_prompt_en, optimized_prompt_zh. "
            "Requirements for optimized_prompt_en: concise, directly usable for model inference, "
            "must include constraints: ignore tiny targets under 12px unless critical; "
            "prefer single class per object even with overlap; mark uncertain objects for human review. "
            f"Class list: {classes}. User intent: {user_prompt}"
        )
    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            out = adapter.client.chat.completions.create(
                model=config.llm_model,
                messages=[
                    {"role": "system", "content": "You output JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                stream=False,
                timeout=45,
            )
            content = out.choices[0].message.content or "{}"
            parsed = _parse_prompt_bundle_text(content)
            if parsed is not None:
                return parsed
            snippet = content.strip().replace("\n", " ")
            if len(snippet) > 220:
                snippet = snippet[:220] + "..."
            console.print(f"[yellow]Remote prompt parse warning:[/yellow] {snippet}")
            return None
        except Exception as exc:
            last_exc = exc
            if attempt < 3:
                console.print(f"[yellow]Remote prompt API retry {attempt}/2...[/yellow]")
                time.sleep(1.2)
    if last_exc is not None:
        console.print(f"[yellow]Remote prompt API error:[/yellow] {type(last_exc).__name__}: {last_exc}")
    return None

    return None


def _generate_prompt_bundle(
    config: RunConfig,
    classes: list[str],
    images_dir: Path,
    user_prompt: str,
    console: Console,
) -> tuple[str, str]:
    remote = _remote_generate_prompt_bundle(config, classes, user_prompt, console)
    if remote is not None:
        console.print("[green]Prompt bundle generated via remote API.[/green]")
        return remote
    console.print("[yellow]Remote prompt API unavailable, fallback to deterministic template prompt.[/yellow]")
    class_text = ", ".join(classes)
    en = (
        "Detect objects from the given class list in this image set. "
        f"Class list: [{class_text}]. "
        f"Task intent: {user_prompt.strip()}. "
        "Output must be strict and detection-focused: ignore tiny targets under 12px unless critical, "
        "prefer a single class per object even in overlap, and mark uncertain objects for human review."
    ).strip()
    zh = (
        "从给定类别列表中检测目标。"
        f"任务意图：{user_prompt.strip()}。"
        "要求：除非关键否则忽略小于12像素的小目标；即使重叠也尽量每个对象只给一个类别；"
        "不确定目标标记为人工复核。"
    )
    return en, zh


def _optimize_prompt(config: RunConfig, classes: list[str], images_dir: Path, user_prompt: str) -> str:
    prompt = (
        "Rewrite the user instruction into a highly targeted object-detection prompt for YOLO pre-annotation. "
        "Return strict JSON: {\"optimized_prompt\": string}. "
        "Constraints: emphasize tight boxes, avoid full-image boxes, output [] if uncertain, and class names must match class list exactly.\n"
        f"Images dir: {images_dir}\n"
        f"Class list: {classes}\n"
        f"User instruction: {user_prompt}"
    )

    if config.llm_provider == "openai":
        try:
            adapter = OpenAILLMAdapter(model=config.llm_model, base_url=config.openai_base_url)
            out = adapter.client.chat.completions.create(
                model=config.llm_model,
                messages=[
                    {"role": "system", "content": "You output JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            content = out.choices[0].message.content or "{}"
            return str(json.loads(content).get("optimized_prompt", user_prompt)).strip() or user_prompt
        except Exception:
            pass

    if config.llm_provider in {"opencode", "openai"}:
        try:
            adapter = OpenCodeLLMAdapter(
                model=config.llm_model,
                executable=config.opencode_executable,
                runner_args=config.opencode_runner_args,
                timeout_sec=config.opencode_timeout_sec,
            )
            out = adapter.build_annotation_plan(classes=["optimized_prompt"], user_prompt=prompt)
            if isinstance(out, dict):
                val = out.get("optimized_prompt")
                if isinstance(val, str) and val.strip():
                    return val.strip()
        except Exception:
            pass

    class_text = ", ".join(classes)
    return (
        "Detect objects from the given class list in this image set. "
        f"Class list: [{class_text}]. "
        f"Task intent: {user_prompt.strip()}. "
        "Output must be strict and detection-focused: ignore tiny targets under 12px unless critical, "
        "prefer a single class per object even in overlap, and mark uncertain objects for human review."
    ).strip()


def _translate_prompt_to_zh(config: RunConfig, prompt_en: str) -> str:
    prompt = (
        "Translate the following English detection prompt into concise, accurate Chinese for human review. "
        "Keep technical constraints intact. Return strict JSON: {\"zh\": string}.\n"
        f"Prompt: {prompt_en}"
    )

    if config.llm_provider == "openai":
        try:
            adapter = OpenAILLMAdapter(model=config.llm_model, base_url=config.openai_base_url)
            out = adapter.client.chat.completions.create(
                model=config.llm_model,
                messages=[
                    {"role": "system", "content": "You output JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            content = out.choices[0].message.content or "{}"
            return str(json.loads(content).get("zh", "")).strip() or prompt_en
        except Exception:
            pass

    if config.llm_provider in {"opencode", "openai"}:
        try:
            adapter = OpenCodeLLMAdapter(
                model=config.llm_model,
                executable=config.opencode_executable,
                runner_args=config.opencode_runner_args,
                timeout_sec=config.opencode_timeout_sec,
            )
            out = adapter.build_annotation_plan(classes=["zh"], user_prompt=prompt)
            if isinstance(out, dict):
                val = out.get("zh")
                if isinstance(val, str) and val.strip():
                    return val.strip()
        except Exception:
            pass

    return (
        "从给定类别列表中检测目标。"
        "任务意图：" + prompt_en + "。"
        "要求：除非关键否则忽略小于12像素的小目标；即使重叠也尽量每个对象只给一个类别；"
        "不确定目标标记为人工复核。"
    )


def _clean_multiline_instruction(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return " ".join(lines).strip()


def create_task_profile(*, base_config: RunConfig, base_config_path: Path, console: Console) -> Path:
    project_root = base_config.abs_path(base_config.project_root)
    tasks_dir = project_root / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    images_dir = _normalize_input_path(typer.prompt("Images directory path"))
    classes_file = _normalize_input_path(typer.prompt("classes file path"))
    if not images_dir.exists() or not images_dir.is_dir():
        raise RuntimeError(f"Images directory not found: {images_dir}")
    if not classes_file.exists() or not classes_file.is_file():
        raise RuntimeError(f"classes file not found: {classes_file}")

    user_prompt = typer.prompt("Natural language instruction")

    classes = [
        line.strip()
        for line in classes_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not classes:
        raise RuntimeError(
            f"classes_file is empty: {classes_file}. Please add at least one class before refining prompt."
        )

    optimized, optimized_zh = _generate_prompt_bundle(base_config, classes, images_dir, user_prompt, console)
    while True:
        console.print("\n[bold cyan]Optimized prompt (EN, used by model)[/bold cyan]")
        console.print(optimized)
        console.print("\n[bold cyan]Prompt explanation (ZH, for review)[/bold cyan]")
        console.print(optimized_zh)
        ok = typer.confirm("Use this prompt?", default=True)
        if ok:
            break
        user_prompt = typer.prompt("Update your instruction and regenerate")
        optimized, optimized_zh = _generate_prompt_bundle(base_config, classes, images_dir, user_prompt, console)

    payload = base_config.model_dump(mode="json")
    payload["images_dir"] = str(images_dir)
    payload["classes_file"] = str(classes_file)
    payload["gpt_prompt"] = optimized
    payload["gpt_prompt_zh"] = optimized_zh
    payload["detector_backend"] = "local_qwen_vl"

    task_id = _next_task_id(tasks_dir)
    task_path = tasks_dir / f"{task_id}.yaml"
    task_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    console.print(f"[green]Task saved:[/green] {task_path}")
    return task_path


def resolve_task_config_path(*, project_root: Path, task_id: int) -> Path:
    return (project_root / "tasks" / f"{task_id}.yaml").resolve()


def list_task_profiles(*, project_root: Path) -> list[dict]:
    tasks_dir = (project_root / "tasks").resolve()
    if not tasks_dir.exists():
        return []

    rows: list[dict] = []
    for p in sorted(tasks_dir.glob("*.yaml"), key=lambda x: int(x.stem) if x.stem.isdigit() else 10**9):
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        stat = p.stat()
        created = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        rows.append(
            {
                "task_id": p.stem,
                "file": str(p),
                "images_dir": str(data.get("images_dir", "")),
                "classes_file": str(data.get("classes_file", "")),
                "created_at": created,
            }
        )
    return rows


def refine_task_prompt(*, task_config_path: Path, console: Console) -> Path:
    if not task_config_path.exists():
        raise RuntimeError(f"Task config not found: {task_config_path}")

    payload = yaml.safe_load(task_config_path.read_text(encoding="utf-8")) or {}
    cfg = RunConfig.model_validate(payload)

    images_dir = Path(str(payload.get("images_dir", cfg.images_dir)))
    classes_file = _normalize_input_path(str(payload.get("classes_file", cfg.classes_file)))
    image_files = [
        p
        for p in sorted(images_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ] if images_dir.exists() and images_dir.is_dir() else []
    classes_preview: list[str] = []
    if classes_file.exists() and classes_file.is_file():
        classes_preview = [
            line.strip()
            for line in classes_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    console.print("\n[bold cyan]Task context before refine[/bold cyan]")
    console.print(f"- task file: {task_config_path}")
    console.print(f"- images_dir: {images_dir}")
    console.print(f"- images count: {len(image_files)}")
    console.print(f"- classes_file: {classes_file}")
    console.print(f"- classes: {classes_preview if classes_preview else '<empty>'}")

    if not typer.confirm("Use current images/classes settings for this refinement?", default=True):
        images_dir = _normalize_input_path(typer.prompt("Images directory path"))
        classes_file = _normalize_input_path(typer.prompt("classes file path"))
        if not classes_file.exists() or not classes_file.is_file():
            raise RuntimeError(f"classes file not found: {classes_file}")
        payload["images_dir"] = str(images_dir)
        payload["classes_file"] = str(classes_file)

    old_prompt_en = str(payload.get("gpt_prompt", cfg.gpt_prompt or "")).strip()
    old_prompt_zh = str(payload.get("gpt_prompt_zh", "")).strip()

    console.print("\n[bold cyan]Current prompt (EN, used by model)[/bold cyan]")
    console.print(old_prompt_en or "<empty>")
    if old_prompt_zh:
        console.print("\n[bold cyan]Current prompt (ZH, for review)[/bold cyan]")
        console.print(old_prompt_zh)

    console.print("\n[cyan]Enter refinement instruction in editor (multiline supported). Save and close to continue.[/cyan]")
    refine_instruction = typer.edit(
        "# Write your refinement instruction below.\n"
        "# Example: Focus on tight box around the whole box, avoid text-only boxes.\n"
    )
    if not refine_instruction or not refine_instruction.strip():
        refine_instruction = typer.prompt("Natural language refinement instruction (single line)")
    refine_instruction = _clean_multiline_instruction(refine_instruction)

    if not classes_file.exists() or not classes_file.is_file():
        console.print(f"[yellow]classes file not found in task:[/yellow] {classes_file}")
        classes_file = _normalize_input_path(typer.prompt("Please input a valid classes file path"))
        if not classes_file.exists() or not classes_file.is_file():
            raise RuntimeError(f"classes file not found: {classes_file}")
        payload["classes_file"] = str(classes_file)
    classes = [
        line.strip()
        for line in classes_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    new_prompt_en, new_prompt_zh = _generate_prompt_bundle(cfg, classes, images_dir, refine_instruction, console)

    while True:
        console.print("\n[bold cyan]===== Refined Prompt (EN, model uses this) =====[/bold cyan]")
        console.print(new_prompt_en)
        console.print("\n[bold cyan]===== Refined Prompt Explanation (ZH, for review) =====[/bold cyan]")
        console.print(new_prompt_zh)
        ok = typer.confirm("Use this refined prompt?", default=True)
        if ok:
            break
        console.print("\n[cyan]Update refinement instruction in editor (multiline supported).[/cyan]")
        refine_instruction = typer.edit(
            "# Update your refinement instruction below.\n"
            f"# Previous input:\n# {refine_instruction}\n"
        )
        if not refine_instruction or not refine_instruction.strip():
            refine_instruction = typer.prompt("Update refinement instruction and regenerate")
        refine_instruction = _clean_multiline_instruction(refine_instruction)
        new_prompt_en, new_prompt_zh = _generate_prompt_bundle(cfg, classes, images_dir, refine_instruction, console)

    payload["gpt_prompt"] = new_prompt_en
    payload["gpt_prompt_zh"] = new_prompt_zh
    task_config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    console.print(f"[green]Task prompt updated:[/green] {task_config_path}")
    return task_config_path
