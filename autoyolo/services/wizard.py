from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from autoyolo.models import RunConfig


def run_wizard(project_root: Path, console: Console) -> RunConfig:
    console.print("[bold]AutoYOLO setup wizard[/bold]")

    images_dir = Path(typer.prompt("Images directory", default=str(project_root / "images")))
    labels_dir = Path(typer.prompt("Labels output directory", default=str(project_root / "labels")))
    reports_dir = Path(typer.prompt("Reports directory", default=str(project_root / "reports")))
    classes_file = Path(typer.prompt("classes.txt path", default=str(project_root / "classes.txt")))

    llm_provider = typer.prompt("LLM provider (mock/openai/opencode)", default="mock").strip().lower()
    if llm_provider not in {"mock", "openai", "opencode"}:
        llm_provider = "mock"

    llm_model = typer.prompt("LLM model", default="deepseek-v4-pro")
    openai_base_url = "https://api.deepseek.com"
    if llm_provider == "openai":
        openai_base_url = typer.prompt(
            "OpenAI-compatible base URL",
            default="https://api.deepseek.com",
        )

    opencode_executable = "npx"
    opencode_runner_args = "opencode run"
    if llm_provider == "opencode":
        opencode_executable = typer.prompt("OpenCode executable", default="npx")
        opencode_runner_args = typer.prompt("OpenCode args", default="opencode run")
    gpt_prompt = typer.prompt(
        "Extra annotation prompt",
        default="Focus on clear visible objects only. Ignore uncertain tiny objects.",
    )

    detector_backend = typer.prompt("Detector backend (mock/grounding_dino/vlm_api/local_qwen_vl)", default="local_qwen_vl").strip().lower()
    if detector_backend not in {"mock", "grounding_dino", "vlm_api", "local_qwen_vl"}:
        detector_backend = "mock"

    grounding_dino_model_id = "IDEA-Research/grounding-dino-base"
    inference_device = "auto"
    if detector_backend == "grounding_dino":
        grounding_dino_model_id = typer.prompt(
            "GroundingDINO model id",
            default="IDEA-Research/grounding-dino-base",
        )
        inference_device = typer.prompt(
            "Inference device (auto/cpu/cuda/mps)",
            default="auto",
        ).strip().lower()
        if inference_device not in {"auto", "cpu", "cuda", "mps"}:
            inference_device = "auto"

    vlm_base_url = "https://api.hanbbq.top/v1"
    vlm_model = "cch/gpt-5.4"
    vlm_api_key_env = "VLM_API_KEY"
    if detector_backend == "vlm_api":
        vlm_base_url = typer.prompt("VLM base URL", default="https://api.hanbbq.top/v1")
        vlm_model = typer.prompt("VLM model", default="cch/gpt-5.4")
        vlm_api_key_env = typer.prompt("VLM key env var", default="VLM_API_KEY")

    local_qwen_model_path = "D:/AI_Models/ModelScope/models/Qwen/Qwen2___5-VL-3B-Instruct"
    local_qwen_device = "cuda"
    local_qwen_max_image_side = 640
    local_qwen_query_classes = ""
    if detector_backend == "local_qwen_vl":
        local_qwen_model_path = typer.prompt(
            "Local Qwen-VL model path",
            default="D:/AI_Models/ModelScope/models/Qwen/Qwen2___5-VL-3B-Instruct",
        )
        local_qwen_device = typer.prompt("Local Qwen-VL device", default="cuda").strip().lower()
        local_qwen_max_image_side = int(typer.prompt("Max image side (0=disable)", default="640"))
        local_qwen_query_classes = typer.prompt(
            "Query classes override (comma-separated, empty=all)",
            default="",
        ).strip()

    return RunConfig(
        profile_name="default",
        project_root=project_root.resolve(),
        images_dir=images_dir,
        labels_dir=labels_dir,
        reports_dir=reports_dir,
        classes_file=classes_file,
        llm_provider=llm_provider,
        llm_model=llm_model,
        openai_base_url=openai_base_url,
        opencode_executable=opencode_executable,
        opencode_runner_args=opencode_runner_args,
        gpt_prompt=gpt_prompt,
        detector_backend=detector_backend,
        grounding_dino_model_id=grounding_dino_model_id,
        inference_device=inference_device,
        vlm_base_url=vlm_base_url,
        vlm_model=vlm_model,
        vlm_api_key_env=vlm_api_key_env,
        local_qwen_model_path=local_qwen_model_path,
        local_qwen_device=local_qwen_device,
        local_qwen_max_image_side=local_qwen_max_image_side,
        local_qwen_query_classes=local_qwen_query_classes,
    )
