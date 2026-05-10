from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class BoxLabel(BaseModel):
    class_id: int
    x_center: float = Field(ge=0.0, le=1.0)
    y_center: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class RunConfig(BaseModel):
    profile_name: str = "default"
    project_root: Path = Path(".")
    images_dir: Path = Path("images")
    labels_dir: Path = Path("labels")
    reports_dir: Path = Path("reports")
    classes_file: Path = Path("classes.txt")
    llm_provider: Literal["mock", "openai", "opencode"] = "mock"
    llm_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.hanbbq.top/v1"
    opencode_executable: str = "npx"
    opencode_runner_args: str = "opencode run"
    opencode_timeout_sec: int = 180
    opencode_fallback_on_openai_error: bool = True
    gpt_prompt: str = ""
    detector_backend: Literal["mock", "grounding_dino", "vlm_api", "ollama_vlm", "local_qwen_vl"] = "mock"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    nms_iou_threshold: float = 0.6
    max_detections_per_class: int = 30
    min_box_area_norm: float = 0.0003
    grounding_dino_model_id: str = "IDEA-Research/grounding-dino-base"
    inference_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    vlm_base_url: str = "https://api.hanbbq.top/v1"
    vlm_model: str = "cch/gpt-5.4"
    vlm_api_key_env: str = "VLM_API_KEY"
    vlm_max_retries: int = 2
    vlm_timeout_sec: int = 90
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5vl:3b"
    ollama_max_retries: int = 2
    ollama_timeout_sec: int = 90
    local_qwen_model_path: str = "D:/AI_Models/ModelScope/models/Qwen/Qwen2___5-VL-3B-Instruct"
    local_qwen_device: str = "cuda"
    local_qwen_max_image_side: int = 640
    local_qwen_max_new_tokens: int = 96
    local_qwen_query_classes: str = ""
    label_naming_mode: Literal["image_name", "sequential"] = "image_name"
    label_sequence_padding: int = 0
    label_output_dir_mode: Literal["base", "sequential_subdir"] = "base"
    mock_add_center_box: bool = False

    def abs_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()
