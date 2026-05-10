from __future__ import annotations

import json
import subprocess

from autoyolo.adapters.llm_base import LLMAdapter


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in output")

    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    raise ValueError("Unclosed JSON object in output")


class OpenCodeLLMAdapter(LLMAdapter):
    def __init__(
        self,
        *,
        model: str,
        executable: str,
        runner_args: str,
        timeout_sec: int,
    ) -> None:
        self.model = model
        self.executable = executable
        self.runner_args = runner_args
        self.timeout_sec = timeout_sec

    def build_annotation_plan(self, classes: list[str], user_prompt: str) -> dict:
        prompt = (
            "You are preparing an object detection annotation spec. "
            "Return strict JSON with keys: rules(list), class_aliases(object), review_priority(list).\n"
            f"Classes: {classes}\n"
            f"User prompt: {user_prompt}"
        )
        cmd = [self.executable, *self.runner_args.split(), "--model", self.model, prompt]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=self.timeout_sec,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"OpenCode failed: {result.stderr.strip() or result.stdout.strip()}")

        raw = result.stdout.strip()
        parsed = json.loads(_extract_json_object(raw))
        return parsed
