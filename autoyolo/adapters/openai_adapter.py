from __future__ import annotations

import json
import os

from openai import OpenAI

from autoyolo.adapters.llm_base import LLMAdapter


class OpenAILLMAdapter(LLMAdapter):
    def __init__(self, model: str, base_url: str | None = None) -> None:
        effective_base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(base_url=effective_base_url)
        self.model = model

    def build_annotation_plan(self, classes: list[str], user_prompt: str) -> dict:
        prompt = (
            "You are preparing an object detection annotation spec. "
            "Return strict JSON with keys: rules(list), class_aliases(object), review_priority(list).\n"
            f"Classes: {classes}\n"
            f"User prompt: {user_prompt}"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You output JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)
