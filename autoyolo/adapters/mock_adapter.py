from __future__ import annotations

from autoyolo.adapters.llm_base import LLMAdapter


class MockLLMAdapter(LLMAdapter):
    def build_annotation_plan(self, classes: list[str], user_prompt: str) -> dict:
        return {
            "class_count": len(classes),
            "classes": classes,
            "rules": [
                "Ignore tiny targets under 12px unless critical.",
                "Prefer single class per object even in overlap.",
                "Mark uncertain objects for human review.",
            ],
            "user_prompt": user_prompt,
        }
