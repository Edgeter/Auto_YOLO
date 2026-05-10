from __future__ import annotations

from abc import ABC, abstractmethod


class LLMAdapter(ABC):
    @abstractmethod
    def build_annotation_plan(self, classes: list[str], user_prompt: str) -> dict:
        raise NotImplementedError
