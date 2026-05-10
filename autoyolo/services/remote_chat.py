from __future__ import annotations

import os

from openai import OpenAI

from autoyolo.models import RunConfig


def run_remote_chat(*, config: RunConfig, message: str) -> dict:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is empty. Please set it before remote chat test.")

    client = OpenAI(api_key=api_key, base_url=config.openai_base_url)
    response = client.chat.completions.create(
        model=config.llm_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message},
        ],
        stream=False,
    )
    content = response.choices[0].message.content or ""
    return {
        "base_url": config.openai_base_url,
        "model": config.llm_model,
        "reply": content,
    }
