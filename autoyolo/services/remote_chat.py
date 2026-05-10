from __future__ import annotations

import os

from openai import OpenAI

from autoyolo.models import RunConfig


def run_remote_chat(*, config: RunConfig, message: str) -> dict:
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    api_key = deepseek_key or openai_key
    key_source = "DEEPSEEK_API_KEY" if deepseek_key else ("OPENAI_API_KEY" if openai_key else "<none>")
    if not api_key:
        raise RuntimeError(
            "API key is empty. Set OPENAI_API_KEY (or DEEPSEEK_API_KEY) in environment/.env before chat-test."
        )

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
        "key_source": key_source,
        "reply": content,
    }
