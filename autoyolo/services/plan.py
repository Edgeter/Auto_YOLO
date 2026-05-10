from __future__ import annotations

from autoyolo.adapters import MockLLMAdapter, OpenAILLMAdapter, OpenCodeLLMAdapter
from autoyolo.models import RunConfig


def build_plan(config: RunConfig, classes: list[str]) -> dict:
    if config.llm_provider == "opencode":
        adapter = OpenCodeLLMAdapter(
            model=config.llm_model,
            executable=config.opencode_executable,
            runner_args=config.opencode_runner_args,
            timeout_sec=config.opencode_timeout_sec,
        )
        plan = adapter.build_annotation_plan(classes=classes, user_prompt=config.gpt_prompt)
        plan["plan_provider"] = "opencode"
        return plan

    if config.llm_provider == "openai":
        try:
            adapter = OpenAILLMAdapter(model=config.llm_model, base_url=config.openai_base_url)
            plan = adapter.build_annotation_plan(classes=classes, user_prompt=config.gpt_prompt)
            plan["plan_provider"] = "openai"
            return plan
        except Exception as exc:
            if config.opencode_fallback_on_openai_error:
                try:
                    adapter = OpenCodeLLMAdapter(
                        model=config.llm_model,
                        executable=config.opencode_executable,
                        runner_args=config.opencode_runner_args,
                        timeout_sec=config.opencode_timeout_sec,
                    )
                    fallback = adapter.build_annotation_plan(classes=classes, user_prompt=config.gpt_prompt)
                    fallback["plan_provider"] = "opencode_fallback"
                    fallback["plan_warning"] = f"OpenAI plan failed: {exc}"
                    return fallback
                except Exception as opencode_exc:
                    fallback = MockLLMAdapter().build_annotation_plan(classes=classes, user_prompt=config.gpt_prompt)
                    fallback["plan_provider"] = "mock_fallback"
                    fallback["plan_warning"] = (
                        f"OpenAI plan failed: {exc}; OpenCode fallback failed: {opencode_exc}"
                    )
                    return fallback
            fallback = MockLLMAdapter().build_annotation_plan(classes=classes, user_prompt=config.gpt_prompt)
            fallback["plan_provider"] = "mock_fallback"
            fallback["plan_warning"] = f"OpenAI plan failed: {exc}"
            return fallback

    adapter = MockLLMAdapter()
    plan = adapter.build_annotation_plan(classes=classes, user_prompt=config.gpt_prompt)
    plan["plan_provider"] = "mock"
    return plan
