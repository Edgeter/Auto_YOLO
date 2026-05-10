from __future__ import annotations

from pathlib import Path

import yaml

from autoyolo.models import RunConfig


DEFAULT_CONFIG_FILE = "autoyolo.yaml"


def load_config(config_path: Path) -> RunConfig:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if "project_root" not in data:
        data["project_root"] = str(config_path.parent.resolve())
    return RunConfig.model_validate(data)


def save_config(config: RunConfig, config_path: Path) -> None:
    payload = config.model_dump(mode="json")
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
