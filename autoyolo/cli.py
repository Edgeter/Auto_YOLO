from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from rich.console import Console

from autoyolo.config import DEFAULT_CONFIG_FILE, load_config, save_config
from autoyolo.io_utils import ensure_dirs
from autoyolo.models import RunConfig
from autoyolo.pipeline import run_pipeline
from autoyolo.services.autotune import run_autotune
from autoyolo.services.qc import run_qc
from autoyolo.services.remote_chat import run_remote_chat
from autoyolo.services.task_profiles import (
    create_task_profile,
    list_task_profiles,
    refine_task_prompt,
    resolve_task_config_path,
)
from autoyolo.services.vision import run_vision_query
from autoyolo.services.wizard import run_wizard

app = typer.Typer(no_args_is_help=True, help="AutoYOLO: GPT-assisted auto-labeling scaffold")
config_app = typer.Typer(help="Read or update config values")
console = Console()

app.add_typer(config_app, name="config")


@app.command()
def init(
    project_root: Path = typer.Option(Path("."), "--project-root", help="Project root directory"),
    force: bool = typer.Option(False, "--force", help="Overwrite default files if they exist"),
) -> None:
    root = project_root.resolve()
    ensure_dirs(root, root / "images", root / "labels", root / "reports")

    classes_file = root / "classes.txt"
    config_file = root / DEFAULT_CONFIG_FILE

    if force or not classes_file.exists():
        classes_file.write_text("person\nhelmet\ncar\n", encoding="utf-8")

    if force or not config_file.exists():
        cfg = RunConfig(project_root=root)
        save_config(cfg, config_file)

    console.print(f"[green]Project initialized:[/green] {root}")
    console.print(f"- Config: {config_file}")
    console.print(f"- Classes: {classes_file}")


@app.command()
def wizard(
    config: Path = typer.Option(Path(DEFAULT_CONFIG_FILE), "--config", help="Config yaml path"),
) -> None:
    cfg_path = config.resolve()
    project_root = cfg_path.parent
    cfg = run_wizard(project_root, console)
    save_config(cfg, cfg_path)
    console.print(f"[green]Config saved:[/green] {cfg_path}")


@app.command()
def run(
    config: Path = typer.Option(Path(DEFAULT_CONFIG_FILE), "--config", help="Config yaml path"),
    images_dir: Path | None = typer.Option(None, "--images-dir", help="Override images directory for this run"),
    task: int | None = typer.Option(None, "--task", help="Run task config by id, e.g. 3 -> tasks/3.yaml"),
) -> None:
    load_dotenv()
    cfg_path = config.resolve()
    if task is not None:
        cfg_path = resolve_task_config_path(project_root=cfg_path.parent.resolve(), task_id=task)
    cfg = load_config(cfg_path)
    if images_dir is not None:
        cfg.images_dir = images_dir
    result = run_pipeline(cfg, console)
    console.print("[bold green]Done.[/bold green]")
    console.print(json.dumps(result, indent=2))


@app.command()
def qc(
    config: Path = typer.Option(Path(DEFAULT_CONFIG_FILE), "--config", help="Config yaml path"),
) -> None:
    cfg = load_config(config.resolve())
    images = sorted(cfg.abs_path(cfg.images_dir).rglob("*.jpg"))
    classes = [line.strip() for line in cfg.abs_path(cfg.classes_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    report_file = cfg.abs_path(cfg.reports_dir) / "qc_report_manual.json"
    report = run_qc(config=cfg, images=images, classes=classes, report_file=report_file)
    console.print(json.dumps(report, indent=2))


@app.command("vision")
def vision(
    image: Path = typer.Option(..., "--image", help="Image path for local model understanding"),
    ask: str = typer.Option(
        "Describe what this image contains for annotation planning.",
        "--ask",
        help="Natural language question/prompt to ask local vision model",
    ),
    config: Path = typer.Option(Path(DEFAULT_CONFIG_FILE), "--config", help="Config yaml path"),
) -> None:
    cfg = load_config(config.resolve())
    img_path = image.resolve()
    result = run_vision_query(config=cfg, image_path=img_path, ask=ask, console=console)
    console.print(json.dumps(result, indent=2, ensure_ascii=False))


@app.command("prompt")
def prompt(
    image: Path = typer.Argument(..., help="Image path"),
    config: Path = typer.Option(Path(DEFAULT_CONFIG_FILE), "--config", "-c", help="Config yaml path"),
) -> None:
    cfg = load_config(config.resolve())
    img_path = image.resolve()
    ask_text = (
        "You are helping build object-detection prompts. "
        "1) Briefly describe the main target(s) and scene challenges. "
        "2) Output 3 high-quality English prompts for detection annotation. "
        "3) Add 3 strict output constraints to reduce malformed boxes (no full-image box, tight box, [] if uncertain). "
        "Return plain text."
    )
    result = run_vision_query(config=cfg, image_path=img_path, ask=ask_text, console=console)
    console.print(result["answer"])


@app.command("task-create")
def task_create(
    config: Path = typer.Option(Path(DEFAULT_CONFIG_FILE), "--config", help="Base config yaml path"),
) -> None:
    load_dotenv()
    cfg_path = config.resolve()
    cfg = load_config(cfg_path)
    create_task_profile(base_config=cfg, base_config_path=cfg_path, console=console)


@app.command("task-list")
def task_list(
    config: Path = typer.Option(Path(DEFAULT_CONFIG_FILE), "--config", help="Base config yaml path"),
) -> None:
    cfg_path = config.resolve()
    rows = list_task_profiles(project_root=cfg_path.parent.resolve())
    if not rows:
        console.print("[yellow]No task profiles found.[/yellow]")
        return
    console.print(json.dumps(rows, indent=2, ensure_ascii=False))


@app.command("task-refine")
def task_refine(
    task: int = typer.Option(..., "--task", help="Task id to refine, e.g. 3 -> tasks/3.yaml"),
    config: Path = typer.Option(Path(DEFAULT_CONFIG_FILE), "--config", help="Base config yaml path"),
) -> None:
    load_dotenv()
    cfg_path = config.resolve()
    task_path = resolve_task_config_path(project_root=cfg_path.parent.resolve(), task_id=task)
    refine_task_prompt(task_config_path=task_path, console=console)


@app.command("chat-test")
def chat_test(
    message: str = typer.Option("Hello! Please reply with one short sentence.", "--message", "-m", help="Test message"),
    config: Path = typer.Option(Path(DEFAULT_CONFIG_FILE), "--config", help="Config yaml path"),
) -> None:
    load_dotenv()
    cfg = load_config(config.resolve())
    result = run_remote_chat(config=cfg, message=message)
    console.print(json.dumps(result, indent=2, ensure_ascii=False))


@app.command()
def autotune(
    profile: str = typer.Option(
        ...,
        "--profile",
        help="Dataset profile in natural language, used to derive convergence objective",
    ),
    max_rounds: int = typer.Option(6, "--max-rounds", min=1, max=30, help="Maximum tuning rounds"),
    target_loss: float = typer.Option(0.25, "--target-loss", min=0.0, help="Stop when objective loss is below this value"),
    probe_images: int = typer.Option(
        10,
        "--probe-images",
        min=1,
        help="Use this many images for fast per-round probe before optional full evaluation",
    ),
    full_eval_trigger_loss: float = typer.Option(
        0.8,
        "--full-eval-trigger-loss",
        min=0.0,
        help="Run full dataset evaluation when probe loss is below this value",
    ),
    config: Path = typer.Option(Path(DEFAULT_CONFIG_FILE), "--config", help="Config yaml path"),
) -> None:
    load_dotenv()
    cfg_path = config.resolve()
    cfg = load_config(cfg_path)
    result = run_autotune(
        config=cfg,
        config_path=cfg_path,
        profile=profile,
        max_rounds=max_rounds,
        target_loss=target_loss,
        probe_images=probe_images,
        full_eval_trigger_loss=full_eval_trigger_loss,
        console=console,
    )
    console.print("[bold green]Autotune finished.[/bold green]")
    console.print(json.dumps(result, indent=2))


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Config field name, e.g. detector_backend"),
    value: str = typer.Argument(..., help="Value, e.g. grounding_dino"),
    config: Path = typer.Option(Path(DEFAULT_CONFIG_FILE), "--config", help="Config yaml path"),
) -> None:
    cfg_path = config.resolve()
    cfg = load_config(cfg_path)
    payload = cfg.model_dump(mode="python")

    if key not in payload:
        available = ", ".join(sorted(payload.keys()))
        raise typer.BadParameter(f"Unknown key '{key}'. Available keys: {available}")

    parsed_value = yaml.safe_load(value)
    payload[key] = parsed_value
    updated = RunConfig.model_validate(payload)
    save_config(updated, cfg_path)
    console.print(f"[green]Updated[/green] {key} = {getattr(updated, key)}")


@config_app.command("show")
def config_show(
    config: Path = typer.Option(Path(DEFAULT_CONFIG_FILE), "--config", help="Config yaml path"),
) -> None:
    cfg = load_config(config.resolve())
    console.print(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))


if __name__ == "__main__":
    app()
