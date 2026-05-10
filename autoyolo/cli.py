from __future__ import annotations

import json
import os
import re
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.json import JSON as RichJSON
from rich.panel import Panel
from rich.table import Table

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

app = typer.Typer(no_args_is_help=True, help="AutoYOLO: GPT-assisted auto-labeling scaffold", rich_markup_mode="rich")
config_app = typer.Typer(help="Read or update config values")
console = Console()

app.add_typer(config_app, name="config")


def _banner() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]AutoYOLO[/bold cyan]\n"
            "[dim]AI-assisted pre-annotation pipeline[/dim]",
            border_style="cyan",
            padding=(0, 2),
        )
    )


def _show_paths(*, config_path: Path, cfg: RunConfig) -> None:
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Item", style="cyan", no_wrap=True)
    table.add_column("Path", style="white")
    table.add_row("Config", str(config_path))
    table.add_row("Images", str(cfg.abs_path(cfg.images_dir)))
    table.add_row("Labels", str(cfg.abs_path(cfg.labels_dir)))
    table.add_row("Reports", str(cfg.abs_path(cfg.reports_dir)))
    table.add_row("Classes", str(cfg.abs_path(cfg.classes_file)))
    console.print(table)


def _print_result(title: str, payload: dict) -> None:
    console.print(Panel.fit(f"[bold green]{title}[/bold green]", border_style="green"))
    console.print(RichJSON.from_data(payload, ensure_ascii=False, indent=2))


def _active_key_source() -> str:
    if (os.getenv("DEEPSEEK_API_KEY", "").strip()):
        return "DEEPSEEK_API_KEY"
    if (os.getenv("OPENAI_API_KEY", "").strip()):
        return "OPENAI_API_KEY"
    return "<none>"


def _print_exec_context(cfg: RunConfig, cfg_path: Path, action: str) -> None:
    table = Table(show_header=False)
    table.add_row("Action", action)
    table.add_row("Config", str(cfg_path))
    table.add_row("Provider", cfg.llm_provider)
    table.add_row("Model", cfg.llm_model)
    table.add_row("Base URL", cfg.openai_base_url)
    table.add_row("Key Source", _active_key_source())
    console.print(Panel.fit("[bold cyan]Execution Context[/bold cyan]", border_style="cyan"))
    console.print(table)


def _extract_json_object(text: str) -> dict | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except Exception:
                    return None
    return None


def _build_context_summary(state: dict[str, object]) -> str:
    last_action = state.get("last_action")
    last_task_ids = state.get("last_task_ids")
    parts: list[str] = []
    if isinstance(last_action, str) and last_action:
        parts.append(f"last_action={last_action}")
    if isinstance(last_task_ids, list) and last_task_ids:
        parts.append(f"last_task_ids={last_task_ids}")
    return ", ".join(parts)


def _direct_context_intent(user_text: str, state: dict[str, object]) -> tuple[str, dict] | None:
    lowered = user_text.lower().strip()
    if any(k in lowered for k in ["具体", "详情", "内容", "展开", "看一下这个"]):
        last_task_ids = state.get("last_task_ids", [])
        if isinstance(last_task_ids, list) and len(last_task_ids) == 1:
            return "task_show", {"task_id": int(last_task_ids[0])}
    if any(k in lowered for k in ["解释", "理解", "说明", "帮我看懂", "总结任务", "分析任务"]):
        m = re.search(r"(?:task|任务|模板|模版)\s*(\d+)", lowered)
        if m:
            return "task_explain", {"task_id": int(m.group(1))}
        last_task_ids = state.get("last_task_ids", [])
        if isinstance(last_task_ids, list) and len(last_task_ids) == 1:
            return "task_explain", {"task_id": int(last_task_ids[0])}
    if any(k in lowered for k in ["修改提示词", "改提示词", "优化提示词", "重写提示词", "refine prompt", "refine"]):
        m = re.search(r"(?:task|任务|模板|模版)\s*(\d+)", lowered)
        if m:
            return "task_refine", {"task_id": int(m.group(1))}
        last_task_ids = state.get("last_task_ids", [])
        if isinstance(last_task_ids, list) and len(last_task_ids) == 1:
            return "task_refine", {"task_id": int(last_task_ids[0])}
    if any(k in lowered for k in ["检查", "可用", "缺失", "完整", "健康", "校验", "验证"]):
        m = re.search(r"(?:task|任务|模板|模版)\s*(\d+)", lowered)
        if m:
            return "task_check", {"task_id": int(m.group(1))}
        last_task_ids = state.get("last_task_ids", [])
        if isinstance(last_task_ids, list) and len(last_task_ids) == 1:
            return "task_check", {"task_id": int(last_task_ids[0])}
    if any(k in lowered for k in ["创建任务", "新建任务", "task create", "task-create"]):
        return "task_create", {}
    if any(k in lowered for k in ["继续", "继续执行", "继续跑", "开始跑"]) and state.get("last_action") in {"task_show", "task_explain", "task_refine"}:
        return "run", {}
    return None


def _route_intent_local(user_text: str) -> tuple[str, dict]:
    t = user_text.strip().lower()
    if any(k in t for k in ["退出", "exit", "quit", "结束"]):
        return "exit", {}
    if any(k in t for k in ["配置", "config", "参数"]):
        return "show_config", {}
    if any(k in t for k in ["任务", "task", "模板", "模版", "提示词", "prompt"]):
        if any(k in t for k in ["本地", "离线", "local", "remote", "远程", "云端", "api"]):
            m_task = re.search(r"(?:task|任务|模板|模版)\s*(\d+)", t)
            task_args = {"task_id": int(m_task.group(1))} if m_task else {}
            if any(k in t for k in ["本地", "离线", "local"]):
                return "task_set_backend", {**task_args, "backend": "local_qwen_vl"}
            if any(k in t for k in ["远程", "云端", "api", "remote"]):
                return "task_set_backend", {**task_args, "backend": "vlm_api"}
        if any(k in t for k in ["检查", "可用", "缺失", "完整", "健康", "校验", "验证"]):
            m3 = re.search(r"(?:task|任务|模板|模版)\s*(\d+)", t)
            if m3:
                return "task_check", {"task_id": int(m3.group(1))}
            return "task_check", {}
        if any(k in t for k in ["修改", "优化", "重写", "refine", "改提示词"]):
            m2 = re.search(r"(?:task|任务|模板|模版)\s*(\d+)", t)
            if m2:
                return "task_refine", {"task_id": int(m2.group(1))}
            return "task_refine", {}
        m = re.search(r"(?:task|任务|模板|模版)\s*(\d+)", t)
        if m:
            return "task_show", {"task_id": int(m.group(1))}
        return "task_list", {}
    if any(k in t for k in ["聊天", "连通", "chat", "deepseek", "测试接口", "test"]):
        return "chat_test", {"message": "请回复：连接成功"}
    if any(k in t for k in ["本地", "离线", "local", "remote", "远程", "云端", "api"]):
        if any(k in t for k in ["切", "换", "改", "使用", "改成", "切到"]):
            if any(k in t for k in ["本地", "离线", "local"]):
                return "task_set_backend", {"backend": "local_qwen_vl"}
            if any(k in t for k in ["远程", "云端", "api", "remote"]):
                return "task_set_backend", {"backend": "vlm_api"}
    if any(k in t for k in ["质检", "qc", "检查标签"]):
        return "qc", {}
    if any(k in t for k in ["调参", "autotune", "优化"]):
        return "autotune", {
            "profile": "single symbol per image, prioritize one clean box",
            "max_rounds": 6,
            "target_loss": 0.25,
            "probe_images": 10,
            "full_eval_trigger_loss": 0.8,
        }
    if any(k in t for k in ["运行", "标注", "run", "开始", "执行"]):
        return "run", {}
    return "ask_clarify", {}


def _route_intent_ai(cfg: RunConfig, user_text: str, *, context_summary: str = "") -> tuple[str, dict] | None:
    try:
        prompt = (
            "你是 AutoYOLO 的命令路由助手。根据用户意图返回严格 JSON，不要多余文字。"
            "\n允许 action: run, qc, chat_test, show_config, autotune, task_list, task_show, task_refine, task_create, task_explain, task_check, task_set_backend, exit, ask_clarify"
            "\n如果 action=autotune，args 中可包含 profile,max_rounds,target_loss,probe_images,full_eval_trigger_loss。"
            "\n如果 action=chat_test，args 中可包含 message。"
            "\n如果 action=task_show 或 task_refine 或 task_explain 或 task_check 或 task_set_backend，args 中可包含 task_id（整数）。"
            "\n如果 action=task_set_backend，args 必须包含 backend，取值 local_qwen_vl 或 vlm_api。"
            f"\n上下文摘要: {context_summary or '无'}"
            f"\n用户输入: {user_text}"
            "\n输出格式: {\"action\":\"...\",\"args\":{...},\"reason\":\"...\"}"
        )
        out = run_remote_chat(config=cfg, message=prompt)
        obj = _extract_json_object(out.get("reply", ""))
        if not obj or not isinstance(obj, dict):
            return None
        action = str(obj.get("action", "ask_clarify")).strip()
        args = obj.get("args", {})
        if action not in {"run", "qc", "chat_test", "show_config", "autotune", "task_list", "task_show", "task_refine", "task_create", "task_explain", "task_check", "task_set_backend", "exit", "ask_clarify"}:
            action = "ask_clarify"
        if not isinstance(args, dict):
            args = {}
        return action, args
    except Exception:
        return None


def _execute_action(action: str, args: dict, cfg_path: Path) -> None:
    if action == "run":
        effective_cfg_path = cfg_path
        task_id = args.get("task_id")
        if isinstance(task_id, int):
            effective_cfg_path = resolve_task_config_path(project_root=cfg_path.parent.resolve(), task_id=task_id)
        cfg = load_config(effective_cfg_path)
        _show_paths(config_path=effective_cfg_path, cfg=cfg)
        result = run_pipeline(cfg, console)
        _print_result("Run complete", result)
        return
    cfg = load_config(cfg_path)
    if action == "qc":
        _show_paths(config_path=cfg_path, cfg=cfg)
        images = sorted(cfg.abs_path(cfg.images_dir).rglob("*.jpg"))
        classes = [line.strip() for line in cfg.abs_path(cfg.classes_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        report_file = cfg.abs_path(cfg.reports_dir) / "qc_report_manual.json"
        report = run_qc(config=cfg, images=images, classes=classes, report_file=report_file)
        _print_result("QC complete", report)
        return
    if action == "chat_test":
        msg = str(args.get("message", "请回复：连接成功"))
        result = run_remote_chat(config=cfg, message=msg)
        _print_result("Chat test complete", result)
        return
    if action == "show_config":
        console.print(Panel.fit("[bold green]Current configuration[/bold green]", border_style="green"))
        console.print(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
        return
    if action == "task_list":
        rows = list_task_profiles(project_root=cfg_path.parent.resolve())
        if not rows:
            console.print("[yellow]No task profiles found.[/yellow]")
            return
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Task", style="cyan")
        table.add_column("Config", style="white")
        table.add_column("Images", style="white")
        for r in rows:
            table.add_row(str(r.get("task_id", "")), str(r.get("file", "")), str(r.get("images_dir", "")))
        console.print(table)
        return
    if action == "task_show":
        task_id = int(args.get("task_id", 1))
        task_path = resolve_task_config_path(project_root=cfg_path.parent.resolve(), task_id=task_id)
        task_cfg = load_config(task_path)
        payload = task_cfg.model_dump(mode="json")
        gpt_prompt = str(payload.get("gpt_prompt", "")).strip()
        gpt_prompt_zh = str(payload.get("gpt_prompt_zh", "")).strip()
        table = Table(show_header=False)
        table.add_row("Task ID", str(task_id))
        table.add_row("Task File", str(task_path))
        table.add_row("Images", str(task_cfg.abs_path(task_cfg.images_dir)))
        table.add_row("Classes", str(task_cfg.abs_path(task_cfg.classes_file)))
        table.add_row("Backend", str(payload.get("detector_backend", "")))
        console.print(Panel.fit("[bold green]Task detail[/bold green]", border_style="green"))
        console.print(table)
        if gpt_prompt:
            console.print(Panel(gpt_prompt, title="Prompt (EN)", border_style="cyan"))
        if gpt_prompt_zh:
            console.print(Panel(gpt_prompt_zh, title="Prompt (ZH)", border_style="cyan"))
        return
    if action == "task_refine":
        task_id = int(args.get("task_id", 1))
        task_path = resolve_task_config_path(project_root=cfg_path.parent.resolve(), task_id=task_id)
        refine_task_prompt(task_config_path=task_path, console=console)
        console.print(f"[green]Task {task_id} prompt refined.[/green]")
        return
    if action == "task_create":
        create_task_profile(base_config=cfg, base_config_path=cfg_path, console=console)
        console.print("[green]Task created.[/green]")
        return
    if action == "task_explain":
        task_id = int(args.get("task_id", 1))
        task_path = resolve_task_config_path(project_root=cfg_path.parent.resolve(), task_id=task_id)
        task_cfg = load_config(task_path)
        payload = task_cfg.model_dump(mode="json")
        prompt_en = str(payload.get("gpt_prompt", "")).strip()
        prompt_zh = str(payload.get("gpt_prompt_zh", "")).strip()
        images_dir = task_cfg.abs_path(task_cfg.images_dir)
        classes_file = task_cfg.abs_path(task_cfg.classes_file)
        image_count = len(sorted(images_dir.rglob("*.jpg"))) + len(sorted(images_dir.rglob("*.png")))
        classes = []
        if classes_file.exists() and classes_file.is_file():
            classes = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]

        summary = {
            "task_id": task_id,
            "task_file": str(task_path),
            "images_dir": str(images_dir),
            "images_count": image_count,
            "classes_file": str(classes_file),
            "classes_count": len(classes),
            "classes_preview": classes[:10],
            "detector_backend": payload.get("detector_backend", ""),
            "prompt_en": prompt_en,
            "prompt_zh": prompt_zh,
        }
        _print_result("Task explanation", summary)
        return
    if action == "task_check":
        task_id = int(args.get("task_id", 1))
        task_path = resolve_task_config_path(project_root=cfg_path.parent.resolve(), task_id=task_id)
        if not task_path.exists():
            raise RuntimeError(f"Task file not found: {task_path}")

        task_cfg = load_config(task_path)
        images_dir = task_cfg.abs_path(task_cfg.images_dir)
        classes_file = task_cfg.abs_path(task_cfg.classes_file)
        labels_dir = task_cfg.abs_path(task_cfg.labels_dir)
        reports_dir = task_cfg.abs_path(task_cfg.reports_dir)

        image_files = []
        if images_dir.exists() and images_dir.is_dir():
            image_files = [
                p for p in images_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ]

        classes = []
        if classes_file.exists() and classes_file.is_file():
            classes = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]

        issues: list[str] = []
        if not images_dir.exists() or not images_dir.is_dir():
            issues.append(f"images_dir missing or not a directory: {images_dir}")
        if len(image_files) == 0:
            issues.append("no images found in images_dir")
        if not classes_file.exists() or not classes_file.is_file():
            issues.append(f"classes_file missing: {classes_file}")
        if len(classes) == 0:
            issues.append("classes_file is empty")
        if not labels_dir.exists():
            issues.append(f"labels_dir missing (will be created at runtime): {labels_dir}")
        if not reports_dir.exists():
            issues.append(f"reports_dir missing (will be created at runtime): {reports_dir}")

        payload = {
            "task_id": task_id,
            "task_file": str(task_path),
            "images_dir": str(images_dir),
            "images_count": len(image_files),
            "classes_file": str(classes_file),
            "classes_count": len(classes),
            "labels_dir": str(labels_dir),
            "reports_dir": str(reports_dir),
            "status": "ready" if len(issues) == 0 else "needs_attention",
            "issues": issues,
        }
        _print_result("Task health check", payload)
        return
    if action == "task_fix":
        task_id = int(args.get("task_id", 1))
        task_path = resolve_task_config_path(project_root=cfg_path.parent.resolve(), task_id=task_id)
        if not task_path.exists():
            raise RuntimeError(f"Task file not found: {task_path}")
        task_cfg = load_config(task_path)
        images_dir = task_cfg.abs_path(task_cfg.images_dir)
        classes_file = task_cfg.abs_path(task_cfg.classes_file)
        labels_dir = task_cfg.abs_path(task_cfg.labels_dir)
        reports_dir = task_cfg.abs_path(task_cfg.reports_dir)

        changes: list[str] = []
        had_labels = labels_dir.exists()
        had_reports = reports_dir.exists()
        ensure_dirs(labels_dir, reports_dir)
        if not had_labels and labels_dir.exists():
            changes.append(f"created labels_dir: {labels_dir}")
        if not had_reports and reports_dir.exists():
            changes.append(f"created reports_dir: {reports_dir}")

        if not classes_file.exists():
            classes_file.write_text("object\n", encoding="utf-8")
            changes.append(f"created classes_file with placeholder class: {classes_file}")

        if not images_dir.exists():
            images_dir.mkdir(parents=True, exist_ok=True)
            changes.append(f"created images_dir: {images_dir}")

        _print_result("Task auto-fix complete", {"task_id": task_id, "changes": changes or ["no changes needed"]})
        return
    if action == "task_set_backend":
        backend = str(args.get("backend", "")).strip()
        if backend not in {"local_qwen_vl", "vlm_api"}:
            raise RuntimeError(f"Unsupported backend: {backend}")
        task_id = args.get("task_id")
        target_cfg_path = cfg_path
        if isinstance(task_id, int):
            target_cfg_path = resolve_task_config_path(project_root=cfg_path.parent.resolve(), task_id=task_id)
        payload_cfg = load_config(target_cfg_path).model_dump(mode="python")
        payload_cfg["detector_backend"] = backend
        updated = RunConfig.model_validate(payload_cfg)
        save_config(updated, target_cfg_path)
        _print_result("Backend switched", {
            "config": str(target_cfg_path),
            "detector_backend": backend,
            "task_id": task_id,
        })
        return
    if action == "autotune":
        result = run_autotune(
            config=cfg,
            config_path=cfg_path,
            profile=str(args.get("profile", "single symbol per image, prioritize one clean box")),
            max_rounds=int(args.get("max_rounds", 6)),
            target_loss=float(args.get("target_loss", 0.25)),
            probe_images=int(args.get("probe_images", 10)),
            full_eval_trigger_loss=float(args.get("full_eval_trigger_loss", 0.8)),
            console=console,
        )
        _print_result("Autotune finished", result)
        return


def _effective_cfg_path_for_action(action: str, args: dict, cfg_path: Path) -> Path:
    if action in {"run", "task_show", "task_refine", "task_explain", "task_check", "task_fix", "task_set_backend"}:
        task_id = args.get("task_id")
        if isinstance(task_id, int):
            return resolve_task_config_path(project_root=cfg_path.parent.resolve(), task_id=task_id)
    return cfg_path


def _menu_header() -> None:
    console.print(Panel.fit("[bold]Interactive Menu[/bold]  (Enter number)", border_style="blue"))


def _menu_choices() -> Table:
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("No.", style="cyan", width=6)
    table.add_column("Action", style="white")
    table.add_row("1", "Run full pipeline")
    table.add_row("2", "Run QC only")
    table.add_row("3", "Chat connectivity test")
    table.add_row("4", "Show current config")
    table.add_row("5", "Autotune")
    table.add_row("0", "Exit")
    return table


@app.callback()
def _app_callback() -> None:
    """AutoYOLO command group callback."""
    _banner()


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

    console.print(Panel.fit("[bold green]Project initialized[/bold green]", border_style="green"))
    table = Table(show_header=False)
    table.add_row("Root", str(root))
    table.add_row("Config", str(config_file))
    table.add_row("Classes", str(classes_file))
    console.print(table)


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
    _show_paths(config_path=cfg_path, cfg=cfg)
    result = run_pipeline(cfg, console)
    _print_result("Run complete", result)


@app.command()
def qc(
    config: Path = typer.Option(Path(DEFAULT_CONFIG_FILE), "--config", help="Config yaml path"),
) -> None:
    cfg = load_config(config.resolve())
    _show_paths(config_path=config.resolve(), cfg=cfg)
    images = sorted(cfg.abs_path(cfg.images_dir).rglob("*.jpg"))
    classes = [line.strip() for line in cfg.abs_path(cfg.classes_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    report_file = cfg.abs_path(cfg.reports_dir) / "qc_report_manual.json"
    report = run_qc(config=cfg, images=images, classes=classes, report_file=report_file)
    _print_result("QC complete", report)


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
    _print_result("Chat test complete", result)


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
    _print_result("Autotune finished", result)


@app.command("menu")
def menu(
    config: Path = typer.Option(Path(DEFAULT_CONFIG_FILE), "--config", help="Config yaml path"),
) -> None:
    """Interactive menu for common AutoYOLO workflows."""
    load_dotenv()
    cfg_path = config.resolve()

    while True:
        _menu_header()
        console.print(_menu_choices())
        choice = typer.prompt("Select", default="1").strip()

        if choice == "0":
            console.print("[bold green]Bye.[/bold green]")
            return

        if choice == "1":
            try:
                cfg = load_config(cfg_path)
                _show_paths(config_path=cfg_path, cfg=cfg)
                result = run_pipeline(cfg, console)
                _print_result("Run complete", result)
            except Exception as exc:
                console.print(f"[bold red]Run failed:[/bold red] {exc}")
            continue

        if choice == "2":
            try:
                cfg = load_config(cfg_path)
                _show_paths(config_path=cfg_path, cfg=cfg)
                images = sorted(cfg.abs_path(cfg.images_dir).rglob("*.jpg"))
                classes = [line.strip() for line in cfg.abs_path(cfg.classes_file).read_text(encoding="utf-8").splitlines() if line.strip()]
                report_file = cfg.abs_path(cfg.reports_dir) / "qc_report_manual.json"
                report = run_qc(config=cfg, images=images, classes=classes, report_file=report_file)
                _print_result("QC complete", report)
            except Exception as exc:
                console.print(f"[bold red]QC failed:[/bold red] {exc}")
            continue

        if choice == "3":
            try:
                msg = typer.prompt("Message", default="请回复：连接成功").strip()
                cfg = load_config(cfg_path)
                result = run_remote_chat(config=cfg, message=msg)
                _print_result("Chat test complete", result)
            except Exception as exc:
                console.print(f"[bold red]Chat test failed:[/bold red] {exc}")
            continue

        if choice == "4":
            try:
                cfg = load_config(cfg_path)
                console.print(Panel.fit("[bold green]Current configuration[/bold green]", border_style="green"))
                console.print(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
            except Exception as exc:
                console.print(f"[bold red]Load config failed:[/bold red] {exc}")
            continue

        if choice == "5":
            try:
                profile = typer.prompt("Profile", default="single symbol per image, prioritize one clean box").strip()
                max_rounds = int(typer.prompt("Max rounds", default="6"))
                target_loss = float(typer.prompt("Target loss", default="0.25"))
                probe_images = int(typer.prompt("Probe images", default="10"))
                full_eval_trigger_loss = float(typer.prompt("Full-eval trigger loss", default="0.8"))
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
                _print_result("Autotune finished", result)
            except Exception as exc:
                console.print(f"[bold red]Autotune failed:[/bold red] {exc}")
            continue

        console.print("[yellow]Unknown option. Please choose 0-5.[/yellow]")


@app.command("assistant")
def assistant(
    config: Path = typer.Option(Path(DEFAULT_CONFIG_FILE), "--config", help="Config yaml path"),
) -> None:
    """Single-mode natural-language assistant for AutoYOLO actions."""
    load_dotenv()
    cfg_path = config.resolve()
    console.print(
        Panel.fit(
            "[bold cyan]AutoYOLO Assistant[/bold cyan]\n"
            "[dim]用自然语言描述需求，例如：\n- 帮我跑一轮标注\n- 先测一下 DeepSeek 连通\n- 看看当前配置\n- 做 6 轮自动调参[/dim]",
            border_style="cyan",
        )
    )
    console.print("[dim]无需切换模式，直接说需求即可。输入“退出”可结束。[/dim]")

    state: dict[str, object] = {"last_action": None, "last_task_ids": []}

    def build_plan(user_text: str, action: str, args: dict) -> list[tuple[str, dict]]:
        t = user_text.lower()
        plan: list[tuple[str, dict]] = []
        if action in {"task_refine", "task_show", "task_explain", "task_check", "task_fix", "task_list"}:
            task_id = int(args.get("task_id", state.get("last_task_ids", [1])[0] if state.get("last_task_ids") else 1))
        else:
            task_id = 1

        if any(k in t for k in ["修复", "补齐", "自动修复"]):
            plan.append(("task_check", {"task_id": task_id}))
            plan.append(("task_fix", {"task_id": task_id}))
            plan.append(("task_check", {"task_id": task_id}))
            return plan

        if action == "task_refine" and any(k in t for k in ["检查", "验证", "可用", "缺失", "完整"]):
            plan.append(("task_refine", {"task_id": task_id}))
            plan.append(("task_check", {"task_id": task_id}))
            return plan

        plan.append((action, args))
        return plan

    while True:
        user_text = typer.prompt("You").strip()
        if not user_text:
            continue

        routed = _direct_context_intent(user_text, state)
        if routed is None:
            cfg = load_config(cfg_path)
            routed = _route_intent_ai(cfg, user_text, context_summary=_build_context_summary(state))
        if routed is None:
            routed = _route_intent_local(user_text)
        action, args = routed

        if action == "exit":
            console.print("[bold green]Assistant exited.[/bold green]")
            return

        if action == "ask_clarify":
            maybe_id = re.search(r"(\d+)", user_text)
            if maybe_id and typer.confirm(f"我猜你的意思是查看 task {maybe_id.group(1)}，要直接执行吗？", default=True):
                action, args = "task_show", {"task_id": int(maybe_id.group(1))}
                console.print(f"[cyan]AI建议动作:[/cyan] [bold]{action}[/bold]")
                console.print(RichJSON.from_data(args, ensure_ascii=False, indent=2))
            else:
                console.print(
                    "[yellow]我还不够确定你的意图。可以直接说：\n"
                    "- 查看任务列表\n- 查看 task 1\n- 解释 task 1\n- 检查 task 1 是否可用\n- 修改 task 1 的提示词\n"
                    "- 把 task 1 切到本地模式 / 切到远程API\n"
                    "- 跑一轮标注\n- 质检\n- 测试连通\n- 看配置\n- 做 6 轮自动调参\n"
                    "如果要新增能力，请直接描述你希望的输入和输出。[/yellow]"
                )
                continue

        console.print(f"[cyan]AI建议动作:[/cyan] [bold]{action}[/bold]")
        if args:
            console.print(RichJSON.from_data(args, ensure_ascii=False, indent=2))

        if action == "task_set_backend" and "task_id" not in args:
            last_task_ids = state.get("last_task_ids", [])
            use_task_scope = False
            if isinstance(last_task_ids, list) and len(last_task_ids) == 1:
                use_task_scope = typer.confirm(
                    f"检测到最近任务为 task {last_task_ids[0]}。是否只切换这个任务？(否则切换当前全局配置)",
                    default=True,
                )
                if use_task_scope:
                    args["task_id"] = int(last_task_ids[0])
            else:
                use_task_scope = typer.confirm(
                    "是否指定一个 task id 进行切换？(否则切换当前全局配置)",
                    default=False,
                )
                if use_task_scope:
                    args["task_id"] = int(typer.prompt("Task ID", default="1"))

            console.print("[cyan]切换范围确认后参数:[/cyan]")
            console.print(RichJSON.from_data(args, ensure_ascii=False, indent=2))

        plan = build_plan(user_text, action, args)
        if len(plan) > 1:
            table = Table(show_header=True, header_style="bold blue")
            table.add_column("Step", style="cyan", width=6)
            table.add_column("Action", style="white")
            table.add_column("Args", style="white")
            for i, (a, p) in enumerate(plan, start=1):
                table.add_row(str(i), a, json.dumps(p, ensure_ascii=False))
            console.print(Panel.fit("[bold]执行计划[/bold]", border_style="blue"))
            console.print(table)

        if not typer.confirm("执行这个动作吗？", default=True):
            console.print("[dim]已取消。本轮未执行。[/dim]")
            continue

        try:
            for step_action, step_args in plan:
                step_cfg_path = _effective_cfg_path_for_action(step_action, step_args, cfg_path)
                step_cfg = load_config(step_cfg_path)
                _print_exec_context(step_cfg, step_cfg_path, step_action)
                _execute_action(step_action, step_args, cfg_path)
                state["last_action"] = step_action

            if state.get("last_action") == "task_list":
                rows = list_task_profiles(project_root=cfg_path.parent.resolve())
                ids = []
                for r in rows:
                    tid = r.get("task_id")
                    if isinstance(tid, int):
                        ids.append(tid)
                    elif isinstance(tid, str) and tid.isdigit():
                        ids.append(int(tid))
                state["last_task_ids"] = ids
            elif state.get("last_action") in {"task_show", "task_refine", "task_explain", "task_check", "task_fix", "task_set_backend"}:
                tid = plan[-1][1].get("task_id") if plan else args.get("task_id")
                if isinstance(tid, int):
                    state["last_task_ids"] = [tid]
        except Exception as exc:
            console.print(f"[bold red]执行失败:[/bold red] {exc}")


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
    console.print(Panel.fit("[bold green]Current configuration[/bold green]", border_style="green"))
    console.print(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))


if __name__ == "__main__":
    app()
