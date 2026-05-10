# AutoYOLO

面向目标检测冷启动场景的自动标注 CLI 工具。AutoYOLO 以「自然语言驱动 + 可执行代理流程」为核心，帮助你完成从任务定义、预标注到质检报告的闭环。

## 功能特点

### 🎯 核心能力
- **Agent 式交互**：`autoyolo assistant` 作为主入口，支持自然语言下达任务。
- **任务工作流**：支持查看任务、改写提示词、可用性检查、执行任务等多步动作。
- **预标注流水线**：自动完成 plan -> pre-annotation -> QC -> report。
- **多后端支持**：本地模型与远程 API 双路线，按场景切换。
- **结果可追踪**：输出标注文件、统计结果和质检报告，便于人工复核。

### 🚀 模型与推理
- **本地检测后端**：支持 `local_qwen_vl`，适合数据隐私场景。
- **远程检测后端**：支持 `vlm_api`，在本地算力不足时可快速上线。
- **GroundingDINO 支持**：可用 `grounding_dino` 做通用目标检测预标注。
- **视觉问答辅助**：`autoyolo vision` / `autoyolo prompt` 可直接对图片提问，辅助写提示词。

### 🧩 工程化能力
- **交互式向导**：`autoyolo wizard` 引导配置数据路径、类别文件、模型与后端。
- **任务配置化**：`tasks/<id>.yaml` 独立保存任务上下文，便于复用和对比。
- **自动调参**：`autoyolo autotune` 按目标指标进行多轮参数搜索。
- **兼容 OpenAI 协议**：支持 `openai_base_url` 与多种兼容服务。

## 安装要求

### 环境要求
- Python 3.10+
- Windows 或 Linux（云端 GPU 推荐 Linux）
- 可选：CUDA + 对应版本 PyTorch（用于 GPU 加速）

### 安装步骤
1. **克隆项目并进入目录**
```bash
git clone https://github.com/Edgeter/Auto_YOLO.git
cd Auto_YOLO
```

2. **创建虚拟环境并安装**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

3. **安装 PyTorch**（按你的 CUDA/CPU 环境选择）
```bash
pip install torch
```

4. **配置环境变量（远程 API 模式）**
```bash
copy .env.example .env
```

常见变量：
- `DEEPSEEK_API_KEY`（优先）
- `OPENAI_API_KEY`（兜底）
- `VLM_API_KEY`（`detector_backend=vlm_api` 时使用）

## 快速开始

### 1) 初始化项目目录
```bash
autoyolo init --project-root .
```

会创建：
- `images/`
- `labels/`
- `reports/`
- `classes.txt`
- `autoyolo.yaml`

### 2) 交互式配置
```bash
autoyolo wizard --config autoyolo.yaml
```

### 3) 推荐主入口（自然语言）
```bash
autoyolo assistant --config autoyolo.yaml
```

示例指令：
- 看看当前任务
- 修改 task 1 提示词
- 检查 task 1 是否可用
- 直接跑一次任务1

### 4) 直接运行流水线
```bash
autoyolo run --config autoyolo.yaml
```

输出包括：
- `reports/annotation_plan.json`
- `labels/**/*.txt`
- `reports/qc_report.json`

## 常用命令

### 任务管理
```bash
autoyolo task-create --config autoyolo.yaml
autoyolo task-list --config autoyolo.yaml
autoyolo task-refine --config autoyolo.yaml --task 1
autoyolo run --config autoyolo.yaml --task 1
```

### 质量检查
```bash
autoyolo qc --config autoyolo.yaml
```

### 连通性测试
```bash
autoyolo chat-test --config autoyolo.yaml --message "Hello from AutoYOLO"
```

### 自动调参
```bash
autoyolo autotune --config autoyolo.yaml --profile "single symbol per image" --max-rounds 6
```

## 部署方式

### A. 本地私有部署（推荐）
- 本地运行 CLI + 本地模型推理。
- 适合私有数据、离线或低外发场景。

### B. 云端 GPU 批处理
- 在云主机执行 `run/autotune`，适合大规模图片批处理。

### C. 混合模式
- 本地编排流程，远程 API 承担模型推理，减少本地算力压力。

## 项目结构

```text
Auto_YOLO/
├── autoyolo/                # 核心代码
│   ├── cli.py               # CLI 与 assistant 入口
│   ├── models.py            # 配置与默认参数
│   ├── adapters/            # 模型/服务适配层
│   └── services/            # 任务编排、预标注、质检、调参等
├── tasks/                   # 任务配置（1.yaml, 2.yaml ...）
├── images/                  # 输入图片
├── labels/                  # 生成标注
├── reports/                 # 报告输出
├── autoyolo.yaml            # 全局配置
└── README.md
```

## 适用场景
- 冷启动数据集构建：先 AI 预标注，再人工复核。
- 提示词迭代实验：快速比较不同任务指令效果。
- 低标注预算项目：缩短首版数据集落地周期。

## 注意事项
- `grounding_dino` 首次运行会下载权重，需要额外网络与磁盘空间。
- 本地模型路径需与 `tasks/<id>.yaml` 或 `autoyolo.yaml` 中配置一致。
- 远程 API 模式建议优先通过 `autoyolo chat-test` 验证 key 与 base URL。

## 当前状态
- CLI 主流程可用，assistant 多步执行与任务路由已落地。
- 本地后端与远程后端都可运行，适合逐步从实验走向生产使用。
