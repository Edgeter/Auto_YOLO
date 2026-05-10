# AutoYOLO CLI Scaffold

This is a production-oriented scaffold for a GPT-assisted auto-labeling workflow.

Current status:
- End-to-end CLI flow is ready.
- LLM planning works with `mock`, OpenAI-compatible API, or OpenCode CLI.
- Pre-annotation supports `grounding_dino` backend.
- QC report generation for YOLO labels is implemented.
- Objective-driven autotune loop is available.

## Who this is for

AutoYOLO is designed for users who need to bootstrap object-detection datasets quickly with an "AI pre-label + human review" workflow, especially in cold-start scenarios where manual labeling capacity is limited.

---

## Deployment options (where to run)

This project can be deployed in three common ways:

1. **Local workstation (recommended for data privacy)**
   - Best for private datasets and iterative prompt tuning.
   - Supports local model inference backends (for example `local_qwen_vl`) if local hardware is sufficient.

2. **Cloud GPU VM (recommended for speed and batch jobs)**
   - Suitable when running large image sets or repeated autotune experiments.
   - Typical platforms: AWS EC2 GPU, GCP Compute Engine GPU, Azure GPU VM, or any equivalent Linux GPU host.

3. **Hybrid mode (local orchestration + remote API inference)**
   - Run CLI locally, but call remote API models (`openai` / `vlm_api`) for planning and detection.
   - Useful when local GPU is limited.

---

## Hardware reference (compute requirements)

The exact requirement depends on detector backend, image resolution, and dataset size.

### Minimum usable baseline (small-scale smoke tests)

- CPU: 4 cores
- RAM: 8 GB
- GPU: optional (CPU-only can run, but slower)
- Storage: 5 GB free

### Recommended for practical daily use (medium dataset)

- CPU: 8+ cores
- RAM: 16-32 GB
- GPU: NVIDIA GPU with 8-12 GB VRAM
- Storage: 20+ GB free (cache + outputs + environment)

### Recommended for larger batches / faster turnaround

- CPU: 12+ cores
- RAM: 32+ GB
- GPU: NVIDIA GPU with 16+ GB VRAM
- Storage: SSD with 50+ GB free

Notes:
- `grounding_dino` first run downloads model weights from Hugging Face and requires additional disk/cache space.
- If using `vlm_api`, local GPU pressure is much lower, but network latency and API throughput become the bottleneck.

---

## Environment requirements

- Python: `>=3.10`
- OS: Windows / Linux (Linux preferred for cloud GPU environments)
- Optional: CUDA-enabled PyTorch for GPU acceleration

## 1) Install

```bash
cd C:\Users\Lenovo\Desktop\Auto_YOLO
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Install PyTorch (choose command from https://pytorch.org/get-started/locally/):

```bash
pip install torch
```

For GPU acceleration, install the CUDA-matched PyTorch build from the official selector.

If using OpenAI provider:

```bash
copy .env.example .env
# then edit .env and set OPENAI_API_KEY
# OPENAI_BASE_URL defaults to https://api.hanbbq.top/v1
# if using detector_backend=vlm_api, also set VLM_API_KEY (separate key supported)
```

You can also set base URL in `autoyolo.yaml` via `openai_base_url`.

## 2) Initialize project files

```bash
autoyolo init --project-root .
```

This creates:
- `images/`
- `labels/`
- `reports/`
- `classes.txt`
- `autoyolo.yaml`

Place raw input images into `images/`, and ensure class names are correctly defined in `classes.txt`.

## 3) Interactive setup

```bash
autoyolo wizard --config autoyolo.yaml
```

You can set:
- dataset paths
- classes file
- GPT prompt
- LLM provider/model (`mock/openai/opencode`)
- detector backend
- GroundingDINO model id/device when backend is `grounding_dino`
- VLM API base/model/key env var when backend is `vlm_api`

Notes on LLM provider behavior:
- If `llm_provider=openai` fails (e.g. provider 503), pipeline can auto-fallback to OpenCode and then mock.
- OpenCode fallback uses command defined by:
  - `opencode_executable` (default `npx`)
  - `opencode_runner_args` (default `opencode run`)

## 4) Run full pipeline

```bash
autoyolo run --config autoyolo.yaml
```

Output:
- `reports/annotation_plan.json`
- `labels/**/*.txt`
- `reports/qc_report.json`

When using `grounding_dino`, the first run downloads model weights from Hugging Face.

## Example: run with GroundingDINO

Use `autoyolo wizard` and set:
- `detector_backend=grounding_dino`
- `grounding_dino_model_id=IDEA-Research/grounding-dino-base`
- `inference_device=auto`

Then run:

```bash
autoyolo run --config autoyolo.yaml
```

## Example: run with VLM API detector

Set in config:
- `detector_backend=vlm_api`
- `vlm_base_url=https://api.hanbbq.top/v1`
- `vlm_model=cch/gpt-5.4`
- `vlm_api_key_env=VLM_API_KEY`

Set key in current shell:

```powershell
$env:VLM_API_KEY="your_vlm_key"
```

Then run:

```bash
autoyolo run --config autoyolo.yaml
```

---

## Typical deployment playbooks

### A) Local private workflow

1. Install environment and dependencies.
2. Run `autoyolo wizard --config autoyolo.yaml`.
3. Configure local backend (`grounding_dino` or `local_qwen_vl`).
4. Run `autoyolo run --config autoyolo.yaml`.
5. Review `reports/qc_report.json` and sampled labels.

### B) Cloud batch workflow

1. Provision Linux GPU VM and install Python/PyTorch.
2. Clone project and run `pip install -e .`.
3. Upload images/classes/config.
4. Run `autoyolo run` or `autoyolo autotune` in batch.
5. Download `labels/` and `reports/` artifacts for review.

### C) API-first lightweight workflow

1. Keep project local.
2. Set API keys (`OPENAI_API_KEY`, `VLM_API_KEY`) and base URLs in config.
3. Use `openai` and/or `vlm_api` backends.
4. Run pipeline with minimal local compute overhead.

## 5) Run QC only

```bash
autoyolo qc --config autoyolo.yaml
```

## 5.1) Ask local vision model (prompt writing helper)

Use local Qwen-VL directly from CLI to understand an image before writing/refining prompts:

```bash
autoyolo vision --config autoyolo.yaml --image images/022_0007.jpg --ask "What is the main object and where is it?"
```

Output includes:
- image path
- original size and runtime size
- model answer text

Fast helper command:

```bash
autoyolo prompt images/022_0007.jpg
```

This command asks the local model to generate prompt-writing guidance and returns plain text directly.

## 5.2) Task automation (interactive)

Create a numbered task profile interactively:

```bash
autoyolo task-create --config autoyolo.yaml
```

It will prompt you for:
- images directory path
- classes file path
- natural language instruction

Then it calls AI to optimize the instruction into a targeted prompt, lets you confirm/regenerate, and saves to:
- `tasks/1.yaml`, `tasks/2.yaml`, ...

During confirmation, it shows bilingual prompt output:
- English prompt (used for model inference)
- Chinese explanation (for human review)

Run by task id only:

```bash
autoyolo run --config autoyolo.yaml --task 1
```

List all tasks with id, image path, and timestamp:

```bash
autoyolo task-list --config autoyolo.yaml
```

Refine prompt only for an existing task (keeps dataset/class paths unchanged):

```bash
autoyolo task-refine --config autoyolo.yaml --task 1
```

This command will:
- show current EN/ZH prompt
- ask for natural language refinement instruction
- regenerate EN prompt + ZH explanation until you confirm
- save back into the same `tasks/<id>.yaml`

## 5.3) Test remote API chat directly

If you want to verify external API connectivity independently:

```bash
autoyolo chat-test --config autoyolo.yaml --message "Hello from AutoYOLO"
```

This uses `openai_base_url`, `llm_model`, and `OPENAI_API_KEY`.

## 6) Autotune parameters from profile

Use natural language to describe dataset characteristics, then let AutoYOLO iterate parameters automatically:

```bash
autoyolo autotune --config autoyolo.yaml --profile "single symbol per image, prioritize one clean box" --max-rounds 6 --target-loss 0.25 --probe-images 10 --full-eval-trigger-loss 0.8
```

Autotune outputs:
- `reports/objective_spec.json`
- `reports/tune_history.json`

How it works:
- Builds objective spec from `llm_provider` (`openai`/`opencode`) with fallback defaults.
- Runs pipeline in rounds and computes metrics from generated labels.
- Uses fast probe rounds on a subset of images, then runs full evaluation only when probe looks promising.
- Adjusts detection thresholds and filtering parameters until convergence or max rounds.

## Next integration points

Main file to extend detector inference:
- `autoyolo/services/preannotate.py`

Recommended direction:
1. Add class aliases from plan into detector prompts.
2. Add SAM/SAM2 refine pass.
3. Add `review-pack` command for active learning prioritization.
4. Add dataset split/export helper for YOLO training.

## Stage update (current)

This project is still under active development, but has reached important milestone results for cold-start labeling:

- End-to-end automation is stable: plan -> pre-annotation -> QC -> reports.
- Local GPU labeling route is validated with `local_qwen_vl` backend.
- Coordinate mapping was fixed to use an explicit resize/inference/original-image mapping chain, improving box localization consistency.
- Single-class precision mode is now available via class query override (example: `6,0`), which is effective for early-stage seed data generation.
- Output quality is now good enough for practical "AI pre-label + human review" workflow in current tests.

### Why this tool is useful

- Solves early "data snowball" problem for detection projects by reducing manual labeling cost.
- Supports natural-language-driven setup and objective-oriented iteration.
- Supports local inference workflows for data-sensitive scenarios.

### Current status

- Development stage: ongoing.
- Phase result: pipeline is production-like for controlled cold-start tasks, with verified local inference and accurate coordinate outputs in focused scenarios.
- Next priorities: multi-class recall stability, speed optimization, and richer debug/visual review tooling.
