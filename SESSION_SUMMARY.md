# Auto_YOLO Session Summary (Detailed)

## 0) About conversation persistence

- This chat context may not be reliably available after restart.
- To avoid losing progress, this file records the full technical state and next steps.

---

## 1) Project goal and direction confirmed

You clarified the real goal is:

- Build a **semi-automatic labeling tool** for YOLO cold-start stage.
- Use AI to generate bounding boxes, then human review to reduce workload.
- Keep CLI workflow and automation, but avoid hardcoding one dataset behavior.
- Support iterative tuning and robust fallback behavior.

Key design decisions made:

- Keep existing CLI scaffold (`run`, `qc`, `config`, `autotune`, etc.).
- Separate concerns:
  - `llm_provider` for planning/objective generation.
  - `detector_backend` for actual box generation.
- Avoid silent failures; fallback paths must be visible in logs/reports.

---

## 2) Current project location

- Project root: `C:\Users\Lenovo\Desktop\Auto_YOLO`

Main config file:

- `C:\Users\Lenovo\Desktop\Auto_YOLO\autoyolo.yaml`

---

## 3) What has been implemented

### 3.1 CLI scaffold and core workflow

Implemented commands:

- `autoyolo init`
- `autoyolo wizard`
- `autoyolo run`
- `autoyolo qc`
- `autoyolo config show`
- `autoyolo config set <key> <value>`
- `autoyolo autotune`

Pipeline behavior (`run`):

1. Load config and env.
2. Read images + classes.
3. Build `annotation_plan.json` via LLM provider with fallback.
4. Run detector backend to generate YOLO labels.
5. Run QC and write report.

### 3.2 LLM planning and fallback chain

Supported `llm_provider`:

- `openai`
- `opencode`
- `mock`

Fallback behavior:

- If `openai` fails (e.g., 503), it can fallback to `opencode`.
- If `opencode` also fails, fallback to `mock`.
- Fallback reason is written into `reports/annotation_plan.json` and printed in console.

### 3.3 GroundingDINO backend

Implemented:

- `detector_backend=grounding_dino`
- GPU/CPU auto device resolution.
- NMS per class.
- area filter and max detections per class.
- progress bar + checkpoints.
- compatibility for `transformers` post-process argument differences (`box_threshold` vs `threshold`).
- in-memory model cache and local-cache fallback for HF transient errors.

### 3.4 OpenCode integration details

Implemented OpenCode adapter for plan generation:

- command pattern: `npx opencode run --model <model> <prompt>` or configured equivalent
- JSON extraction from CLI output

Windows path issue fixed by supporting `.cmd` executable in config.

### 3.5 Autotune (objective-driven)

Implemented `autoyolo autotune` with:

- objective generation from profile text (OpenAI/OpenCode/default fallback)
- per-round metrics and loss
- best-config persistence to `autoyolo.yaml` while tuning
- replay-best behavior with transient error tolerance
- probe-based fast rounds:
  - `--probe-images` for subset evaluation each round
  - `--full-eval-trigger-loss` to decide when full dataset run is needed

Outputs:

- `reports/objective_spec.json`
- `reports/tune_history.json`

### 3.6 New detector backend added: `vlm_api`

As requested, added a new backend to use multimodal API directly for box generation.

Key features:

- `detector_backend=vlm_api`
- per-image VLM call with strict JSON instruction
- expected payload fields: `class`, `x1`, `y1`, `x2`, `y2`, `confidence`
- retries and parse safeguards
- conversion to YOLO txt format
- uses **independent API key env var** (`VLM_API_KEY` by default), separate from OpenCode/OpenAI plan key

---

## 4) Config fields currently supported

Important fields in `autoyolo.yaml` now include:

- planning related:
  - `llm_provider`
  - `llm_model`
  - `openai_base_url`
  - `opencode_executable`
  - `opencode_runner_args`
  - `opencode_timeout_sec`
  - `opencode_fallback_on_openai_error`

- detector related:
  - `detector_backend` (`mock` / `grounding_dino` / `vlm_api`)
  - `box_threshold`
  - `text_threshold`
  - `nms_iou_threshold`
  - `max_detections_per_class`
  - `min_box_area_norm`
  - `grounding_dino_model_id`
  - `inference_device`

- VLM API detector fields:
  - `vlm_base_url`
  - `vlm_model`
  - `vlm_api_key_env`
  - `vlm_max_retries`
  - `vlm_timeout_sec`

---

## 5) Known issues and observations from testing

### 5.1 GroundingDINO quality mismatch on symbol/digit task

- For this dataset type, GroundingDINO can underperform or over-detect.
- Lower thresholds increased recall but caused huge box counts/noise.
- Root cause likely model-task mismatch (character/symbol localization).

### 5.2 503 provider errors in OpenAI-compatible planning

- `openai` planning frequently got `503 no_available_providers`.
- Fallback chain now handles this, often ending at `opencode_fallback`.

### 5.3 HF network instability

- Occasional Hugging Face transient connection errors were observed.
- Some robustness added (cache reuse/local fallback), but internet instability can still affect runs.

---

## 6) Environment and dependencies status (what was done)

- Python 3.13 env had PyTorch compatibility problems.
- Created/used Python 3.12 environment (`.venv312`) and installed compatible CUDA build.
- GPU detected successfully for RTX 5060 laptop.

---

## 7) Files changed significantly in this session

- `pyproject.toml`
- `.env.example`
- `autoyolo.yaml`
- `README.md`
- `autoyolo/models.py`
- `autoyolo/cli.py`
- `autoyolo/pipeline.py`
- `autoyolo/services/wizard.py`
- `autoyolo/services/plan.py`
- `autoyolo/services/preannotate.py`
- `autoyolo/services/autotune.py`
- `autoyolo/adapters/__init__.py`
- `autoyolo/adapters/openai_adapter.py`
- `autoyolo/adapters/opencode_adapter.py`

---

## 8) Current strategic decision at end of session

You decided to try **local VLM model** route with:

- Model family: `Qwen3-VL-2B` GGUF quantized
- Runtime preference: `Ollama`
- Model storage path planned:
  - `D:\AI_Models\Qwen\Qwen3-VL-2B-GGUF\`

Reason:

- Better fit to user-intended “AI understands image and gives coordinates” workflow.
- Potentially better quality for your specific single-symbol scenario than GroundingDINO.

---

## 9) Immediate next steps after restart

### Step A: finish local model download

If PowerShell `Invoke-WebRequest` fails, use:

```powershell
curl.exe -L --retry 8 --retry-delay 5 -o "D:\AI_Models\Qwen\Qwen3-VL-2B-GGUF\qwen3-vl-2b-q4_k_m.gguf" "https://huggingface.co/Qwen/Qwen3-VL-2B-GGUF/resolve/main/qwen3-vl-2b-q4_k_m.gguf"
```

Or use HF CLI download method.

### Step B: set up Ollama model

1. Ensure Ollama installed.
2. Create `Modelfile` referencing GGUF.
3. `ollama create ...`.
4. `ollama run ...` sanity test.

### Step C: integrate Auto_YOLO with local VLM endpoint

Current code already has `vlm_api` backend for remote OpenAI-compatible multimodal API.

Next integration decision:

- Either point `vlm_api` to local OpenAI-compatible server around Ollama,
- Or add dedicated `ollama_vlm` backend in code.

---

## 10) Suggested acceptance criteria (for next phase)

Define and lock before further changes:

1. coverage rate (images with >=1 box)
2. single-box rate (for this dataset profile)
3. area-valid rate (reasonable normalized area band)
4. runtime budget per N images
5. QC hard constraints (format, bounds, class id validity)

This prevents pseudo-convergence and keeps optimization aligned to task, not recent prompt wording.

---

## 11) Resume prompt template (copy after restart)

Use this message after reopening chat:

```text
Please read C:\Users\Lenovo\Desktop\Auto_YOLO\SESSION_SUMMARY.md and continue from section 9.
Current priority: finish Ollama local VLM path and connect Auto_YOLO detector backend for local multimodal inference.
Keep OpenCode/OpenAI planning key path independent from VLM detection key path.
```

---

## 12) New progress update (today)

### 12.1 Major direction change confirmed

- Confirmed that current Windows Ollama route is unstable/slow on RTX 5060 (sm_120) in this setup.
- Confirmed local `transformers` + PyTorch `cu128` route is the active path for GPU inference.
- Focus moved from "whether pipeline runs" to "bbox position quality".

### 12.2 New backends and config added in code

Implemented in codebase:

- Added `detector_backend=ollama_vlm` (local Ollama API path, regex fallback, fail-fast on pure text/no coords).
- Added `detector_backend=local_qwen_vl` (local Transformers path).
- Added new config fields in `RunConfig`:
  - `ollama_base_url`, `ollama_model`, `ollama_max_retries`, `ollama_timeout_sec`
  - `local_qwen_model_path`, `local_qwen_device`, `local_qwen_max_image_side`, `local_qwen_max_new_tokens`
- Wizard now supports both `ollama_vlm` and `local_qwen_vl`.

### 12.3 Local Qwen-VL behavior observed

Run with full set (36 images):

- Completed end-to-end with `local_qwen_vl` on CUDA.
- Runtime about 19 minutes for 36 images.
- Produced `total_boxes=17`, QC format issues remained 0.

Key quality issue:

- Labels are often semantically correct, but bbox positions/sizes are frequently poor.
- Frequent failure pattern before fixes: empty `[]`, degenerate boxes (`x2<=x1` or `y2<=y1`), and oversized/unstable boxes.

### 12.4 Coordinate/format robustness changes already applied

- Added stricter retry prompts for structured output.
- Added handling for degenerate boxes (minimal width/height repair).
- Changed empty output handling: if model returns `[]`, write empty label and continue (no hard crash).
- Added coordinate mapping helper that can interpret normalized vs pixel-like outputs.
- Disabled manual pre-resize by default for local Qwen path to reduce coordinate-reference confusion:
  - `local_qwen_max_image_side` default changed to `0` (disable manual resize).

### 12.5 Smoke tests performed

Three-image smoke set used:

- `images_smoke3/022_0007.jpg`
- `images_smoke3/022_0008.jpg`
- `images_smoke3/022_0012.jpg`

Single-class smoke files created:

- `classes_smoke6.txt` (only class `6`)
- `autoyolo_smoke3_six.yaml`

Single-class prompt strategy update:

- For one-class mode, prompt changed to "find number 6 and return ONLY `[x1,y1,x2,y2]`" style.
- Parser updated to accept one-box list format for one-class mode.

Latest smoke result (single-class):

- Pipeline runs on CUDA and completes.
- 3 images -> 1 detected box total (recall still low).

### 12.6 Current status at end of today

- Infrastructure integration is working (CLI, backend routing, model loading, label write, QC).
- Main blocker is now **bbox localization quality**, not connectivity or schema.
- Speed is still high-cost; deferred until position quality is stabilized.

### 12.7 Priority for next session

1. Fix localization quality first (highest priority):
   - tighten one-class and then multi-class prompts,
   - add geometric post-filters (reject near-full-image boxes),
   - add debug artifact logging per image (raw output + mapped coords).
2. Validate quality on smoke3 first, then scale to 36 images.
3. After quality baseline is acceptable, optimize runtime.

### 12.8 Resume prompt template (updated)

```text
Please read C:\Users\Lenovo\Desktop\Auto_YOLO\SESSION_SUMMARY.md and continue from section 12.
Current priority: fix bbox localization quality for local_qwen_vl (not speed first).
Use smoke3 first (022_0007.jpg, 022_0008.jpg, 022_0012.jpg), then expand to 36 images.
```

---

## 13) New progress update (today, task automation + prompt engineering)

### 13.1 Major milestone

- Local Qwen inference path achieved a key quality breakthrough after fixing resize/mapping chain.
- In single-class precision mode, 3-image smoke test reached accurate box localization.
- Main issue shifted from "can it run" to "task workflow and prompt quality tooling".

### 13.2 What was added to CLI

New commands implemented:

- `autoyolo vision --image ... --ask ...`
  - Query local vision model directly for image understanding.
- `autoyolo prompt <image>`
  - Fast helper that asks local model to output prompt-writing guidance.
- `autoyolo task-create --config ...`
  - Interactive task profile creation.
  - Prompts user for image directory, classes file, natural-language instruction.
  - Generates optimized EN prompt + ZH explanation with confirmation loop.
  - Saves numbered task config to `tasks/<id>.yaml`.
- `autoyolo task-list --config ...`
  - Lists task id, image path, classes file, and timestamp.
- `autoyolo task-refine --config ... --task <id>`
  - Prompt-only refinement for existing task.
  - Shows current EN/ZH prompt, accepts refinement instruction, regenerates and confirms.
- `autoyolo chat-test --config ... --message ...`
  - Direct remote API chat connectivity test.

Also updated:

- `autoyolo run` now supports `--task <id>` and `--images-dir` override.

### 13.3 Label output behavior updates

Implemented configurable label output strategy:

- `label_naming_mode`: `image_name` or `sequential`
- `label_output_dir_mode`: `base` or `sequential_subdir`

Current preferred behavior for your workflow:

- Keep label filename aligned with source image name.
- Optionally create sequential run folders under `labels/` (e.g. `labels/1/`, `labels/2/`...).

### 13.4 Root-cause debugging done today

#### A) CUDA unavailable issue fixed

Observed:

- `torch.cuda.is_available() == False` while `nvidia-smi` looked normal.

Root cause:

- Environment variable `CUDA_VISIBLE_DEVICES=1` masked the only GPU (`index 0`).

Fix:

- Removed/cleared `CUDA_VISIBLE_DEVICES`, then CUDA became visible again (`True`, count `1`).

#### B) Task path serialization issue fixed

Observed:

- `RepresenterError: cannot represent an object WindowsPath(...)` during task save.

Fix:

- Switched task payload dump to JSON-compatible mode before YAML write.

#### C) Quoted path issue fixed

Observed:

- Task image path saved with quotes and became invalid (`...\"C:\...`).

Fix:

- Added path normalization: trim surrounding quotes and resolve absolute path.

### 13.5 Prompt generation architecture change

Decision confirmed:

- Prompt engineering (EN/ZH rewrite) should be outsourced to remote API when available.
- Local model remains primary for detection inference.

Implemented now:

- Task prompt generation/refinement tries remote API first.
- On remote failure, falls back to local model.
- Added retry/timeout and clearer remote error logs.

DeepSeek config validated:

- `openai_base_url=https://api.deepseek.com`
- `llm_model=deepseek-v4-pro`
- `llm_provider=openai`
- `autoyolo chat-test` succeeded with expected response.

### 13.6 Known issue still open at end of session

- `task-refine` remote prompt call sometimes reports `APIConnectionError: Connection error` while standalone `chat-test` succeeds.
- This indicates unstable/conditional connectivity for longer prompt-generation calls, not a total API misconfiguration.
- Temporary behavior is acceptable (auto-fallback to local), but needs stricter mode + better diagnostics next session.

### 13.7 Practical status summary

- Detection side: usable and improving, key localization path is validated.
- Task automation side: now significantly more productive than manual config editing.
- Prompt engineering side: workflow is in place, but remote reliability for refine path needs one more debug pass.

### 13.8 Priority for next session

1. Add `--remote-only` mode for `task-refine` / `task-create` prompt generation.
2. Add richer remote diagnostics (request id/error body classification) for DeepSeek failures.
3. Cache valid classes path back into task file permanently after correction.
4. Run task on new dataset (`box` / `box_all`) with finalized prompt and verify label quality end-to-end.

### 13.9 Resume prompt template (latest)

```text
Please read C:\Users\Lenovo\Desktop\Auto_YOLO\SESSION_SUMMARY.md and continue from section 13.
Current priority: stabilize remote DeepSeek prompt-refine path (chat-test already passes), then run task pipeline on box dataset with local_qwen_vl inference.
Keep ollama path disabled.
```
