
# llm-finetune-sft-dpo

**Goal:** Fine-tune a small instruction model with **Supervised Fine-Tuning (SFT)** and compare against **preference optimization** (**DPO**). Uses **LoRA** adapters (parameter-efficient), optional **4/8‑bit quantization**, local **MLflow** tracking, and a simple **FastAPI** server that loads the LoRA at inference.

> **Default base model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (works on CPU albeit slow). You can swap for `gpt2` for a super-light demo or any 7B+ model if you have GPU VRAM.

---

## Features
- **SFT** on a small curated instruction dataset
- **DPO** (Direct Preference Optimization) on paired comparisons (`chosen` vs `rejected`)
- **LoRA** adapters via `peft` (keeps base weights frozen by default)
- Optional **4-bit**/8-bit loading (`bitsandbytes`) to reduce VRAM
- **MLflow** for runs, params, and artifacts (local `./mlruns/`)
- **FastAPI** server that loads the LoRA at inference (or merged weights)
- Minimal **eval** hook via `lm-eval-harness` (optional)
- **De-identification** pre-processing with Presidio (optional, for data governance)

---

## Quickstart

### 0) Environment
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# (Optional, GPU/NVIDIA) quantization + speedups
pip install -r requirements.gpu.txt
# (Optional) PII de-identification
pip install -r requirements.extra.txt
```

> If `bitsandbytes` doesn’t install (CPU-only or macOS), set `quantization.load_in_4bit: false` in the config.

### 1) Configure
```bash
cp configs/config.example.yaml configs/config.yaml
# edit configs/config.yaml to pick base model, LoRA params, training hyperparams
```

### 2) (Optional) De‑identify your dataset
```bash
python scripts/deid_presidio.py   --in data/sft/sample_sft.jsonl   --out data/sft/sample_sft.deid.jsonl
```
Update `configs/config.yaml` to point SFT to the de‑identified file.

### 3) Supervised Fine-Tuning (SFT)
```bash
python scripts/train_sft.py --config configs/config.yaml
```
Artifacts: `outputs/sft_adapter/` (LoRA weights), tokenizer, and MLflow run in `mlruns/`.

### 4) Direct Preference Optimization (DPO)
```bash
python scripts/train_dpo.py --config configs/config.yaml
```
By default, DPO initializes from the SFT adapter (`outputs/sft_adapter/`). Artifacts in `outputs/dpo_adapter/`.

### 5) Serve for inference (FastAPI)
```bash
# Uses base model + LoRA adapter at runtime
uvicorn serve.fastapi_server:app --reload --port 8000
# Then: curl -X POST http://localhost:8000/v1/generate -H "Content-Type: application/json" #   -d '{"prompt": "Write a two-sentence project summary.", "max_new_tokens": 128}'
```

### 6) (Optional) Merge LoRA → full model
```bash
python scripts/export_lora.py   --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0   --adapter outputs/dpo_adapter   --out_dir outputs/merged_full_model
```
> Merged weights are large; prefer adapters for sharing.

### 7) (Optional) LM Eval Harness
```bash
# Example: tiny subset for a smoke test
python eval/run_lmeval.py --model_path outputs/dpo_adapter --task hellaswag --limit 50
```

---

## Data formats

### SFT (JSONL)
Each line:
```json
{"instruction": "Summarize the text", "input": "....", "output": "A concise summary ..."}
```
If `input` is empty, it’s omitted in the prompt template.

### DPO (JSONL)
Each line:
```json
{"prompt": "How to handle secrets?", "chosen": "Use a password manager...", "rejected": "Email the password."}
```

---

## MLflow
```bash
mlflow ui --backend-store-uri mlruns --port 5000
```
Open http://localhost:5000 to browse runs, params, and metrics.

---

## Config overview (`configs/config.yaml`)
- `base_model_id`: HF model id (e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, `gpt2`)
- `tokenizer_id`: usually same as base
- `quantization.load_in_4bit`: true/false (requires bitsandbytes + CUDA)
- `lora`: rank/alpha/dropout + `target_modules` (use `auto` to infer for LLaMA vs GPT2-like)
- `sft` & `dpo`: dataset paths + training hyperparameters
- `serve`: adapter to load at runtime
- `tracking.mlflow_tracking_uri`: defaults to local `mlruns`

---

## Notes & Limits
- Training large models requires GPU VRAM. For CPU-only, use tiny models (`gpt2`, `TinyLlama`) and small batch sizes.
- `bitsandbytes` is optional; you can disable quantized loading entirely.
- This repo uses a tiny sample dataset for demonstration only; replace with your domain data.
- Always de‑identify sensitive data before training if privacy matters.

---

## License
MIT (see LICENSE)
