# genai-private-assistant (Local + Privacy-first)

A **privacy-first, local-only AI assistant** showcasing:
- **Local inference** (Ollama)
- **FastAPI** API
- **Gradio** UI
- **No persistence by default**
- Optional: **PII scrubbing** (Presidio)
- Optional: **policy guardrails** via YAML

> This repo is designed as a learning + portfolio demo. It focuses on **local-first design** and clear toggles so you can show security-minded engineering in interviews.

---

## Features

- Runs **100% locally** using **Ollama** (no cloud calls required).
- FastAPI API (`/v2/chat`, `/v2/chat_with_files`) and Gradio UI.
- Privacy-by-design defaults:
  - telemetry disabled
  - no chat history stored
  - no request logging
- Config-driven behavior in `configs/config.yaml`.
- Optional:
  - PII scrub with Presidio
  - file ingestion (PDF, DOCX, TXT, images w/ OCR)

---

## Quickstart (Windows / PowerShell)

### 0) Install prerequisites

- Python 3.10+ (3.11 recommended)
- Ollama: https://ollama.com/download

Pull a model:
```bash
ollama pull llama3:instruct
```

### 1) Install Python environment
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2) Configure the app

Example baseline: 

```bash
llm:
  provider: ollama
  model: llama3:instruct
  temperature: 0.2
  max_new_tokens: 512

privacy:
  telemetry: false
  persist_history: false
  log_requests: false

safety:
  guardrails: false
  pii_scrub: false

paths:
  policy_file: policies/guardrails.yaml
```

### 3) Run the API

```bash
python -m uvicorn private_assistant.server.api:app --host 127.0.0.1 --port 8000 --no-access-log

```

Health check: 
```bash
GET http://127.0.0.1:8000/healthz

```

### 4) Run the UI

```bash
python private_assistant/ui/chat_ui.py

```
Open in browser at http://127.0.0.1:8001



---

## Architecture

┌───────────────┐
│   Gradio UI   │  (private_assistant/ui/chat_ui.py)
└───────┬───────┘
        │ HTTP POST /v2/chat
┌───────▼───────┐
│   FastAPI API │  (private_assistant/server/api.py)
└───────┬───────┘
        │ local HTTP call
┌───────▼────────────┐
│ Ollama Runtime      │  http://127.0.0.1:11434
│ (Local model)       │
└─────────────────────┘


## Flow diagram

User → UI/API
   ↓
(optional) preprocess input:
   - policy checks
   - PII scrubbing
   - file ingestion (PDF/DOCX/TXT/OCR)
   ↓
Prompt → Ollama (/api/generate)
   ↓
LLM output
   ↓
(optional) postprocess:
   - policy checks
   - PII scrub
   ↓
Return response (not stored by default)


## Config toggles

###LLM behavior

llm.provider: ollama (default)

llm.model: e.g. llama3:instruct

llm.temperature: higher = more creative, lower = more strict

llm.max_new_tokens: output length cap

###Privacy behavior

privacy.telemetry: keep false for privacy

privacy.persist_history: false keeps everything in memory only

privacy.log_requests: false disables request logging (recommended)

###Safety behavior

safety.guardrails: enable YAML policy checks (demo)

safety.pii_scrub: enable Presidio anonymization


###Limits

Speed and quality depend on your local model + hardware.

No streaming output in this simple demo (sync responses).

OCR quality depends on image quality; low-res scans will be weak.

Guardrails are basic demo logic (extend for real production use).

Presidio requires extra dependencies and is English-focused by default.


