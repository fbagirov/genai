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
