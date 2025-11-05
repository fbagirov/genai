# genai
This is a repository of a privacy-first, local-only AI chat assistant built to demonstrate secure on-device inference, minimal data exposure, and configurable guardrails.


# Features

- Runs 100 % locally using Ollama or llama.cpp — no cloud calls required.
- FastAPI API and Gradio UI for chat.
- Privacy-by-design defaults: no telemetry, no log persistence, no chat history storage.
- Configurable guardrails with YAML policy rules.
- Optional PII scrubbing via Microsoft Presidio.
- Docker & Makefile for reproducible builds.
- Cross-platform: works on Windows, macOS, and Linux.


# Quickstart Demo

## Install preprequisits

- Python 3.10+
- Ollama (https://ollama.com/download)

bash ```
ollama pull llama3:instruct
```

## Setup the environment
bash ```

python -m venv .venv
# PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt

```

## Configure configs/config.yaml

bash ```
llm:
  provider: ollama
  model: llama3:instruct
privacy:
  telemetry: false
  persist_history: false
  log_requests: false

```

## Run the API

bash ```

uvicorn server.api:app --host 127.0.0.1 --port 8000 --no-access-log
```

## Run health check

bash ```
GET http://127.0.0.1:8000/healthz

```

## Launch UI
bash ```
python private_assistant/ui/chat_ui.py
```

## Test (curl)
bash ```
curl -s http://127.0.0.1:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Explain retrieval-augmented generation in two sentences."}]}'

```
# How to use Docker
bash ```
make docker-build                 # build the image
make docker-run                   # start API at http://localhost:8000
make docker-logs                  # tail logs
make docker-stop                  # stop containers
make docker-dev                   # run with source mounted & reload (dev)

```


# Architecture

          ┌────────────┐
          │   User UI  │  ← Gradio frontend
          └──────┬─────┘
                 │ HTTP
          ┌──────▼─────┐
          │  FastAPI   │  ← server.api
          └──────┬─────┘
      ┌──────────┼───────────┐
      │ Privacy Middleware   │ disables logs, history
      │ Guardrails Wrapper   │ policy validation
      │ PII Scrubber         │ anonymizes text (optional)
      └──────────┬───────────┘
                 │
          ┌──────▼─────┐
          │  Ollama /  │
          │ llama.cpp  │  ← local LLM runtime
          └────────────┘


# Flow diagram

User → (UI or API)
   │
   ▼
[Privacy Middleware]
   │
   ▼
[Guardrails / PII Scrub]
   │
   ▼
[LLM Inference: Ollama or llama.cpp]
   │
   ▼
[Guardrails Post-filter]
   │
   ▼
Response → returned to user (not stored)

# Config Toggles

| Section   | Key               | Description                        | Default                    |
| --------- | ----------------- | ---------------------------------- | -------------------------- |
| `llm`     | `provider`        | LLM runtime (`ollama` or `openai`) | `ollama`                   |
|           | `model`           | Model name/tag                     | `llama3:instruct`          |
|           | `temperature`     | Sampling temperature               | 0.2                        |
| `privacy` | `telemetry`       | Enable outbound telemetry          | false                      |
|           | `persist_history` | Save chats to disk                 | false                      |
|           | `log_requests`    | HTTP request logging               | false                      |
| `safety`  | `guardrails`      | Use policy YAML                    | true                       |
|           | `pii_scrub`       | Enable Presidio scrub              | false                      |
| `paths`   | `policy_file`     | Path to guardrails YAML            | `policies/guardrails.yaml` |

# Limits

- Model size: constrained by your hardware; small models (< 8 GB VRAM) recommended.
- No persistence unless explicitly enabled.
- No streaming yet (synchronous responses).
- PII scrub requires extra deps + spaCy model (en_core_web_sm).
- Guardrails are minimal demos—extend for enterprise use.