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

Temperature defines how "creative" the answers are - the higher the temp, the more creative they are. The lower the temp, the more the answers are on point. For example, randomness 0 = strict; 1+ = creative

Telemetry 
   - false - never send your telemetry (privacy reasons)

log_requests  
   - true (for debugging)
   - false (no HTTP access, no API body & access logs)

guardrails: 
   - true - app checks prompts/outputs against policies/guardrails.yaml (tweak the yaml policy to be stricter (refuse  harmful requests, mask sensitive requests, etc.) or loser). 

pii_scrub
   - true - runs Presidio to detect and anonymize PII in prompt queries (requires the Presidio and english SpaCy model installed)


bash ```
llm:
  provider: ollama        # ollama or openai
  model: llama3:instruct  # model name (llama or gpt2 or >gpt3 (paid version))
  temperature: 0.2        # randomness 0 = strict; 1+ = creative
  max_new_tokens: 512     # maximum tokens generated per reply

privacy:
  telemetry: false        # false by default to protect your privacy.
  persist_history: false  # keep chat only in memory, no files
  log_requests: false     # false - no HTTP access, no API body & access logs. 

safety: 
  guardrails: true        # apply policies/guardrails.yaml
  pii_scrub: false        # scrub PII via Presidio pre/post stages

paths:
  policy_file: policies/guardrails.yaml

```

### Sample configurations

#### Ultra private, offline

This configuration will generate short, consistent replies. Nothing will be written on your hard drive, no longs. Ensure you donwload the LLM locally and no keys are set. 

bash ```

llm:
  provider: ollama
  model: llama3:instruct
  temperature: 0.2
  max_new_tokens: 256
privacy:
  telemetry: false
  persist_history: false
  log_requests: false
safety:
  guardrails: true
  pii_scrub: false

```

### Longer answers

More eloquent wording, longer outputs 

bash```

llm:
  temperature: 0.5
  max_new_tokens: 800

```

### Creative/brainstorming

More creative, novel ideas. Set the guardrails to be able to still catch unsafe content. 

bash```

llm:
  temperature: 0.9
  max_new_tokens: 512
safety:
  guardrails: true


```
### Strict/enforced compliance/sensitive data

PII in the input is scrubbed. Nothing is logged. 

bash```

privacy:
  persist_history: false
  log_requests: false
safety:
  guardrails: true
  pii_scrub: true

```

# Setting policies




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


# Troubleshooting

- Model not found
   - make sure you pull the model
- Empty responses 
   - reduce temperature
- PII scrubbing 
   - turning this on without installing Presidio will raise an error. Install extras
   - Install SpaCy (python -m spacy download en_core_web_[sm|lg])