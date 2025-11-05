# =========================
# Dockerized Private Assistant
# =========================
IMAGE ?= genai-private-assistant:latest
PORT ?= 8000            # API (FastAPI)
UI_PORT ?= 8001         # Optional Gradio UI (if you run it in-container)
CONFIG ?= configs/config.yaml
# Ollama on the host (default port 11434). On macOS/Windows Docker Desktop use host.docker.internal
OLLAMA_URL ?= http://host.docker.internal:11434

.PHONY: docker-build docker-run docker-dev docker-logs docker-stop docker-sh

docker-build:
	@echo "==> Building $(IMAGE)"
	docker build -t $(IMAGE) .

# Production-ish run: read-only container, just the API
docker-run:
	@echo "==> Running $(IMAGE) on :$(PORT)"
	docker run --rm -d \
		--name genai-private-assistant \
		-p $(PORT):8000 \
		-e OLLAMA_URL=$(OLLAMA_URL) \
		-e CONFIG_PATH=$(CONFIG) \
		--read-only --tmpfs /tmp:rw,size=64m \
		$(IMAGE)

# Dev mode: mounts source so you can edit & hot-reload (requires uvicorn --reload in image/cmd)
docker-dev:
	@echo "==> Dev run with bind mount & hot reload"
	docker run --rm -it \
		--name genai-private-assistant-dev \
		-p $(PORT):8000 \
		-v $(PWD):/app \
		-e OLLAMA_URL=$(OLLAMA_URL) \
		-e CONFIG_PATH=$(CONFIG) \
		$(IMAGE) \
		uvicorn server.api:app --host 0.0.0.0 --port 8000 --reload --no-access-log

docker-logs:
	docker logs -f genai-private-assistant || true

docker-stop:
	- docker stop genai-private-assistant genai-private-assistant-dev 2>/dev/null || true

docker-sh:
	docker exec -it genai-private-assistant /bin/sh


# .PHONY: dev ui api bench test

# dev:
# \tuvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# ui:
# \tpython -m ui.chat_ui

# api:
# \tpython -m app.main

# bench:
# \tpython -m eval.bench

# test:
# \tpytest -q
