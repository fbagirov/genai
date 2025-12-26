# =========================
# Dockerized Private Assistant
# =========================
IMAGE ?= genai-private-assistant:latest
PORT ?= 8000
CONFIG ?= configs/config.yaml

# Ollama on host default port 11434
# On macOS/Windows Docker Desktop: host.docker.internal works.
OLLAMA_URL ?= http://host.docker.internal:11434

.PHONY: docker-build docker-run docker-dev docker-logs docker-stop docker-sh

docker-build:
	@echo "==> Building $(IMAGE)"
	docker build -t $(IMAGE) .

# Production-ish run: just the API
docker-run:
	@echo "==> Running $(IMAGE) on :$(PORT)"
	docker run --rm -d \
		--name genai-private-assistant \
		-p $(PORT):8000 \
		-e OLLAMA_URL=$(OLLAMA_URL) \
		-e CONFIG_PATH=/app/$(CONFIG) \
		--read-only --tmpfs /tmp:rw,size=128m \
		$(IMAGE)

# Dev mode: bind mount source
docker-dev:
	@echo "==> Dev run with bind mount"
	docker run --rm -it \
		--name genai-private-assistant-dev \
		-p $(PORT):8000 \
		-v $(PWD):/app \
		-e OLLAMA_URL=$(OLLAMA_URL) \
		-e CONFIG_PATH=/app/$(CONFIG) \
		$(IMAGE) \
		uvicorn private_assistant.server.api:app --host 0.0.0.0 --port 8000 --reload --no-access-log

docker-logs:
	docker logs -f genai-private-assistant || true

docker-stop:
	- docker stop genai-private-assistant genai-private-assistant-dev 2>/dev/null || true

docker-sh:
	docker exec -it genai-private-assistant /bin/sh
