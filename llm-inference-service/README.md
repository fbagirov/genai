
# llm-inference-service

**Goal:** A production-ready inference microservice with **FastAPI**, optional upstream **vLLM/Triton**, **Prometheus** metrics, **OpenTelemetry** traces, **Docker + K8s** manifests (with **HPA**), and a **bench harness** (Locust).

> Supports OpenAI-compatible upstreams (e.g., vLLM's `/v1/completions` or `/v1/chat/completions`) via `VLLM_BASE_URL`.

---

## Features
- **FastAPI** gateway with `/v1/generate`, `/healthz`, `/metrics` (Prometheus)
- **Observability**: Prometheus counters/histograms + OpenTelemetry tracing (OTLP)
- **Config**: `configs/config.yaml` (provider mode, timeouts, defaults)
- **Docker** & **Kubernetes**: Deployment/Service/HPA + Prom scrape annotations
- **Bench**: Locust load test with p95 latency and tokens/sec
- **CI**: GitHub Action (lint, tests, build)

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# Upstream (example): run vLLM OpenAI server separately (choose your model)
# pip install vllm; python -m vllm.entrypoints.openai.api_server --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --port 8001
# Or point to an existing OpenAI-compatible endpoint.

export VLLM_BASE_URL="http://127.0.0.1:8001/v1"    # Windows PS: $env:VLLM_BASE_URL="http://127.0.0.1:8001/v1"
export MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Run the gateway
uvicorn server.main:app --host 0.0.0.0 --port 8080 --workers 1
```

### Test
```bash
curl -s http://127.0.0.1:8080/healthz
curl -s -X POST http://127.0.0.1:8080/v1/generate   -H "Content-Type: application/json"   -d '{"prompt":"Write one short sentence.", "max_new_tokens":64}'
```

### Metrics
```
curl -s http://127.0.0.1:8080/metrics | head
```

---

## Bench (Locust)
```bash
# in one terminal
uvicorn server.main:app --port 8080

# in another
locust -f scripts/locustfile.py --users 50 --spawn-rate 10 --host http://127.0.0.1:8080
```
Open http://127.0.0.1:8089 for charts. Inspect p95 latency and throughput (req/s, tokens/s).

---

## Docker
```bash
docker build -t yourrepo/llm-inference-service:latest .
docker run --rm -p 8080:8080 \
  -e VLLM_BASE_URL=http://host.docker.internal:8001/v1 \
  -e MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  yourrepo/llm-inference-service:latest
```

Optional docker-compose with a vLLM upstream is easy to add.

---

## ☸️ Kubernetes
```bash
kubectl apply -f k8s/
kubectl get pods -l app=llm-inference-service
```
Manifests include **HPA** (CPU-based) and **Prometheus scrape** annotations.

---

## 🔭 Grafana
Import `grafana/dashboard.json` in Grafana (set your Prometheus datasource). Panels show:
- RPS, error rate
- p95 latency (histogram quantile)
- tokens/sec

---

## ⚙️ Config (`configs/config.example.yaml`)
- `server.host`, `server.port` — where this service listens
- `upstream.base_url` — OpenAI-compatible base URL (e.g., vLLM)
- `inference.*` — default generation params
- `timeouts.*` — HTTP timeouts
- `otel.*` — OTLP endpoint & service name

Copy to `configs/config.yaml` and edit or override via env.

---

## 🧰 Limits & Notes
- vLLM/Triton run **outside** this gateway; point `VLLM_BASE_URL` to them.
- Ensure models fit your GPU/CPU; tune max tokens/batch.
- Use one process per pod and scale via HPA; prefer **k8s horizontal scaling** over multi-workers if you need Prometheus multiproc.

---

## 📁 Repo structure
```
llm-inference-service/
  server/
    main.py
    settings.py
  configs/
    config.example.yaml
  grafana/
    dashboard.json
  k8s/
    deployment.yaml
    service.yaml
    hpa.yaml
  scripts/
    locustfile.py
  tests/
    test_health.py
  .github/workflows/ci.yml
  Dockerfile
  Makefile
  requirements.txt
  README.md
  .env.example
  .gitignore
```
