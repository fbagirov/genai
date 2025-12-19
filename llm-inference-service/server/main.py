
import os, time, asyncio
from typing import Optional
from fastapi import FastAPI, APIRouter, Response, Request, HTTPException
from pydantic import BaseModel, Field
import httpx

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from loguru import logger

from server.settings import load_settings

# ---------------- Prometheus metrics ----------------
REQUESTS = Counter("inference_requests_total", "Total /v1/generate requests", ["status_code","model"])
TOKENS = Counter("tokens_generated_total", "Total tokens generated", ["model"])
INFLIGHT = Gauge("inference_inflight_requests", "In-flight requests")
LATENCY = Histogram(
    "inference_request_latency_seconds",
    "Latency of /v1/generate",
    buckets=[0.05,0.1,0.25,0.5,1,2,3,5,8,13]
)

# ---------------- Models ----------------
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Plain text prompt")
    max_new_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.2
    model: Optional[str] = None

class GenerateResponse(BaseModel):
    output: str
    model: str
    tokens: Optional[int] = None
    latency_sec: float

# ---------------- App ----------------
cfg = load_settings()
app = FastAPI(title="LLM Inference Service", version="1.0.0")
api = APIRouter()

# ------------- OpenTelemetry (optional) -------------
try:
    if cfg.otel.enabled:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        resource = Resource.create({"service.name": cfg.otel.service_name})
        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=cfg.otel.otlp_endpoint))
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        logger.info("OpenTelemetry tracing enabled: {}", cfg.otel.otlp_endpoint)
except Exception as e:
    logger.warning("Failed to init OpenTelemetry: {}", e)

# ---------------- Helpers ----------------
def upstream_headers():
    return {"Content-Type":"application/json"}

def parse_tokens_from_openai(resp_json) -> Optional[int]:
    usage = resp_json.get("usage")
    if usage and "completion_tokens" in usage:
        return int(usage["completion_tokens"])
    # Try to approximate from text (fallback)
    try:
        txt = resp_json.get("choices",[{}])[0].get("text") or resp_json.get("choices",[{}])[0].get("message",{}).get("content","")
        return max(1, len(txt.split()))
    except Exception:
        return None

async def call_upstream(payload: dict) -> dict:
    timeout = httpx.Timeout(connect=cfg.timeouts.connect, read=cfg.timeouts.read)
    async with httpx.AsyncClient(timeout=timeout, base_url=cfg.upstream.base_url) as client:
        if cfg.upstream.mode == "completions":
            r = await client.post("/completions", json=payload, headers=upstream_headers())
        else:
            # chat mode
            messages = [{"role":"user","content": payload.get("prompt","")}]
            chat_payload = {
                "model": payload.get("model"),
                "messages": messages,
                "max_tokens": payload.get("max_tokens"),
                "temperature": payload.get("temperature"),
            }
            r = await client.post("/chat/completions", json=chat_payload, headers=upstream_headers())
        r.raise_for_status()
        return r.json()

# ---------------- Routes ----------------
@api.get("/healthz")
async def healthz():
    return {"status":"ok", "upstream": cfg.upstream.base_url, "mode": cfg.upstream.mode}

@api.post("/v1/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, request: Request):
    model = req.model or cfg.inference.model_name
    payload = {
        "model": model,
        "prompt": req.prompt,
        "max_tokens": req.max_new_tokens or cfg.inference.max_new_tokens,
        "temperature": req.temperature if req.temperature is not None else cfg.inference.temperature,
    }

    INFLIGHT.inc()
    t0 = time.time()
    status = "200"
    try:
        upstream = await call_upstream(payload)
        # Extract output text
        text = ""
        if "choices" in upstream and upstream["choices"]:
            # completions: text; chat: message.content
            text = upstream["choices"][0].get("text") or upstream["choices"][0].get("message",{}).get("content","")
        tokens = parse_tokens_from_openai(upstream)
        dt = time.time() - t0
        LATENCY.observe(dt)
        REQUESTS.labels(status_code=status, model=model).inc()
        if tokens:
            TOKENS.labels(model=model).inc(tokens)
        return GenerateResponse(output=text, model=model, tokens=tokens, latency_sec=round(dt,3))
    except httpx.HTTPStatusError as e:
        status = str(e.response.status_code)
        REQUESTS.labels(status_code=status, model=model).inc()
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        status = "500"
        REQUESTS.labels(status_code=status, model=model).inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        INFLIGHT.dec()

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

app.include_router(api)
