
import os, yaml
from pydantic import BaseModel

DEFAULT_CONFIG_PATH = os.getenv("CONFIG_PATH", "configs/config.yaml")

class Timeouts(BaseModel):
    connect: float = 5.0
    read: float = 60.0

class InferenceDefaults(BaseModel):
    model_name: str = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    max_new_tokens: int = 256
    temperature: float = 0.2

class Upstream(BaseModel):
    base_url: str = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1")
    mode: str = "completions"  # completions | chat

class Server(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080

class OTel(BaseModel):
    enabled: bool = False
    service_name: str = "llm-inference-service"
    otlp_endpoint: str = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")

class Settings(BaseModel):
    server: Server = Server()
    upstream: Upstream = Upstream()
    inference: InferenceDefaults = InferenceDefaults()
    timeouts: Timeouts = Timeouts()
    otel: OTel = OTel()

def load_settings() -> Settings:
    path = DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        # Try example
        path = "configs/config.example.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # Merge env overrides via Pydantic defaults above
    return Settings(**data)
