# private_assistant/server/api.py
import os
import yaml
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

# Load config (default to package path)
CONFIG_PATH = os.getenv("CONFIG_PATH", "private_assistant/configs/config.yaml")
CFG = yaml.safe_load(open(CONFIG_PATH, "r"))

LLM_PROVIDER = CFG["llm"]["provider"]
MODEL = CFG["llm"]["model"]
TEMP = float(CFG["llm"].get("temperature", 0.2))
MAX_NEW = int(CFG["llm"].get("max_new_tokens", 512))

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_new_tokens: Optional[int] = None

class ChatResponse(BaseModel):
    content: str

app = FastAPI(title="Private Assistant (Local)")

@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "provider": LLM_PROVIDER,
        "model": MODEL,
        "config": CONFIG_PATH,
    }

@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # very simple: use the last user message as the prompt
    prompt = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    temperature = req.temperature if req.temperature is not None else TEMP
    max_new = req.max_new_tokens if req.max_new_tokens is not None else MAX_NEW

    if LLM_PROVIDER != "ollama":
        return ChatResponse(content="Cloud providers are disabled in this minimal server. Set llm.provider: ollama.")

    base = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{base}/api/generate",
            json={
                "model": MODEL,
                "prompt": prompt,
                "options": {"temperature": temperature, "num_predict": max_new},
                "stream": False,
            },
        )
        r.raise_for_status()
        data = r.json()
        text = (data.get("response") or "").strip()
        return ChatResponse(content=text if text else "(no response)")
