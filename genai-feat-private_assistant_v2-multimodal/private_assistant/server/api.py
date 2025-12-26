# private_assistant/server/api.py
import os
import yaml
import httpx
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import tempfile

from private_assistant.attachments.pipeline import build_attachment_context, format_context_block


# --- Config loading (repo-root configs/config.yaml) ---
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

LLM_PROVIDER = CFG["llm"]["provider"]
MODEL = CFG["llm"]["model"]
TEMP = float(CFG["llm"].get("temperature", 0.2))
MAX_NEW = int(CFG["llm"].get("max_new_tokens", 512))


# --- Pydantic models ---
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
        "config_path": str(CONFIG_PATH),
    }


async def _ollama_generate(prompt: str, temperature: float, max_new: int) -> str:
    """
    Minimal Ollama generate call: localhost:11434/api/generate
    """
    base = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    async with httpx.AsyncClient(timeout=300) as client:
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
        return (data.get("response") or "").strip()


@app.post("/v2/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Use the last user message as prompt (simple baseline)
    prompt = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    temperature = req.temperature if req.temperature is not None else TEMP
    max_new = req.max_new_tokens if req.max_new_tokens is not None else MAX_NEW

    if LLM_PROVIDER != "ollama":
        return ChatResponse(content="Cloud providers are disabled in this minimal server. Set llm.provider: ollama.")

    text = await _ollama_generate(prompt, temperature, max_new)
    return ChatResponse(content=text if text else "(no response)")


@app.post("/v2/chat_with_files")
async def chat_with_files(
    message: str = Form(...),
    files: List[UploadFile] = File(default=[]),
    temperature: Optional[float] = Form(None),
    max_new_tokens: Optional[int] = Form(None),
):
    """
    Multi-modal (local) endpoint:
      - Extract text from PDFs/DOCX/TXT
      - OCR images
      - Inject extracted content into the prompt
    """
    if LLM_PROVIDER != "ollama":
        return {"answer": "Cloud providers are disabled. Set llm.provider: ollama.", "attachments_used": []}

    temperature = float(temperature) if temperature is not None else TEMP
    max_new = int(max_new_tokens) if max_new_tokens is not None else MAX_NEW

    temp_paths: List[Path] = []
    try:
        # Save uploads to temp
        for f in files:
            suffix = Path(f.filename).suffix
            fd, tmp = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            tmp_path = Path(tmp)
            content = await f.read()
            tmp_path.write_bytes(content)
            temp_paths.append(tmp_path)

        contexts = build_attachment_context(temp_paths, CFG)
        ctx_block = format_context_block(contexts)

        # Inject attachments context into prompt
        prompt = message.strip()
        if ctx_block:
            prompt = f"{ctx_block}\n\nUSER QUESTION:\n{prompt}"

        answer = await _ollama_generate(prompt, temperature, max_new)

        return {
            "answer": answer if answer else "(no response)",
            "attachments_used": [c.source for c in contexts],
        }

    finally:
        # delete temp files (privacy-first)
        for p in temp_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
