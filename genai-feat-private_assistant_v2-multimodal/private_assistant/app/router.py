from fastapi import APIRouter, HTTPException
from .schemas import ChatRequest, ChatResponse
from .settings import get_settings
from .privacy import maybe_scrub, guard_input
from .ollama_client import OllamaClient

router = APIRouter()
client = OllamaClient()
cfg = get_settings()

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Privacy: sanitize input (optional) + no logging by default
    sanitized = []
    for m in req.messages:
        content = maybe_scrub(m.content)
        sanitized.append({"role": m.role, "content": content})
    guard_input({"messages": [m.dict() for m in req.messages]})

    try:
        out = await client.chat(sanitized)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(content=out, model=client.model)
