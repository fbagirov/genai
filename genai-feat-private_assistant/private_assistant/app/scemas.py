from pydantic import BaseModel, Field
from typing import List, Optional

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None

class ChatResponse(BaseModel):
    content: str = Field(..., description="Model response text")
    model: str
    usage_tokens: Optional[int] = None
