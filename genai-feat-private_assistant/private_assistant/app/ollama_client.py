import httpx
from typing import List, Dict
from .settings import get_settings

class OllamaClient:
    def __init__(self):
        cfg = get_settings()
        self.model = cfg["model"]["name"]
        self.temperature = cfg["model"]["temperature"]
        self.max_tokens = cfg["model"]["max_tokens"]
        self.top_p = cfg["model"]["top_p"]
        self.url = "http://localhost:11434/api/chat"  # Ollama default

    async def chat(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "top_p": self.top_p
            },
            "stream": False
        }
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(self.url, json=payload)
            r.raise_for_status()
            data = r.json()
            # Ollama returns {"message":{"role":"assistant","content":"..."}}
            return data.get("message", {}).get("content", "")
