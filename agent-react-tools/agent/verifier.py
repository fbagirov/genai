
from typing import Dict, Any, List

class Verifier:
    def __init__(self, cfg: Dict[str, Any]):
        self.max_len = int(cfg.get("verifier", {}).get("max_answer_chars", 2000))
        self.require_citation = bool(cfg.get("verifier", {}).get("require_citation_for_web", True))

    def should_stop(self, history: List[str]) -> bool:
        return any(h.startswith("FINAL:") for h in history)

    def enforce(self, final_answer: str, used_web: bool) -> str:
        text = final_answer.strip()
        if self.require_citation and used_web:
            if "[source:" not in text.lower():
                text += "\n\n[source: web_search]"
        if len(text) > self.max_len:
            text = text[: self.max_len] + "..."
        return text
