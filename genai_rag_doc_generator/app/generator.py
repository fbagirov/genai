import os
import re
from typing import Dict, Any, List, Tuple

def _redact_emails(text: str) -> str:
    return re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[REDACTED_EMAIL]", text)

class Generator:
    # Provider toggle:
    # - mock: offline template generator (always runnable)
    # - openai: uses OpenAI (requires OPENAI_API_KEY)
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.provider = cfg.get("generation", {}).get("provider", "mock")
        self.temperature = float(cfg.get("generation", {}).get("temperature", 0.3))
        self.max_new_tokens = int(cfg.get("generation", {}).get("max_new_tokens", 320))
        self.redact = bool(cfg.get("privacy", {}).get("redact_emails_in_logs", True))

    def _build_prompt(self, req: Dict[str, Any], examples: List[Dict[str, Any]]) -> str:
        header = (
            "Write a concise, ethical B2B sales email.
"
            "Use the successful examples as style reference, but do not copy.
"
            "Return:
"
            "Subject: <one line>
"
            "<blank line>
"
            "<email body>
"
        )
        ctx = ""
        for i, ex in enumerate(examples[:5], start=1):
            ctx += f"\n--- Successful Example {i} ---\nSubject: {ex.get('subject','')}\n{ex.get('body','')}\n"

        target = (
            f"\n--- Target Scenario ---\n"
            f"Industry: {req['industry']}\nPersona: {req['persona']}\n"
            f"Product: {req['product']}\nValue prop: {req['value_prop']}\n"
            f"Tone: {req.get('tone','consultative')}\nGoal: {req.get('goal','book a 15-min call')}\n"
        )
        if req.get("company_name"):
            target += f"Prospect company: {req['company_name']}\n"
        if req.get("constraints"):
            target += f"Constraints: {req['constraints']}\n"
        target += f"Sender: {req.get('sender_name','Feyzi')} ({req.get('sender_title','')}, {req.get('sender_company','')})\n"

        prompt = header + ctx + target
        return _redact_emails(prompt) if self.redact else prompt

    def _mock_generate(self, req: Dict[str, Any], examples: List[Dict[str, Any]]) -> Tuple[str, str, List[str]]:
        tone = (req.get("tone") or "consultative").lower()
        industry = req["industry"]
        persona = req["persona"]
        product = req["product"]
        value_prop = req["value_prop"]
        company = req.get("company_name") or "your team"

        subject = f"{product}: idea for {company}"
        opener = {
            "executive": "I’ll keep this brief.",
            "direct": "Quick note—",
            "friendly": "Hope you’re doing well.",
            "warm": "Hope your week is going well.",
            "consultative": "Not sure if this is on your radar, but",
        }.get(tone, "Quick note—")

        body = f"""{opener}

I’m reaching out because teams in {industry} are using {product} to improve {value_prop}—especially for leaders like a {persona}.

If helpful, I can share a short 2-minute walkthrough and a low-lift pilot plan so you can validate fit quickly.

Would you be open to a 15-minute call to see whether this is relevant for {company}?

Best,
{req.get('sender_name','Feyzi')}
"""
        notes = []
        notes.append(f"Conditioned on {min(len(examples),5)} retrieved successful example(s)." if examples else "No close matches retrieved; draft is generic.")
        return subject.strip(), body.strip(), notes

    def _openai_generate(self, prompt: str) -> Tuple[str, str]:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = self.cfg.get("generation", {}).get("openai_model", "gpt-4o-mini")

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You write high-converting, ethical B2B sales emails."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        text = resp.choices[0].message.content or ""
        subject = "Quick question"
        body = text.strip()
        for line in text.splitlines():
            if line.lower().startswith("subject:"):
                subject = line.split(":", 1)[1].strip() or subject
                body = "\n".join(text.splitlines()[1:]).strip()
                break
        return subject, body

    def generate(self, req: Dict[str, Any], examples: List[Dict[str, Any]]) -> Tuple[str, str, List[str]]:
        prompt = self._build_prompt(req, examples)
        if self.provider == "openai" and os.getenv("OPENAI_API_KEY"):
            subject, body = self._openai_generate(prompt)
            return subject, body, [f"Generated via OpenAI model={self.cfg.get('generation', {}).get('openai_model','gpt-4o-mini')}"]
        if self.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            subject, body, notes = self._mock_generate(req, examples)
            notes.append("OPENAI_API_KEY not set; fell back to provider=mock.")
            return subject, body, notes
        return self._mock_generate(req, examples)
