import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel, Field
import yaml

from app.retriever import Retriever
from app.generator import Generator
from app.scorer import SuccessScorer

CONFIG_PATH_DEFAULT = "configs/config.yaml"

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class GenerateEmailRequest(BaseModel):
    industry: str
    persona: str
    product: str
    value_prop: str
    goal: str = Field(default="book a 15-min call")
    tone: str = Field(default="consultative")
    constraints: Optional[str] = None
    company_name: Optional[str] = None
    sender_name: str = "Feyzi"
    sender_title: str = "AI/ML Engineer"
    sender_company: str = "Outcome Writer"
    include_examples: bool = False

class GenerateEmailResponse(BaseModel):
    subject: str
    body: str
    retrieved_count: int
    retrieved_examples: Optional[List[Dict[str, Any]]] = None
    success_likelihood: Optional[float] = None
    notes: List[str] = []

app = FastAPI(title="Outcome Writer API", version="0.1.0")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/v1/generate_email", response_model=GenerateEmailResponse)
def generate_email(req: GenerateEmailRequest):
    config_path = os.getenv("OUTCOME_WRITER_CONFIG", CONFIG_PATH_DEFAULT)
    cfg = load_config(config_path)

    retriever = Retriever(cfg)
    generator = Generator(cfg)
    scorer = SuccessScorer(cfg)

    examples = retriever.retrieve_success_examples(
        industry=req.industry,
        persona=req.persona,
        product=req.product,
        value_prop=req.value_prop,
        tone=req.tone,
        k=int(cfg["retrieval"]["k"]),
    )

    subject, body, notes = generator.generate(req=req.model_dump(), examples=examples)

    score = None
    if cfg.get("generation", {}).get("include_scoring", True):
        score = scorer.predict_success_probability(subject=subject, body=body)

    out = {
        "subject": subject,
        "body": body,
        "retrieved_count": len(examples),
        "success_likelihood": score,
        "notes": notes,
    }
    if req.include_examples or cfg.get("generation", {}).get("include_examples_in_output", False):
        out["retrieved_examples"] = examples
    return out
