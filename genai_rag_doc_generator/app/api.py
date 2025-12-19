"""
FastAPI service for Outcome Writer:
Generates sales emails conditioned on historically successful examples.

Architecture:
- TF-IDF retrieval over "won" emails
- Simple generation logic (mock / placeholder for LLM)
- Optional outcome scoring stub
"""

import yaml
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List

from app.retriever_tfidf import TfidfRetriever


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

CONFIG_PATH = "configs/config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)


# ---------------------------------------------------------------------
# Initialize components
# ---------------------------------------------------------------------

retriever = TfidfRetriever(
    data_path=CFG["data"]["dataset_path"]
)

app = FastAPI(
    title="Outcome Writer API",
    description="Generate sales emails conditioned on previously successful examples",
    version="0.1.0",
)


# ---------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------

class GenerateEmailRequest(BaseModel):
    goal: str
    customer_context: Optional[str] = ""
    tone: Optional[str] = "professional"
    k_examples: Optional[int] = 3


class GenerateEmailResponse(BaseModel):
    generated_email: str
    retrieved_examples: List[str]


# ---------------------------------------------------------------------
# Helper: simple generation logic (LLM placeholder)
# ---------------------------------------------------------------------

def generate_email(
    goal: str,
    context: str,
    examples: List[str],
    tone: str,
) -> str:
    """
    Simple template-based generator.
    Replace this with an LLM call later.
    """
    header = f"Subject: {goal}\n\n"
    body = f"Hi,\n\n"

    if context:
        body += f"I hope this message finds you well. Based on your context ({context}), "

    body += (
        f"I wanted to reach out regarding {goal.lower()}.\n\n"
        f"Below are proven patterns from similar successful outreach:\n\n"
    )

    for i, ex in enumerate(examples, 1):
        body += f"Example {i}:\n{ex}\n\n"

    body += (
        "Using these principles, I believe we can move forward effectively.\n\n"
        "Best regards,\n"
        "Your Name"
    )

    return header + body


# ---------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------

@app.post("/v1/generate_email", response_model=GenerateEmailResponse)
def generate_email_endpoint(req: GenerateEmailRequest):
    """
    Generate a sales email using retrieved successful examples.
    """

    # Retrieve similar successful emails
    examples = retriever.retrieve(
        query=req.goal + " " + (req.customer_context or ""),
        k=req.k_examples,
    )

    # Generate new email
    email = generate_email(
        goal=req.goal,
        context=req.customer_context or "",
        examples=examples,
        tone=req.tone,
    )

    return GenerateEmailResponse(
        generated_email=email,
        retrieved_examples=examples,
    )


# ---------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}
