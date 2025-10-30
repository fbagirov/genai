import argparse
import os
import time
import yaml
import json
from pydantic import BaseModel
from typing import Dict, Any, List


from retriever.hybrid import HybridRetriever
from openai import OpenAI
import httpx

class Config(BaseModel):
    embedding: Dict[str, Any]
    llm: Dict[str, Any]
    retrieval: Dict[str, Any]
    reranker: Dict[str, Any]
    paths: Dict[str, str]

def read_config(path: str) -> Config:
    with open(path, "r") as f:
        return Config(**yaml.safe_load(f))

def format_citations(hits: List[Dict[str, Any]]) -> str:
    # Deduplicate / compress per doc
    buckets: Dict[str, List[int]] = {}
    for h in hits:
        m = h["metadata"]
        key = f"{m['doc_id']}"
        buckets.setdefault(key, []).append(m.get("page", 0))
    parts = []
    for doc, pages in buckets.items():
        pages = sorted(set(pages))
        if len(pages) <= 4:
            pg = ",".join(str(p) for p in pages)
        else:
            pg = f"{pages[0]}–{pages[-1]}"
        parts.append(f"{doc}:{pg}")
    return "; ".join(parts)

# def build_prompt(question: str, hits: List[Dict[str, Any]]) -> str:
#     context = "\n\n".join([f"[Source {i+1}]\n{h['text']}" for i, h in enumerate(hits)])
#     return f"""
# You are a careful assistant. Answer the user's question **using only the context** below.
# If the answer is not contained in the context, say you don't know. Include inline citations like [doc_id:page] where relevant.

# Context:\n{context}

# Question: {question}
# Answer:
# """.strip()

def build_prompt(question: str, hits: list[dict]) -> str:
    # Build short, provenance-tagged blocks so citations are obvious
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h["metadata"]; doc = m.get("doc_id","?"); pg = m.get("page","?")
        blocks.append(f"[{i}] {doc} p.{pg}\n{h['text']}")
    context = "\n\n".join(blocks)

    return f"""
        Answer the user's question using ONLY the context below.
        - Write ONE concise sentence that states the repository's purpose if the context indicates it.
        - Include at least one inline citation like [doc_id:page].
        - If and only if the context is unrelated, say "I don't know."

        Context:
        {context}

        Question: {question}
        Answer:
        """.strip()

def call_llm_ollama(model: str, prompt: str, temperature: float = 0.2) -> str:
    url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {
            "temperature": 0.0,   # be decisive
            "top_p": 1.0,
            "top_k": 50,
            "repeat_penalty": 1.05
        },
        "stream": False
    }
    with httpx.Client(timeout=None) as client:
        with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            out = []
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    j = json.loads(line)
                    if "response" in j:
                        out.append(j["response"])
                except json.JSONDecodeError:
                    continue
            return "".join(out).strip()
    
# If you are using open ai
# def call_llm_openai(model: str, prompt: str, temperature: float = 0.2) -> str:
#     client = OpenAI()
#     resp = client.chat.completions.create(
#         model=model,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=temperature,
#         )
#     return resp.choices[0].message.content

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--use_reranker", type=bool, default=None)
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = read_config(args.config)
    if args.use_reranker is not None:
        cfg.reranker["enabled"] = bool(args.use_reranker)

    retriever = HybridRetriever(cfg)

    t0 = time.time()
    hits, conf = retriever.retrieve(
        query=args.question,
        k_dense=cfg.retrieval["k_dense"],
        k_bm25=cfg.retrieval["k_bm25"],
        w_dense=cfg.retrieval["weight_dense"],
        w_bm25=cfg.retrieval["weight_bm25"],
        topk=args.k,
    )
    t_retrieve = time.time() - t0

    prompt = build_prompt(args.question, hits)

    t1 = time.time()
    if cfg.llm["provider"] == "ollama":
        answer = call_llm_ollama(cfg.llm["model"], prompt, cfg.llm.get("temperature", 0.2))
    else:
        answer = call_llm_openai(cfg.llm["model"], prompt, cfg.llm.get("temperature", 0.2))
    t_generate = time.time() - t1

    cites = format_citations(hits)

    print("\n=== ANSWER ===\n")
    print(answer.strip())
    print("\n---")
    print(f"Citations: {cites}")
    print(f"Confidence (retrieval): {conf:.3f}")
    print(f"Latency: retrieve={t_retrieve:.2f}s, generate={t_generate:.2f}s")




if __name__ == "__main__":
    main()