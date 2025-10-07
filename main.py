import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


# ETL


def load_gutenberg_alice(download: bool = True, pasted_text: Optional[str] = None) -> str:
    """
    Returns a cleaned text of 'Alice’s Adventures in Wonderland' (public domain).
    Either downloads from Gutenberg or uses pasted_text if provided.
    """
    if pasted_text:
        raw = pasted_text
    elif download:
        import requests
        URL = "https://www.gutenberg.org/files/11/11-0.txt"
        resp = requests.get(URL, timeout=60)
        resp.raise_for_status()
        raw = resp.text
    else:
        raise ValueError("Provide pasted_text or set download=True.")

    # Strip Gutenberg header/footer if present
    start = re.search(r"\*\*\* START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK .* \*\*\*", raw, re.I)
    end   = re.search(r"\*\*\* END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK .* \*\*\*", raw, re.I)
    text  = raw[start.end():end.start()] if (start and end) else raw

    # Normalize whitespace
    text  = re.sub(r"\r\n?", "\n", text)
    text  = re.sub(r"[ \t]+", " ", text).strip()
    return text


# CHUNKING
## Chunking is better due to: 
## - Search works better on small passages
## - LLM prompts are token-limited
## - prcise ciations are easier


def split_into_chunks(text: str, n: int = 1200, overlap: int = 200) -> List[str]:
    """
    Simple word-window chunker with overlap. Adjust n/overlap to your LLM/prompt size.
    """
    words = text.split(" ")
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+n]).strip()
        if chunk:
            chunks.append(chunk)
        i += max(1, n - overlap)
    return chunks



# EMBEDDINGS (sentence-transformers)


# Lazy-load heavy libs on first use for faster startup

_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embed_model

def build_embeddings(chunks: List[str]) -> np.ndarray:
    model = get_embed_model()
    embs = model.encode(chunks, batch_size=64, show_progress_bar=True, normalize_embeddings=False)
    return embs.astype(np.float32)


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, eps)



# RETRIEVAL (cosine sim with NumPy)


def top_k_search(query: str, df: pd.DataFrame, embeddings_norm: np.ndarray, k: int = 4) -> List[Dict]:
    model = get_embed_model()
    q = model.encode([query], normalize_embeddings=False)[0].astype(np.float32)
    q = q / max(np.linalg.norm(q), 1e-12)
    sims = embeddings_norm @ q
    idx = np.argsort(-sims)[:k]
    results = []
    for i in idx:
        results.append({
            "path":  df.iloc[i]["path"],
            "chunk": df.iloc[i]["chunk"],
            "score": float(sims[i])
        })
    return results


# GENERATION (FLAN-T5)
## for CPU we use FLAN-T5
## for stronger outputs switch to flan-t5-large (requires more RAM) or use other LLM that you prefer.

_gen = None
_tok = None

def get_generator():
    global _gen, _tok
    if _gen is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
        gen_model_name = "google/flan-t5-base"  # try 'small' if RAM is tight, 'large' if you have it
        _tok = AutoTokenizer.from_pretrained(gen_model_name)
        mdl  = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)
        _gen = pipeline("text2text-generation", model=mdl, tokenizer=_tok)
    return _gen

def build_context(passages: List[Dict]) -> str:
    lines = []
    for i, p in enumerate(passages, 1):
        lines.append(f"[{i}] {p['chunk']}\n(Source: {p['path']})")
    return "\n\n".join(lines)

def rag_answer(question: str, df: pd.DataFrame, embeddings_norm: np.ndarray,
               k: int = 4, max_new_tokens: int = 200) -> Tuple[str, List[Dict]]:
    hits = top_k_search(question, df, embeddings_norm, k=k)
    context = build_context(hits)
    prompt = (
        "You are a helpful assistant. Use ONLY the context to answer. "
        "If unsure, say you don't know. Cite sources as [1], [2], etc.\n\n"
        f"Question: {question}\n\nContext:\n{context}\n\nAnswer (with citations):"
    )
    gen = get_generator()
    out = gen(prompt, max_new_tokens=max_new_tokens, temperature=0.2, do_sample=False)[0]["generated_text"]
    return out, hits


# GRADIO UI
## UI that is more fun


def run_ui(df: pd.DataFrame, embeddings_norm: np.ndarray, port: int = 7860):
    import gradio as gr
    def ui_answer(q):
        if not q or not q.strip():
            return "Please enter a question."
        a, _ = rag_answer(q, df, embeddings_norm, k=4)
        return a

    demo = gr.Interface(fn=ui_answer,
                        inputs=gr.Textbox(label="Ask a question about Alice in Wonderland"),
                        outputs="text",
                        title="RAG over Alice in Wonderland")
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)


# CLI entrypoint


def main():
    # 1) Load/clean text
    text = load_gutenberg_alice(download=True)

    # 2) Chunk
    chunks = split_into_chunks(text, n=1200, overlap=200)
    df = pd.DataFrame({"path": [f"alice/{i:05d}" for i in range(len(chunks))], "chunk": chunks})

    # 3) Embeddings
    embeddings = build_embeddings(df["chunk"].tolist())
    embeddings_norm = l2_normalize(embeddings)

    # 4) Quick CLI Q&A
    print("RAG ready. Type a question (or 'ui' to launch web UI, or 'exit'):")
    while True:
        q = input("> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if q.lower() == "ui":
            print("Launching local UI at http://localhost:7860 ...")
            run_ui(df, embeddings_norm, port=7860)
            continue
        if not q:
            continue
        ans, ctx = rag_answer(q, df, embeddings_norm, k=4)
        print("\nAnswer:\n", ans, "\n")
        print("Citations:")
        for i, p in enumerate(ctx, 1):
            print(f"[{i}] {p['path']} (score={p['score']:.3f})")
        print("-" * 60)

if __name__ == "__main__":
    main()
