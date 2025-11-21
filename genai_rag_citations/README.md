# genai-rag-citations

# genai-rag-citations


A compact RAG system that ingests local PDFs, builds **hybrid search** (BM25 + dense), and answers questions with **inline citations** and a **confidence score**. Includes **RAGAS** evaluation, simple latency profiling, and optional reranking.


## Features
- Ingestion from PDFs/markdown/txt via `pypdf` & simple loaders
- Embeddings with `sentence-transformers` (default: `intfloat/e5-base-v2`) or OpenAI (toggle in `configs/config.yaml`)
- Local vector store via **Chroma**
- Sparse retriever via **rank_bm25**
- Optional reranker via **bge-reranker-base** (FlagEmbedding)
- Generator via **Ollama** (default, offline) or OpenAI (toggle)
- Inline citation formatting `[doc_id:page]` and confidence
- **RAGAS** eval (faithfulness, answer_relevancy, context_precision, context_recall)
- `Makefile` tasks + `Dockerfile` + `pytest` smoke tests


## Quickstart Demo
```bash
# 1) Python env
uv venv .venv # or: python -m venv .venv
source .venv/bin/activate # or: .venv/Script/Activate.ps1 (for Windows PowerShell) 

# 2) Install deps
pip install -r requirements.txt


# 3) (Optional) Start Ollama & pull a model
# https://ollama.com/download
ollama pull llama3:instruct


# 4) Configure
cp configs/config.yaml
# Edit configs/config.yaml to pick embedding/LLM providers (Ollama or licensed OpenAI)


# 5) Ingest a few PDFs/texts
python ingest/ingest_docs.py --input_dir sample_docs --persist_dir .chroma


# 6) Ask questions
python rag_pipeline.py --question "What are the main topics?" --k 6 --use_reranker false
```

--k 6 means "retrieve top 6 relevant chunks"

The reason reranker above is set to false is because the sstem fetches a batch of top-K chunks from the corpus using approximate similarity measures: 
 - Dense retrieval (compares embedding vectors on semantic similarity)
 - sparse retrieval (BM25) (counts overlapping words and phrases)
 While these methods are fast, they score chunks independently and don't read the actual text of both question and chunk together. 
 Reranking re-reads those top candidates one by one and takes both query and each chunk. It then recomputes a deep similarity score by reading them jointly and sorts them again by a better score. 

The reason I set to false by default is because it is computationally expenseive (cross-encodes once per candidate chunk). On CPU that can take several seconds per query, on GPU a bit faster, but still has latency. You could re-enable (apply cross-encoder) to improve accuracy on long documents/your passages are irrelevant, but for demo purposes, speed outweights accuracy or you are running it on CPU. 


```bash
# 7) Evaluate with RAGAS (example dataset under eval/datasets)
python eval/ragas_eval.py --dataset_path eval/datasets/sample_eval.jsonl
```


# To add your documents
1. Add a new folder (private_docs) 
2. re-run step 5 (make sure you are not including scanned PDFs. If you do, it's better to copy and past the text into a txt and save it to the folder) 
3. re-run step 6 with your own question

## Architecture
```
[Loaders] -> [TextSplitter] -> [Embeddings] -> [Chroma]
| ^
v |
[BM25 Index] ------------------+


Query -> HybridRetriever (BM25 + Dense [+ optional reranker]) -> Generator(LLM) ->
Inline citations + confidence
```

# Using Docker

Build the container

bash ```
docker build -t genai-rag-citations:latest .
```

Run the app 
bash ```
docker run --rm -p 7860:7860 \
  -e OLLAMA_URL=http://host.docker.internal:11434 \
  -e CONFIG_PATH=configs/config.yaml \
  genai-rag-citations:latest

```

## Flow Diagram
```

                    ┌────────────────────────────┐
                    │         Your Query         │
                    └────────────┬───────────────┘
                                 │
                     ┌───────────▼───────────┐
                     │  Hybrid Retriever     │
                     │  (Dense + BM25)       │
                     └───────────┬───────────┘
                                 │
              Top-K results (e.g., 6) ────────────────┐
                                 │                   │
                                 │                   │
                  (optional)     ▼                   │
                  ┌──────────────────────────┐        │
                  │   Cross-Encoder Reranker │        │
                  │ (e.g., bge-reranker-base)│        │
                  └───────────┬──────────────┘        │
                              │                      │
                   Re-ordered & rescored chunks      │
                              │                      │
                              ▼                      │
                 ┌────────────────────────────┐       │
                 │  Prompt Builder (citations)│◄──────┘
                 └───────────┬────────────────┘
                             │
                             ▼
                 ┌────────────────────────────┐
                 │       LLM Generator        │
                 │ (Ollama / OpenAI model)    │
                 └───────────┬────────────────┘
                             │
                             ▼
                 ┌────────────────────────────┐
                 │ Final Answer w/ Citations  │
                 └────────────────────────────┘
```


## Config toggles (`configs/config.yaml`)

### Ingestion
- input_dir / include_glob – which files are ingested. Change to point at your own corpus; re-run ingest when you add/update docs.
- chunk_size – how big each document chunk is.
  - Smaller (300–600) → better recall/grounding, more chunks → higher latency.
  - Larger (800–1200) → fewer chunks (faster), but risk mixing topics in one chunk.
- chunk_overlap – keeps context continuity across chunks. 10–20% of chunk_size is a good start; too large slows indexing & increases redundancy.
- splitter – how text is split. Sentence-aware splitters improve semantic integrity; recursive often balances size/semantics.
- add_metadata.filename/page_number – needed for citations. Keep true to show source info like “(Doc.pdf, p. 12)”.

### Embeddings
- provider / model_name – dense vector model. e5/bge “base” → good general quality; “small” → faster; “large” → better recall but heavier. If you change models, you must rebuild the index (vectors are not compatible).
- device – cuda accelerates both ingestion and query.
- normalize – L2-normalization for cosine similarity. Usually set to true for e5/bge; keep consistent between ingestion and query.
- batch_size – ingestion/query throughput. Increase on GPU; reduce on CPU to avoid OOM.

### Vector store
- type – storage backend. Chroma = easy local; pgvector = Postgres; FAISS = in-memory or file-based speed.
- persist_dir – where the index lives on disk (e.g., .chroma). Moving it or deleting it changes which dataset you’re querying. Keep it out of git.
- collection_name – useful when hosting multiple corpora.

### BM25 (sparse retriever)
- enabled – toggles lexical search. Enabling BM25 often boosts recall (especially for exact phrases/acronyms).
- index_path – the serialized BM25 index (e.g., bm25.pkl). Regenerate when docs change (your ingest script likely does this).

### Retriever
- mode – "dense", "sparse", or "hybrid". Hybrid usually performs best; pure dense can miss rare terms; pure sparse can miss paraphrases.
- k – how many top chunks you send to the generator. Raise k (8–12) for safety-sensitive/long docs; lower (3–6) for latency-critical paths.
- fetch_k – how many candidates to fetch before re-ranking. Set to a multiple of k (e.g., 3×–5×) if using a reranker.
- alpha – blending weight in hybrid search (0=sparse-only, 1=dense-only). Tune on a dev set via RAGAS; e.g., 0.4–0.6 is a typical sweet spot.

### Reranker
- use_reranker – cross-encoder refinement. Increase quality, especially when initial recall is noisy; Increase latency & compute. You previously asked: --use_reranker false skips this to improve speed.
- model_name – e.g., BAAI/bge-reranker-base or cross-encoder/ms-marco-*. Base models are decent; large models are slower but a bit better.
- top_n – how many chunks to keep after re-ranking (usually equal to retriever k).
- device – use GPU if available for big gains.

### Generator (LLM)
- provider / model – where answers are synthesized (Ollama local vs OpenAI). Local models (llama3/mistral) preserve privacy; OpenAI can be stronger but requires key.
- temperature – randomness in answer generation. Keep low (0–0.3) for factual RAG—less risk of hallucination.
- max_new_tokens – upper bound of answer length. Increase for long-form answers; keep modest for latency & cost control.
- prompt_template – template that enforces inline citations and structure. Tuning your template is one of the highest ROI levers for factuality.

### Citations
- max_chunks – cap how many distinct sources are cited. Keep 3–5 to avoid citation spam while still being transparent.
- style – inline numeric vs other styles; inline is easiest to parse.
- deduplicate – remove repeat citations.

### Server
- host / port – API binding; change to 0.0.0.0 for access on LAN.

### Eval
- dataset_path – JSON/JSONL with {question, answer, references} for RAGAS.
- ragas.* – weighting between faithfulness and answer relevancy metrics. Tune these when tracking regressions across parameter changes.


## Trustworthy Citations
Every retrieved chunk carries metadata `{doc_id, source_path, page}`. Answers reference them inline like:


> "...the framework recommends hybrid retrieval [doc3:12; doc1:5]."


The pipeline also prints a confidence (normalized score from top results + reranker if enabled).


## RAGAS Evaluation
We provide a tiny example dataset to demonstrate the flow. For robust evaluation, expand the dataset.


Metrics reported:
- Faithfulness
- Answer relevancy
- Context precision/recall


## Limits
- Toy BM25 (in‑memory) is fine for <10k chunks. For larger corpora, switch to Elasticsearch/OpenSearch.
- Reranking adds latency; leave off for CPU-only.


