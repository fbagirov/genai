# genai-rag-citations
# 2) Install deps
pip install -r requirements.txt


# 3) (Optional) Start Ollama & pull a model
# https://ollama.com/download
ollama pull llama3:instruct


# 4) Configure
cp configs/config.example.yaml configs/config.yaml
# Edit configs/config.yaml to pick embedding/LLM providers (Ollama vs OpenAI)


# 5) Ingest a few PDFs/texts
python ingest/ingest_docs.py --input_dir sample_docs --persist_dir .chroma


# 6) Ask questions
python rag_pipeline.py --question "What are the main topics?" --k 6 --use_reranker false


# 7) Evaluate with RAGAS (example dataset under eval/datasets)
python eval/ragas_eval.py --dataset_path eval/datasets/sample_eval.jsonl
```


## Architecture
```
[Loaders] -> [TextSplitter] -> [Embeddings] -> [Chroma]
| ^
v |
[BM25 Index] ------------------+


Query -> HybridRetriever (BM25 + Dense [+ optional reranker]) -> Generator(LLM) ->
Inline citations + confidence
```


## Config toggles (`configs/config.yaml`)
- `embedding.provider`: `sentence_transformers` | `openai`
- `embedding.model`: e.g., `intfloat/e5-base-v2`
- `llm.provider`: `ollama` | `openai`
- `llm.model`: e.g., `llama3:instruct` or `gpt-4o-mini`
- `retrieval.k`: top-K for dense & BM25, score fusion weights
- `reranker.enabled`: true|false with model name


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