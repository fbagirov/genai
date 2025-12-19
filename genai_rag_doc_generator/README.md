# outcome-writer — Outcome-Conditioned Sales Email Generator

**Goal:** Generate e-commerce sales emails that are *more likely* to succeed (result in sales in the outboudn email campaign) by learning from emails that previously restuled in successful outcomes.

This repository is a **reference implementation** for learning and portfolio demonstration:
- Uses a **simulated dataset** with labeled outcomes (`won`, `lost`, `no_response`)
- Builds a local **vector index** over *successful* emails
- Retrieves the most relevant successful examples for your scenario
- Generates a draft email (provider toggle: `mock` (offline) or `openai`)

> This system improves *consistency and style* by conditioning on successful examples, but it does **not guarantee sales outcomes**.

For this use case I intentionally used TF-IDF retrieval to avoid native dependencies and keep the system fully portable. The retrieval interface is abstracted, so swapping to a vector DB(like ChromaDB) is necessary if the scale requires it.

---

## Features
- **Outcome-conditioned generation** via retrieval of successful examples
- **Local retrieval** (Chroma + SentenceTransformers embeddings)
- **Config toggles** for retrieval/generation/scoring
- **Local “success likelihood” scorer** (toy model trained on simulated labels)
- **FastAPI** endpoint (`/v1/generate_email`)
- **Streamlit UI** for interactive drafting

---

## Quickstart

### 0) Create venv & install dependencies
```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

### 1) (Optional) Build the retrieval index (not required for this versino with tfidf, but is required if you build with vector database (chromadb, etc.))


### 2) Run the API server
```bash
uvicorn app.api:app --host 127.0.0.1 --port 8088 --reload
streamlit run ui/streamlit_app.py

```

Test it:
```bash
curl -X POST http://127.0.0.1:8088/v1/generate_email \
  -H "Content-Type: application/json" \
  -d '{"industry":"FinTech","persona":"CFO","product":"CloudCostGuard","value_prop":"cloud cost optimization","tone":"executive","goal":"book a 15-min intro call"}'
```

### 3) Run the Streamlit UI
```bash
streamlit run ui/streamlit_app.py
```

---

## Architecture
- `data/` — simulated labeled emails + preference pairs
- `ingest/` — builds embeddings index (Chroma)
- `app/`
  - `retriever.py` — fetches top-k successful examples
  - `generator.py` — provider toggle (`mock` offline or `openai`)
  - `scorer.py` — toy “success likelihood” model
  - `api.py` — FastAPI endpoint
- `ui/` — Streamlit UI
- `eval/` — smoke evaluation script

---

## Flow diagram
```mermaid
flowchart TD
  A[Scenario inputs] --> B[Retriever]
  B --> C[Chroma Search<br/>(success-only)]
  C --> D[Top-k successful examples]
  D --> E[Prompt builder]
  E --> F[Generator<br/>mock or openai]
  F --> G[Draft email]
  G --> H[Optional scorer]
  H --> I[Return draft + score]
```

---

## Config toggles (configs/config.yaml)

### Retrieval
- `retrieval.k`: how many successful examples to retrieve
- `retrieval.min_similarity`: filter weak matches
- `vectorstore.metadata_filter.only_success`: index/retrieve only successful (`won`) examples

### Generation
- `generation.provider`: `mock` (offline) or `openai`
- `generation.temperature`: creativity vs determinism (0.1–0.3 is good for sales)
- `generation.max_new_tokens`: email length cap
- `generation.include_scoring`: include local score in responses

### Privacy
- `privacy.store_user_inputs`: keep false by default
- Use `.env` for provider keys; never commit secrets

---

## Dataset
### Labeled emails: `data/simulated_emails.jsonl`
Each record includes:
- `subject`, `body`
- `industry`, `persona`, `product`, `tone`
- `outcome`: `won` | `lost` | `no_response`

### Preference pairs: `data/preference_pairs.jsonl`
Each record includes `prompt`, `chosen`, `rejected` — useful if you later add DPO training.

---

## Evaluation
Run a small smoke eval (requires API running):
```bash
python eval/eval.py --config configs/config.yaml
```

---

## Limits
- Dataset is **simulated**; scorer is a **toy model** (for demo only).
- Real outcomes depend on many factors beyond text.
- If you enable OpenAI provider, content is sent to the provider; use local models for strict privacy.

---

## Troubleshooting
### “No results retrieved”
- Run ingest again
- Lower `retrieval.min_similarity`
- Ensure `.chroma/` exists

### PowerShell curl issues
PowerShell aliases `curl` to `Invoke-WebRequest`. Use:
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8088/v1/generate_email `
  -ContentType "application/json" `
  -Body '{"industry":"FinTech","persona":"CFO","product":"CloudCostGuard","value_prop":"cloud cost optimization","tone":"executive","goal":"book a 15-min intro call"}'
```

