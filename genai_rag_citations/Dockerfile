FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV OLLAMA_URL=http://host.docker.internal:11434/api/generate
CMD ["bash", "-lc", "pytest -q && python ingest/ingest_docs.py --input_dir sample_docs --persist_dir .chroma && python rag_pipeline.py --question 'What does this repo implement?' --k 6"]