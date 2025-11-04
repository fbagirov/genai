
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt requirements.gpu.txt requirements.extra.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# GPU optional: uncomment for CUDA images
# RUN pip install --no-cache-dir -r requirements.gpu.txt
# Optional PII tools
# RUN pip install --no-cache-dir -r requirements.extra.txt && python -m spacy download en_core_web_lg || true

COPY . .

# Default command: quick SFT smoke run on sample data (CPU-friendly, tiny steps)
CMD ["bash","-lc","python scripts/train_sft.py --config configs/config.example.yaml && python scripts/train_dpo.py --config configs/config.example.yaml"]
