.PHONY: fmt lint test ingest ask eval docker-build


fmt:
	python -m pip install pre-commit && pre-commit run --all-files || true


lint:
	ruff check . || true


test:
	pytest -q


ingest:
	python ingest/ingest_docs.py --input_dir sample_docs --persist_dir .chroma


ask:
	python rag_pipeline.py --question "What is this repo?" --k 6


eval:
	python eval/ragas_eval.py --dataset_path eval/datasets/sample_eval.jsonl


docker-build:
	docker build -t genai-rag-citations:latest .