.PHONY: install ingest api ui eval test

install:
	pip install -r requirements.txt

ingest:
	python ingest/ingest.py --config configs/config.yaml

api:
	uvicorn app.api:app --host 127.0.0.1 --port 8088 --reload

ui:
	streamlit run ui/streamlit_app.py

eval:
	python eval/eval.py --config configs/config.yaml

test:
	pytest -q
