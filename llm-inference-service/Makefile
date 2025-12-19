
.PHONY: dev bench docker-build docker-run k8s-apply k8s-delete

dev:
	uvicorn server.main:app --host 0.0.0.0 --port 8080 --reload

bench:
	locust -f scripts/locustfile.py --host http://127.0.0.1:8080

docker-build:
	docker build -t yourrepo/llm-inference-service:latest .

docker-run:
	docker run --rm -p 8080:8080 -e VLLM_BASE_URL=http://host.docker.internal:8001/v1 -e MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0 yourrepo/llm-inference-service:latest

k8s-apply:
	kubectl apply -f k8s/

k8s-delete:
	kubectl delete -f k8s/
