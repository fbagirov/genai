#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="agent-react-rag-lambda"
TAG="latest"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker CLI not found."
  echo "Install Docker Desktop and enable WSL 2 integration, or run this script in an environment where docker is available."
  exit 1
fi

echo "Building Docker image ${IMAGE_NAME}:${TAG}"
docker build -t "${IMAGE_NAME}:${TAG}" .

echo "Built ${IMAGE_NAME}:${TAG}"
echo "To run locally (Lambda runtime API):"
echo "  docker run -p 9000:8080 ${IMAGE_NAME}:${TAG}"
echo "Then invoke the function with:"
echo "  curl -XPOST 'http://localhost:9000/2015-03-31/functions/function/invocations' -d '{\"question\": \"Hello\"}'"
