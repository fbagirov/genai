#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-agent-react-rag-lambda}"
TAG="${2:-latest}"

# Resolve project root (one level up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker CLI not found. Install Docker Desktop and ensure it is on your PATH." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Error: Docker daemon is not running. Start Docker Desktop and try again." >&2
  exit 1
fi

echo "Building ${IMAGE_NAME}:${TAG} from ${PROJECT_ROOT}"
docker build --prefer-binary -t "${IMAGE_NAME}:${TAG}" "$PROJECT_ROOT"

echo ""
echo "Build complete: ${IMAGE_NAME}:${TAG}"
echo ""
echo "Test locally (Lambda runtime emulator):"
echo "  docker run --rm -p 9000:8080 \\"
echo "    -e HUGGINGFACEHUB_API_TOKEN=your_token_here \\"
echo "    ${IMAGE_NAME}:${TAG}"
echo ""
echo "Query endpoint:"
echo "  curl -s -XPOST 'http://localhost:9000/2015-03-31/functions/function/invocations' \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"question\": \"What is retrieval-augmented generation?\"}'"
echo ""
echo "Ingest a PDF:"
echo "  B64=\$(base64 -w0 /path/to/your.pdf)"
echo "  curl -s -XPOST 'http://localhost:9000/2015-03-31/functions/function/invocations' \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d \"{\\\"path\\\": \\\"/ingest\\\", \\\"httpMethod\\\": \\\"POST\\\", \\\"body\\\": \\\"{\\\\\\\"pdf_base64\\\\\\\":\\\\\\\"\$B64\\\\\\\",\\\\\\\"filename\\\\\\\":\\\\\\\"your.pdf\\\\\\\"}\\\"}\" "
