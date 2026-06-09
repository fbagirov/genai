param(
    [string]$ImageName = "agent-react-rag-lambda",
    [string]$Tag = "latest"
)

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker CLI not found. Install Docker Desktop or Docker Engine and ensure it is on your PATH."
    exit 1
}

if (-not (docker info > $null 2>&1)) {
    Write-Error "Docker daemon is not running or not accessible. Start Docker Desktop or Docker Engine and try again."
    exit 1
}

Write-Host "Building Docker image ${ImageName}:${Tag}"

docker build -t "${ImageName}:${Tag}" .
if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker build failed. Make sure the Docker daemon is running and accessible."
    exit $LASTEXITCODE
}

Write-Host "Built ${ImageName}:${Tag}"
Write-Host "To run locally (Lambda runtime API):"
Write-Host "  docker run -p 9000:8080 ${ImageName}:${Tag}"
Write-Host "Then invoke the function with:`n  curl -XPOST 'http://localhost:9000/2015-03-31/functions/function/invocations' -d '{"question":"Hello"}'"
