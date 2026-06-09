This repository has an agentic RAG AI system, that, given a question on the documentation, evaluates if the documents are sufficient enough to answer the question, and if they are not, re-writes the query and tries again.  

Tech Stack: 
Python 3.12
Chroma

Usage Instructions: 
1. Go to Huggingface > Profile > Settings > Access Token > Create new token > Wait until it is approved. 
2. Once approved, in powershell, set the environmental variable to your token: 
        $env:HUGGINGFACEHUB_API_TOKEN = "Your Token"
    or use Huggingface CLI login: 
        huggingface-cli login
    If you need to change your token in your environment, use: 
        setx HUGGINGFACEHUB_API_TOKEN "YOUR_TOKEN"

3. Load your pdfs into Chroma (data/chroma)

Running locally (Docker / Lambda runtime)
---------------------------------------

Run the project locally inside the AWS Lambda Python runtime using Docker. This is useful for testing the same runtime you will deploy to Lambda.

Build the image:

```bash
cd genai-main/agent-react-rag
./scripts/build_image.sh
```

Notes:
- `./scripts/build_image.sh` only builds the Docker image, it does not start a container.
- On Windows, run the script from Git Bash, WSL, or use PowerShell with `bash ./scripts/build_image.sh`.
- If the build succeeds, the image will appear in `docker images`, not `docker ps`.

Or build directly:

```bash
docker build -t agent-react-rag-lambda:latest .
```

Run the container (binds local port 9000 to the Lambda runtime API):

Bash / WSL:

```bash
docker run -p 9000:8080 \
    -e HUGGINGFACEHUB_API_TOKEN="$HUGGINGFACEHUB_API_TOKEN" \
    -v $(pwd)/src/data:/var/task/src/data \
    agent-react-rag-lambda:latest
```

PowerShell:

```powershell
docker run -p 9000:8080 `
    -e HUGGINGFACEHUB_API_TOKEN="$env:HUGGINGFACEHUB_API_TOKEN" `
    -v "${PWD}\src\data:/var/task/src/data" `
    agent-react-rag-lambda:latest
```

Or use a single-line PowerShell command:

```powershell
docker run -p 9000:8080 -e HUGGINGFACEHUB_API_TOKEN="$env:HUGGINGFACEHUB_API_TOKEN" -v "${PWD}\src\data:/var/task/src/data" agent-react-rag-lambda:latest
```

Notes:
- In PowerShell, use backtick (`) for line continuation instead of backslash (`\`).
- `docker ps` shows running containers only; use `docker images` to verify the built image.
- Mounting `src/data` allows the container to access your Chroma DB.

Invoke the Lambda function locally (Lambda runtime API):

```bash
curl -XPOST 'http://localhost:9000/2015-03-31/functions/function/invocations' \
    -d '{"question":"What are the main points of the documents in the database?"}'
```

PowerShell example:

```powershell
curl.exe -Method POST -Uri 'http://localhost:9000/2015-03-31/functions/function/invocations' \
    -Body '{"question":"What are the main points of the documents in the database?"}' \
    -ContentType 'application/json'
```

If you use PowerShell without `curl.exe`, you can also use `Invoke-RestMethod`:

```powershell
Invoke-RestMethod -Method POST -Uri 'http://localhost:9000/2015-03-31/functions/function/invocations' -Body '{"question":"What are the main points of the documents in the database?"}' -ContentType 'application/json'
```

Deploying to AWS Lambda (container image)
-----------------------------------------

1. Build and tag the image for ECR (replace `<aws-account>` and `<region>`):

```bash
docker build -t agent-react-rag-lambda:latest .
docker tag agent-react-rag-lambda:latest <aws-account>.dkr.ecr.<region>.amazonaws.com/agent-react-rag-lambda:latest
```

2. Create an ECR repo (if needed) and push the image:

```bash
aws ecr create-repository --repository-name agent-react-rag-lambda --region <region> || true
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <aws-account>.dkr.ecr.<region>.amazonaws.com
docker push <aws-account>.dkr.ecr.<region>.amazonaws.com/agent-react-rag-lambda:latest
```

3. Create a Lambda function from the container image (Console or CLI). Example CLI:

```bash
aws lambda create-function \
    --function-name agent-react-rag \
    --package-type Image \
    --code ImageUri=<aws-account>.dkr.ecr.<region>.amazonaws.com/agent-react-rag-lambda:latest \
    --role arn:aws:iam::<aws-account>:role/<lambda-exec-role> \
    --region <region>
```

4. Configure environment variables for the Lambda function (e.g., `HUGGINGFACEHUB_API_TOKEN`, `SECRET_ARN`) and ensure the Lambda execution role has permission to read Secrets Manager if you store API keys there.

Security notes
- Do NOT commit secrets into the repo. Use AWS Secrets Manager and set the secret value from CI (GitHub Actions OIDC) or the Console.
- For demos prefer using external LLM APIs (OpenAI / Hugging Face Inference API) instead of bundling a large local model in the container — Lambda has size and memory limits.
- Do NOT commit `src/data` or `src/data/chroma` to GitHub. This repo should stay code-only.

Storing large data on AWS (free tier)
- AWS S3 is the best free-tier option for storing PDFs or Chroma data outside the repo.
- AWS Free Tier includes: 5 GB of Amazon S3 Standard storage, 20,000 GET requests, and 2,000 PUT requests per month for the first 12 months.

Example S3 workflow:

```bash
cd genai-main/agent-react-rag
aws s3 mb s3://my-agent-react-rag-data --region <region>
aws s3 sync src/data s3://my-agent-react-rag-data
```

To use S3 data in a Lambda deployment, keep the image code-only and download the files at runtime or during a cold-start step. For example, in your Lambda startup code:

```python
import os
import boto3
from pathlib import Path

s3 = boto3.client("s3")
BUCKET = os.environ["DATA_BUCKET"]
TARGET_DIR = Path("/tmp/src/data")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

s3.download_file(BUCKET, "chroma/chroma_db/chroma.sqlite3", str(TARGET_DIR / "chroma.sqlite3"))
```

If you prefer not to download at runtime, use a small build or init step on an EC2/builder host to fetch the S3 files and populate the Lambda-friendly storage path.

Troubleshooting
- If the container fails with import errors for heavy ML libs (transformers/torch), either remove those dependencies for Lambda builds or use a larger container (ECS/Fargate or EC2) instead.
- To test the handler locally, ensure `src/main.py` exposes `answer_from_input(question_or_dict)` as used by `lambda_app.py`.
