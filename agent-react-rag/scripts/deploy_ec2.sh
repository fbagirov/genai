#!/usr/bin/env bash
# Deploy agent-react-rag to AWS EC2 using Terragrunt.
#
# Prerequisites (run once):
#   - AWS CLI configured: aws configure
#   - Terraform >= 1.5 installed
#   - Terragrunt >= 0.53 installed
#   - Your HuggingFace token stored in SSM (step 1 below)
#
# Usage:
#   bash scripts/deploy_ec2.sh [plan|apply|destroy]
#   Default action is "plan" (safe — no changes made).
set -euo pipefail

LIVE_DIR="$(cd "$(dirname "$0")/.." && pwd)/infrastructure/live/demo"
ACTION="${1:-plan}"
REGION="us-east-1"
SSM_PATH="/agent-react-rag/hf-token"

# ── Step 1: Store HuggingFace token in SSM (run once) ─────────────────────────
# Uncomment and run this block once before the first deploy:
#
# read -rsp "Paste your HuggingFace token: " HF_TOKEN; echo
# aws ssm put-parameter \
#   --name "$SSM_PATH" \
#   --value "$HF_TOKEN" \
#   --type SecureString \
#   --region "$REGION" \
#   --overwrite
# echo "Token stored in SSM at $SSM_PATH"

# ── Step 2: Deploy with Terragrunt ────────────────────────────────────────────

echo "==> Running: terragrunt run-all $ACTION (live/demo)"

case "$ACTION" in
  plan)
    (cd "$LIVE_DIR" && terragrunt run-all plan --terragrunt-non-interactive)
    echo ""
    echo "Review the plan above. Run 'bash scripts/deploy_ec2.sh apply' to deploy."
    ;;
  apply)
    (cd "$LIVE_DIR" && terragrunt run-all apply --terragrunt-non-interactive)
    echo ""
    echo "==> Deployment complete."
    echo "    App URL: http://$(cd "$LIVE_DIR/ec2" && terragrunt output -raw app_url)"
    echo ""
    echo "==> Test the deployment:"
    APP_URL=$(cd "$LIVE_DIR/ec2" && terragrunt output -raw app_url)
    echo "    Health:  curl $APP_URL/health"
    echo "    Query:   curl -s -XPOST $APP_URL/query \\"
    echo "               -H 'Content-Type: application/json' \\"
    echo "               -d '{\"question\": \"What is RAG?\"}'"
    ;;
  destroy)
    echo "WARNING: this will delete all resources. Press Ctrl-C to abort."
    sleep 5
    (cd "$LIVE_DIR" && terragrunt run-all destroy --terragrunt-non-interactive)
    ;;
  *)
    echo "Unknown action: $ACTION. Use plan, apply, or destroy." >&2
    exit 1
    ;;
esac
