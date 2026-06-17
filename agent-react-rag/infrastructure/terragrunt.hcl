# Root Terragrunt configuration.
# Generates the AWS provider and sets up a local Terraform state backend.
# To use remote state (recommended before going to production) uncomment the
# remote_state block below and create the S3 bucket + DynamoDB table first.

locals {
  aws_region = "us-east-1"   # free-tier resources are region-specific; us-east-1 has the widest coverage
  project    = "agent-react-rag"
  env        = "demo"
}

generate "provider" {
  path      = "provider.tf"
  if_exists = "overwrite_terragrunt"
  contents  = <<-EOF
    provider "aws" {
      region = "${local.aws_region}"

      default_tags {
        tags = {
          Project     = "${local.project}"
          Environment = "${local.env}"
          ManagedBy   = "terragrunt"
        }
      }
    }

    terraform {
      required_version = ">= 1.5"
      required_providers {
        aws = {
          source  = "hashicorp/aws"
          version = "~> 5.0"
        }
      }
    }
  EOF
}

# ── Local state (default for demo) ────────────────────────────────────────────
# State files are written to .terraform/ inside each live/* directory.
# Fine for a single-developer demo; switch to remote_state for team use.

# ── Remote state (uncomment when ready) ───────────────────────────────────────
# Pre-requisite: create the bucket and table manually once:
#   aws s3 mb s3://agent-react-rag-tfstate-demo --region us-east-1
#   aws dynamodb create-table --table-name agent-react-rag-tfstate-lock \
#     --attribute-definitions AttributeName=LockID,AttributeType=S \
#     --key-schema AttributeName=LockID,KeyType=HASH \
#     --billing-mode PAY_PER_REQUEST --region us-east-1
#
# remote_state {
#   backend = "s3"
#   generate = {
#     path      = "backend.tf"
#     if_exists = "overwrite_terragrunt"
#   }
#   config = {
#     bucket         = "${local.project}-tfstate-${local.env}"
#     key            = "${path_relative_to_include()}/terraform.tfstate"
#     region         = local.aws_region
#     encrypt        = true
#     dynamodb_table = "${local.project}-tfstate-lock"
#   }
# }
