variable "project" {
  type        = string
  description = "Project name."
}

variable "env" {
  type        = string
  description = "Environment name."
}

variable "vpc_id" {
  type        = string
  description = "VPC ID from the networking module."
}

variable "subnet_id" {
  type        = string
  description = "Public subnet ID from the networking module."
}

variable "s3_bucket_name" {
  type        = string
  description = "S3 bucket name for PDFs (from the s3 module)."
}

variable "s3_bucket_arn" {
  type        = string
  description = "S3 bucket ARN for IAM policy."
}

variable "ssm_hf_token_path" {
  type        = string
  description = "SSM Parameter Store path where the HuggingFace token is stored."
  default     = "/agent-react-rag/hf-token"
}

variable "instance_type" {
  type        = string
  description = "EC2 instance type. t2.micro is free-tier eligible (750 hrs/month)."
  default     = "t2.micro"
}

variable "app_port" {
  type        = number
  description = "Port the Flask server listens on."
  default     = 8080
}

variable "github_repo" {
  type        = string
  description = "GitHub repo to clone the app from."
  default     = "https://github.com/fbagirov/genai.git"
}

variable "app_subdir" {
  type        = string
  description = "Sub-directory inside the cloned repo that contains the app."
  default     = "agent-react-rag"
}

variable "allowed_cidr" {
  type        = string
  description = "CIDR allowed to reach the app port. Default is open to all (demo only)."
  default     = "0.0.0.0/0"
}
