variable "project" {
  type        = string
  description = "Project name used in resource names and tags."
}

variable "env" {
  type        = string
  description = "Environment name (e.g. demo, staging, prod)."
}

variable "vpc_cidr" {
  type        = string
  description = "CIDR block for the VPC."
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidr" {
  type        = string
  description = "CIDR block for the single public subnet."
  default     = "10.0.1.0/24"
}

variable "availability_zone" {
  type        = string
  description = "AZ for the public subnet."
  default     = "us-east-1a"
}
