include "root" {
  path = find_in_parent_folders()
}

terraform {
  source = "../../../modules/ec2"
}

# Pull VPC and S3 outputs from sibling modules — no manual copy-paste of IDs.
dependency "networking" {
  config_path = "../networking"

  mock_outputs = {
    vpc_id           = "vpc-00000000"
    public_subnet_id = "subnet-00000000"
  }
  mock_outputs_allowed_terraform_commands = ["validate", "plan"]
}

dependency "s3" {
  config_path = "../s3"

  mock_outputs = {
    bucket_name = "mock-bucket"
    bucket_arn  = "arn:aws:s3:::mock-bucket"
  }
  mock_outputs_allowed_terraform_commands = ["validate", "plan"]
}

inputs = {
  project        = "agent-react-rag"
  env            = "demo"
  vpc_id         = dependency.networking.outputs.vpc_id
  subnet_id      = dependency.networking.outputs.public_subnet_id
  s3_bucket_name = dependency.s3.outputs.bucket_name
  s3_bucket_arn  = dependency.s3.outputs.bucket_arn

  # SSM path where you store the HuggingFace token (see setup steps below).
  ssm_hf_token_path = "/agent-react-rag/hf-token"

  instance_type = "t2.micro"   # free-tier eligible
  app_port      = 8080
  allowed_cidr  = "0.0.0.0/0" # restrict to your IP for better security
}
