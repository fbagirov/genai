# Demo environment — shared inputs inherited by all child modules.
locals {
  project = "agent-react-rag"
  env     = "demo"
  region  = "us-east-1"
}

# Pull in the root-level provider + backend config.
include "root" {
  path = find_in_parent_folders()
}
