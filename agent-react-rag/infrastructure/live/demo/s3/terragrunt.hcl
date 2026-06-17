include "root" {
  path = find_in_parent_folders()
}

terraform {
  source = "../../../modules/s3"
}

inputs = {
  project = "agent-react-rag"
  env     = "demo"
}
