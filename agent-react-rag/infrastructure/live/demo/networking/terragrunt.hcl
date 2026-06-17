include "root" {
  path = find_in_parent_folders()
}

terraform {
  source = "../../../modules/networking"
}

inputs = {
  project            = "agent-react-rag"
  env                = "demo"
  vpc_cidr           = "10.0.0.0/16"
  public_subnet_cidr = "10.0.1.0/24"
  availability_zone  = "us-east-1a"
}
