locals {
  name_prefix = "${var.project}-${var.env}"
}

# ── Latest Amazon Linux 2023 AMI ──────────────────────────────────────────────

data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# ── Security group ────────────────────────────────────────────────────────────

resource "aws_security_group" "app" {
  name        = "${local.name_prefix}-sg"
  description = "Allow inbound on app port and all outbound"
  vpc_id      = var.vpc_id

  ingress {
    description = "App port"
    from_port   = var.app_port
    to_port     = var.app_port
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
  }

  # SSH is intentionally omitted — use SSM Session Manager instead (no key pair needed).
  # To enable SSH uncomment the block below and set allowed_cidr to your IP.
  # ingress {
  #   description = "SSH"
  #   from_port   = 22
  #   to_port     = 22
  #   protocol    = "tcp"
  #   cidr_blocks = [var.allowed_cidr]
  # }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${local.name_prefix}-sg" }
}

# ── IAM role ──────────────────────────────────────────────────────────────────

resource "aws_iam_role" "app" {
  name = "${local.name_prefix}-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

# S3: read + write to the PDF bucket only
resource "aws_iam_role_policy" "s3_access" {
  name = "s3-pdf-access"
  role = aws_iam_role.app.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ]
      Resource = [
        var.s3_bucket_arn,
        "${var.s3_bucket_arn}/*"
      ]
    }]
  })
}

# SSM: read the HuggingFace token secret
resource "aws_iam_role_policy" "ssm_access" {
  name = "ssm-hf-token-read"
  role = aws_iam_role.app.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["ssm:GetParameter"]
      Resource = "arn:aws:ssm:*:*:parameter${var.ssm_hf_token_path}"
    }]
  })
}

# SSM Session Manager (replaces SSH — free, no key pair needed)
resource "aws_iam_role_policy_attachment" "ssm_managed" {
  role       = aws_iam_role.app.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "app" {
  name = "${local.name_prefix}-instance-profile"
  role = aws_iam_role.app.name
}

# ── EC2 instance ──────────────────────────────────────────────────────────────
# t2.micro: 1 vCPU, 1 GB RAM — free tier 750 hrs/month.
# A 2 GB swap file is created in user_data so sentence-transformers + torch
# don't OOM. Inference will be slow (~30–60 s) but functional for a demo.

resource "aws_instance" "app" {
  ami                    = data.aws_ami.al2023.id
  instance_type          = var.instance_type
  subnet_id              = var.subnet_id
  vpc_security_group_ids = [aws_security_group.app.id]
  iam_instance_profile   = aws_iam_instance_profile.app.name

  root_block_device {
    volume_type = "gp3"   # gp3 is free-tier eligible (up to 30 GB)
    volume_size = 8
    encrypted   = true
  }

  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    region            = data.aws_region.current.name
    ssm_hf_token_path = var.ssm_hf_token_path
    s3_bucket_name    = var.s3_bucket_name
    github_repo       = var.github_repo
    app_subdir        = var.app_subdir
    app_port          = var.app_port
  })

  tags = { Name = "${local.name_prefix}-app" }

  lifecycle {
    # Replacing the instance on every user_data change is expensive during
    # development; set to true when you want immutable deployments.
    ignore_changes = [user_data]
  }
}

data "aws_region" "current" {}
