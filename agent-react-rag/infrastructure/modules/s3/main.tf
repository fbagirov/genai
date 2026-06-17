# S3 bucket for PDF uploads.
# Free tier: 5 GB storage, 20,000 GET requests, 2,000 PUT requests per month.

resource "aws_s3_bucket" "pdfs" {
  # Bucket names must be globally unique; the random suffix ensures that.
  bucket = "${var.project}-${var.env}-pdfs-${random_id.suffix.hex}"

  tags = { Name = "${var.project}-${var.env}-pdfs" }
}

resource "random_id" "suffix" {
  byte_length = 4
}

# Block all public access — the EC2 instance accesses via IAM role, not public URLs.
resource "aws_s3_bucket_public_access_block" "pdfs" {
  bucket                  = aws_s3_bucket.pdfs.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "pdfs" {
  bucket = aws_s3_bucket.pdfs.id
  versioning_configuration {
    status = "Enabled"
  }
}
