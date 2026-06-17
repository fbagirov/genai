output "bucket_name" {
  value       = aws_s3_bucket.pdfs.bucket
  description = "S3 bucket name for PDF uploads."
}

output "bucket_arn" {
  value       = aws_s3_bucket.pdfs.arn
  description = "S3 bucket ARN."
}
