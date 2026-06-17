output "instance_id" {
  value       = aws_instance.app.id
  description = "EC2 instance ID."
}

output "public_ip" {
  value       = aws_instance.app.public_ip
  description = "Public IP address of the EC2 instance."
}

output "public_dns" {
  value       = aws_instance.app.public_dns
  description = "Public DNS of the EC2 instance."
}

output "app_url" {
  value       = "http://${aws_instance.app.public_ip}:${var.app_port}"
  description = "Base URL for the RAG API."
}
