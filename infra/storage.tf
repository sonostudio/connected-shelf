# Inventory database
resource "aws_dynamodb_table" "inventory" {
  name         = "ConnectedShelf-Inventory"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "sku_id"

  attribute {
    name = "sku_id"
    type = "S" # String
  }

  tags = {
    Purpose = "Inventory State"
  }
}

# Asset warehouse
resource "aws_s3_bucket" "assets" {
  bucket = "${var.project_name}-assets-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_public_access_block" "assets_block" {
  bucket = aws_s3_bucket.assets.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ML Data Storage
resource "aws_s3_bucket" "ml_data" {
  bucket = "${var.project_name}-ml-data-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_public_access_block" "ml_data_block" {
  bucket = aws_s3_bucket.ml_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Outputs
output "bucket_name" {
  value = aws_s3_bucket.assets.id
}