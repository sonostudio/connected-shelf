# Auto-Zip
data "archive_file" "lambda_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../src/backend"
  output_path = "${path.module}/lambda_payload.zip"
}

# IAM Role & Permissions
resource "aws_iam_role" "lambda_role" {
  name = "ConnectedShelf-LambdaRole"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "lambda_permissions" {
  name = "ConnectedShelf-LambdaPolicy"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # Allow Logging
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      # Allow Database Access
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:UpdateItem"
        ]
        Resource = aws_dynamodb_table.inventory.arn
      },
      # Allow IoT Publishing
      {
        Effect = "Allow"
        Action = [ "iot:Publish" ]
        Resource = "*"
      }
    ]
  })
}

# Function
resource "aws_lambda_function" "backend_logic" {
  function_name = "ConnectedShelf-Logic"
  filename         = data.archive_file.lambda_zip.output_path
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256
  handler = "main.lambda_handler"
  runtime = "python3.11"
  role = aws_iam_role.lambda_role.arn

  environment {
    variables = {
      TABLE_NAME = aws_dynamodb_table.inventory.name
    }
  }
}