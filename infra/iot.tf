resource "aws_iot_thing" "connectedshelf_node" {
# Raspberry Pi
  name = "ConnectedShelf-01"

  attributes = {
    project = "connected-shelf"
    model   = "RPi4"
  }
}

resource "aws_iot_policy" "connectedshelf_policy" {
  name = "ConnectedShelfAccessPolicy"

  # This JSON defines what the shelf is allowed to do
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "iot:Connect",
          "iot:Publish",
          "iot:Subscribe",
          "iot:Receive"
        ]
        Resource = "*"
      }
    ]
  })
}

# 3. Output the Endpoint (You will need this for your Python script later)
data "aws_iot_endpoint" "current" {
  endpoint_type = "iot:Data-ATS"
}

output "iot_endpoint" {
  value = data.aws_iot_endpoint.current.endpoint_address
}