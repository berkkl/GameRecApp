#!/bin/bash
set -e

CONTAINER_NAME="gamerecapp_dev"
IMAGE_NAME="gamerecapp"
PORT="8000"

echo "🐳 Building Docker image..."
docker build -t $IMAGE_NAME .

if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "🛑 Stopping and removing existing container..."
    docker rm -f $CONTAINER_NAME
fi

echo "🚀 Starting new container..."
# Run with volume mount for logs and consistent name
docker run -d \
  --name $CONTAINER_NAME \
  -p $PORT:8000 \
  -v "$(pwd)/data:/app/data" \
  $IMAGE_NAME

echo "✅ Deployed! Logs:"
docker logs -f $CONTAINER_NAME
