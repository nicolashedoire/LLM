#!/bin/bash

echo "Starting Docker container..."
docker run -p 8000:8000 small-llm-app
echo "Docker container started. Access the application at http://localhost:8000"