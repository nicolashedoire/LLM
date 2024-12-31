#!/bin/bash

echo "Starting Docker build process..."
docker build -t small-llm-app .
echo "Docker build process completed."