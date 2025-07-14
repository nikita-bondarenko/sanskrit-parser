#!/bin/bash

# Build base image with dependencies for faster subsequent builds

echo "🔨 Building base image with dependencies..."

# Build only up to the dependencies installation stage
docker build --target builder -t sanskrit-parser-base:latest ./backend

echo "✅ Base image built successfully!"
echo "📦 Image size:"
docker images sanskrit-parser-base:latest --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

echo ""
echo "💡 Now you can use ./deploy.sh for faster builds!" 