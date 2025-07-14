#!/bin/bash

# Fast Sanskrit Parser Deployment Script

echo "⚡ Starting fast Sanskrit Parser deployment..."

# Check if Traefik network exists
if ! docker network ls | grep -q "traefik_net"; then
    echo "❌ Traefik network not found. Please start Traefik first."
    exit 1
fi

# Check if base image exists
if ! docker images sanskrit-parser-base:latest | grep -q "sanskrit-parser-base"; then
    echo "📦 Base image not found. Building base image first..."
    ./build-base-image.sh
fi

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose -f docker-compose.dev.yml down

# Use fast Dockerfile for backend
echo "🔨 Building with fast Dockerfile..."
docker-compose -f docker-compose.dev.yml build --build-arg DOCKERFILE=Dockerfile.fast

# Start services
echo "🚀 Starting services..."
docker-compose -f docker-compose.dev.yml up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are running
echo "✅ Checking service status..."
docker-compose -f docker-compose.dev.yml ps

# Show logs
echo "📋 Recent logs:"
docker-compose -f docker-compose.dev.yml logs --tail=20

echo "🎉 Fast deployment completed!"
echo "🌐 Application should be available at: https://sanskrit-parser.bondarenko-nikita.ru"
echo "⚡ Build time significantly reduced!" 