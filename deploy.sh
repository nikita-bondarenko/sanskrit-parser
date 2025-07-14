#!/bin/bash

# Sanskrit Parser Deployment Script (Development Mode)

echo "🚀 Starting Sanskrit Parser deployment in development mode..."

# Check if Traefik network exists
if ! docker network ls | grep -q "traefik_net"; then
    echo "❌ Traefik network not found. Please start Traefik first."
    exit 1
fi

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose -f docker-compose.dev.yml down

# Remove old images
echo "🗑️ Removing old images..."
docker rmi sanskrit-parser-backend:latest 2>/dev/null || true
docker rmi sanskrit-parser-frontend:latest 2>/dev/null || true

# Build and start services
echo "🔨 Building and starting services in development mode..."
docker-compose -f docker-compose.dev.yml up -d --build

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 15

# Check if services are running
echo "✅ Checking service status..."
docker-compose -f docker-compose.dev.yml ps

# Show logs
echo "📋 Recent logs:"
docker-compose -f docker-compose.dev.yml logs --tail=20

echo "🎉 Development deployment completed!"
echo "🌐 Application should be available at: https://sanskrit-parser.bondarenko-nikita.ru"
echo "🔧 Development mode: Hot reload enabled for both frontend and backend" 