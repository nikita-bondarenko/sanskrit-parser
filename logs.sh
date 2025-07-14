#!/bin/bash

# Sanskrit Parser Logs Script

echo "📋 Sanskrit Parser Development Logs"
echo "=================================="

# Check if containers are running
if ! docker-compose -f docker-compose.dev.yml ps | grep -q "Up"; then
    echo "❌ No containers are running. Please start the application first with ./deploy.sh"
    exit 1
fi

# Show logs with follow option
echo "📊 Following logs (press Ctrl+C to exit)..."
docker-compose -f docker-compose.dev.yml logs -f --tail=50 