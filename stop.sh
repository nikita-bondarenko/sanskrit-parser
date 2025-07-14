#!/bin/bash

# Sanskrit Parser Stop Script

echo "🛑 Stopping Sanskrit Parser development environment..."

# Stop containers
docker-compose -f docker-compose.dev.yml down

echo "✅ All containers stopped successfully!" 