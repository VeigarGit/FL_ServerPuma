#!/bin/bash
set -e

echo "🔄 Cleanup completo..."
docker compose --project-directory .. -f docker-compose.generated.yml down -v --rmi all 2>/dev/null || true
docker system prune -a -f

echo "🏗️ Build otimizado..."
docker compose --project-directory .. -f docker-compose.generated.yml build --no-cache --parallel

python generate_compose.py

docker compose --project-directory .. -f docker-compose.generated.yml up