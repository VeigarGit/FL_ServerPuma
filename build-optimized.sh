#!/bin/bash
set -e

echo "🔄 Cleanup completo..."
#docker-compose down -v --rmi all 2>/dev/null || true
#docker system prune -a -f

echo "🏗️ Build otimizado..."
docker-compose build #--no-cache --parallel


#python generate_compose.py

docker-compose -f docker-compose.generated.yml up