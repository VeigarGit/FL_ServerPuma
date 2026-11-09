#!/bin/bash
# Responsabilidade: Limpa contêineres e imagens Docker, reconstrói as imagens, gera o Compose
# centralizado e inicia os serviços do experimento.

set -e

echo "🔄 Cleanup completo..."
docker compose --project-directory .. -f docker-compose.generated.yml down -v --rmi all 2>/dev/null || true
docker system prune -a -f

echo "🏗️ Build otimizado..."
docker compose --project-directory .. -f docker-compose.generated.yml build --no-cache --parallel

uv run python -m fl_serverpuma.simulation.docker.compose

docker compose --project-directory .. -f docker-compose.generated.yml up