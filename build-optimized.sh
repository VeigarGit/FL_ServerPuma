#!/bin/bash
echo "Limpando cache do Docker..."
docker system prune -a -f

echo "Build otimizado das imagens..."
docker-compose build --no-cache --parallel

echo "Verificando tamanhos das imagens..."
echo "=== TAMANHO DAS IMAGENS ==="
docker images | grep fl-server
docker images | grep fl-client

echo "=== ANÁLISE DE ESPAÇO ==="
docker run -it --rm $(docker images -q fl-client | head -1) du -h --max-depth=3 /opt/conda 2>/dev/null || echo "Imagem não encontrada"