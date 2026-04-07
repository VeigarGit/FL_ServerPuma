#!/bin/bash

# 1. Cria a topologia de diretórios localmente PRIMEIRO (pertencendo ao seu usuário david)
mkdir -p src/results
mkdir -p src/system/dados_compartilhados

# 2. Injeta os UIDs seguros no arquivo oculto .env que o Docker Compose lê nativamente
echo "LOCAL_UID=$(id -u)" > .env
echo "LOCAL_GID=$(id -g)" >> .env

# 3. Gera o arquivo YML e sobe o sistema
uv run generate_compose.py --clients 5 --dataset Cifar100 --rounds 5 --prune 1 --ala 0
docker compose -f docker-compose.generated.yml up --build -d

# 4. Inicia o rastreio dos logs
docker compose -f docker-compose.generated.yml logs -f