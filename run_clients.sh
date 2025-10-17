#!/bin/bash

# Número de clientes (padrão: 3)
CLIENT_COUNT=${1:-3}

echo "🔄 Construindo imagem do cliente..."
docker build -t fl-client .

echo "🔄 Iniciando $CLIENT_COUNT clientes..."

# Iniciar cada cliente com ID único
for i in $(seq 0 $(($CLIENT_COUNT-1))); do
  echo "🚀 Iniciando cliente $i..."
  docker run -d \
    --name "fl-client-$i" \
    --env CLIENT_ID=$i \
    fl-client \
    python client.py --client-idx $i --host server --dataset MNIST
done

echo "✅ $CLIENT_COUNT clientes iniciados!"
echo "📋 Containers ativos:"
docker ps --filter "name=fl-client"