echo "🔄 Cleanup completo..."
docker-compose down -v --rmi all 2>/dev/null || true
docker system prune -a -f

# Clona e inicia o docker-tc se não existir
if [ ! -d "docker-tc" ]; then
    git clone https://github.com/lukaszlach/docker-tc.git
fi

# Tenta atualizar docker-compose se estiver com versão antiga
echo "🔧 Verificando versão do docker-compose..."
DOCKER_COMPOSE_VERSION=$(docker-compose --version | grep -oP '[0-9]+\.[0-9]+\.[0-9]+')
echo "📋 Versão atual: $DOCKER_COMPOSE_VERSION"

# Clona e inicia o docker-tc se não existir
if [ ! -d "docker-tc" ]; then
    git clone https://github.com/lukaszlach/docker-tc.git
fi

cd docker-tc

# Usa método compatível baseado na versão
if docker-compose up -d; then
    echo "✅ docker-tc iniciado com sucesso"
else
    echo "⚠️  Método padrão falhou, tentando alternativas..."
    
    # Método 1: Usar imagem diretamente
    if docker run -d --name docker-tc \
        --cap-add=NET_ADMIN \
        --network host \
        -v /var/run/docker.sock:/var/run/docker.sock \
        lukaszlach/docker-tc; then
        echo "✅ docker-tc via container direto"
    else
        echo "⚠️  Todos os métodos falharam, continuando sem docker-tc"
    fi
fi

cd ../
ls
echo "🚀 Iniciando aplicação principal..."
docker-compose -f docker-composeclient.yml up
