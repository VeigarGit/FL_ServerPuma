echo "🔄 Cleanup completo..."
docker-compose down -v --rmi all 2>/dev/null || true
docker system prune -a -f

echo "🚀 Iniciando aplicação principal..."
docker-compose -f docker-composeclient.yml up

