from pathlib import Path

COMPOSE_FILE = "docker-compose.v2x.yml"

def generate(
    client_indices: list[int],
    dataset: str,
    rounds: int,
    prune: int,
    ala: int,
    output_path: Path,
) -> str:
    """
    Gera um docker-compose.yml customizado contendo APENAS os clientes (veículos)
    para o modo V2X descentralizado.
    
    Retorna o caminho do arquivo gerado.
    """
    
    template = f"""
x-client-template: &client
  build:
    context: .
    dockerfile: dockerfile.client
    args:
      - NO_CACHE=true
  image: fl-client-image
  restart: 'no'
  working_dir: /app/src/system
  environment: 
    - PYTHONUNBUFFERED=1
    - PYTHONPATH=/app/src/system:/app/src
  volumes:
    - ./src/dataset:/app/src/dataset
    - ./src/results:/app/src/results
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
  networks:
    - docker-tc
  labels:
    - "com.docker-tc.enabled=1"
    - "com.docker-tc.loss=10%"

services:
"""

    for i, client_idx in enumerate(client_indices):
        template += f"""
  client-v2x-{i}:
    <<: *client
    container_name: fl-client-v2x-{i}
    command: ["python", "client.py", "--mode", "decentralized", "--client-idx", "{client_idx}", "--dataset", "{dataset}", "--rounds", "{rounds}", "--ala", "{ala}"]
"""

    template += """
networks:
  docker-tc:
    driver: bridge
"""

    compose_path = output_path / COMPOSE_FILE
    with open(compose_path, "w") as f:
        f.write(template)
    return str(compose_path)

if __name__ == "__main__":
    print("Use o orchestrator.")