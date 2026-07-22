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
    Gera um docker-compose.yml customizado contendo APENAS os clientes
    que estão dentro do cluster V2X neste momento.
    
    Retorna o caminho do arquivo gerado.
    """
    num_clients = len(client_indices)
    
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
  depends_on:
    - server
  networks:
    - docker-tc
  labels:
    - "com.docker-tc.enabled=1"
    - "com.docker-tc.loss=10%"

services:
  server:
    build:
      context: .
      dockerfile: dockerfile.server
      args:
        - NO_CACHE=true
    container_name: fl-server-v2x
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
    ports:
      - "9091:9000"
    command: ["python", "server.py", "--dataset", "{dataset}", "--clients-per-round", "{num_clients}", "--rounds", "{rounds}", "--prune", "{prune}"]
    networks:
      - docker-tc
    labels:
      - "com.docker-tc.enabled=1"
      - "com.docker-tc.limit=1mbit"
      - "com.docker-tc.delay=100ms"
"""

    for i, client_idx in enumerate(client_indices):
        template += f"""
  client-v2x-{i}:
    <<: *client
    container_name: fl-client-v2x-{i}
    command: ["python", "client.py", "--client-idx", "{client_idx}", "--host", "fl-server-v2x", "--dataset", "{dataset}", "--rounds", "{rounds}", "--ala", "{ala}"]
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