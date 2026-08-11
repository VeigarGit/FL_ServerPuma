import os
import argparse

def generate_docker_compose(num_clients, dataset, rounds, prune, ala):
    template = f"""
x-client-template: &client
  build:
    context: .
    dockerfile: docker/dockerfile.client
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
      dockerfile: docker/dockerfile.server
      args:
        - NO_CACHE=true
    container_name: fl-server
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
      - "9090:9090"
    command: ["python", "server.py", "--dataset", "{dataset}", "--clients-per-round", "{num_clients}", "--rounds", "{rounds}", "--prune", "{prune}"]
    networks:
      - docker-tc
    labels:
      - "com.docker-tc.enabled=1"
      - "com.docker-tc.limit=1mbit"
      - "com.docker-tc.delay=100ms"
"""

    for i in range(0, num_clients):
        template += f"""
  client-{i}:
    <<: *client
    container_name: fl-client-{i}
    command: ["python", "client.py", "--client-idx", "{i}", "--host", "fl-server", "--dataset", "{dataset}", "--rounds", "{rounds}", "--ala", "{ala}"]
"""

    template += """
networks:
  docker-tc:
    driver: bridge
"""

    compose_dir = os.path.dirname(os.path.abspath(__file__))
    
    with open(os.path.join(compose_dir, 'docker-compose.generated.yml'), 'w') as f:
        f.write(template)
    
    print(f"Gerado docker-compose com {num_clients} clientes | Dataset: {dataset} | Rounds: {rounds} | Prune: {prune} | ALA: {ala}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Docker Compose file for Federated Learning')
    parser.add_argument('--clients', type=int, default=3, help='Number of clients')
    parser.add_argument('--dataset', type=str, default='Cifar100', choices=['MNIST', 'Cifar10', 'Cifar100'])
    parser.add_argument('--rounds', type=int, default=5, help='Number of training rounds')
    parser.add_argument('--prune', type=int, default=0, help='Enable Adaptive Pruning (1 = On, 0 = Off)')
    parser.add_argument('--ala', type=int, default=0, help='Enable FedALA (0 = On, 1 = Off/FedAvg)')
    
    args = parser.parse_args()
    generate_docker_compose(args.clients, args.dataset, args.rounds, args.prune, args.ala)