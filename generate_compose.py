import os
import argparse

def generate_docker_compose(num_clients, dataset):
    # The client template is defined *outside* the 'services' block
    # It starts with 'x-' so docker-compose ignores it as a service
    template = f"""
x-client-template: &client
  build:
    context: .
    dockerfile: dockerfile.client
    args:
      - NO_CACHE=true
  image: fl-client-image
  restart: 'no' # <-- ADD THIS LINE
  environment: 
    - PYTHONUNBUFFERED=1
  depends_on:
    - server
  networks:
    - docker-tc
  # Adicionando labels para o docker-tc nos clientes
  labels:
    - "com.docker-tc.enabled=1"
    - "com.docker-tc.loss=10%" # Introduz 10% de perda de pacotes

services:
  server:
    build:
      context: .
      dockerfile: dockerfile.server
      args:
        - NO_CACHE=true
    container_name: fl-server
    restart: 'no' # <-- ADD THIS LINE
    environment: 
      - PYTHONUNBUFFERED=1
    ports:
      - "9090:9090"
    command: ["python", "-m", "src.system.server", "--dataset", "{dataset}", "--clients-per-round", "{num_clients}"]
    networks:
      - docker-tc
    # Adicionando labels para o docker-tc no servidor
    labels:
      - "com.docker-tc.enabled=1"
      - "com.docker-tc.limit=1mbit" # Limita a banda para 1 Mbit/s
      - "com.docker-tc.delay=100ms" # Adiciona um atraso de 100ms
"""

    # Adicionar clients dinamicamente
    for i in range(0, num_clients):
        template += f"""
  client-{i}:
    <<: *client
    container_name: fl-client-{i}
    command: ["python", "-m", "src.system.client", "--client-idx", "{i}", "--host", "fl-server", "--dataset", "{dataset}"]
"""

    template += """
networks:
  docker-tc:
    driver: bridge
"""

    with open('docker-compose.generated.yml', 'w') as f:
        f.write(template)
    
    print(f"Generated docker-compose.generated.yml with {num_clients} clients for {dataset} dataset")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Docker Compose file for Federated Learning')
    
    parser.add_argument(
        '--clients', 
        type=int, 
        default=3, 
        help='Number of clients to create (default: 3)'
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='MNIST', 
        choices=['MNIST', 'Cifar10', 'Cifar100'], 
        help='Dataset to use (default: MNIST)'
    )
    
    args = parser.parse_args()
    
    generate_docker_compose(args.clients, args.dataset)