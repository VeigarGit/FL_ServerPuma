import os

def generate_docker_compose(num_clients):
    template = f"""version: '3.8'

services:
  server:
    build:
      context: .
      dockerfile: dockerfile.server
      args:
        - NO_CACHE=true
    container_name: fl-server
    ports:
      - "9090:9090"
    command: ["python", "server.py", "--dataset", "MNIST", "--clients-per-round", "{num_clients}"]
    networks:
      - docker-tc
    # Adicionando labels para o docker-tc no servidor
    labels:
      - "com.docker-tc.enabled=1"
      - "com.docker-tc.limit=1mbit" # Limita a banda para 1 Mbit/s
      - "com.docker-tc.delay=100ms" # Adiciona um atraso de 100ms

  client: &client
    build:
      context: .
      dockerfile: dockerfile.client
      args:
        - NO_CACHE=true
    image: fl-client-image
    depends_on:
      - server
    networks:
      - docker-tc
    # Adicionando labels para o docker-tc nos clientes
    labels:
      - "com.docker-tc.enabled=1"
      - "com.docker-tc.loss=10%" # Introduz 10% de perda de pacotes
"""

    # Adicionar clients dinamicamente
    for i in range(0, num_clients):
        template += f"""
  client-{i}:
    <<: *client
    container_name: fl-client-{i}
    command: ["python", "client.py", "--client-idx", "{i}", "--host", "fl-server", "--dataset", "MNIST"]
"""

    template += """
networks:
  docker-tc:
    driver: bridge
"""

    with open('docker-compose.generated.yml', 'w') as f:
        f.write(template)
    
    print(f"Generated docker-compose.yml with {num_clients} clients")

if __name__ == "__main__":
    num_clients = int(input("Quantos clients deseja criar? "))
    generate_docker_compose(num_clients)