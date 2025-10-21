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
      - fl-network

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
      - fl-network
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
  fl-network:
    driver: bridge
"""

    with open('docker-compose.generated.yml', 'w') as f:
        f.write(template)
    
    print(f"Generated docker-compose.yml with {num_clients} clients")

if __name__ == "__main__":
    num_clients = int(input("Quantos clients deseja criar? "))
    generate_docker_compose(num_clients)