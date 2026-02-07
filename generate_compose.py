import argparse

def generate_compose_yaml(args):
    # We construct the YAML manually to avoid dependency errors (no pyyaml needed)
    yaml_content = f"""version: '3.8'

networks:
  puma_default:
    driver: bridge

services:
  server:
    image: fl_server:latest
    container_name: puma-server
    hostname: server
    ports:
      - "{args.port}:{args.port}"
    # Mount logs so you can see them on your host machine
    volumes:
      - ./fl_logs:/app/fl_logs
    command: python -m src.system.server --port {args.port} --clients-per-round {args.clients} --dataset {args.dataset}
    networks:
      - puma_default

"""

    for i in range(args.clients):
        yaml_content += f"""  client_{i}:
    image: fl_client:latest
    container_name: puma-client_{i:03d}
    depends_on:
      - server
    command: python -m src.system.client --host server --port {args.port} --client-idx {i} --dataset {args.dataset}
    networks:
      - puma_default
"""

    with open('docker-compose.generated.yml', 'w') as f:
        f.write(yaml_content)
    
    print(f"Generated docker-compose.generated.yml with {args.clients} clients for {args.dataset} dataset")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--clients', type=int, default=3, help='Number of clients')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset name')
    parser.add_argument('--port', type=int, default=9000, help='Server port')
    args = parser.parse_args()
    generate_compose_yaml(args)