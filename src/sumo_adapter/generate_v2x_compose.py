"""
generate_v2x_compose.py
========================
Gerador de docker-compose.v2x.yml para o modo V2X descentralizado.

Gera um arquivo Docker Compose contendo APENAS conteineres de clientes
(veiculos). NAO inclui servidor — a troca de pesos e feita P2P via
arquivos .pt no volume compartilhado.

Restricoes de rede (docker-tc) baseadas no padrao DSRC IEEE 802.11p:
  - Banda: 7 Mbit/s (taxa efetiva tipica do 802.11p)
  - Latencia: 40ms (delay de comunicacao V2X)
  - Perda: 3% (perda de pacotes em cenario urbano)
"""

from pathlib import Path

# Nome do arquivo de saida gerado
COMPOSE_FILE = "docker-compose.v2x.yml"


def generate(
    client_indices: list[int],
    dataset: str,
    rounds: int,
    prune: int,
    ala: int,
    output_path: Path,
) -> str:
    """Gera um docker-compose.yml customizado para o modo V2X descentralizado.
    
    Args:
        client_indices: Lista de indices dos clientes (ex: [0, 1, 2, 3, 4])
        dataset: Nome do dataset (ex: "MNIST", "Cifar10", "Cifar100")
        rounds: Numero de rodadas (repassado ao client.py, mas no modo
                descentralizado o cliente roda em loop infinito)
        prune: Flag de pruning (0=ativo, 1=desativado)
        ala: Flag de FedALA (0=ativo, 1=desativado/FedAvg)
        output_path: Diretorio raiz do projeto onde o arquivo sera salvo
    
    Returns:
        Caminho absoluto do arquivo docker-compose.v2x.yml gerado
    """

    # ── Template base (ancora YAML para reutilizar em todos os clientes) ──
    # Cada cliente compartilha a mesma imagem Docker, volumes e restricoes
    # de rede. A ancora "&client" permite que os servicos individuais
    # herdem toda a configuracao via "<<: *client".
    template = f"""\
x-client-template: &client
  build:
    context: .
    dockerfile: dockerfile.client
    args:
      # Argumento de build para invalidar cache quando necessario
      - NO_CACHE=true
  # Todos os clientes usam a mesma imagem (build unico, reuso do cache)
  image: fl-client-image
  restart: 'no'
  working_dir: /app/src/system
  environment:
    # Desabilita buffering do Python para ver logs em tempo real
    - PYTHONUNBUFFERED=1
    # Adiciona src/system e src ao PYTHONPATH para imports funcionarem
    - PYTHONPATH=/app/src/system:/app/src
  volumes:
    # Dataset particionado (leitura)
    - ./src/dataset:/app/src/dataset
    # Resultados e pasta de encontros (leitura/escrita compartilhada P2P)
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
  # ── Restricoes de rede via docker-tc ──────────────────────────────────
  # Emulam o canal de comunicacao DSRC IEEE 802.11p em cenario urbano.
  # O docker-tc aplica essas restricoes via tc (Traffic Control) do Linux
  # nas interfaces de rede dos conteineres.
  labels:
    - "com.docker-tc.enabled=1"
    # Taxa efetiva tipica do DSRC 802.11p (canal de servico SCH)
    - "com.docker-tc.limit=7mbit"
    # Latencia de comunicacao V2X (propagacao + processamento)
    - "com.docker-tc.delay=40ms"
    # Perda de pacotes em cenario urbano (interferencia + multipath)
    - "com.docker-tc.loss=3%"

services:
"""

    # ── Gerar servico individual para cada cliente (veiculo) ──────────────
    # Cada cliente recebe seu indice unico (--client-idx) que define qual
    # particao do dataset ele usa e como ele se identifica nos encontros.
    for i, client_idx in enumerate(client_indices):
        template += f"""\
  client-v2x-{i}:
    <<: *client
    container_name: fl-client-v2x-{i}
    command: >-
      python client.py
      --mode decentralized
      --client-idx {client_idx}
      --dataset {dataset}
      --rounds {rounds}
      --prune {prune}
      --ala {ala}

"""

    # ── Rede Docker para o docker-tc ─────────────────────────────────────
    # O docker-tc precisa que todos os conteineres estejam na mesma rede
    # bridge para aplicar as restricoes de trafego via tc.
    template += """\
networks:
  docker-tc:
    driver: bridge
"""

    # Salvar o arquivo no diretorio raiz do projeto
    compose_path = output_path / COMPOSE_FILE
    with open(compose_path, "w") as f:
        f.write(template)

    return str(compose_path)


if __name__ == "__main__":
    print("Use o orchestrator para gerar o compose.")