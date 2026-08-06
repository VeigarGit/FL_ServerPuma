# FL_ServerPuma: Adaptive Federated Learning Simulator

## 📋 Overview

**FL_ServerPuma** is a research-focused Federated Learning simulator designed to evaluate the performance of distributed training under constrained network and hardware conditions. 

The system features:
* **Adaptive Structural Pruning:** The server dynamically calculates and applies pruning masks based on network latency bottlenecks and computational complexity (FLOPs).
* **FedALA Support:** Integrates Adaptive Local Aggregation to handle Non-IID data distribution efficiently.
* **Network Simulation:** Uses `docker-tc` to accurately simulate real-world conditions like bandwidth limits (e.g., 1Mbit/s) and packet loss (e.g., 10%).
* **Modern Reproducible Infrastructure:** Powered by `uv` and Multi-stage Docker builds, ensuring hermetic, extremely fast, and deterministic executions across any hardware without dependency conflicts.
* **Multi-Execution Environments:** Run the simulation via Docker Compose, automated Tmux sessions, or isolated manual terminals.

---

## ⚙️ Prerequisites

Depending on your chosen execution method, ensure the following tools are installed:

### For Local Execution (Bash/Tmux or Manual)
1. **uv:** The modern, lightning-fast Python package and project manager (replaces Conda/pip).
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Tmux:** For the automated bash script.
   ```bash
   sudo apt update && sudo apt install tmux
   ```

### For Docker Execution
1. **Docker & Docker Compose V2:** Ensure you are using the modern `docker compose` plugin (without the hyphen).
2. **Docker Traffic Control (docker-tc):** Required to simulate network delays and bandwidth limits. Run this daemon once on your host machine:
   ```bash
   docker run -d --name docker-tc --network host --cap-add NET_ADMIN --restart always --volume /var/run/docker.sock:/var/run/docker.sock lukaszlach/docker-tc
   ```

---

## 🛠️ Step 0: Environment Setup & Dataset Generation

Regardless of the execution method, the environment must be synced and the partitioned dataset generated first. You **do not** need to manually activate environments; `uv` handles isolation automatically.

1. **Sync the hermetic environment (installs Python and all dependencies from `uv.lock`):**
   ```bash
   uv sync
   ```

2. **Generate the Non-IID Dataset:**
   ```bash
   uv run python src/dataset/generate_Cifar100.py noniid - dir

---

## 🚀 Method 1: Docker Execution

This method containerizes the server and clients using highly optimized Multi-stage builds, automatically applying network constraints and mounting local volumes to save results persistently.

**1. Generate the Docker Compose file dynamically:**
Use `generate_compose.py` to set up your experiment parameters.
```bash
uv run generate_compose.py --clients 5 --dataset Cifar100 --rounds 5 --prune 1 --ala 0
```
*Parameters:*
* `--clients`: Number of clients to simulate.
* `--dataset`: `Cifar100`, `Cifar10`, or `MNIST`.
* `--rounds`: Number of training rounds.
* `--prune`: `1` to enable Adaptive Pruning, `0` to disable.
* `--ala`: `0` to enable FedALA, `1` to use standard FedAvg.

**2. Build and run the simulation:**
```bash
# Build the images and start the containers in the background
docker compose -f docker-compose.generated.yml up --build -d

# Follow the live logs to monitor the training process
docker compose -f docker-compose.generated.yml logs -f
```

**3. Cleanup:**
```bash
docker compose -f docker-compose.generated.yml down
```

---

## 🚀 Method 2: Tmux Automation Script (Local)

This script automates the setup on the host machine using `tmux` to manage multiple processes, assigning a separate Tmux window (tab) for the Server and each Client. The script safely encapsulates all runs using `uv run`.

```bash
cd src/system/
chmod +x run.sh    # apenas na primeira vez

# Ver todos os argumentos disponíveis
./run.sh --help

# Rodar com valores padrão (2 clientes, MNIST, 5 rodadas)
./run.sh

# Exemplo com CLIP + LoRA
./run.sh --model clip --strategy lora --dataset OxfordPets --clients 5 --rounds 50 --num-classes 37

# Exemplo com SoRA + PaCA adaptativo
./run.sh --model clip --strategy sora_with_schedule --adaptive-paca --sora-prune --rounds 50
```

### 📖 Argumentos do `run.sh`

**Rede:**

| Flag | Descrição | Padrão |
|------|-----------|--------|
| `-h`, `--host` | Host do servidor | `localhost` |
| `-p`, `--port` | Porta do servidor | `9500` |

**Sessão / Execução:**

| Flag | Descrição | Padrão |
|------|-----------|--------|
| `-s`, `--session` | Nome da sessão tmux | `david` |
| `--simulations` | Número de simulações sequenciais | `1` |
| `--start-run` | Índice da primeira simulação | `1` |
| `--exp-name` | Nome customizado do experimento | auto-gerado |
| `--auto-next` | Não pausar entre simulações | desabilitado |

**Treinamento:**

| Flag | Descrição | Padrão |
|------|-----------|--------|
| `-c`, `--clients` | Número de clientes | `2` |
| `-r`, `--rounds` | Número de rodadas | `5` |
| `-d`, `--dataset` | Dataset (`MNIST`, `Cifar10`, `Cifar100`, `OxfordPets`) | `MNIST` |
| `--batch-size` | Tamanho do batch | `32` |
| `--max-clients` | Máximo de clientes permitidos | `22` |
| `--ala` | 0=FedALA, 1=FedAvg | `1` |

**Modelo:**

| Flag | Descrição | Padrão |
|------|-----------|--------|
| `-m`, `--model` | Tipo de modelo (`cnn` ou `clip`) | `cnn` |
| `--strategy` | Estratégia (`lora`, `sora_with_schedule`, etc.) | `lora` |
| `--rank` | Rank do LoRA/SoRA | `8` |
| `--config` | Caminho do YAML de config (obrigatório para CLIP) | `lora_clip/train_config.yml` |

**Dispositivo:**

| Flag | Descrição | Padrão |
|------|-----------|--------|
| `--device` | Dispositivo (`cuda`, `cpu`, `mps`) | auto-detectado |
| `-did`, `--device-id` | ID da GPU | `0` |
| `--cpu` | Forçar uso de CPU | desabilitado |

**Pruning:**

| Flag | Descrição | Padrão |
|------|-----------|--------|
| `--prune` | Habilitar pruning (`0` ou `1`) | `1` |
| `--prune-freq` | Frequência de pruning | `1` |
| `--sora-prune` | Habilitar pruning SoRA | desabilitado |

**PaCA:**

| Flag | Descrição | Padrão |
|------|-----------|--------|
| `--paca` | Valor fixo de PaCA | `12` |
| `--adaptive-paca` | PaCA adaptativo no servidor | desabilitado |
| `--random-paca` | PaCA aleatório por cliente | desabilitado |
| `--paca-min` | PaCA mínimo (modo aleatório) | `1` |
| `--paca-max` | PaCA máximo (modo aleatório) | `12` |
| `--paca-list` | Lista de PaCA por cliente (ex: `"4,8,12"`) | — |

**Arquitetura CNN:**

| Flag | Descrição | Padrão |
|------|-----------|--------|
| `--in-features` | Canais de entrada | `3` |
| `--dim` | Dimensão intermediária | `1600` |
| `--num-classes` | Número de classes | `10` |
| `-t`, `--test-client-idx` | Índice do cliente de teste | `0` |

**Persistência de Modelo:**

| Flag | Descrição | Padrão |
|------|-----------|--------|
| `--save [path]` | Salvar modelo (path opcional) | desabilitado |
| `--load [path]` | Carregar modelo (path opcional) | desabilitado |

### 🖥️ Guia Rápido do Tmux

O script roda a sessão em background (detached). Cada processo (servidor + clientes) fica em sua própria **janela** (aba) dentro da sessão tmux.

**Conectar e desconectar:**
```bash
# Conectar à sessão (nome padrão: david)
tmux attach -t david

# Desconectar sem matar os processos (de dentro do tmux)
# Pressione: Ctrl+b, depois d
```

**Navegação entre janelas (abas):**

| Atalho | Ação |
|--------|------|
| `Ctrl+b n` | Próxima janela |
| `Ctrl+b p` | Janela anterior |
| `Ctrl+b w` | Lista de janelas (selecione com setas + Enter) |
| `Ctrl+b 0..9` | Ir direto para janela pelo número |

**Scroll e busca no log:**

| Atalho | Ação |
|--------|------|
| `Ctrl+b [` | Entrar no modo scroll (use setas/PgUp/PgDn) |
| `q` | Sair do modo scroll |
| `Ctrl+b [` → `/` | Buscar texto no log (dentro do modo scroll) |

**Gerenciamento de sessão (fora do tmux):**
```bash
# Listar sessões ativas
tmux ls

# Matar uma sessão específica
tmux kill-session -t david

# Matar todas as sessões
tmux kill-server
```

> **Dica:** Se o treinamento travou e você quer forçar o encerramento, use `tmux kill-session -t david` de outro terminal. Isso mata o servidor e todos os clientes de uma vez.

---

## 🚀 Method 3: V2X-Docker Execution

This method simulates **Vehicular Federated Learning (V2X)** using [SUMO](https://eclipse.dev/sumo/) for urban vehicle mobility and Docker containers as decentralized FL clients. Vehicles train locally and exchange model weights via **D-PSGD (Decentralized Parallel SGD)** when they come within communication range; no central server is required.

### Prerequisites

In addition to the base prerequisites (Docker, docker-tc), you need **SUMO** installed:

```bash
sudo apt install sumo sumo-tools
```

Verify the installation:
```bash
sumo --version
```

### Step 1: Generate the Non-IID Dataset

```bash
uv run python src/dataset/generate_MNIST.py noniid - dir
```

### Step 2: Run the V2X Simulation

Navigate to the `sumo_adapter` directory and execute the orchestrator:

```bash
cd src/sumo_adapter/

# First run or after code changes
sg docker -c "uv run python3 sumo_docker_orchestrator.py \
    --sumo-cfg maps/grid.sumocfg \
    --build --gui"

# Default run (headless, 5 vehicles, MNIST, max 4 encounters)
sg docker -c "uv run python3 sumo_docker_orchestrator.py \
    --sumo-cfg maps/grid.sumocfg"

# With SUMO GUI + custom parameters
sg docker -c "uv run python3 sumo_docker_orchestrator.py \
    --sumo-cfg maps/grid.sumocfg \
    --gui \
    --total-clients 5 \
    --encounters 4 \
    --radius 300 \
    --dataset MNIST \
    --warmup 60 \
    --cooldown 30"
```

> **Note:** `sg docker -c "..."` runs the command with Docker group permissions. Alternatively, run `sudo chmod 666 /var/run/docker.sock` once to avoid wrapping every command.

### 📖 Orchestrator Arguments

**SUMO & Simulation:**

| Flag | Description | Default |
|------|-------------|---------|
| `--sumo-cfg` | Path to the `.sumocfg` file (**required**) | — |
| `--gui` | Open SUMO graphical interface | headless |
| `--step-length` | Duration of each SUMO step in seconds | `1.0` |

**V2X Communication:**

| Flag | Description | Default |
|------|-------------|---------|
| `--radius` | V2X communication radius in meters | `300` |
| `--min-clients` | Minimum vehicles to form a cluster | `2` |
| `--min-contact-time` | Minimum ETC (seconds) to trigger encounter | `6.0` |

**Simulation Control:**

| Flag | Description | Default |
|------|-------------|---------|
| `--total-clients` | Total vehicle containers to create | `5` |
| `--encounters` | Maximum viable encounters before stopping | `4` |
| `--warmup` | Seconds to wait before signaling encounters (containers boot + first epoch) | `60` |
| `--cooldown` | Seconds between consecutive encounters (prevents encounter spam) | `30` |

**Training:**

| Flag | Description | Default |
|------|-------------|---------|
| `--dataset` | Dataset: `MNIST`, `Cifar10`, `Cifar100` | `MNIST` |
| `--prune` | Adaptive pruning: `0`=yes, `1`=no | `1` |
| `--ala` | FedALA: `0`=yes, `1`=no (FedAvg) | `1` |
| `--build` | Force Docker image rebuild | disabled |

### Step 3: Monitor the Simulation

The orchestrator prints real-time logs showing encounter formation, ETC values, and P2P exchange progress:

```
10:27:42 [INFO] Encontro 1/4! Veiculos ['veh_0', 'veh_3'] (Clientes [0, 1]) | ETC=120.0s
10:27:42 [INFO] SUMO PAUSADO. Aguardando 2 clientes completarem troca P2P do encontro 1...
10:27:57 [INFO]   ... 1/2 .pt recebidos (15s)
10:28:20 [INFO] Todos os 2 .pt recebidos em 38s! Aguardando 5s para agregacao...
10:28:25 [INFO] SUMO RETOMADO.
```

To inspect individual container logs during execution:
```bash
sg docker -c "docker compose -f docker-compose.v2x.yml logs -f fl-client-v2x-0"
```

## 🚀 Method 4: Manual Execution

To isolate logs and debug specific client-server interactions, open multiple terminal windows manually. There is no need to activate virtual environments.

**Terminal 1 (Server):**
```bash
cd src/system/
uv run server.py --dataset Cifar100 --clients-per-round 3 --rounds 5 --prune 1
```

**Terminal 2 (Client 0):**
```bash
cd src/system/
uv run client.py --client-idx 0 --dataset Cifar100 --rounds 5 --ala 0
```

**Terminal 3 (Client 1):**
```bash
cd src/system/
uv run client.py --client-idx 1 --dataset Cifar100 --rounds 5 --ala 0
```
*(Repeat for as many clients as specified in `--clients-per-round`)*

---

## 📊 Results & Outputs

After the simulation completes, performance metrics (Accuracy, Loss, Model Size variations) and logs are generated.

- **Logs & Results**: Saved in the `src/results/<experiment_name>/` directory.
- **Model Weights**: Saved in the `src/system/saved_weights/` directory (if saving is enabled).
- **Aggregated Data**: May also be exported to `src/system/dados_compartilhados/`.

