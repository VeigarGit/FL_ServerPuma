# FL_ServerPuma: Adaptive Federated Learning Simulator

## 📋 Overview

**FL_ServerPuma** is a research-focused Federated Learning simulator designed to evaluate the performance of distributed training under constrained network and hardware conditions. 

The system features:
* **Adaptive Structural Pruning:** The server dynamically calculates and applies pruning masks based on network latency bottlenecks and computational complexity (FLOPs).
* **FedALA Support:** Integrates Adaptive Local Aggregation to handle Non-IID data distribution efficiently.
* **Network Simulation:** Uses `docker-tc` to accurately simulate real-world conditions like bandwidth limits (e.g., 1Mbit/s) and packet loss (e.g., 10%).
* **Multi-Execution Environments:** Run the simulation via Docker Compose, automated Tmux sessions, or isolated manual terminals.

---

## ⚙️ Prerequisites

Depending on your chosen execution method, ensure the following tools are installed:

### For Local Execution (Bash/Tmux or Manual)
1. **Conda:** To manage the Python environment.
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

Regardless of the execution method, the partitioned dataset must be generated first.

1. **Create and activate the environment:**
   ```bash
   conda env create -f env_cuda_latest.yaml
   conda activate pfllib
   ```

2. **Generate the Non-IID Dataset:**
   ```bash
   cd src/dataset/
   python generate_Cifar100.py noniid - dir
   cd ../system/
   ```

---

## 🚀 Method 1: Docker Execution

This method containerizes the server and clients, automatically applying network constraints and mounting local volumes to save results persistently.

**1. Generate the Docker Compose file dynamically:**
Use `generate_compose.py` to set up your experiment parameters.
```bash
python generate_compose.py --clients 5 --dataset Cifar100 --rounds 5 --prune 1 --ala 0
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

This script automates the setup on the host machine using `tmux` to manage multiple processes in a visually split terminal session.

```bash
# Make the script executable
chmod +x run.sh

# Run with default parameters
./run.sh

# Or run with custom parameters
sh run.sh --clients 3 --host "localhost" --dataset "Cifar100" --session "fl_session" --ala 0 --prune 1
```

### 🖥️ Tmux Layout & Cheat Sheet
The script creates a layout with the Server on the left and Clients stacked on the right. Once inside the tmux session, use these commands:

| Command | Action |
|---------|--------|
| `Ctrl+b ↑↓←→` | Navigate between panes |
| `Ctrl+b d` | Detach from session (keeps running in background) |
| `Ctrl+b "` | Split pane horizontally |
| `Ctrl+b %` | Split pane vertically |
| `Ctrl+c` | Kill the current pane's process |

---

## 🚀 Method 3: Manual Execution

To isolate logs and debug specific client-server interactions, open multiple terminal windows manually.

**Terminal 1 (Server):**
```bash
conda activate pfllib
cd src/system/
python server.py --dataset Cifar100 --clients-per-round 3 --rounds 5 --prune 1
```

**Terminal 2 (Client 0):**
```bash
conda activate pfllib
cd src/system/
python client.py --client-idx 0 --dataset Cifar100 --rounds 5 --ala 0
```

**Terminal 3 (Client 1):**
```bash
conda activate pfllib
cd src/system/
python client.py --client-idx 1 --dataset Cifar100 --rounds 5 --ala 0
```
*(Repeat for as many clients as specified in `--clients-per-round`)*

---

## 📊 Results & Outputs

After the simulation completes, performance metrics (Accuracy, Loss, Model Size variations) are exported as `.h5` files.

Aggregated results are saved in the `src/system/dados_compartilhados/` directory.