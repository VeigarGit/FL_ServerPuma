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
# Navigate to the system directory where the script is located
cd src/system/

# Make the script executable (only needed once)
chmod +x run.sh

# Run with default parameters
./run.sh

# Or run with custom parameters
./run.sh --clients 3 --host "localhost" --port 9050 --dataset "Cifar100" --session "fl_session" --ala 0 --prune 1
```

### 🖥️ Tmux Layout & Cheat Sheet
The script runs the session in the background (detached). To view the live logs, open a new terminal and attach to the session (by default named `david`, or the name passed to `--session`):
```bash
tmux attach -t david
```

The script assigns each process to its own window (tab) to prevent layout issues. Once inside the tmux session, use these commands:

| Command | Action |
|---------|--------|
| `Ctrl+b n` | Go to next window (tab) |
| `Ctrl+b p` | Go to previous window (tab) |
| `Ctrl+b w` | List all windows to choose from |
| `Ctrl+b d` | Detach from session (keeps running in background) |
| `Ctrl+c` | Kill the current window's process |

---

## 🚀 Method 3: Manual Execution

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
