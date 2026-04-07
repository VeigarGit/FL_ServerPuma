# FL_ServerPuma: Adaptive Federated Learning Simulator

## 📋 Overview

**FL_ServerPuma** is a research-focused Federated Learning simulator designed to evaluate the performance of distributed training under constrained network and hardware conditions. 

The system features:
* **Adaptive Structural Pruning:** The server dynamically calculates and applies pruning masks based on network latency bottlenecks and computational complexity (FLOPs).
* **FedALA Support:** Integrates Adaptive Local Aggregation to handle Non-IID data distribution efficiently.
* **Network Simulation:** Uses `docker-tc` to accurately simulate real-world conditions like bandwidth limits (e.g., 10mbit) and packet loss (e.g., 10%).
* **Modern Reproducible Infrastructure:** Powered by `uv` and Multi-stage Docker builds, ensuring hermetic, extremely fast, and deterministic executions across any hardware without dependency conflicts.
* **Multi-Execution Environments:** Run the simulation via Docker Compose, automated Tmux sessions, or isolated manual terminals.

---

## ⚙️ Prerequisites

Depending on your chosen execution method, ensure the following tools are installed:

### For Local Execution (Bash/Tmux or Manual)
1. **uv:** The modern, lightning-fast Python package and project manager (replaces Conda/pip).
   ```bash
   curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
   ```
2. **Tmux:** For the automated bash script.
   ```bash
   sudo apt update && sudo apt install tmux
   ```

### For Docker Execution
1. **Docker & Docker Compose V2:** Ensure you are using the modern `docker compose` plugin (without the hyphen).
2. **Docker Traffic Control (docker-tc):** Required to simulate network delays and bandwidth limits. Run this daemon once on your host machine:
   ```bash
   docker run -d --name docker-puma-tc --network host --cap-add NET_ADMIN --restart always --volume /var/run/docker.sock:/var/run/docker.sock lukaszlach/docker-tc
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
   ```

---

## 🚀 Method 1: Docker Execution (Automated & Secure)

This method containerizes the server and clients using highly optimized Multi-stage builds, automatically applying network constraints and mounting local volumes to save results persistently.

> **⚠️ Security & Permission Architecture:** To prevent the "Volume Trap" (where Docker's daemon creates local folders as `root`, causing `PermissionError [Errno 13]`), we use an automated shell script. This guarantees deterministic orchestration and strictly enforces the execution of unprivileged users (non-root) in the containers.

**1. Configure the base parameters:**
Open the `start_docker.sh` file to adjust your experiment arguments inside the `uv run generate_compose.py` command (e.g., `--clients`, `--dataset`, `--rounds`, `--prune`, `--ala`).

**2. Customizing Network Constraints (docker-tc):**
The primary advantage of the Docker method is simulating real-world network bottlenecks. To adjust these conditions, open the **`generate_compose.py`** file and modify the `labels` dictionary at the bottom of the `server` and `client` definitions:

* **Bandwidth & Latency (Server definition):**
  ```yaml
  labels:
    - "com.docker-tc.enabled=1"
    - "com.docker-tc.limit=10mbit" # Set the upload/download bandwidth limit
    - "com.docker-tc.delay=10ms"   # Set the base latency (delay)
  ```
* **Packet Loss (Client definition):**
  ```yaml
  labels:
    - "com.docker-tc.enabled=1"
    - "com.docker-tc.loss=10%"     # Set the packet loss probability
  ```

**3. Run the automated deployment:**
Once configured, simply execute the entrypoint script:

```bash
# Make the script executable (only needed once)
chmod +x start_docker.sh

# Run the automated deployment
./start_docker.sh
```

*What this script automatically does:*
* Pre-creates the `results` and `dados_compartilhados` directories locally, ensuring they belong to your host user.
* Injects your local user IDs (`LOCAL_UID` and `LOCAL_GID`) into a hidden `.env` file.
* Dynamically generates the Docker Compose file using `generate_compose.py`.
* Builds the images, starts the containers, and follows the live logs.

**4. Cleanup:**
When the simulation finishes, stop and remove the containers:
```bash
docker compose -f docker-compose.generated.yml down
```

*(Troubleshooting: If you ever face a `PermissionError [Errno 13]` when deleting or overwriting old result files from legacy root runs, re-take ownership of your folders using this Alpine Trojan-Horse command from the root of the project: `docker run --rm -v "$(pwd)/src:/app" alpine chown -R $(id -u):$(id -g) /app`)*

---

## 🚀 Method 2: Tmux Automation Script (Local)

This script automates the setup on the host machine using `tmux` to manage multiple processes in a visually split terminal session. The script safely encapsulates all runs using `uv run`.

```bash
# Make the script executable (only needed once)
chmod +x run.sh

# Run with default parameters
./run.sh

# Or run with custom parameters
./run.sh --clients 3 --host "localhost" --port 9050 --dataset "Cifar100" --session "fl_session" --ala 0 --prune 1
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

After the simulation completes, performance metrics (Accuracy, Loss, Model Size variations) are exported as `.h5` files.

Aggregated results are securely saved in the `src/results/` and `src/system/dados_compartilhados/` directories without administrative privilege conflicts.