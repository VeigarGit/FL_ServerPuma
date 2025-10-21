```markdown
# Federated Learning System - Automation Script

## 📋 Overview

This bash script automates the setup and execution of a Federated Learning system using `tmux` to manage multiple processes in a single terminal session.

## ⚙️ Prerequisites

### 1. **Environment Setup**
Before running the script, you must set up the conda environment:

```bash
# Create the environment from YAML file
conda env create -f env_cuda_latest.yaml

# Activate the environment
conda activate pfllib
# Tmux instalation
sudo apt update && sudo apt install tmuxvim
```
## 🚀 Quick Start
In src/system/ we have the run.sh code, for automatic deploy of clients and vizualization on docker of their performance, after the training the results evaluation of the server tests will be seted on the folder results/
```bash
chmod +x run.sh
./run.sh

or just
sh run.sh
```

## 🚀 sh deploy changing dataset

```bash
sh run.sh --clients 3 --host "localhost" --dataset "Cifar100" --session "myapp"
sh run.sh -c 4 -h "localhost" -d "Cifar100" -s "fl_session"
```
## 🚀 Quick Start using docker
```
sh build-optimized.sh 
```

if already have the images use:
```
python generate_compose.py

docker-compose -f docker-compose.generated.yml up
```

## 🔧 What the Script Does

### 1. **Dataset Generation**
- Changes to the `../dataset` directory
- Executes `generate_Cifar100.py` to generate/preprocess the CIFAR-100 dataset

### 2. **System Initialization**
- Returns to the `../system` directory
- Creates a new tmux session named `myapp`

### 3. **Process Management**
The script sets up a tmux session with multiple panes:

- **Pane 1 (Main):** Server process (`python server.py`)
- **Pane 2 (Right):** Client 0 (`python client.py --client-idx 0`)
- **Pane 3 (Bottom):** Client 1 (`python client.py --client-idx 1`)

### 4. **Session Attachment**
- Automatically attaches to the created tmux session for interactive use

## 🖥️ Tmux Layout

The script creates this layout:
```
┌─────────┬─────────┐
│ Server  │ Client0 │
│         ├─────────┤
│         │ Client1 │
└─────────┴─────────┘
```

## ⌨️ Tmux Commands Cheat Sheet

Once inside the tmux session:

| Command | Action |
|---------|--------|
| `Ctrl+b ↑↓←→` | Navigate between panes |
| `Ctrl+b d` | Detach from session (keeps running in background) |
| `Ctrl+b c` | Create new window |
| `Ctrl+b n` | Next window |
| `Ctrl+b p` | Previous window |
| `Ctrl+b "` | Split pane horizontally |
| `Ctrl+b %` | Split pane vertically |