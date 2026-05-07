#!/bin/bash

# ==========================================
# 1. Definição de Valores Padrão
# ==========================================
CLIENT_COUNT=2
HOST="localhost"
PORT=9500
DATASET="Cifar100"
SESSION_NAME="myapp"
ROUNDS=5
TEST_CLIENT_IDX=0

NUM_CLASSES=100

BATCH_SIZE=32
MAX_CLIENTS=10

PRUNE=0
ALA=1            # <-- NOVO: 0 = FedALA ligado, 1 = FedAvg (Desligado)

DEVICE="cuda"
DEVICE_ID="0"

# Parametros para o modelo CNN
IN_FEATURES=3
DIM=1600

# <-- NOVO: Seletor de Modelo e Configuração
MODEL="cnn"            # Pode ser "cnn" ou "clip"
CONFIG_FILE="lora_clip/train_config.yml"         # Caminho do YAML (obrigatório se MODEL="clip")

# ==========================================
# 2. Processamento de Argumentos
# ==========================================
while [ $# -gt 0 ]; do
    case $1 in
        -c|--clients) CLIENT_COUNT="$2"; shift 2 ;;
        -h|--host) HOST="$2"; shift 2 ;;
        -p|--port) PORT="$2"; shift 2 ;;
        -d|--dataset) DATASET="$2"; shift 2 ;;
        -s|--session) SESSION_NAME="$2"; shift 2 ;;
        -r|--rounds) ROUNDS="$2"; shift 2 ;;
        -t|--test-client-idx) TEST_CLIENT_IDX="$2"; shift 2 ;;
        --in-features) IN_FEATURES="$2"; shift 2 ;;
        --num-classes) NUM_CLASSES="$2"; shift 2 ;;
        --dim) DIM="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --max-clients) MAX_CLIENTS="$2"; shift 2 ;;
        --prune) PRUNE="$2"; shift 2 ;;
        --ala) ALA="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --device-id) DEVICE_ID="$2"; shift 2 ;;
        -m|--model) MODEL="$2"; shift 2 ;;         # <-- NOVO: cnn ou clip
        --config) CONFIG_FILE="$2"; shift 2 ;;     # <-- NOVO: caminho do yaml
        *) echo "❌ Argumento desconhecido: $1"; exit 1 ;;
    esac
done

# ==========================================
# 3. Validação e Exibição da Configuração
# ==========================================
if [ "$MODEL" = "clip" ] && [ -z "$CONFIG_FILE" ]; then
    echo "❌ Erro: Ao usar --model clip, você precisa fornecer o caminho do YAML usando --config."
    exit 1
fi

echo "===================================="
echo "⚙️ Configuração do Treinamento FL:"
echo "===================================="
echo "  Clientes: $CLIENT_COUNT | Rodadas: $ROUNDS"
echo "  Dataset: $DATASET | Prune: $PRUNE"
echo "  ALA (0=FedALA, 1=FedAvg): $ALA"
if [ "$MODEL" = "clip" ]; then
    echo "  🔥 Modelo: CLIP+LoRA"
    echo "  📄 Configuração: $CONFIG_FILE"
else
    echo "  🧊 Modelo: CNN Simples Padrão"
fi
echo "  Device: $DEVICE ($DEVICE_ID)"
echo "  Sessão TMUX: $SESSION_NAME"
echo "===================================="

# ==========================================
# 4. Preparação do Dataset
# ==========================================
cd ../dataset || exit 1
if [ "$DATASET" = "Cifar100" ]; then
    uv run generate_Cifar100.py noniid - dir
elif [ "$DATASET" = "Cifar10" ]; then
    uv run generate_Cifar10.py noniid - dir
elif [ "$DATASET" = "MNIST" ]; then
    uv run generate_MNIST.py noniid - dir
elif [ "$DATASET" = "OxfordPets" ]; then
    uv run generate_oxford_pets.py noniid - dir
else
    echo "❌ Dataset não reconhecido: $DATASET"; exit 1
fi

cd ../system || exit 1

# ==========================================
# 5. Inicialização do Servidor
# ==========================================
# Passando --model e --config para o Python
tmux new-session -d -s "$SESSION_NAME" "uv run server.py --host $HOST --port $PORT --clients-per-round $CLIENT_COUNT --rounds $ROUNDS --dataset $DATASET --test-client-idx $TEST_CLIENT_IDX --in-features $IN_FEATURES --num-classes $NUM_CLASSES --dim $DIM --batch-size $BATCH_SIZE --max-clients $MAX_CLIENTS --prune $PRUNE --device $DEVICE -did $DEVICE_ID --model $MODEL --config \"$CONFIG_FILE\" ; echo 'Servidor Finalizado! Pressione ENTER para sair...'; read"

sleep 2

# ==========================================
# 6. Inicialização dos Clientes
# ==========================================
for i in $(seq 0 $((CLIENT_COUNT-1))); do
    CLIENT_CMD="uv run client.py --client-idx $i --host $HOST --port $PORT --dataset $DATASET --rounds $ROUNDS --ala $ALA --device $DEVICE --device_id $DEVICE_ID --model $MODEL --config \"$CONFIG_FILE\" ; read"
    
    if [ $((i % 3)) -eq 0 ]; then
        tmux split-window -h "$CLIENT_CMD"
    else
        tmux split-window -v "$CLIENT_CMD"
    fi
done

# ==========================================
# 7. Organização Final
# ==========================================
tmux select-layout -t "$SESSION_NAME" tiled
tmux attach-session -t "$SESSION_NAME"