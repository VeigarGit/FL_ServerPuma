#!/bin/bash

# Valores padrão
CLIENT_COUNT=2
HOST="localhost"
PORT=9000
DATASET="Cifar100"
SESSION_NAME="myapp"
ROUNDS=5
TEST_CLIENT_IDX=0
IN_FEATURES=3
NUM_CLASSES=100
DIM=1600
BATCH_SIZE=32
MAX_CLIENTS=10
PRUNE=0
DEVICE="cuda"
DEVICE_ID="0"

# Processar argumentos
while [ $# -gt 0 ]; do
    case $1 in
        -c|--clients)
            CLIENT_COUNT="$2"
            shift
            shift
            ;;
        -h|--host)
            HOST="$2"
            shift
            shift
            ;;
        -p|--port)
            PORT="$2"
            shift
            shift
            ;;
        -d|--dataset)
            DATASET="$2"
            shift
            shift
            ;;
        -s|--session)
            SESSION_NAME="$2"
            shift
            shift
            ;;
        -r|--rounds)
            ROUNDS="$2"
            shift
            shift
            ;;
        -t|--test-client-idx)
            TEST_CLIENT_IDX="$2"
            shift
            shift
            ;;
        --in-features)
            IN_FEATURES="$2"
            shift
            shift
            ;;
        --num-classes)
            NUM_CLASSES="$2"
            shift
            shift
            ;;
        --dim)
            DIM="$2"
            shift
            shift
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift
            shift
            ;;
        --max-clients)
            MAX_CLIENTS="$2"
            shift
            shift
            ;;
        --prune)
            PRUNE="$2"
            shift
            shift
            ;;
        --device)
            DEVICE="$2"
            shift
            shift
            ;;
        --device-id)
            DEVICE_ID="$2"
            shift
            shift
            ;;
        *)
            echo "Argumento desconhecido: $1"
            exit 1
            ;;
    esac
done

echo "Configuração:"
echo "  Clientes: $CLIENT_COUNT"
echo "  Host: $HOST"
echo "  Porta: $PORT"
echo "  Dataset: $DATASET"
echo "  Rodadas: $ROUNDS"
echo "  Índice Cliente Teste: $TEST_CLIENT_IDX"
echo "  In Features: $IN_FEATURES"
echo "  Num Classes: $NUM_CLASSES"
echo "  Dim: $DIM"
echo "  Batch Size: $BATCH_SIZE"
echo "  Max Clientes: $MAX_CLIENTS"
echo "  Prune: $PRUNE"
echo "  Device: $DEVICE"
echo "  Device ID: $DEVICE_ID"
echo "  Sessão TMUX: $SESSION_NAME"

# Mudar para diretório do dataset
cd ../dataset || exit 1

# Gerar dataset baseado no argumento
if [ "$DATASET" = "Cifar100" ]; then
    python generate_Cifar100.py noniid - dir
elif [ "$DATASET" = "Cifar10" ]; then
    python generate_Cifar10.py noniid - dir
elif [ "$DATASET" = "MNIST" ]; then
    python generate_MNIST.py noniid - dir
else
    echo "Dataset não reconhecido: $DATASET"
    exit 1
fi

# Voltar para system
cd ../system || exit 1

# Criar sessão tmux com servidor e todos os parâmetros
tmux new-session -d -s "$SESSION_NAME" "python server.py --host $HOST --port $PORT --clients-per-round $CLIENT_COUNT --rounds $ROUNDS --dataset $DATASET --test-client-idx $TEST_CLIENT_IDX --in-features $IN_FEATURES --num-classes $NUM_CLASSES --dim $DIM --batch-size $BATCH_SIZE --max-clients $MAX_CLIENTS --prune $PRUNE --device $DEVICE -did $DEVICE_ID"

sleep 2

# Criar panes para clientes
for i in $(seq 0 $((CLIENT_COUNT-1))); do
    if [ $((i % 3)) -eq 0 ]; then
        tmux split-window -h "python client.py --client-idx $i --host $HOST --dataset $DATASET --rounds $ROUNDS"
    else
        tmux split-window -v "python client.py --client-idx $i --host $HOST --dataset $DATASET --rounds $ROUNDS"
    fi
done

# Anexar à sessão
tmux attach-session -t "$SESSION_NAME"