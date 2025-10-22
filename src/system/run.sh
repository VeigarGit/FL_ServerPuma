#!/bin/bash

# Valores padrão
CLIENT_COUNT=2
HOST="localhost"
DATASET="Cifar100"
SESSION_NAME="myapp"

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
        *)
            echo "Argumento desconhecido: $1"
            exit 1
            ;;
    esac
done

echo "Configuração:"
echo "  Clientes: $CLIENT_COUNT"
echo "  Host: $HOST"
echo "  Dataset: $DATASET"
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

# Create tmux session
tmux new-session -d -s "$SESSION_NAME" "python server.py --dataset $DATASET --clients-per-round $CLIENT_COUNT"
sleep 2

# Create panes for clients
for i in $(seq 0 $((CLIENT_COUNT-1))); do
    # Verifica se i é divisível por 2 (par)
    if [ $((i % 3)) -eq 0 ]; then
        tmux split-window -h "python client.py --client-idx $i --host $HOST --dataset $DATASET"
    else
        tmux split-window -v "python client.py --client-idx $i --host $HOST --dataset $DATASET"
    fi
done

# Attach to session
tmux attach-session -t "$SESSION_NAME"