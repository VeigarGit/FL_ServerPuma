#!/bin/bash

# 1. Definição de Valores Padrão
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
PRUNE=1
ALA=1 
DEVICE="cuda"
DEVICE_ID="0"
IN_FEATURES=3
DIM=1600
MODEL="cnn"
CONFIG_FILE="lora_clip/train_config.yml"
PRUNE_FREQ=1

# --- NOVO: Valor padrão para a estratégia ---

# todo acho que podemos resolver o 1 e 3, o 2 não, mas então o maior gargalo é o evaluet né? sugere alguma coisa ? novamente, deixe os sleeps como estão, a parte de do break e de usar a melhor gpu tudo bem.
STRATEGY="lora" 
PACA=12
RANK=8

# --- PaCA Heterogêneo ---
RANDOM_PACA=0
PACA_MIN=1
PACA_MAX=12
PACA_LIST=""

WEIGHTS_DIR="saved_weights"
SAVE_MODEL_FLAG=""
LOAD_MODEL_FLAG=""
SIMULATIONS=1
AUTO_NEXT=0

# 2. Processamento de Argumentos
while [ $# -gt 0 ]; do
    case $1 in
        -c|--clients) CLIENT_COUNT="$2"; shift 2 ;;
        -h|--host) HOST="$2"; shift 2 ;;
        -p|--port) PORT="$2"; shift 2 ;;
        -d|--dataset) DATASET="$2"; shift 2 ;;
        -s|--session) SESSION_NAME="$2"; shift 2 ;;
        --paca) PACA="$2"; shift 2 ;;
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
        -m|--model) MODEL="$2"; shift 2 ;;
        --config) CONFIG_FILE="$2"; shift 2 ;;
        --prune-freq) PRUNE_FREQ="$2"; shift 2 ;;
        --simulations) SIMULATIONS="$2"; shift 2 ;; 
        --auto-next) AUTO_NEXT=1; shift 1 ;;
        
        # --- NOVO: Captura da flag strategy ---
        --strategy) STRATEGY="$2"; shift 2 ;;
        --rank) RANK="$2"; shift 2 ;;
        
        # --- PaCA Heterogêneo ---
        --random-paca) RANDOM_PACA=1; shift 1 ;;
        --paca-min) PACA_MIN="$2"; shift 2 ;;
        --paca-max) PACA_MAX="$2"; shift 2 ;;
        --paca-list) PACA_LIST="$2"; shift 2 ;;
        
        --save) 
            if [[ ! $2 =~ ^- ]] && [[ -n $2 ]]; then
                SAVE_MODEL_PATH="$2"; shift 2
            else
                SAVE_MODEL_PATH="$WEIGHTS_DIR/${MODEL}_${STRATEGY}_rank${RANK}_prune${PRUNE}_freq${PRUNE_FREQ}_${DATASET}.pt"; shift 1 # <-- ALTERADO
            fi
            SAVE_MODEL_FLAG="--save-model $SAVE_MODEL_PATH"
            ;;
        --load)
            if [[ ! $2 =~ ^- ]] && [[ -n $2 ]]; then
                LOAD_MODEL_PATH="$2"; shift 2
            else
                LOAD_MODEL_PATH="$WEIGHTS_DIR/${MODEL}_${STRATEGY}_rank${RANK}_prune${PRUNE}_freq${PRUNE_FREQ}_${DATASET}.pt"; shift 1 # <-- ALTERADO
            fi
            LOAD_MODEL_FLAG="--load-model $LOAD_MODEL_PATH"
            ;;
        *) echo "❌ Argumento desconhecido: $1"; exit 1 ;;
    esac
done

# 3. Validação e Exibição da Configuração
if [ "$MODEL" = "clip" ] && [ -z "$CONFIG_FILE" ]; then
    echo "❌ Erro: Ao usar --model clip, forneça o caminho do YAML usando --config."
    exit 1
fi

echo "===================================="
echo "⚙️ Configuração do Treinamento FL:"
echo "===================================="
echo "  Clientes: $CLIENT_COUNT | Rodadas: $ROUNDS | Simulações: $SIMULATIONS"
echo "  Dataset: $DATASET | Prune: $PRUNE"
echo "  ALA (0=FedALA, 1=FedAvg): $ALA"
if [ "$MODEL" = "clip" ]; then
    echo "  🔥 Modelo: CLIP | Estratégia: $STRATEGY | Config: $CONFIG_FILE"
else
    echo "  🧊 Modelo: CNN Simples Padrão"
fi
echo "  Device: $DEVICE ($DEVICE_ID) | Sessão TMUX: $SESSION_NAME"

if [ "$RANDOM_PACA" -eq 1 ]; then
    echo "  🎲 PaCA Heterogêneo: ATIVADO (aleatório entre $PACA_MIN e $PACA_MAX)"
elif [ -n "$PACA_LIST" ]; then
    echo "  🎲 PaCA Heterogêneo: ATIVADO (lista: $PACA_LIST)"
else
    echo "  PaCA: $PACA (fixo para todos)"
fi

if [ -n "$SAVE_MODEL_FLAG" ]; then echo "  Salvar modelo: SIM ($SAVE_MODEL_PATH)"; fi
if [ -n "$LOAD_MODEL_FLAG" ]; then echo "  Carregar modelo: SIM ($LOAD_MODEL_PATH)"; fi
echo "===================================="

# 4. Preparação do Dataset
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

if [ -n "$SAVE_MODEL_FLAG" ]; then
    mkdir -p "$WEIGHTS_DIR"
fi

# 5. Loop de Simulações
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_NAME="${SESSION_NAME}_${TIMESTAMP}"
echo "📁 Diretório de Resultados: ../results/$EXP_NAME"

for RUN in $(seq 1 $SIMULATIONS); do
    echo "========================================================="
    echo "🚀 INICIANDO SIMULAÇÃO $RUN DE $SIMULATIONS"
    echo "========================================================="

    # --- Determinar PaCA do servidor: se heterogêneo, forçar o máximo ---
    if [ "$RANDOM_PACA" -eq 1 ]; then
        SERVER_PACA=$PACA_MAX
    elif [ -n "$PACA_LIST" ]; then
        # Encontrar o maior valor da lista para usar no servidor
        SERVER_PACA=$(echo "$PACA_LIST" | tr ',' '\n' | sort -rn | head -1)
    else
        SERVER_PACA=$PACA
    fi

    # --- NOVO: Adicionado --strategy $STRATEGY no comando do server.py ---
    SERVER_CMD="uv run server.py --host $HOST --port $PORT --clients-per-round $CLIENT_COUNT --rounds $ROUNDS --dataset $DATASET --test-client-idx $TEST_CLIENT_IDX --in-features $IN_FEATURES --num-classes $NUM_CLASSES --dim $DIM --batch-size $BATCH_SIZE --max-clients $MAX_CLIENTS --prune $PRUNE --device $DEVICE -did $DEVICE_ID --model $MODEL --strategy $STRATEGY --rank $RANK --paca $SERVER_PACA --config \"$CONFIG_FILE\" --prune-freq $PRUNE_FREQ $SAVE_MODEL_FLAG $LOAD_MODEL_FLAG --run-id $RUN --exp-name $EXP_NAME"
    if [ "$AUTO_NEXT" -eq 1 ]; then
        tmux new-session -d -s "$SESSION_NAME" "$SERVER_CMD"
    else
        tmux new-session -d -s "$SESSION_NAME" "$SERVER_CMD ; echo 'Pressione ENTER nesta tela do servidor para fechar a sessão do tmux...'; read; tmux kill-session -t $SESSION_NAME"
    fi
    sleep 2

    for i in $(seq 0 $((CLIENT_COUNT-1))); do
        # --- Montar flags de PaCA heterogêneo para o cliente ---
        PACA_FLAGS="--paca $PACA"
        if [ "$RANDOM_PACA" -eq 1 ]; then
            PACA_FLAGS="--random-paca --paca-min $PACA_MIN --paca-max $PACA_MAX"
        elif [ -n "$PACA_LIST" ]; then
            PACA_FLAGS="--paca-list $PACA_LIST"
        fi

        # --- Distribuição de clientes entre GPUs (round-robin) ---
        if [ $((i % 2)) -eq 0 ]; then
            CLIENT_DEVICE_ID="0"
        else
            CLIENT_DEVICE_ID="1"
        fi

        CLIENT_CMD="uv run client.py --client-idx $i --host $HOST --port $PORT --dataset $DATASET --rounds $ROUNDS --ala $ALA --device $DEVICE --device_id $CLIENT_DEVICE_ID --model $MODEL --strategy $STRATEGY --rank $RANK $PACA_FLAGS --config \"$CONFIG_FILE\" --run-id $RUN --exp-name $EXP_NAME"
        if [ "$AUTO_NEXT" -eq 0 ]; then
            CLIENT_CMD="$CLIENT_CMD ; read"
        fi
        if [ $((i % 3)) -eq 0 ]; then
            tmux split-window -h "$CLIENT_CMD"
        else
            tmux split-window -v "$CLIENT_CMD"
        fi
    done

    tmux select-layout -t "$SESSION_NAME" tiled
    
    echo "⏳ Aguardando a simulação $RUN terminar em segundo plano..."
    # Loop que verifica a cada 5 segundos se a sessão do tmux ainda está ativa
    while tmux has-session -t "$SESSION_NAME" 2>/dev/null; do
        sleep 5
    done
    
    echo "✅ Simulação $RUN concluída com sucesso!"
    sleep 2 
done