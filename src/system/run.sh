#!/bin/bash

# =============================================================================
# run.sh — Script principal para execução de simulações de Federated Learning
#
# Uso: ./run.sh [opções]
# Execute ./run.sh --help para ver todos os argumentos disponíveis.
# =============================================================================

# Hugging Face: modo offline por padrão para evitar chamadas de rede redundantes a cada cliente/servidor
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}

# Previne que o PyTorch tente usar todos os núcleos (100+ threads) para cada um dos 10 clientes, o que causa Thrashing
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# Compilador C: necessário para o Triton (JIT de kernels CUDA em modelos Transformers)
# Usa o gcc do micromamba local se o sistema não tiver um instalado
if [ -z "$CC" ] && ! command -v cc &>/dev/null && ! command -v gcc &>/dev/null; then
    MAMBA_GCC="$HOME/.local/micromamba/envs/gcc/bin/x86_64-conda-linux-gnu-gcc"
    if [ -x "$MAMBA_GCC" ]; then
        export CC="$MAMBA_GCC"
    fi
fi

# ---- Valores Padrão ----

# Rede
HOST="localhost"
PORT=9500

# Sessão
SESSION_NAME="david"

# Treinamento
CLIENT_COUNT=2
ROUNDS=5
DATASET="MNIST"
BATCH_SIZE=32
MAX_CLIENTS=22
SIMULATIONS=1
START_RUN=1
EXP_NAME_OVERRIDE=""
AUTO_NEXT=0

# Modelo
MODEL="cnn"
STRATEGY="lora"
RANK=8
CONFIG_FILE=""

# Dispositivo
DEVICE="cuda"
DEVICE_ID="0"
CUSTOM_DID=0
USE_CPU=0

# Pruning
PRUNE=1
PRUNE_FREQ=0
SORA_PRUNE=0

# ALA
ALA=1

# Arquitetura (CNN)
IN_FEATURES=3
DIM=1600
NUM_CLASSES=10
TEST_CLIENT_IDX=0

# PaCA
PACA=12
CUSTOM_PACA=0
ADAPTIVE_PACA=0
RANDOM_PACA=0
PACA_MIN=1
PACA_MAX=12
CUSTOM_PACA_MAX=0
PACA_LIST=""

# Rank Adaptativo
ADAPTIVE_RANK=0
ADAPTIVE_RANK_MIN=2
ADAPTIVE_RANK_MAX=8

# Persistência de Modelo
WEIGHTS_DIR="saved_weights"
SAVE_MODEL_FLAG=""
LOAD_MODEL_FLAG=""

# Otimização de Avaliação
SKIP_POST_EVAL=0
SKIP_TRAIN_EVAL=0

# Delta Coding (LHDQ)
DELTA_CODING=0

# ---- Função de Ajuda ----
show_help() {
    cat << 'EOF'
Uso: ./run.sh [opções]

Opções de Rede:
  -h, --host <host>           Host do servidor (padrão: localhost)
  -p, --port <port>           Porta do servidor (padrão: 9500)

Opções de Sessão:
  -s, --session <nome>        Nome da sessão tmux (padrão: david)
  --auto-next                 Não pausar entre simulações

Opções de Treinamento:
  -c, --clients <n>           Número de clientes (padrão: 2)
  -r, --rounds <n>            Número de rodadas (padrão: 5)
  -d, --dataset <nome>        Dataset: Nome do dataset (deve existir um generate_Nome.py) (padrão: MNIST)
  --batch-size <n>            Tamanho do batch (padrão: 32)
  --max-clients <n>           Máximo de clientes permitidos (padrão: 22)
  --simulations <n>           Número de simulações (padrão: 1)
  --start-run <n>             Índice da primeira simulação (padrão: 1)
  --exp-name <nome>           Nome customizado do experimento

Opções de Modelo:
  -m, --model <tipo>          Modelo: cnn ou clip (padrão: cnn)
  --strategy <tipo>           Estratégia: lora, sora_with_schedule, etc. (padrão: lora)
  --rank <n>                  Rank do LoRA/SoRA (padrão: 8)
  --config <path>             Caminho do YAML de config CLIP (padrão: lora_clip/train_config.yml)

Opções de Dispositivo:
  --device <tipo>             Dispositivo: cuda, cpu, mps (padrão: auto-detectado)
  -did, --device-id <id>      ID da GPU (padrão: 0)
  --cpu                       Forçar uso de CPU

Opções de Pruning:
  --prune <0|1>               Habilitar pruning (padrão: 1)
  --prune-freq <n>            Frequência de pruning (padrão: 1)
  --sora-prune                Habilitar pruning SoRA

Opções de ALA:
  --ala <0|1>                 0=FedALA, 1=FedAvg (padrão: 1)

Opções de Arquitetura (CNN):
  --in-features <n>           Canais de entrada (padrão: 3)
  --dim <n>                   Dimensão intermediária (padrão: 1600)
  --num-classes <n>           Número de classes (padrão: 10)
  -t, --test-client-idx <n>   Índice do cliente de teste (padrão: 0)

Opções de PaCA:
  --paca <n>                  Valor fixo de PaCA (padrão: 12)
  --adaptive-paca             Habilitar PaCA adaptativo no servidor
  --random-paca               PaCA aleatório por cliente (entre --paca-min e --paca-max)
  --paca-min <n>              PaCA mínimo para modo aleatório (padrão: 1)
  --paca-max <n>              PaCA máximo para modo aleatório (padrão: 12)
  --paca-list <lista>         Lista de PaCA por cliente (ex: "4,8,12")

Opções de Rank Adaptativo:
  --adaptive-rank             Habilitar rank adaptativo no servidor
  --adaptive-rank-min <n>     Rank mínimo no rank adaptativo (padrão: 2)
  --adaptive-rank-max <n>     Rank máximo no rank adaptativo (padrão: 8)

Persistência de Modelo:
  --save [path]               Salvar modelo (path opcional, gera nome automático)
  --load [path]               Carregar modelo (path opcional, gera nome automático)

Otimização de Avaliação:
  --skip-post-eval            Pular todas as avaliações pós-treino (mantém apenas Global Model Test Acc)
  --skip-train-eval           Pular apenas a avaliação no conjunto de treino pós-treino

Compressão:
  --delta-coding              Ativa LHDQ (Low Huffman-coded Delta Quantization) em vez de int8

Outros:
  --help                      Exibir esta mensagem
EOF
    exit 0
}

# ---- Processamento de Argumentos ----
while [ $# -gt 0 ]; do
    case $1 in
        --help) show_help ;;

        # Rede
        -h|--host) HOST="$2"; shift 2 ;;
        -p|--port) PORT="$2"; shift 2 ;;

        # Sessão
        -s|--session) SESSION_NAME="$2"; shift 2 ;;
        --auto-next) AUTO_NEXT=1; shift 1 ;;

        # Treinamento
        -c|--clients) CLIENT_COUNT="$2"; shift 2 ;;
        -r|--rounds) ROUNDS="$2"; shift 2 ;;
        -d|--dataset) DATASET="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --max-clients) MAX_CLIENTS="$2"; shift 2 ;;
        --simulations) SIMULATIONS="$2"; shift 2 ;;
        --start-run) START_RUN="$2"; shift 2 ;;
        --exp-name) EXP_NAME_OVERRIDE="$2"; shift 2 ;;

        # Modelo
        -m|--model) MODEL="$2"; shift 2 ;;
        --strategy) STRATEGY="$2"; shift 2 ;;
        --rank) RANK="$2"; shift 2 ;;
        --config) CONFIG_FILE="$2"; shift 2 ;;

        # Dispositivo
        --device) DEVICE="$2"; shift 2 ;;
        -did|--device-id) DEVICE_ID="$2"; CUSTOM_DID=1; shift 2 ;;
        --cpu) USE_CPU=1; shift 1 ;;

        # Pruning
        --prune) PRUNE="$2"; shift 2 ;;
        --prune-freq) PRUNE_FREQ="$2"; shift 2 ;;
        --sora-prune) SORA_PRUNE=1; shift 1 ;;

        # ALA
        --ala) ALA="$2"; shift 2 ;;

        # Arquitetura (CNN)
        --in-features) IN_FEATURES="$2"; shift 2 ;;
        --dim) DIM="$2"; shift 2 ;;
        --num-classes) NUM_CLASSES="$2"; shift 2 ;;
        -t|--test-client-idx) TEST_CLIENT_IDX="$2"; shift 2 ;;

        # PaCA
        --paca) PACA="$2"; CUSTOM_PACA=1; shift 2 ;;
        --adaptive-paca) ADAPTIVE_PACA=1; shift 1 ;;
        --random-paca) RANDOM_PACA=1; shift 1 ;;
        --paca-min) PACA_MIN="$2"; shift 2 ;;
        --paca-max) PACA_MAX="$2"; CUSTOM_PACA_MAX=1; shift 2 ;;
        --paca-list) PACA_LIST="$2"; shift 2 ;;

        # Rank Adaptativo
        --adaptive-rank) ADAPTIVE_RANK=1; shift 1 ;;
        --adaptive-rank-min) ADAPTIVE_RANK_MIN="$2"; shift 2 ;;
        --adaptive-rank-max) ADAPTIVE_RANK_MAX="$2"; shift 2 ;;
        --allow-paca-upscale) ALLOW_PACA_UPSCALE=1; shift 1 ;;
        --allow-rank-upscale) ALLOW_RANK_UPSCALE=1; shift 1 ;;

        # Persistência de Modelo
        --save)
            if [[ ! $2 =~ ^- ]] && [[ -n $2 ]]; then
                SAVE_MODEL_PATH="$2"; shift 2
            else
                SAVE_MODEL_PATH="$WEIGHTS_DIR/${MODEL}_${STRATEGY}_rank${RANK}_prune${PRUNE}_freq${PRUNE_FREQ}_${DATASET}.pt"; shift 1
            fi
            SAVE_MODEL_FLAG="--save-model $SAVE_MODEL_PATH"
            ;;
        --load)
            if [[ ! $2 =~ ^- ]] && [[ -n $2 ]]; then
                LOAD_MODEL_PATH="$2"; shift 2
            else
                LOAD_MODEL_PATH="$WEIGHTS_DIR/${MODEL}_${STRATEGY}_rank${RANK}_prune${PRUNE}_freq${PRUNE_FREQ}_${DATASET}.pt"; shift 1
            fi
            LOAD_MODEL_FLAG="--load-model $LOAD_MODEL_PATH"
            ;;

        # Otimização de Avaliação
        --skip-post-eval) SKIP_POST_EVAL=1; shift 1 ;;
        --skip-train-eval) SKIP_TRAIN_EVAL=1; shift 1 ;;

        # Delta Coding (LHDQ)
        --delta-coding) DELTA_CODING=1; shift 1 ;;

        *) echo "❌ Argumento desconhecido: $1"; exit 1 ;;
    esac
done

# ---- Auto-detecção de Dispositivo (GPU / MPS / CPU) ----
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
MPS_AVAILABLE=0
if [ "$(uname)" = "Darwin" ]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_CMD=python
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_CMD=python3
    fi
    if [ -n "$PYTHON_CMD" ]; then
        MPS_AVAILABLE=$($PYTHON_CMD -c 'import torch, sys
print(int(getattr(torch.backends.mps, "is_available", lambda: False)()))' 2>/dev/null || echo 0)
    fi
fi
MPS_AVAILABLE=${MPS_AVAILABLE:-0}

if [ "$USE_CPU" -eq 1 ]; then
    DEVICE="cpu"
    NUM_GPUS=0
elif [ "$DEVICE" = "mps" ] && [ "$MPS_AVAILABLE" -eq 1 ]; then
    NUM_GPUS=0
elif [ "$NUM_GPUS" -eq 0 ]; then
    if [ "$MPS_AVAILABLE" -eq 1 ]; then
        DEVICE="mps"
    else
        DEVICE="cpu"
    fi
else
    DEVICE="cuda"
fi

# ---- Validação ----
if [ "$MODEL" = "clip" ] || [ "$MODEL" = "slm" ]; then
    if [ -z "$CONFIG_FILE" ]; then
        if [ "$MODEL" = "slm" ]; then
            CONFIG_FILE="lora_slm/train_config.yml"
        elif [ "$MODEL" = "clip" ]; then
            CONFIG_FILE="lora_clip/train_config.yml"
        fi
    fi
    
    # Auto-ajuste de PaCA conforme o modelo (SLM=28 camadas, CLIP=12 camadas)
    if [ "$CUSTOM_PACA" -eq 0 ]; then
        if [ "$MODEL" = "slm" ]; then
            PACA=28
        elif [ "$MODEL" = "clip" ]; then
            PACA=12
        fi
    fi
    if [ "$CUSTOM_PACA_MAX" -eq 0 ]; then
        if [ "$MODEL" = "slm" ]; then
            PACA_MAX=28
        elif [ "$MODEL" = "clip" ]; then
            PACA_MAX=12
        fi
    fi
fi

# ---- Exibição da Configuração ----
echo "===================================="
echo "⚙️ Configuração do Treinamento FL:"
echo "===================================="
echo "  Clientes: $CLIENT_COUNT | Rodadas: $ROUNDS | Simulações: $SIMULATIONS"
echo "  Dataset: $DATASET | Prune: $PRUNE"
echo "  ALA (0=FedALA, 1=FedAvg): $ALA"
if [ "$MODEL" = "clip" ]; then
    echo "  🔥 Modelo: CLIP | Estratégia: $STRATEGY | Config: $CONFIG_FILE"
elif [ "$MODEL" = "slm" ]; then
    echo "  🧠 Modelo: SLM (Qwen-VL) | Estratégia: $STRATEGY | Config: $CONFIG_FILE"
else
    echo "  🧊 Modelo: CNN Simples Padrão"
fi
echo "  Device: $DEVICE ($DEVICE_ID) | Sessão TMUX: $SESSION_NAME"

if [ "$RANDOM_PACA" -eq 1 ]; then
    echo "  🎲 PaCA Heterogêneo: ATIVADO (aleatório entre $PACA_MIN e $PACA_MAX)"
elif [ -n "$PACA_LIST" ]; then
    echo "  🎲 PaCA Heterogêneo: ATIVADO (lista: $PACA_LIST)"
else
    if [ "$ADAPTIVE_PACA" -eq 1 ]; then
        echo "  PaCA: $PACA (MÁXIMO) | ⚡ PaCA Adaptativo (Server): ATIVADO"
    else
        echo "  PaCA: $PACA (fixo para todos)"
    fi
fi

if [ -n "$SAVE_MODEL_FLAG" ]; then echo "  Salvar modelo: SIM ($SAVE_MODEL_PATH)"; fi
if [ -n "$LOAD_MODEL_FLAG" ]; then echo "  Carregar modelo: SIM ($LOAD_MODEL_PATH)"; fi
if [ "$DELTA_CODING" -eq 1 ]; then echo "  🗜️  Compressão: LHDQ (Delta Coding ~1.67 bits/param)"; else echo "  🗜️  Compressão: INT8 padrão (8 bits/param)"; fi
echo "===================================="

# ---- Preparação do Dataset ----
cd ../dataset || exit 1
    # Encontra o script ignorando case
    SCRIPT_NAME=$(find . -maxdepth 1 -iname "generate_${DATASET}.py" -print -quit)
    if [ -n "$SCRIPT_NAME" ]; then
        uv run "$SCRIPT_NAME" noniid - dir "$CLIENT_COUNT"
    else
        echo "❌ Dataset não reconhecido ou script de geração não encontrado para: $DATASET"
        exit 1
    fi
cd ../system || exit 1

if [ -n "$SAVE_MODEL_FLAG" ]; then
    mkdir -p "$WEIGHTS_DIR"
fi

# ---- Nome do Experimento ----
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [ -n "$EXP_NAME_OVERRIDE" ]; then
    EXP_NAME="$EXP_NAME_OVERRIDE"
else
    if [ "$ADAPTIVE_PACA" -eq 1 ]; then
        EXP_NAME="${SESSION_NAME}_${MODEL}_${STRATEGY}_prune${PRUNE}_ala${ALA}_adaptpaca"
    elif [ "$RANDOM_PACA" -eq 1 ]; then
        EXP_NAME="${SESSION_NAME}_${MODEL}_${STRATEGY}_prune${PRUNE}_ala${ALA}_randompaca"
    elif [ -n "$PACA_LIST" ]; then
        EXP_NAME="${SESSION_NAME}_${MODEL}_${STRATEGY}_prune${PRUNE}_ala${ALA}_pacalist"
    else
        EXP_NAME="${SESSION_NAME}_${MODEL}_${STRATEGY}_prune${PRUNE}_ala${ALA}_paca${PACA}"
    fi
    # Sufixo de compressão LHDQ
    if [ "$DELTA_CODING" -eq 1 ]; then
        EXP_NAME="${EXP_NAME}_deltacoding"
    fi
fi
echo "📁 Diretório de Resultados: ../results/$EXP_NAME"

# ---- Loop de Simulações ----
END_RUN=$((START_RUN + SIMULATIONS - 1))
for RUN in $(seq $START_RUN $END_RUN); do
    echo "========================================================="
    echo "🚀 INICIANDO SIMULAÇÃO $RUN DE $SIMULATIONS"
    echo "========================================================="

    # PaCA do servidor: se heterogêneo, usar o máximo
    if [ "$RANDOM_PACA" -eq 1 ]; then
        SERVER_PACA=$PACA_MAX
    elif [ -n "$PACA_LIST" ]; then
        SERVER_PACA=$(echo "$PACA_LIST" | tr ',' '\n' | sort -rn | head -1)
    else
        SERVER_PACA=$PACA
    fi

    # Diretório de logs
    LOG_DIR="../results/$EXP_NAME/logs/sim_$RUN"
    mkdir -p "$LOG_DIR"

    # Comando do servidor
    SERVER_CMD="CUDA_VISIBLE_DEVICES=$DEVICE_ID uv run server.py --host $HOST --port $PORT --clients-per-round $CLIENT_COUNT --rounds $ROUNDS --dataset $DATASET --test-client-idx $TEST_CLIENT_IDX --in-features $IN_FEATURES --num-classes $NUM_CLASSES --dim $DIM --batch-size $BATCH_SIZE --max-clients $MAX_CLIENTS --prune $PRUNE --device $DEVICE -did $DEVICE_ID --model $MODEL --strategy $STRATEGY --rank $RANK --paca $SERVER_PACA --config \"$CONFIG_FILE\" --prune-freq $PRUNE_FREQ $SAVE_MODEL_FLAG $LOAD_MODEL_FLAG --run-id $RUN --exp-name $EXP_NAME --timestamp $TIMESTAMP"
    if [ "$ADAPTIVE_PACA" -eq 1 ]; then
        SERVER_CMD="$SERVER_CMD --adaptive-paca"
    fi
    if [ "$ADAPTIVE_RANK" -eq 1 ]; then
        SERVER_CMD="$SERVER_CMD --adaptive-rank --adaptive-rank-min $ADAPTIVE_RANK_MIN --adaptive-rank-max $ADAPTIVE_RANK_MAX"
    fi
    if [ "${ALLOW_PACA_UPSCALE:-0}" -eq 1 ]; then
        SERVER_CMD="$SERVER_CMD --allow-paca-upscale"
    fi
    if [ "${ALLOW_RANK_UPSCALE:-0}" -eq 1 ]; then
        SERVER_CMD="$SERVER_CMD --allow-rank-upscale"
    fi
    if [ "$SORA_PRUNE" -eq 1 ]; then
        SERVER_CMD="$SERVER_CMD --sora-prune"
    fi
    if [ "$DELTA_CODING" -eq 1 ]; then
        SERVER_CMD="$SERVER_CMD --delta-coding"
    fi
    SERVER_CMD="$SERVER_CMD 2>&1 | tee $LOG_DIR/server.log"

    if [ "$AUTO_NEXT" -eq 1 ]; then
        tmux new-session -d -s "$SESSION_NAME" "$SERVER_CMD"
    else
        tmux new-session -d -s "$SESSION_NAME" "$SERVER_CMD ; echo 'Pressione ENTER nesta tela do servidor para fechar a sessão do tmux...'; read; tmux kill-session -t $SESSION_NAME"
    fi
    sleep 2

    # Lançar clientes
    for i in $(seq 0 $((CLIENT_COUNT-1))); do
        # Flags de PaCA para o cliente
        PACA_FLAGS="--paca $PACA"
        if [ "$RANDOM_PACA" -eq 1 ]; then
            PACA_FLAGS="--random-paca --paca-min $PACA_MIN --paca-max $PACA_MAX"
        elif [ -n "$PACA_LIST" ]; then
            PACA_FLAGS="--paca-list $PACA_LIST"
        fi

        # Flags de Otimização de Avaliação
        EVAL_FLAGS=""
        if [ "$SKIP_POST_EVAL" -eq 1 ]; then
            EVAL_FLAGS="$EVAL_FLAGS --skip-post-eval"
        elif [ "$SKIP_TRAIN_EVAL" -eq 1 ]; then
            EVAL_FLAGS="$EVAL_FLAGS --skip-train-eval"
        fi

        # Distribuição round-robin de GPUs
        if [ "$CUSTOM_DID" -eq 1 ]; then
            CLIENT_DEVICE_ID=$DEVICE_ID
        elif [ "$NUM_GPUS" -gt 0 ]; then
            CLIENT_DEVICE_ID=$((i % NUM_GPUS))
        else
            CLIENT_DEVICE_ID="0"
        fi

        CLIENT_CMD="CUDA_VISIBLE_DEVICES=$CLIENT_DEVICE_ID uv run client.py --client-idx $i --host $HOST --port $PORT --dataset $DATASET --rounds $ROUNDS --ala $ALA --device $DEVICE --device_id $CLIENT_DEVICE_ID --model $MODEL --strategy $STRATEGY --rank $RANK $PACA_FLAGS $EVAL_FLAGS --config \"$CONFIG_FILE\" --num-classes $NUM_CLASSES --in-features $IN_FEATURES --run-id $RUN --exp-name $EXP_NAME --timestamp $TIMESTAMP"
        if [ "$DELTA_CODING" -eq 1 ]; then
            CLIENT_CMD="$CLIENT_CMD --delta-coding"
        fi
        CLIENT_CMD="$CLIENT_CMD 2>&1 | tee $LOG_DIR/client_$i.log"
        if [ "$AUTO_NEXT" -eq 0 ]; then
            CLIENT_CMD="$CLIENT_CMD ; read"
        fi
        tmux new-window -t "$SESSION_NAME" -n "Client-$i" "$CLIENT_CMD"
    done

    echo "⏳ Aguardando a simulação $RUN terminar em segundo plano..."
    while tmux has-session -t "$SESSION_NAME" 2>/dev/null; do
        sleep 5
    done

    echo "✅ Simulação $RUN concluída com sucesso!"
    sleep 2
done