#!/bin/bash
# Executa simulações federadas com o Python do Conda ativo e painéis no tmux.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CALLER_DIR=$PWD
DATASET_DIR="$SCRIPT_DIR/../dataset"

# Valores padrão
CLIENT_COUNT=2
HOST=localhost
PORT=9000
DATASET=Cifar100
SESSION_NAME=myapp
ROUNDS=5
TEST_CLIENT_IDX=0
IN_FEATURES=3
NUM_CLASSES=
DIM=1600
BATCH_SIZE=32
MAX_CLIENTS=
PRUNE=0
PRUNE_FREQ=0
ALA=1
MODEL=cnn
STRATEGY=lora
RANK=8
CONFIG_FILE=
DEVICE=auto
DEVICE_ID=
PACA=
PACA_MIN=1
PACA_MAX=
PACA_LIST=
ADAPTIVE_PACA_MIN=2
ADAPTIVE_RANK_MIN=2
ADAPTIVE_RANK_MAX=
SIMULATIONS=1
START_RUN=1
EXP_NAME=
AUTO_NEXT=0
SAVE_MODEL=0
LOAD_MODEL=0
SAVE_PATH=
LOAD_PATH=
SERVER_FLAGS=()
CLIENT_FLAGS=()
RANDOM_PACA=0
ADAPTIVE_PACA=0
ADAPTIVE_RANK=0

die() {
    printf '[ERRO] %s\n' "$*" >&2
    exit 1
}

show_help() {
    cat <<'EOF'
Uso: conda activate puma && ./run_conda.sh [opções]

  -c, --clients N              Clientes por rodada (2)
  -r, --rounds N               Rodadas por simulação (5)
  -d, --dataset NOME           Dataset com gerador em src/dataset (Cifar100)
  -h, --host HOST              Endereço do servidor (localhost)
  -p, --port PORTA             Porta TCP (9000)
  -s, --session NOME           Sessão tmux (myapp)
  -t, --test-client-idx N      Partição de teste inicial (0)
  --in-features N              Canais de entrada da CNN (3)
  --dim N                      Dimensão intermediária da CNN (1600)
  --num-classes N              Classes (inferidas para os datasets conhecidos)
  --batch-size N               Tamanho do lote no servidor e nos clientes (32)
  --max-clients N              Fila TCP do servidor (padrão de pelo menos 10)
  -m, --model cnn|clip|slm|lfm Modelo (cnn)
  --strategy ESTRATÉGIA        lora, adalora, sora_with_schedule ou sora_no_schedule
  --config ARQUIVO             YAML de modelo (padrão em lora_clip/, lora_slm/ ou lora_lfm/)
  --rank N                     Rank inicial dos adaptadores (8)
  --device auto|cuda|cpu|mps   Dispositivo (auto)
  -did, --device-id ID         GPU usada por todos os processos
  --cpu                        Usar CPU
  --prune 0|1                  Poda estrutural da CNN (0)
  --prune-freq N               Frequência de poda SoRA (0)
  --sora-prune                 Habilitar a flag de poda SoRA no servidor
  --ala 0|1                    0=FedALA, 1=FedAvg (1)
  --paca N                     PaCA inicial (12 para CLIP, 28 para SLM)
  --adaptive-paca              Ativar PaCA adaptativo
  --adaptive-paca-min N        Mínimo de camadas para PaCA adaptativo (2)
  --random-paca                Sortear o PaCA inicial dos clientes
  --paca-min N / --paca-max N  Limites do sorteio (1 e o PaCA inicial)
  --paca-list LISTA            PaCA inicial dos clientes, por exemplo 4,8,12
  --allow-paca-upscale          Permitir aumento de PaCA
  --adaptive-rank              Ativar rank adaptativo
  --adaptive-rank-min N        Rank adaptativo mínimo (2)
  --adaptive-rank-max N        Rank adaptativo máximo (rank inicial)
  --allow-rank-upscale          Permitir aumento de rank
  --simulations N              Quantidade de simulações (1)
  --start-run N                Índice da primeira simulação (1)
  --exp-name NOME              Subdiretório do experimento em src/results
  --auto-next                  Executar sem anexar ao tmux e sem pausas
  --save [ARQUIVO]             Salvar pesos (caminho padrão em saved_weights/)
  --load [ARQUIVO]             Carregar pesos (caminho padrão em saved_weights/)
  --skip-train-eval            Pular avaliação local do conjunto de treino
  --skip-post-eval             Pular avaliações locais após o treino
  --delta-coding               Usar compressão LHDQ
  --qlora                      Ativar QLoRA 4-bit NF4 (slm/lfm + lora)
  --help                       Exibir esta ajuda
EOF
}

# Processar argumentos
while (( $# )); do
    case "$1" in
        --help) show_help; exit 0 ;;
        -c|--clients) option=CLIENT_COUNT ;;
        -h|--host) option=HOST ;;
        -p|--port) option=PORT ;;
        -d|--dataset) option=DATASET ;;
        -s|--session) option=SESSION_NAME ;;
        -r|--rounds) option=ROUNDS ;;
        -t|--test-client-idx) option=TEST_CLIENT_IDX ;;
        --in-features) option=IN_FEATURES ;;
        --num-classes) option=NUM_CLASSES ;;
        --dim) option=DIM ;;
        --batch-size) option=BATCH_SIZE ;;
        --max-clients) option=MAX_CLIENTS ;;
        --prune) option=PRUNE ;;
        --prune-freq) option=PRUNE_FREQ ;;
        --ala) option=ALA ;;
        -m|--model) option=MODEL ;;
        --strategy) option=STRATEGY ;;
        --rank) option=RANK ;;
        --config) option=CONFIG_FILE ;;
        --device) option=DEVICE ;;
        -did|--device-id) option=DEVICE_ID ;;
        --paca) option=PACA ;;
        --paca-min) option=PACA_MIN ;;
        --paca-max) option=PACA_MAX ;;
        --paca-list) option=PACA_LIST ;;
        --adaptive-paca-min) option=ADAPTIVE_PACA_MIN ;;
        --adaptive-rank-min) option=ADAPTIVE_RANK_MIN ;;
        --adaptive-rank-max) option=ADAPTIVE_RANK_MAX ;;
        --simulations) option=SIMULATIONS ;;
        --start-run) option=START_RUN ;;
        --exp-name) option=EXP_NAME ;;
        --cpu) DEVICE=cpu; shift; continue ;;
        --auto-next) AUTO_NEXT=1; shift; continue ;;
        --random-paca) RANDOM_PACA=1; shift; continue ;;
        --adaptive-paca) ADAPTIVE_PACA=1; shift; continue ;;
        --adaptive-rank) ADAPTIVE_RANK=1; shift; continue ;;
        --sora-prune|--allow-paca-upscale|--allow-rank-upscale)
            SERVER_FLAGS+=("$1"); shift; continue ;;
        --skip-train-eval|--skip-post-eval)
            CLIENT_FLAGS+=("$1"); shift; continue ;;
        --straggler-idx|--straggler-delay|--straggler-round)
            [[ -n ${2-} && $2 != -* ]] || die "Informe o valor de $1."
            CLIENT_FLAGS+=("$1" "$2"); shift 2; continue ;;
        --delta-coding|--qlora)
            SERVER_FLAGS+=("$1"); CLIENT_FLAGS+=("$1"); shift; continue ;;
        --save|--load)
            action=$1
            shift
            path=
            if [[ -n ${1-} && $1 != -* ]]; then
                path=$1
                [[ $path == /* ]] || path="$CALLER_DIR/$path"
                shift
            fi
            if [[ $action == --save ]]; then
                SAVE_MODEL=1
                SAVE_PATH=$path
            else
                LOAD_MODEL=1
                LOAD_PATH=$path
            fi
            continue ;;
        *) die "Argumento desconhecido: $1" ;;
    esac
    [[ -n ${2-} && $2 != -* ]] || die "Informe o valor de $1."
    printf -v "$option" '%s' "$2"
    shift 2
done

case "$DATASET" in
    MNIST|Cifar10|FashionMNIST) NUM_CLASSES=${NUM_CLASSES:-10} ;;
    OxfordPets) NUM_CLASSES=${NUM_CLASSES:-37} ;;
    DTD) NUM_CLASSES=${NUM_CLASSES:-47} ;;
    Banking77) NUM_CLASSES=${NUM_CLASSES:-77} ;;
    Flowers102) NUM_CLASSES=${NUM_CLASSES:-102} ;;
    *) NUM_CLASSES=${NUM_CLASSES:-100} ;;
esac
case "$MODEL" in
    cnn) PACA=${PACA:-12} ;;
    clip) PACA=${PACA:-12}; CONFIG_FILE=${CONFIG_FILE:-lora_clip/train_config.yml} ;;
    slm) PACA=${PACA:-28}; CONFIG_FILE=${CONFIG_FILE:-lora_slm/train_config.yml} ;;
    lfm) PACA=${PACA:-24}; CONFIG_FILE=${CONFIG_FILE:-lora_lfm/train_config.yml} ;;
    *) die "Modelo inválido: $MODEL" ;;
esac
case "$STRATEGY" in
    lora|adalora|sora_with_schedule|sora_no_schedule) ;;
    *) die "Estratégia inválida: $STRATEGY" ;;
esac
PACA_MAX=${PACA_MAX:-$PACA}
ADAPTIVE_RANK_MAX=${ADAPTIVE_RANK_MAX:-$RANK}

[[ $CLIENT_COUNT =~ ^[1-9][0-9]*$ ]] || die "--clients deve ser um inteiro positivo."
if [[ -z $MAX_CLIENTS ]]; then
    # A fila padrão acomoda também os 25 clientes dos lotes PUMA-GT.
    MAX_CLIENTS=$(( CLIENT_COUNT > 10 ? CLIENT_COUNT : 10 ))
fi
for name in CLIENT_COUNT PORT ROUNDS IN_FEATURES NUM_CLASSES DIM BATCH_SIZE MAX_CLIENTS \
            RANK PACA_MIN ADAPTIVE_PACA_MIN ADAPTIVE_RANK_MIN ADAPTIVE_RANK_MAX SIMULATIONS START_RUN; do
    [[ ${!name} =~ ^[1-9][0-9]*$ ]] || die "$name deve ser um inteiro positivo."
done
for name in TEST_CLIENT_IDX PRUNE_FREQ PACA PACA_MAX; do
    [[ ${!name} =~ ^(0|[1-9][0-9]*)$ ]] || die "$name deve ser um inteiro não negativo."
done
[[ $PRUNE =~ ^[01]$ && $ALA =~ ^[01]$ ]] || die "--prune e --ala aceitam 0 ou 1."
(( PORT <= 65535 )) || die "Porta fora do intervalo TCP."
[[ $DATASET =~ ^[[:alnum:]_-]+$ ]] || die "Nome de dataset inválido."
[[ $SESSION_NAME =~ ^[[:alnum:]_-]+$ ]] || die "Nome de sessão inválido."
[[ -z $EXP_NAME || $EXP_NAME =~ ^[[:alnum:]_.-]+$ ]] && [[ $EXP_NAME != . && $EXP_NAME != .. ]] \
    || die "--exp-name deve ser um nome de diretório simples."

# Selecionar PaCA e rank antes de iniciar o servidor.
SERVER_PACA=$PACA
CLIENT_PACA=(--paca "$PACA")
if (( RANDOM_PACA )); then
    [[ -z $PACA_LIST ]] || die "Escolha --random-paca ou --paca-list."
    (( PACA_MIN <= PACA_MAX )) || die "--paca-min deve ser menor ou igual a --paca-max."
    SERVER_PACA=$PACA_MAX
    CLIENT_PACA=(--random-paca --paca-min "$PACA_MIN" --paca-max "$PACA_MAX")
elif [[ -n $PACA_LIST ]]; then
    layer_count='[[:space:]]*(0|[1-9][0-9]*)[[:space:]]*'
    [[ $PACA_LIST =~ ^$layer_count(,$layer_count)*$ ]] || die "Use uma lista como --paca-list 4,8,12."
    IFS=, read -r -a paca_values <<< "$PACA_LIST"
    SERVER_PACA=0
    for value in "${paca_values[@]}"; do
        if (( value > SERVER_PACA )); then SERVER_PACA=$((value)); fi
    done
    CLIENT_PACA=(--paca-list "$PACA_LIST")
fi
if (( ADAPTIVE_PACA )); then
    (( ADAPTIVE_PACA_MIN <= SERVER_PACA )) || die "O mínimo do PaCA adaptativo excede o PaCA inicial do servidor."
    SERVER_FLAGS+=(--adaptive-paca --adaptive-paca-min "$ADAPTIVE_PACA_MIN")
fi
if (( ADAPTIVE_RANK )); then
    (( ADAPTIVE_RANK_MIN <= ADAPTIVE_RANK_MAX && ADAPTIVE_RANK_MAX <= RANK )) \
        || die "Use mínimo <= máximo <= rank inicial para o rank adaptativo."
    SERVER_FLAGS+=(--adaptive-rank --adaptive-rank-min "$ADAPTIVE_RANK_MIN" --adaptive-rank-max "$ADAPTIVE_RANK_MAX")
fi

if [[ -n $CONFIG_FILE ]]; then
    if [[ $CONFIG_FILE != /* ]]; then
        if [[ -f "$CALLER_DIR/$CONFIG_FILE" ]]; then
            CONFIG_FILE="$CALLER_DIR/$CONFIG_FILE"
        else
            CONFIG_FILE="$SCRIPT_DIR/$CONFIG_FILE"
        fi
    fi
    [[ -f $CONFIG_FILE ]] || die "Configuração não encontrada: $CONFIG_FILE"
fi
default_weights="$SCRIPT_DIR/saved_weights/${MODEL}_${STRATEGY}_rank${RANK}_prune${PRUNE}_freq${PRUNE_FREQ}_${DATASET}.pt"
if (( SAVE_MODEL )); then
    SAVE_PATH=${SAVE_PATH:-$default_weights}
    mkdir -p -- "$(dirname -- "$SAVE_PATH")"
    SERVER_FLAGS+=(--save-model "$SAVE_PATH")
fi
if (( LOAD_MODEL )); then
    LOAD_PATH=${LOAD_PATH:-$default_weights}
    [[ -f $LOAD_PATH ]] || die "Pesos não encontrados: $LOAD_PATH"
    SERVER_FLAGS+=(--load-model "$LOAD_PATH")
fi

# Usar o mesmo ambiente Conda no gerador e nos painéis.
[[ -n ${CONDA_PREFIX-} ]] || die "Ative um ambiente Conda, por exemplo com 'conda activate puma'."
PYTHON="$CONDA_PREFIX/bin/python"
[[ -x $PYTHON ]] || die "Python do ambiente Conda não encontrado: $PYTHON"
command -v tmux >/dev/null || die "tmux não encontrado."
if tmux has-session -t "=$SESSION_NAME" 2>/dev/null; then
    die "Já existe uma sessão tmux chamada $SESSION_NAME. Escolha outro nome com --session."
fi

export PYTHONPATH="$SCRIPT_DIR/..${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false

case "$DEVICE" in
    auto|cuda|mps)
        device_info=$("$PYTHON" -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0, int(torch.backends.mps.is_available()))') \
            || die "Falha ao consultar os dispositivos com o PyTorch do Conda."
        read -r GPU_COUNT HAS_MPS <<< "$device_info"
        ;;
    cpu) GPU_COUNT=0; HAS_MPS=0 ;;
    *) die "Dispositivo inválido: $DEVICE" ;;
esac
if [[ $DEVICE == auto ]]; then
    if (( GPU_COUNT > 0 )); then DEVICE=cuda
    elif (( HAS_MPS )); then DEVICE=mps
    else DEVICE=cpu
    fi
fi

GPU_IDS=()
SERVER_GPU=0
if [[ $DEVICE == cuda ]]; then
    (( GPU_COUNT > 0 )) || die "CUDA indisponível no PyTorch do Conda."
    if [[ -n ${CUDA_VISIBLE_DEVICES-} ]]; then
        IFS=, read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
        GPU_IDS=("${GPU_IDS[@]:0:GPU_COUNT}")
    else
        for ((gpu=0; gpu<GPU_COUNT; gpu++)); do GPU_IDS+=("$gpu"); done
    fi
    SERVER_GPU=${DEVICE_ID:-${GPU_IDS[0]}}
    CUDA_VISIBLE_DEVICES="$SERVER_GPU" "$PYTHON" -c 'import torch; torch.ones(1, device="cuda"); torch.cuda.synchronize()' \
        || die "Não foi possível executar PyTorch na GPU $SERVER_GPU."
elif [[ $DEVICE == mps ]]; then
    (( HAS_MPS )) || die "MPS indisponível no PyTorch do Conda."
fi

if [[ -z $EXP_NAME ]]; then
    EXP_NAME="${SESSION_NAME}_${DATASET}_${MODEL}_${STRATEGY}_prune${PRUNE}_ala${ALA}"
    if (( ADAPTIVE_PACA )); then EXP_NAME+=_adaptpaca
    elif (( RANDOM_PACA )); then EXP_NAME+=_randompaca
    elif [[ -n $PACA_LIST ]]; then EXP_NAME+=_pacalist
    else EXP_NAME+="_paca${PACA}"
    fi
    if (( ADAPTIVE_RANK )); then EXP_NAME+=_adaptrank; fi
    if [[ " ${SERVER_FLAGS[*]} " == *' --delta-coding '* ]]; then EXP_NAME+=_deltacoding; fi
    if [[ " ${SERVER_FLAGS[*]} " == *' --qlora '* ]]; then EXP_NAME+=_qlora; fi
fi
RESULTS_DIR="$SCRIPT_DIR/../results/$EXP_NAME"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
printf 'Ambiente Conda: %s\nDataset: %s (%s classes)\nModelo: %s | Estratégia: %s\n' \
    "$CONDA_PREFIX" "$DATASET" "$NUM_CLASSES" "$MODEL" "$STRATEGY"
printf 'Clientes: %s | Rodadas: %s | Simulações: %s | Dispositivo: %s\n' \
    "$CLIENT_COUNT" "$ROUNDS" "$SIMULATIONS" "$DEVICE"
printf 'Servidor: %s:%s | Sessão tmux: %s\nResultados: %s\n' "$HOST" "$PORT" "$SESSION_NAME" "$RESULTS_DIR"

# Gerar o dataset no diretório esperado pelo leitor de partições.
GENERATOR="generate_${DATASET}.py"
[[ -f "$DATASET_DIR/$GENERATOR" ]] || die "Gerador não encontrado: $GENERATOR"
(
    cd -- "$DATASET_DIR"
    case "$DATASET" in
        MNIST|Cifar10|Cifar100) "$PYTHON" "$GENERATOR" noniid - dir --num-clients "$CLIENT_COUNT" ;;
        DTD)
            # A CLI antiga do DTD usa uma contagem fixa; a função aceita a contagem desejada.
            "$PYTHON" -c 'import sys; from generate_DTD import dir_path, generate_dataset; generate_dataset(dir_path, int(sys.argv[1]), True, False, "dir")' "$CLIENT_COUNT"
            ;;
        *) "$PYTHON" "$GENERATOR" noniid - dir "$CLIENT_COUNT" ;;
    esac
) || die "Falha ao preparar o dataset $DATASET."
for ((i=0; i<CLIENT_COUNT; i++)); do
    for split in train test; do
        [[ -f "$DATASET_DIR/$DATASET/$split/$i.npz" ]] || die "Partição ausente: $DATASET/$split/$i.npz"
    done
done
[[ -f "$DATASET_DIR/$DATASET/test/$TEST_CLIENT_IDX.npz" ]] \
    || die "Partição de teste $TEST_CLIENT_IDX ausente em $DATASET."

# Controlar apenas a sessão criada por esta execução.
session_id=
watch_pid=
cleanup() {
    if [[ -n $watch_pid ]]; then kill "$watch_pid" 2>/dev/null || true; fi
    if [[ -n $session_id ]]; then tmux kill-session -t "$session_id" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

watch_simulation() {
    local panes pane dead status running count
    while panes=$(tmux list-panes -t "$session_id:" -F '#{pane_id} #{pane_dead} #{pane_dead_status}' 2>/dev/null); do
        running=0
        count=0
        while read -r pane dead status; do
            (( count += 1 ))
            if [[ $dead == 0 ]]; then
                (( running += 1 ))
            elif [[ $status != 0 ]]; then
                printf '[ERRO] Painel %s terminou com falha. Logs em %s\n' "$pane" "$LOG_DIR" >&2
                tmux kill-session -t "$session_id" 2>/dev/null || true
                return 1
            fi
        done <<< "$panes"
        if (( count != CLIENT_COUNT + 1 )); then
            tmux kill-session -t "$session_id" 2>/dev/null || true
            return 1
        fi
        if (( running == 0 )); then
            tmux kill-session -t "$session_id" 2>/dev/null || true
            return 0
        fi
        sleep 1
    done
    return 1
}

# O tmux pode reutilizar um servidor já aberto com outro PATH e PYTHONPATH.
PANE_ENV=("PATH=$PATH" "CONDA_PREFIX=$CONDA_PREFIX" "PYTHONPATH=$PYTHONPATH"
    "HF_HUB_OFFLINE=$HF_HUB_OFFLINE" "TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
    "OMP_NUM_THREADS=$OMP_NUM_THREADS" "MKL_NUM_THREADS=$MKL_NUM_THREADS"
    "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS" "VECLIB_MAXIMUM_THREADS=$VECLIB_MAXIMUM_THREADS"
    "NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS" "TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM")
PANE_RUNNER=("$BASH" -o pipefail -c 'log=$1; shift; "$@" 2>&1 | tee "$log"' run_conda_pane)
CLIENT_HOST=$HOST
if [[ $HOST == 0.0.0.0 ]]; then CLIENT_HOST=127.0.0.1; fi
INTERACTIVE=0
if (( ! AUTO_NEXT )) && [[ -t 0 && -t 1 ]]; then INTERACTIVE=1; fi

for ((run=START_RUN; run<START_RUN+SIMULATIONS; run++)); do
    LOG_DIR="$RESULTS_DIR/logs/sim_$run"
    mkdir -p -- "$LOG_DIR"
    COMMON_ARGS=(--port "$PORT" --dataset "$DATASET" --rounds "$ROUNDS" --device "$DEVICE"
        --model "$MODEL" --strategy "$STRATEGY" --rank "$RANK" --config "$CONFIG_FILE"
        --in-features "$IN_FEATURES" --num-classes "$NUM_CLASSES" --dim "$DIM" --batch-size "$BATCH_SIZE"
        --prune "$PRUNE" --run-id "$run" --exp-name "$EXP_NAME" --timestamp "$TIMESTAMP")
    printf 'Simulação %s | Logs em %s\n' "$run" "$LOG_DIR"

    session_id=$(tmux new-session -d -s "$SESSION_NAME" -n treino -c "$SCRIPT_DIR" \
        -x 160 -y 50 -P -F '#{session_id}' \
        "${PANE_RUNNER[@]}" "$LOG_DIR/server.log" env "${PANE_ENV[@]}" "CUDA_VISIBLE_DEVICES=$SERVER_GPU" \
        "$PYTHON" -u server.py "${COMMON_ARGS[@]}" --host "$HOST" --clients-per-round "$CLIENT_COUNT" \
        --test-client-idx "$TEST_CLIENT_IDX" --max-clients "$MAX_CLIENTS" --prune-freq "$PRUNE_FREQ" \
        -did "$SERVER_GPU" --paca "$SERVER_PACA" "${SERVER_FLAGS[@]}")
    tmux set-option -w -t "$session_id:" remain-on-exit on
    sleep 2

    for ((i=0; i<CLIENT_COUNT; i++)); do
        CLIENT_GPU=$SERVER_GPU
        if [[ $DEVICE == cuda && -z $DEVICE_ID ]]; then CLIENT_GPU=${GPU_IDS[i % GPU_COUNT]}; fi
        split=-v
        if (( i % 3 == 0 )); then split=-h; fi
        tmux split-window "$split" -t "$session_id:" -c "$SCRIPT_DIR" \
            "${PANE_RUNNER[@]}" "$LOG_DIR/client_$i.log" env "${PANE_ENV[@]}" "CUDA_VISIBLE_DEVICES=$CLIENT_GPU" \
            "$PYTHON" -u client.py "${COMMON_ARGS[@]}" --client-idx "$i" --host "$CLIENT_HOST" \
            --ala "$ALA" -did "$CLIENT_GPU" "${CLIENT_PACA[@]}" "${CLIENT_FLAGS[@]}"
        if (( CLIENT_COUNT > 2 )); then tmux select-layout -t "$session_id:" tiled >/dev/null; fi
    done

    watch_simulation &
    watch_pid=$!
    if (( INTERACTIVE )); then
        if [[ -n ${TMUX-} ]]; then
            tmux switch-client -t "$session_id" || true
        else
            tmux attach-session -t "$session_id" || true
        fi
    fi
    if ! wait "$watch_pid"; then
        watch_pid=
        die "Simulação $run interrompida. Consulte $LOG_DIR."
    fi
    watch_pid=
    session_id=
    printf 'Simulação %s concluída.\n' "$run"
    if (( INTERACTIVE && run + 1 < START_RUN + SIMULATIONS )); then
        read -r -p 'Pressione Enter para iniciar a próxima simulação.'
    fi
done
