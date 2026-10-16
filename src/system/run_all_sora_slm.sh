#!/bin/bash

# =============================================================================
# run_all_sora_slm.sh — Executa simulações SoRA Padrão (Baseline) com SLM (Qwen-VL)
#
# Configuração SoRA Padrão (Baseline SLM):
#   - Modelo: SLM (Qwen/Qwen3-VL-2B-Instruct)
#   - Estratégia: sora_with_schedule
#   - Frequência de Pruning: 1 (--prune-freq 1)
#   - PaCA: Fixo em 28 camadas (todas as camadas, sem --adaptive-paca)
#   - Rank: Fixo em 8 (sem --adaptive-rank)
#   - Clientes: 12 (mesma partição Non-IID Dirichlet dos outros experimentos)
#   - Batch Size: 2
#   - Rodadas: 100
#   - Simulações por dataset: 10
#
# Datasets em ordem:
#   1. OxfordPets   (37 classes)
#   2. DTD          (47 classes)
#   3. FGVCAircraft (100 classes)
#   4. Flowers102   (102 classes)
#
# Uso recomendado dentro de uma sessão tmux:
#   tmux new -s orquestrador_sora_slm
#   cd /home/rafael.teixeira.silva/David/FL_ServerPuma/src/system
#   ./run_all_sora_slm.sh [opções adicionais como -did 0, --simulations 10, etc.]
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

# Parâmetros padrão do SoRA Baseline com SLM
SESSION_NAME="fl_sora_slm"
SIMULATIONS=10
START_RUN=1
CLIENTS=12
BATCH_SIZE=2
ROUNDS=100
STRATEGY="sora_with_schedule"
PRUNE_FREQ=1
MODEL="slm"
DEVICE_ID="0"
TARGET_DATASET=""
EXTRA_ARGS=()

# Evita conflito se a sessão tmux externa tiver o mesmo nome da interna usada pelo run.sh
if [ -n "$TMUX" ]; then
    CURRENT_TMUX_SESSION=$(tmux display-message -p '#S' 2>/dev/null || true)
    if [ "$CURRENT_TMUX_SESSION" = "$SESSION_NAME" ] || [ "$CURRENT_TMUX_SESSION" = "fl_puma" ] || [ "$CURRENT_TMUX_SESSION" = "fl_puma_v2" ] || [ "$CURRENT_TMUX_SESSION" = "fl_pumagt_plus" ]; then
        echo "❌ [ERRO] Sua sessão tmux atual se chama '$CURRENT_TMUX_SESSION'!"
        echo "O script cria internamente uma sessão com esse nome para o servidor e clientes."
        echo "Por favor, execute dentro de uma sessão com nome diferente (ex: tmux new -s orquestrador_sora_slm)."
        exit 1
    fi
fi

# Função de ajuda
show_help() {
    cat << 'EOF'
Uso: ./run_all_sora_slm.sh [opções]

Executa automaticamente as 10 simulações de SoRA (Baseline) com SLM (Qwen-VL)
(SoRA com schedule, prune-freq 1, PaCA fixo em 28 camadas, rank fixo em 8)
para os 4 datasets na ordem:
  1. OxfordPets   (37 classes)
  2. DTD          (47 classes)
  3. FGVCAircraft (100 classes)
  4. Flowers102   (102 classes)

Opções disponíveis:
  -s, --session <nome>       Nome da sessão tmux interna (padrão: fl_sora_slm)
  --simulations <n>          Número de simulações por dataset (padrão: 10)
  --start-run <n>            Número da primeira simulação (padrão: 1)
  --clients <n>              Número de clientes (padrão: 12)
  --batch-size <n>           Tamanho do batch local (padrão: 2)
  --rounds <n>               Número de rodadas por simulação (padrão: 100)
  --prune-freq <n>           Frequência de poda SoRA (padrão: 1)
  -did, --device-id <id>     ID da GPU a utilizar (padrão: 0)
  -d, --dataset <nome>       Executa apenas um dataset específico
  -h, --help                 Exibe esta mensagem de ajuda

Exemplos:
  ./run_all_sora_slm.sh
  ./run_all_sora_slm.sh -did 0 --simulations 10
  ./run_all_sora_slm.sh -d OxfordPets --simulations 1 --rounds 5
EOF
    exit 0
}

# Processamento de argumentos
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--session)
            SESSION_NAME="$2"
            shift 2
            ;;
        --simulations)
            SIMULATIONS="$2"
            shift 2
            ;;
        --start-run)
            START_RUN="$2"
            shift 2
            ;;
        --clients)
            CLIENTS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --rounds)
            ROUNDS="$2"
            shift 2
            ;;
        --prune-freq)
            PRUNE_FREQ="$2"
            shift 2
            ;;
        -did|--device-id)
            DEVICE_ID="$2"
            shift 2
            ;;
        -d|--dataset)
            TARGET_DATASET="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift 1
            ;;
    esac
done

# Definição dos datasets: "nome:num_classes"
DATASETS=(
    "OxfordPets:37"
    "DTD:47"
    "FGVCAircraft:100"
    "Flowers102:102"
)

if [ -n "$TARGET_DATASET" ]; then
    case "$TARGET_DATASET" in
        OxfordPets|oxfordpets) DATASETS=("OxfordPets:37") ;;
        DTD|dtd) DATASETS=("DTD:47") ;;
        FGVCAircraft|fgvcaircraft) DATASETS=("FGVCAircraft:100") ;;
        Flowers102|flowers102) DATASETS=("Flowers102:102") ;;
        *) echo "❌ [ERRO] Dataset '$TARGET_DATASET' desconhecido."; exit 1 ;;
    esac
fi

TOTAL_DATASETS=${#DATASETS[@]}

echo "================================================================="
echo "        ORQUESTRADOR SoRA PADRÃO (SLM: Qwen-VL Baseline)"
echo "================================================================="
echo "  Modelo:             $MODEL (Qwen3-VL-2B-Instruct)"
echo "  Estratégia:         $STRATEGY (prune_freq=$PRUNE_FREQ)"
echo "  PaCA:               FIXO em 28 camadas (todas as camadas)"
echo "  Rank:               FIXO em 8 (sem adaptive-rank)"
echo "  Simulações/dataset: $SIMULATIONS"
echo "  Clientes:           $CLIENTS"
echo "  Batch Size:         $BATCH_SIZE"
echo "  Rodadas:            $ROUNDS"
echo "  GPU Device ID:      $DEVICE_ID"
echo "  Datasets na fila:   ${TOTAL_DATASETS} datasets"
echo "================================================================="
echo ""

START_TOTAL_TIME=$(date +%s)

IDX=1
for ENTRY in "${DATASETS[@]}"; do
    DATASET_NAME="${ENTRY%%:*}"
    NUM_CLASSES="${ENTRY##*:}"

    DATASET_START_TIME=$(date +%s)
    echo "================================================================="
    echo "[$IDX/$TOTAL_DATASETS] Iniciando SoRA SLM: $DATASET_NAME ($NUM_CLASSES classes)"
    echo "Horário de início: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================="

    CMD=(
        ./run.sh
        -s "$SESSION_NAME"
        --simulations "$SIMULATIONS"
        --start-run "$START_RUN"
        --clients "$CLIENTS"
        --batch-size "$BATCH_SIZE"
        --strategy "$STRATEGY"
        --prune-freq "$PRUNE_FREQ"
        --model "$MODEL"
        --rounds "$ROUNDS"
        --dataset "$DATASET_NAME"
        --num-classes "$NUM_CLASSES"
        --auto-next
        --skip-train-eval
        --skip-post-eval
        -did "$DEVICE_ID"
        "${EXTRA_ARGS[@]}"
    )

    echo "Comando a executar: ${CMD[*]}"
    echo ""

    "${CMD[@]}"

    DATASET_END_TIME=$(date +%s)
    DATASET_DURATION=$((DATASET_END_TIME - DATASET_START_TIME))
    DURATION_H=$((DATASET_DURATION / 3600))
    DURATION_M=$(((DATASET_DURATION % 3600) / 60))
    DURATION_S=$((DATASET_DURATION % 60))

    echo ""
    echo "✅ [$IDX/$TOTAL_DATASETS] Concluído: $DATASET_NAME"
    echo "Horário de término: $(date '+%Y-%m-%d %H:%M:%S')"
    printf "Tempo decorrido: %02dh:%02dm:%02ds\n" "$DURATION_H" "$DURATION_M" "$DURATION_S"
    echo "================================================================="
    echo ""

    IDX=$((IDX + 1))
done

END_TOTAL_TIME=$(date +%s)
TOTAL_DURATION=$((END_TOTAL_TIME - START_TOTAL_TIME))
TOTAL_H=$((TOTAL_DURATION / 3600))
TOTAL_M=$(((TOTAL_DURATION % 3600) / 60))
TOTAL_S=$((TOTAL_DURATION % 60))

echo "🎉 TODOS OS EXPERIMENTOS SoRA (SLM) FORAM CONCLUÍDOS COM SUCESSO!"
echo "Horário final: $(date '+%Y-%m-%d %H:%M:%S')"
printf "Tempo total de execução: %02dh:%02dm:%02ds\n" "$TOTAL_H" "$TOTAL_M" "$TOTAL_S"
echo "================================================================="
