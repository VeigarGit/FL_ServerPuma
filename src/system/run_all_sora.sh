#!/bin/bash

# =============================================================================
# run_all_sora.sh — Executa a sequência de simulações PUMA-GT nos 4 datasets
#
# Configuração PUMA-GT:
#   - Estratégia: sora_with_schedule
#   - Frequência de Pruning: 3 (--prune-freq 3)
#   - PaCA Adaptativo / APL: Ativado (--adaptive-paca)
#   - Modelo: CLIP
#   - Rodadas: 150
#   - Clientes: 25
#   - Simulações por dataset: 10
#
# Datasets em ordem:
#   1. OxfordPets   (37 classes)
#   2. DTD          (47 classes)
#   3. FGVCAircraft (100 classes)
#   4. Flowers102   (102 classes)
#
# Uso recomendado dentro de uma sessão tmux:
#   tmux new -s orquestrador
#   cd /home/rafael.teixeira.silva/David/FL_ServerPuma/src/system
#   ./run_all_sora.sh [opções adicionais como -did 1, --simulations 5, etc.]
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

# Evita conflito se a sessão tmux externa tiver o mesmo nome da interna usada pelo run.sh
if [ -n "$TMUX" ]; then
    CURRENT_TMUX_SESSION=$(tmux display-message -p '#S' 2>/dev/null || true)
    if [ "$CURRENT_TMUX_SESSION" = "fl_puma" ]; then
        echo "❌ [ERRO] Sua sessão tmux atual se chama 'fl_puma'!"
        echo "O script run.sh cria internamente uma sessão chamada 'fl_puma' para o servidor e clientes."
        echo "Por favor, renomeie sua sessão externa ou crie outra:"
        echo "  tmux rename-session -t fl_puma orquestrador"
        echo "ou inicie uma nova com: tmux new -s orquestrador"
        exit 1
    fi
fi

# Parâmetros padrão do PUMA-GT
SIMULATIONS=10
CLIENTS=25
ROUNDS=150
STRATEGY="sora_with_schedule"
PRUNE_FREQ=3
MODEL="clip"
DEVICE_ID="0"
EXTRA_ARGS=()

# Função de ajuda
show_help() {
    cat << 'EOF'
Uso: ./run_all_sora.sh [opções]

Executa automaticamente as 10 simulações de PUMA-GT
(SoRA com schedule, prune-freq 3 e adaptive-paca ativo)
para os 4 datasets na ordem:
  1. OxfordPets   (37 classes)
  2. DTD          (47 classes)
  3. FGVCAircraft (100 classes)
  4. Flowers102   (102 classes)

Opções:
  --simulations <n>           Número de simulações por dataset (padrão: 10)
  -c, --clients <n>           Número de clientes (padrão: 25)
  -r, --rounds <n>            Número de rodadas (padrão: 150)
  --prune-freq <n>            Frequência de pruning (padrão: 3)
  -did, --device-id <id>      ID da GPU (padrão: 0)
  -h, --help                  Exibir esta ajuda
EOF
    exit 0
}

# Processa argumentos passados na linha de comando
while [ $# -gt 0 ]; do
    case $1 in
        -h|--help) show_help ;;
        --simulations) SIMULATIONS="$2"; shift 2 ;;
        --clients|-c) CLIENTS="$2"; shift 2 ;;
        --rounds|-r) ROUNDS="$2"; shift 2 ;;
        --prune-freq) PRUNE_FREQ="$2"; shift 2 ;;
        -did|--device-id) DEVICE_ID="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift 1 ;;
    esac
done

DATASETS=(
    "OxfordPets:37"
    "DTD:47"
    "FGVCAircraft:100"
    "Flowers102:102"
)

TOTAL_DATASETS=${#DATASETS[@]}

echo "================================================================="
echo "        ORQUESTRADOR DE EXPERIMENTOS PUMA-GT (CLIP)"
echo "================================================================="
echo "  Estratégia:         $STRATEGY (prune_freq=$PRUNE_FREQ)"
echo "  PaCA Adaptativo:    ATIVADO (--adaptive-paca)"
echo "  Modelo:             $MODEL"
echo "  Simulações/dataset: $SIMULATIONS"
echo "  Clientes:           $CLIENTS"
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
    echo "[$IDX/$TOTAL_DATASETS] Iniciando PUMA-GT: $DATASET_NAME ($NUM_CLASSES classes)"
    echo "Horário de início: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================="

    CMD=(
        ./run.sh
        --simulations "$SIMULATIONS"
        --clients "$CLIENTS"
        --strategy "$STRATEGY"
        --prune-freq "$PRUNE_FREQ"
        --adaptive-paca
        --model "$MODEL"
        --rounds "$ROUNDS"
        --dataset "$DATASET_NAME"
        --num-classes "$NUM_CLASSES"
        --auto-next
        --skip-train-eval
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

echo "🎉 TODOS OS EXPERIMENTOS PUMA-GT FORAM CONCLUÍDOS COM SUCESSO!"
echo "Horário final: $(date '+%Y-%m-%d %H:%M:%S')"
printf "Tempo total de execução: %02dh:%02dm:%02ds\n" "$TOTAL_H" "$TOTAL_M" "$TOTAL_S"
echo "================================================================="
