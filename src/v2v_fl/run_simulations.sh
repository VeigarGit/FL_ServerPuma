#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

# CONFIGURAÇÃO — edite os valores abaixo e execute o arquivo sem argumentos.
# Parâmetros compartilhados pelas duas simulações.
DATASET="Cifar10"
DATASET_PROFILES=("alpha01" "alpha1")
DATASET_DIRS=("src/dataset/Cifar10" "src/dataset/Cifar10_alpha1")
TOTAL_CLIENTS=10
ENCOUNTERS=30
SEED=36
RUN_ID="new-simu"                    # Prefixo; vazio gera um ID para as quatro execuções.
MAX_EPOCHS=30                # 0: sem limite adicional ao número de encontros.
BUILD=0                     # Reconstrói somente na primeira execução V2V.
GUI=0                          # 1: abre a interface gráfica do SUMO.
WARMUP=30
COOLDOWN=15
STEP_LENGTH=1.0
ALA=1                          # 0: habilitado; 1: desabilitado.
PRUNE=1                        # 0: habilitado; 1: desabilitado.
MODEL="cnn"
CONFIG="lora_clip/train_config.yml"
STRATEGY="lora"
RANK=8
PACA=12
# Caminhos relativos à raiz do projeto ou caminhos absolutos.
SUMO_CFG="src/v2v_fl/maps/marambaia_recorte.sumocfg"
RSU_CONFIG="src/v2v_fl/maps/rsu_config_marambaia_recorte.json"

# Parâmetros específicos do V2V.
V2V_RADIUS=300
V2V_MIN_CLIENTS=3
V2V_MAX_ENCOUNTERS_PER_PAIR=0
V2V_MIN_CONTACT_TIME=9
V2V_EXCHANGE_TIMEOUT=400

# Parâmetros específicos do V2I.
V2I_MIN_CLIENTS=3              # Mínimo de veículos viáveis por encontro na mesma RSU.
V2I_MIN_CONTACT_TIME=9
V2I_EXCHANGE_TIMEOUT=400
V2I_EVALUATION_TIMEOUT=300
V2I_OTA_EPSILON=0.1
V2I_OTA_BANDWIDTH_HZ=100000
# FIM DA CONFIGURAÇÃO

fail() {
    printf 'Erro: %s\n' "$*" >&2
    exit 2
}

(($# == 0)) || fail "Edite a CONFIGURAÇÃO no arquivo e execute sem argumentos."
[[ "$MODEL" == cnn ]] || fail "MODEL deve ser cnn: a execução V2I suporta somente CNN."
[[ "$BUILD" =~ ^[01]$ && "$GUI" =~ ^[01]$ ]] || fail "BUILD e GUI devem ser 0 ou 1."
[[ -n "$RUN_ID" ]] || RUN_ID="marambaia-recorte-$(date +%Y%m%d-%H%M%S)-$$"
[[ "$SUMO_CFG" == /* ]] || SUMO_CFG="$PROJECT_ROOT/$SUMO_CFG"
[[ "$RSU_CONFIG" == /* ]] || RSU_CONFIG="$PROJECT_ROOT/$RSU_CONFIG"
[[ -f "$SUMO_CFG" ]] || fail "Arquivo SUMO não encontrado: $SUMO_CFG"
[[ -f "$RSU_CONFIG" ]] || fail "Configuração das RSUs não encontrada: $RSU_CONFIG"
[[ "$RUN_ID" =~ [^[:space:]] && "$RUN_ID" != . && "$RUN_ID" != .. && "$RUN_ID" != */* ]] \
    || fail "RUN_ID deve ser um nome de diretório não vazio."
[[ "$TOTAL_CLIENTS" =~ ^[0-9]+$ ]] && ((10#$TOTAL_CLIENTS >= 2)) \
    || fail "TOTAL_CLIENTS deve ser um inteiro maior ou igual a 2."
[[ "$V2I_MIN_CLIENTS" =~ ^[0-9]+$ ]] \
    && ((10#$V2I_MIN_CLIENTS >= 2 && 10#$V2I_MIN_CLIENTS <= 10#$TOTAL_CLIENTS)) \
    || fail "V2I_MIN_CLIENTS deve ser um inteiro entre 2 e TOTAL_CLIENTS."
[[ "$ENCOUNTERS" =~ ^[0-9]+$ ]] && ((10#$ENCOUNTERS >= 1)) \
    || fail "ENCOUNTERS deve ser um inteiro positivo."
[[ "$MAX_EPOCHS" =~ ^[0-9]+$ ]] || fail "MAX_EPOCHS deve ser um inteiro não negativo."
((10#$MAX_EPOCHS == 0 || 10#$MAX_EPOCHS >= 10#$ENCOUNTERS)) \
    || fail "MAX_EPOCHS deve ser 0 ou pelo menos o número de encontros."
(( ${#DATASET_PROFILES[@]} > 0 && ${#DATASET_PROFILES[@]} == ${#DATASET_DIRS[@]} )) \
    || fail "Informe um diretório para cada perfil de dataset."
for index in "${!DATASET_PROFILES[@]}"; do
    profile="${DATASET_PROFILES[$index]}"
    [[ "$profile" =~ ^[a-zA-Z0-9_-]+$ ]] || fail "Perfil inválido: $profile"
    directory="${DATASET_DIRS[$index]}"
    [[ "$directory" == /* ]] || directory="$PROJECT_ROOT/$directory"
    DATASET_DIRS[$index]="$directory"
    [[ -f "$directory/config.json" ]] || fail "Dataset não encontrado: $directory"
    for ((client=0; client<TOTAL_CLIENTS; client++)); do
        [[ -f "$directory/train/$client.npz" && -f "$directory/test/$client.npz" ]] \
            || fail "Partição ausente para cliente $client em $directory"
    done
    for mode in v2v v2i; do
        [[ ! -e "$PROJECT_ROOT/src/results/runs/$mode/$RUN_ID-$profile" ]] \
            || fail "Já existem resultados para $mode/$RUN_ID-$profile. Altere RUN_ID."
    done
done
command -v uv >/dev/null 2>&1 || fail "uv não encontrado no PATH."
command -v setsid >/dev/null 2>&1 || fail "setsid não encontrado no PATH."

cd -- "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED="$SEED"
comparison_dir="$PROJECT_ROOT/src/results/comparisons/$RUN_ID"
[[ ! -e "$comparison_dir" ]] || fail "Comparação já existe: $comparison_dir"
mkdir -p -- "$comparison_dir"
uv run python -m v2v_fl.validate_mobility \
    --sumo-cfg "$SUMO_CFG" --rsu-config "$RSU_CONFIG" \
    --total-clients "$TOTAL_CLIENTS" --encounters "$ENCOUNTERS" --seed "$SEED" \
    --radius "$V2V_RADIUS" --min-clients "$V2V_MIN_CLIENTS" \
    --min-contact-time "$V2V_MIN_CONTACT_TIME" --cooldown "$COOLDOWN" \
    --v2i-min-clients "$V2I_MIN_CLIENTS" --v2i-min-contact-time "$V2I_MIN_CONTACT_TIME" \
    --step-length "$STEP_LENGTH" --output "$comparison_dir/mobility.json"
common_args=(
    --dataset "$DATASET" --total-clients "$TOTAL_CLIENTS" --encounters "$ENCOUNTERS"
    --seed "$SEED" --max-epochs "$MAX_EPOCHS" --sumo-cfg "$SUMO_CFG"
    --warmup "$WARMUP" --cooldown "$COOLDOWN" --step-length "$STEP_LENGTH"
    --ala "$ALA" --prune "$PRUNE" --model "$MODEL" --config "$CONFIG"
    --strategy "$STRATEGY" --rank "$RANK" --paca "$PACA"
)
v2v_args=(
    --radius "$V2V_RADIUS" --min-clients "$V2V_MIN_CLIENTS"
    --max-encounters-per-pair "$V2V_MAX_ENCOUNTERS_PER_PAIR"
    --min-contact-time "$V2V_MIN_CONTACT_TIME" --exchange-timeout "$V2V_EXCHANGE_TIMEOUT"
)
v2i_args=(
    --min-clients "$V2I_MIN_CLIENTS"
    --rsu-config "$RSU_CONFIG" --min-contact-time "$V2I_MIN_CONTACT_TIME"
    --exchange-timeout "$V2I_EXCHANGE_TIMEOUT" --evaluation-timeout "$V2I_EVALUATION_TIMEOUT"
    --ota-epsilon "$V2I_OTA_EPSILON" --ota-bandwidth-hz "$V2I_OTA_BANDWIDTH_HZ"
)
if ((GUI)); then
    common_args+=(--gui)
fi

child_pid=""
stop_run() {
    trap '' INT TERM
    if [[ -n "$child_pid" ]]; then
        kill -INT -- "-$child_pid" 2>/dev/null \
            || kill -INT "$child_pid" 2>/dev/null || true
        wait "$child_pid" || true
    fi
    exit "$1"
}
trap 'stop_run 130' INT
trap 'stop_run 143' TERM

run_simulation() {
    local mode="$1" status=0
    shift
    printf '\n>>> %s | execução %s\n' "${mode^^}" "$current_run_id"
    (
        trap - INT TERM
        exec setsid uv run python -m "v2v_fl.${mode}_docker_orchestrator" \
            "${common_args[@]}" "$@"
    ) > >(tee "$comparison_dir/$profile-$mode.log") 2>&1 &
    child_pid=$!
    wait "$child_pid" || status=$?
    child_pid=""
    if ((status)); then
        printf 'Falha na simulação %s (código %s).\n' "$mode" "$status" >&2
        return "$status"
    fi
}

report_args=()
run_and_report() {
    local mode="$1" status=0
    run_simulation "$@" || status=$?
    report_args+=(--run-dir "$PROJECT_ROOT/src/results/runs/$mode/$current_run_id")
    if ! uv run python -m v2v_fl.experiment_report "${report_args[@]}" \
        --output-dir "$comparison_dir" --require-complete; then
        ((status)) || status=1
    fi
    return "$status"
}

build_pending="$BUILD"
for index in "${!DATASET_PROFILES[@]}"; do
    profile="${DATASET_PROFILES[$index]}"
    current_run_id="$RUN_ID-$profile"
    profile_args=(--run-id "$current_run_id" --dataset-dir "${DATASET_DIRS[$index]}")
    build_args=()
    if ((build_pending)); then
        build_args=(--build)
    fi
    run_and_report v2v "${v2v_args[@]}" "${profile_args[@]}" "${build_args[@]}"
    build_pending=0
    run_and_report v2i "${v2i_args[@]}" "${profile_args[@]}"
done
printf '\nSimulações finalizadas. Relatórios e logs:\n  %s\n' "$comparison_dir"
