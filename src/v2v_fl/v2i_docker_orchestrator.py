#!/usr/bin/env python3
"""
v2i_docker_orchestrator.py
============================
Orquestrador V2I (Vehicle-to-Infrastructure) para Treinamento Federado
com antenas fixas (RSUs — Road Side Units).

Serve como BASELINE de comparação contra o modo V2V-OTA descentralizado.

Diferenças em relação ao V2V (v2v_docker_orchestrator.py):
  - As RSUs são pontos fixos no mapa (definidos em rsu_config.json).
  - A detecção é veículo ↔ RSU (não veículo ↔ veículo).
  - ≥1 veículo no raio da RSU é suficiente para agregar.
  - A agregação OTA é feita NO HOST (neste script), não nos containers.
  - As RSUs são stateless: só agregam os modelos do encontro atual.
  - O conhecimento permanece nos veículos, sem modelo global persistido na RSU.

Fluxo:
  1. Carrega posições das RSUs do rsu_config.json.
  2. Inicia containers Docker dos veículos (mesmos do V2V).
  3. Inicia SUMO e conecta via TraCI.
  4. A cada step do SUMO:
     a) Detecta veículos dentro do raio de cada RSU.
     b) Calcula o ETC de cada veículo com a RSU.
     c) Se ETC suficiente: gera sinal de encontro V2I.
     d) Espera veículos salvarem seus .pt no volume compartilhado.
     e) Carrega os .pt e simula canal, beamforming e superposição OTA na RSU.
     f) Salva o modelo agregado para os veículos carregarem.
     g) Descarta os pesos temporários após a confirmação dos veículos.
  5. Encerra quando o número máximo de encontros é atingido.

Uso:
    cd src/v2v_fl/
    sg docker -c "uv run python3 v2i_docker_orchestrator.py \\
        --sumo-cfg maps/marambaia.sumocfg \\
        --rsu-config maps/rsu_config_marambaia.json \\
        --total-clients 5 \\
        --encounters 6 \\
        --dataset MNIST \\
        --build"
"""

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from utils.model_utils import save_state_dict_atomic

# Importar o gerador de compose do modo V2V (mesmo formato de containers)
from v2v_fl.generate_v2v_compose import generate as compose_generate

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("v2i_orchestrator")

# ── Constantes ────────────────────────────────────────────────────────────────

# Duração de cada step de simulação do SUMO (segundos)
DEFAULT_STEP_LENGTH = 1.0

# Tempo mínimo de contato (ETC) em segundos para que um encontro V2I
# seja considerado viável. Menor que o V2V porque a RSU é fixa
# e a transmissão é mais confiável.
DEFAULT_MIN_CONTACT_TIME = 4.0

# Teto máximo para o ETC em segundos. Valores acima são truncados.
MAX_ETC = 120.0

# Segundos de warmup antes de começar a sinalizar encontros.
DEFAULT_WARMUP = 60

# Segundos de cooldown entre encontros na MESMA RSU.
# Diferente do V2V, o cooldown é por RSU (não global).
DEFAULT_COOLDOWN = 15

DEFAULT_EXCHANGE_TIMEOUT = 300
DEFAULT_OTA_EPSILON = 0.1
DEFAULT_OTA_BANDWIDTH_HZ = 100e3

# Raiz do projeto (FL_ServerPuma/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── Estruturas de Dados ──────────────────────────────────────────────────────

@dataclass
class VehicleState:
    """Estado instantâneo de um veículo no SUMO (idêntico ao V2V)."""
    veh_id: str
    x: float
    y: float
    speed: float
    edge: str
    angle: float = 0.0


@dataclass
class V2IEvent:
    """Evento de log da simulação V2I para análise posterior."""
    sim_time: float
    event_type: str
    rsu_id: str
    vehicles: list[str]
    details: dict = field(default_factory=dict)


# ── Funções de Geometria ─────────────────────────────────────────────────────

def distance_to_rsu(vehicle: VehicleState, rsu: dict) -> float:
    """Calcula a distância euclidiana entre um veículo e uma RSU."""
    return math.sqrt((vehicle.x - rsu["x"]) ** 2 + (vehicle.y - rsu["y"]) ** 2)


def estimate_contact_time_v2i(
    vehicle: VehicleState,
    rsu: dict,
) -> float | str:
    """Estima o Tempo de Contato (ETC) entre um veículo e uma RSU fixa.

    A RSU é um ponto fixo (speed=0), então a equação se simplifica:
      - dp = posição relativa (rsu - veículo)
      - dv = velocidade relativa = -velocidade do veículo (RSU parada)
      - A = |dv|^2 = speed^2
      - B = 2 * (dp · dv)
      - C = |dp|^2 - R^2

    Args:
        vehicle: Estado do veículo (posição, velocidade, ângulo).
        rsu: Dicionário com "x", "y", "radius" da RSU.

    Returns:
        ETC em segundos (truncado em MAX_ETC).
        "Carro Parado" se o veículo está essencialmente parado.
        0.0 se o veículo já está fora do raio.
    """
    radius = rsu["radius"]
    dist = distance_to_rsu(vehicle, rsu)

    # Se já está fora do raio, ETC = 0
    if dist >= radius:
        return 0.0

    # Decompor velocidade do veículo em componentes X e Y
    # Convenção SUMO: 0=Norte, sentido horário
    angle_rad = math.radians(vehicle.angle)
    vx = vehicle.speed * math.sin(angle_rad)   # Componente Leste
    vy = vehicle.speed * math.cos(angle_rad)   # Componente Norte

    # Posição relativa (RSU - veículo)
    dp_x = rsu["x"] - vehicle.x
    dp_y = rsu["y"] - vehicle.y

    # Velocidade relativa (0 - vel_veículo) = (-vx, -vy)
    dv_x = -vx
    dv_y = -vy

    # Coeficientes da equação quadrática
    A = dv_x * dv_x + dv_y * dv_y         # |dv|^2 = speed^2
    B = 2.0 * (dp_x * dv_x + dp_y * dv_y)
    C = dp_x * dp_x + dp_y * dp_y - radius * radius

    # Caso especial: veículo parado (A ≈ 0)
    if A < 1e-9:
        if vehicle.speed < 0.1:
            return "Carro Parado"
        remaining = radius - dist
        return min(remaining / vehicle.speed, MAX_ETC)

    # Discriminante
    discriminante = B * B - 4.0 * A * C
    if discriminante < 0:
        return MAX_ETC

    sqrt_d = math.sqrt(discriminante)
    t_exit = (-B + sqrt_d) / (2.0 * A)

    if t_exit <= 0:
        return 0.0

    return min(t_exit, MAX_ETC)


# ── Detecção de Proximidade V2I ──────────────────────────────────────────────

def find_vehicles_in_rsu_range(
    vehicles: list[VehicleState],
    rsus: list[dict],
) -> dict[str, list[tuple[VehicleState, float]]]:
    """Para cada RSU, retorna a lista de veículos dentro do seu raio.

    Args:
        vehicles: Lista de estados dos veículos ativos na simulação.
        rsus: Lista de dicts com "id", "x", "y", "radius".

    Returns:
        Dict {rsu_id: [(veículo, distância), ...]}
        Apenas RSUs com pelo menos 1 veículo no raio são incluídas.
    """
    result = {}
    for rsu in rsus:
        in_range = []
        for v in vehicles:
            dist = distance_to_rsu(v, rsu)
            if dist <= rsu["radius"]:
                in_range.append((v, dist))
        if in_range:
            result[rsu["id"]] = in_range
    return result


# ── Escrita Atômica de Sinais de Encontro ────────────────────────────────────

def write_encounter_signal(
    encounters_dir: Path,
    encounter_data: dict,
    encounter_id: int,
) -> Path:
    """Escreve o sinal de encontro V2I (JSON) de forma ATÔMICA.

    Idêntico ao V2V: write-to-temp + rename para evitar leituras parciais.
    """
    final_path = encounters_dir / f"encounter_{encounter_id}.json"

    fd, tmp_path = tempfile.mkstemp(
        dir=str(encounters_dir),
        prefix=".encounter_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as tmp_file:
            json.dump(encounter_data, tmp_file)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.rename(tmp_path, final_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return final_path


# ── Controle Docker (reutilizado do V2V) ─────────────────────────────────────

def wait_all_clients_epoch_done(
    encounters_dir: Path,
    client_indices: list[int],
    round_num: int,
    timeout: int = 300,
):
    """Espera todos os clientes sinalizarem que completaram a época."""
    expected = [encounters_dir / f".epoch_done_client_{c}" for c in client_indices]

    start_round_file = encounters_dir / f".start_round_{round_num}"
    if not start_round_file.exists():
        start_round_file.touch()

    log.info(
        "Aguardando todos os %d clientes completarem a época da rodada %d...",
        len(client_indices), round_num,
    )

    for tick in range(timeout):
        if all(f.exists() for f in expected):
            for f in expected:
                f.unlink(missing_ok=True)
            log.info("Todos os clientes completaram a época!")
            return True

        if tick > 0 and tick % 15 == 0:
            count = sum(f.exists() for f in expected)
            log.info("  ... %d/%d clientes concluídos (%ds)", count, len(expected), tick)

        time.sleep(1)

    log.warning("Timeout (%ds) aguardando clientes completarem a época!", timeout)
    return False


def docker_compose_up(compose_file: Path, project_root: Path, build: bool = False) -> bool:
    """Inicia todos os containers de veículos."""
    cmd = ["docker", "compose", "--project-directory", ".", "-f", str(compose_file), "up", "-d"]
    if build:
        cmd.insert(-1, "--build")
    log.info("Iniciando containers Docker...")

    result = subprocess.run(cmd, cwd=str(project_root))
    if result.returncode != 0:
        log.error("docker compose up falhou (exit code %d)", result.returncode)
        return False
    log.info("Containers iniciados com sucesso!")
    return True


def docker_compose_down(compose_file: str, project_root: Path) -> None:
    """Destrói os containers ao final da simulação."""
    log.info("Encerrando containers Docker...")
    subprocess.run(
        ["docker", "compose", "--project-directory", ".", "-f", compose_file,
         "down", "--remove-orphans", "-v"],
        cwd=str(project_root),
        capture_output=True, text=True, timeout=120,
    )
    log.info("Containers removidos.")


# ── Agregação OTA Stateless no Host ──────────────────────────────────────────

def ota_aggregate(
    client_states: dict[int, dict],
    positions: dict[int, tuple[float, float]],
    rsu: dict,
    *,
    epsilon: float = DEFAULT_OTA_EPSILON,
    bandwidth_hz: float = DEFAULT_OTA_BANDWIDTH_HZ,
) -> dict:
    """Agrega apenas os veículos deste encontro pelo canal OTA veículo → RSU.

    Todos os veículos são transmissores com peso uniforme. A RSU é somente
    receptora, sem pesos próprios ou memória de modelos de encontros anteriores.
    Usa o mesmo pipeline de path loss, fading, beamforming e ruído do V2V.
    """
    import numpy as np

    from v2v_fl.over_the_air.ota_aggregation import (
        apply_ota_channel,
        compute_weight_variance,
        mmse_beamforming,
    )
    from v2v_fl.over_the_air.ota_channel import build_channel_matrices, compute_noise_variance

    if not client_states:
        raise ValueError("Agregação V2I requer pelo menos um veículo transmissor.")
    n_antennas = rsu.get("n_antennas", 16)
    if not isinstance(n_antennas, int) or n_antennas < 1:
        raise ValueError("A RSU deve ter um número inteiro positivo de antenas.")
    if not 0 <= epsilon < 1:
        raise ValueError("O erro de estimação OTA deve estar no intervalo [0, 1).")
    if not math.isfinite(bandwidth_hz) or bandwidth_hz <= 0:
        raise ValueError("A largura de banda OTA deve ser positiva e finita.")

    transmitter_ids = sorted(client_states)
    distances = np.array([
        math.hypot(positions[c][0] - rsu["x"], positions[c][1] - rsu["y"])
        for c in transmitter_ids
    ])
    # Mesma distância mínima do V2V para evitar singularidade no path loss.
    distances = np.maximum(distances, 1.0)
    n_samples = np.ones(len(transmitter_ids), dtype=np.float64)
    H_est, H_true, _, _ = build_channel_matrices(
        distances, n_antennas, n_samples, epsilon=epsilon,
    )
    variances, _ = compute_weight_variance(client_states)
    weight_vars = np.maximum([variances[c] for c in transmitter_ids], 1e-12)
    f_prime, transmit_scalars = mmse_beamforming(H_est, weight_vars, n_samples)

    return apply_ota_channel(
        state_dicts=client_states,
        transmitter_ids=transmitter_ids,
        n_samples=n_samples,
        H_true=H_true,
        transmit_scalars=transmit_scalars,
        f_prime=f_prime,
        noise_variance=compute_noise_variance(bandwidth_hz),
        weight_variances=weight_vars,
    )


# ── Loop Principal ───────────────────────────────────────────────────────────

def run_orchestrator(args: argparse.Namespace) -> None:
    """Loop principal do orquestrador V2I."""
    import torch

    # Importar TraCI
    try:
        import traci  # pyright: ignore [reportMissingImports]
    except ImportError:
        log.error(
            "O pacote 'traci' não foi encontrado. "
            "Instale o SUMO (sumo-tools) ou via: pip install traci"
        )
        sys.exit(1)

    # ── Carregar configuração das RSUs ────────────────────────────────────
    rsu_config_path = Path(args.rsu_config).resolve()
    if not rsu_config_path.exists():
        log.error("Arquivo de configuração das RSUs não encontrado: %s", rsu_config_path)
        sys.exit(1)

    with open(rsu_config_path, "r") as f:
        rsu_data = json.load(f)
    rsus = rsu_data["rsus"]
    log.info("Carregadas %d RSUs do arquivo %s", len(rsus), rsu_config_path.name)

    # Validar configuração do SUMO
    sumo_cfg = Path(args.sumo_cfg).resolve()
    if not sumo_cfg.exists():
        log.error("Arquivo SUMO config não encontrado: %s", sumo_cfg)
        sys.exit(1)

    sumo_cmd = [
        "sumo-gui" if args.gui else "sumo",
        "-c", str(sumo_cfg),
        "--step-length", str(args.step_length),
    ]

    # ── Banner de configuração ────────────────────────────────────────────
    log.info("=" * 60)
    log.info("V2I Orchestrator - FL OTA (RSUs stateless)")
    log.info("=" * 60)
    log.info("  Total Clientes    : %d", args.total_clients)
    log.info("  RSUs              : %d", len(rsus))
    log.info("  Max Encontros     : %d", args.encounters)
    log.info("  Min Tempo Contato : %.0f s (ETC)", args.min_contact_time)
    log.info("  Warmup            : %d s", args.warmup)
    log.info("  Cooldown (por RSU): %d s", args.cooldown)
    log.info("  Dataset           : %s", args.dataset)
    log.info("  Erro CSI OTA      : %.2f", args.ota_epsilon)
    log.info("  Banda OTA         : %.0f Hz", args.ota_bandwidth_hz)
    for rsu in rsus:
        log.info("  RSU %s: (%.0f, %.0f) raio=%.0fm antenas=%d",
                 rsu["id"], rsu["x"], rsu["y"], rsu["radius"],
                 rsu.get("n_antennas", 16))
    log.info("=" * 60)

    # ── PASSO 1: Limpar e criar pasta de sinalização ─────────────────────
    encounters_dir = PROJECT_ROOT / "src" / "results" / "encounters"
    encounters_dir.mkdir(parents=True, exist_ok=True)

    # Limpar arquivos de execuções anteriores
    for pattern in ["*.json", "*.pt", ".done_enc_*", ".start_round_*",
                     ".epoch_done_*", ".simulation_done", ".encounter_*.tmp"]:
        for f in encounters_dir.glob(pattern):
            f.unlink()

    # ── PASSO 2: Gerar Docker Compose e iniciar containers ───────────────
    client_indices = list(range(args.total_clients))

    compose_file = compose_generate(
        client_indices=client_indices,
        dataset=args.dataset,
        rounds=3,
        prune=args.prune,
        ala=args.ala,
        exp_name='default_exp_v2i',
        max_epochs=args.max_epochs,
        model=args.model,
        config=args.config,
        strategy=args.strategy,
        rank=args.rank,
        paca=args.paca,
    )

    if not docker_compose_up(compose_file, PROJECT_ROOT, build=args.build):
        sys.exit(1)

    # ── PASSO 3: Warmup ──────────────────────────────────────────────────
    if args.warmup > 0:
        log.info("Aguardando %ds de warmup...", args.warmup)
        time.sleep(args.warmup)
        log.info("Warmup concluído!")

    # ── PASSO 3.5: Treino Inicial (Rodada 0) ─────────────────────────────
    all_client_indices = list(range(args.total_clients))
    log.info("--- Iniciando Treino (Rodada 0) ---")
    wait_all_clients_epoch_done(encounters_dir, all_client_indices, round_num=0, timeout=300)

    # ── PASSO 4: Iniciar simulação SUMO ──────────────────────────────────
    log.info("Iniciando SUMO...")
    traci.start(sumo_cmd)

    # Contadores e estado da simulação
    encounter_count = 0
    rejected_count = 0
    event_log: list[V2IEvent] = []
    veh_to_client_idx: dict[str, int] = {}
    next_client_idx = 0

    # Cooldown POR RSU (cada RSU tem seu próprio timer)
    last_encounter_per_rsu: dict[str, float] = {
        rsu["id"]: -float('inf') for rsu in rsus
    }

    # Controle de duplicação: evitar re-sinalizar o mesmo grupo na mesma RSU
    prev_clients_per_rsu: dict[str, frozenset[int]] = {}

    try:
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            sim_time = traci.simulation.getTime()
            step += 1

            # ── Coletar estado dos veículos ───────────────────────────────
            veh_ids = traci.vehicle.getIDList()
            vehicles = []

            for vid in veh_ids:
                if vid not in veh_to_client_idx and next_client_idx < args.total_clients:
                    veh_to_client_idx[vid] = next_client_idx
                    next_client_idx += 1
                    log.info(
                        "Veículo SUMO '%s' vinculado ao container cliente %d",
                        vid, veh_to_client_idx[vid],
                    )

                if vid in veh_to_client_idx:
                    x, y = traci.vehicle.getPosition(vid)
                    speed = traci.vehicle.getSpeed(vid)
                    edge = traci.vehicle.getRoadID(vid)
                    angle = traci.vehicle.getAngle(vid)
                    vehicles.append(VehicleState(
                        veh_id=vid, x=x, y=y,
                        speed=speed, edge=edge, angle=angle,
                    ))

            if not vehicles:
                continue

            # ── Detectar veículos no raio de cada RSU ─────────────────────
            vehicles_by_rsu = find_vehicles_in_rsu_range(vehicles, rsus)

            for rsu in rsus:
                rsu_id = rsu["id"]

                if rsu_id not in vehicles_by_rsu:
                    # Nenhum veículo no raio desta RSU
                    if rsu_id in prev_clients_per_rsu:
                        del prev_clients_per_rsu[rsu_id]
                    continue

                # ── Verificar cooldown desta RSU ──────────────────────────
                if (sim_time - last_encounter_per_rsu[rsu_id]) < args.cooldown:
                    continue

                vehs_with_dist = vehicles_by_rsu[rsu_id]

                # ── Filtrar veículos com ETC suficiente ───────────────────
                viable = []
                for v, dist in vehs_with_dist:
                    etc = estimate_contact_time_v2i(v, rsu)
                    if isinstance(etc, str):
                        # "Carro Parado" — considerar viável
                        viable.append((v, dist, MAX_ETC))
                    elif etc >= args.min_contact_time:
                        viable.append((v, dist, etc))

                if not viable:
                    rejected_count += 1
                    continue

                # ── Verificar se é um NOVO grupo (não repetição) ──────────
                c_indices = sorted(
                    veh_to_client_idx[v.veh_id] for v, _, _ in viable
                )
                current_clients_key = frozenset(c_indices)

                if prev_clients_per_rsu.get(rsu_id) == current_clients_key:
                    continue

                # ── ETC suficiente: gerar encontro V2I ────────────────────
                encounter_count += 1
                min_etc = min(etc_val for _, _, etc_val in viable)

                veh_names = [v.veh_id for v, _, _ in viable]
                log.info(
                    "Encontro V2I %d/%d! RSU %s ← Veículos %s (Clientes %s) | ETC=%.1fs",
                    encounter_count, args.encounters,
                    rsu_id, veh_names, c_indices, min_etc,
                )

                # ── Montar sinal de encontro V2I ──────────────────────────
                encounter_data = {
                    "encounter_id": encounter_count,
                    "type": "v2i",
                    "rsu_id": rsu_id,
                    "clients": c_indices,
                    "timestamp": sim_time,
                    "etc_seconds": min_etc,
                    "positions": {
                        veh_to_client_idx[v.veh_id]: (v.x, v.y)
                        for v, _, _ in viable
                    },
                    "rsu_position": (rsu["x"], rsu["y"]),
                    "rsu_n_antennas": rsu.get("n_antennas", 16),
                    "exchange_timeout": DEFAULT_EXCHANGE_TIMEOUT,
                }

                # Escrita atômica do JSON
                write_encounter_signal(encounters_dir, encounter_data, encounter_count)

                # Registrar evento
                event_log.append(V2IEvent(
                    sim_time=sim_time,
                    event_type="encounter_formed",
                    rsu_id=rsu_id,
                    vehicles=veh_names,
                    details={
                        "encounter_id": encounter_count,
                        "clients": c_indices,
                        "etc_seconds": round(min_etc, 1),
                        "aggregation": "ota",
                        "n_antennas": rsu.get("n_antennas", 16),
                        "ota_epsilon": args.ota_epsilon,
                        "ota_bandwidth_hz": args.ota_bandwidth_hz,
                        "distances": {
                            veh_to_client_idx[v.veh_id]: round(dist, 1)
                            for v, dist, _ in viable
                        },
                    },
                ))

                # Atualizar estado
                prev_clients_per_rsu[rsu_id] = current_clients_key
                last_encounter_per_rsu[rsu_id] = sim_time

                # ── PAUSAR SUMO e esperar veículos salvarem .pt ───────────
                log.info(
                    "SUMO PAUSADO. Aguardando %d clientes salvarem pesos para RSU %s...",
                    len(c_indices), rsu_id,
                )

                exchange_start = time.time()
                exchange_timeout = DEFAULT_EXCHANGE_TIMEOUT

                # Esperar que cada veículo salve seu .pt
                expected_pt_files = [
                    encounters_dir / f"client_{c}_enc_{encounter_count}.pt"
                    for c in c_indices
                ]

                for tick in range(exchange_timeout):
                    received = [f.exists() for f in expected_pt_files]
                    if all(received):
                        log.info("  Todos os %d .pt recebidos!", len(c_indices))
                        break
                    if tick > 0 and tick % 15 == 0:
                        log.info(
                            "  ... %d/%d .pt recebidos (%ds)",
                            sum(received), len(expected_pt_files), tick,
                        )
                    time.sleep(1)
                else:
                    received_count = sum(f.exists() for f in expected_pt_files)
                    log.warning(
                        "Timeout (%ds) esperando .pt do encontro %d (%d/%d recebidos)",
                        exchange_timeout, encounter_count,
                        received_count, len(expected_pt_files),
                    )

                # ── AGREGAÇÃO OTA STATELESS NO HOST ───────────────────────
                log.info("  Realizando agregação OTA na RSU %s...", rsu_id)

                # Carregar pesos dos veículos
                client_states = {}
                for c in c_indices:
                    pt_path = encounters_dir / f"client_{c}_enc_{encounter_count}.pt"
                    if pt_path.exists():
                        client_states[c] = torch.load(
                            pt_path, map_location="cpu", weights_only=True,
                        )

                if not client_states:
                    log.warning("  Nenhum .pt carregado! Pulando agregação.")
                else:
                    positions = {
                        veh_to_client_idx[v.veh_id]: (v.x, v.y)
                        for v, _, _ in viable
                    }
                    aggregated = ota_aggregate(
                        client_states, positions, rsu,
                        epsilon=args.ota_epsilon,
                        bandwidth_hz=args.ota_bandwidth_hz,
                    )
                    event_log[-1].details["received_clients"] = sorted(client_states)

                    # Salvar resultado para cada veículo carregar
                    for c in c_indices:
                        result_path = encounters_dir / (
                            f"rsu_result_enc_{encounter_count}_client_{c}.pt"
                        )
                        save_state_dict_atomic(aggregated, result_path)

                    log.info("  Resultado disponível para %d clientes.", len(c_indices))
                    del aggregated
                # Nenhum tensor é retido pela RSU para o próximo encontro.
                del client_states

                # ── Esperar confirmação .done dos clientes ────────────────
                expected_done_files = [
                    encounters_dir / f".done_enc_{encounter_count}_client_{c}"
                    for c in c_indices
                ]

                for tick in range(exchange_timeout):
                    received = [f.exists() for f in expected_done_files]
                    if all(received):
                        exchange_time = time.time() - exchange_start
                        log.info(
                            "Todas as %d confirmações recebidas em %.0fs!",
                            len(expected_done_files), exchange_time,
                        )
                        event_log[-1].details["exchange_time"] = round(exchange_time, 1)
                        time.sleep(3)
                        break
                    if tick > 0 and tick % 15 == 0:
                        count = sum(f.exists() for f in expected_done_files)
                        log.info(
                            "  ... %d/%d confirmações (%ds)",
                            count, len(expected_done_files), tick,
                        )
                    time.sleep(1)
                else:
                    exchange_time = time.time() - exchange_start
                    received_count = sum(f.exists() for f in expected_done_files)
                    log.warning(
                        "Timeout (%ds) esperando confirmações do encontro %d (%d/%d)",
                        exchange_timeout, encounter_count,
                        received_count, len(expected_done_files),
                    )
                    event_log[-1].details["exchange_time"] = round(exchange_time, 1)

                # Os arquivos .pt são apenas transporte temporário deste encontro.
                for c in c_indices:
                    (encounters_dir / f"client_{c}_enc_{encounter_count}.pt").unlink(
                        missing_ok=True,
                    )
                    (encounters_dir / f"rsu_result_enc_{encounter_count}_client_{c}.pt").unlink(
                        missing_ok=True,
                    )

                log.info("SUMO RETOMADO.")

                # Não iniciar uma época adicional depois do último encontro.
                if encounter_count >= args.encounters:
                    log.info(
                        "Limite de %d encontros atingido. Encerrando simulação.",
                        args.encounters,
                    )
                    break

                # ── Iniciar nova rodada de treino ─────────────────────────
                log.info(f"--- Iniciando Treino (Rodada {encounter_count}) ---")
                wait_all_clients_epoch_done(
                    encounters_dir, all_client_indices,
                    round_num=encounter_count, timeout=300,
                )

            # Verificar limite após iterar todas as RSUs
            if encounter_count >= args.encounters:
                break

    except KeyboardInterrupt:
        log.info("\nSimulação interrompida pelo usuário.")
    finally:
        # ── PASSO 5: Limpeza geral ───────────────────────────────────────
        log.info("Encerrando simulação V2I...")
        done_file = encounters_dir / ".simulation_done"
        done_file.touch()
        time.sleep(5)
        docker_compose_down(compose_file, PROJECT_ROOT)
        traci.close()
        log.info("SUMO encerrado.")
        # Também descartar pesos temporários se um encontro foi interrompido.
        for pattern in ["client_*_enc_*.pt", "rsu_result_enc_*.pt"]:
            for path in encounters_dir.glob(pattern):
                path.unlink(missing_ok=True)

        # Estatísticas finais
        log.info("=" * 60)
        log.info("ESTATÍSTICAS DA SIMULAÇÃO V2I")
        log.info("=" * 60)
        log.info("  Encontros V2I viáveis  : %d", encounter_count)
        log.info("  Encontros rejeitados   : %d (ETC insuficiente)", rejected_count)
        log.info("  Total de steps SUMO    : %d", step)
        for rsu in rsus:
            rsu_id = rsu["id"]
            rsu_encounters = sum(
                1 for e in event_log
                if e.event_type == "encounter_formed" and e.rsu_id == rsu_id
            )
            log.info("  %s: %d encontros OTA (RSU stateless)", rsu_id, rsu_encounters)
        log.info("=" * 60)

        # Salvar log de eventos
        log_path = PROJECT_ROOT / "src" / "results" / "v2i_events.jsonl"
        with open(log_path, "w") as f:
            for event in event_log:
                f.write(json.dumps({
                    "sim_time": event.sim_time,
                    "type": event.event_type,
                    "rsu_id": event.rsu_id,
                    "vehicles": event.vehicles,
                    **event.details,
                }, default=str) + "\n")
        log.info("Log de eventos salvo em: %s", log_path)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="V2I FL Orchestrator — OTA com RSUs stateless (antenas fixas)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sumo-cfg", required=True,
        help="Caminho para o arquivo .sumocfg do SUMO",
    )
    parser.add_argument(
        "--rsu-config", required=True,
        help="Caminho para o rsu_config.json com posições das RSUs",
    )
    parser.add_argument(
        "--total-clients", type=int, default=5,
        help="Total de veículos (containers) a criar no Docker (default: 5)",
    )
    parser.add_argument(
        "--encounters", type=int, default=6,
        help="Número máximo de encontros V2I viáveis (default: 6)",
    )
    parser.add_argument(
        "--min-contact-time", type=float, default=DEFAULT_MIN_CONTACT_TIME,
        help=f"Tempo mínimo de contato (ETC) em segundos (default: {DEFAULT_MIN_CONTACT_TIME})",
    )
    parser.add_argument(
        "--dataset", type=str, default="MNIST",
        choices=["MNIST", "Cifar10", "Cifar100"],
        help="Dataset para o treinamento (default: MNIST)",
    )
    parser.add_argument(
        "--prune", type=int, default=1, choices=[0, 1],
        help="Habilitar pruning adaptativo (0=sim, 1=não, default: 1)",
    )
    parser.add_argument(
        "--ala", type=int, default=1, choices=[0, 1],
        help="Habilitar FedALA (0=sim, 1=não/FedAvg, default: 1)",
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Usar sumo-gui (interface gráfica)",
    )
    parser.add_argument(
        "--build", action="store_true",
        help="Forçar o rebuild das imagens Docker",
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP,
        help=f"Segundos de warmup (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--cooldown", type=int, default=DEFAULT_COOLDOWN,
        help=f"Segundos de cooldown por RSU (default: {DEFAULT_COOLDOWN})",
    )
    parser.add_argument(
        "--step-length", type=float, default=DEFAULT_STEP_LENGTH,
        help=f"Duração de cada step do SUMO (default: {DEFAULT_STEP_LENGTH})",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=0,
        help="Limite de épocas por cliente (0 = sem limite, default: 0)",
    )
    parser.add_argument(
        "--ota-epsilon", type=float, default=DEFAULT_OTA_EPSILON,
        help="Erro de estimação do canal OTA no intervalo [0, 1) (default: 0.1)",
    )
    parser.add_argument(
        "--ota-bandwidth-hz", type=float, default=DEFAULT_OTA_BANDWIDTH_HZ,
        help="Largura de banda usada no ruído térmico OTA em Hz (default: 100000)",
    )
    # ── Parâmetros do Modelo ──────────────────────────────────────────────
    parser.add_argument("--model", type=str, default="cnn",
                        choices=["cnn", "clip"],
                        help="Modelo a ser utilizado (default: cnn)")
    parser.add_argument("--config", type=str,
                        default="lora_clip/train_config.yml",
                        help="Caminho do config YAML para o CLIP")
    parser.add_argument("--strategy", type=str, default="lora",
                        choices=["lora", "adalora", "sora_with_schedule",
                                 "sora_no_schedule"],
                        help="Estratégia de PEFT")
    parser.add_argument("--rank", type=int, default=8,
                        help="Rank para LoRA/SoRA")
    parser.add_argument("--paca", type=int, default=12,
                        help="Número de camadas upper para PaCA")

    args = parser.parse_args()
    if not 0 <= args.ota_epsilon < 1:
        parser.error("--ota-epsilon deve estar no intervalo [0, 1).")
    if not math.isfinite(args.ota_bandwidth_hz) or args.ota_bandwidth_hz <= 0:
        parser.error("--ota-bandwidth-hz deve ser positivo e finito.")
    run_orchestrator(args)


if __name__ == "__main__":
    main()
