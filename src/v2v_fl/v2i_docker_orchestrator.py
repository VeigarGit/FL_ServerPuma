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
  - São necessários ≥2 veículos no raio da RSU com ETC suficiente para agregar.
  - A agregação OTA é feita no serviço único rsu-service, em CPU.
  - As RSUs são stateless: só agregam os modelos do encontro atual.
  - O conhecimento permanece nos veículos, sem modelo global persistido na RSU.

Fluxo:
  1. Carrega posições das RSUs do rsu_config.json.
  2. Inicia containers Docker dos veículos e o rsu-service.
  3. Inicia SUMO e conecta via TraCI.
  4. A cada step do SUMO:
     a) Detecta veículos dentro do raio de cada RSU.
     b) Calcula o ETC de cada veículo com a RSU.
     c) Com ≥2 veículos elegíveis: publica identificadores, endereço e posições em JSON.
     d) O serviço recebe os pesos por TCP, agrega e aguarda APPLIED dos veículos.
     e) Espera o status terminal do servidor e a avaliação dos clientes.
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
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from runpy import run_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

from utils.model_utils import (  # noqa: E402
    prepare_run,
    summarize_encounter_metrics,
    write_json_atomic,
)
from v2v_fl.over_the_air.ota_aggregation import (  # noqa: E402
    DEFAULT_OTA_BANDWIDTH_HZ,
    DEFAULT_OTA_EPSILON,
)

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
DEFAULT_EVALUATION_TIMEOUT = 300
DEFAULT_RSU_HOST = "rsu-service"
DEFAULT_RSU_PORT = 9000

# Raiz do projeto (FL_ServerPuma/)
compose_generate = run_path(str(PROJECT_ROOT / "docker/generate_v2v_compose.py"))["generate"]


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
    vx = vehicle.speed * math.sin(angle_rad)  # Componente Leste
    vy = vehicle.speed * math.cos(angle_rad)  # Componente Norte

    # Posição relativa (RSU - veículo)
    dp_x = rsu["x"] - vehicle.x
    dp_y = rsu["y"] - vehicle.y

    # Velocidade relativa (0 - vel_veículo) = (-vx, -vy)
    dv_x = -vx
    dv_y = -vy

    # Coeficientes da equação quadrática
    A = dv_x * dv_x + dv_y * dv_y  # |dv|^2 = speed^2
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
    write_json_atomic(encounter_data, final_path)

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
        len(client_indices),
        round_num,
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
    """Inicia os veículos após o healthcheck TCP do rsu-service."""
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
        [
            "docker",
            "compose",
            "--project-directory",
            ".",
            "-f",
            compose_file,
            "down",
            "--remove-orphans",
            "-v",
        ],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        timeout=120,
    )
    log.info("Containers removidos.")


# ── Loop Principal ───────────────────────────────────────────────────────────


def check_client_failures(
    encounters_dir: Path, run_id: str, rsu_id: str, encounter_id: int, clients: list[int]
) -> None:
    """Falhas do transporte ou da avaliação não podem parecer encontros concluídos."""
    for client_id in clients:
        path = encounters_dir / f".failed_enc_{encounter_id}_client_{client_id}.json"
        if not path.exists():
            continue
        with path.open() as stream:
            failure = json.load(stream)
        if not isinstance(failure, dict):
            raise ValueError("Registro de falha do cliente inválido.")
        if all(
            failure.get(key) == value
            for key, value in (
                ("run_id", run_id),
                ("rsu_id", rsu_id),
                ("encounter_id", encounter_id),
                ("client_id", client_id),
            )
        ):
            raise RuntimeError(
                f"Cliente {client_id}: {failure.get('error', 'Falha V2I sem descrição.')}"
            )


def wait_for_rsu_status(
    encounters_dir: Path,
    *,
    run_id: str,
    rsu_id: str,
    encounter_id: int,
    clients: list[int],
    timeout: float,
) -> dict:
    """Espera o status final da RSU; um arquivo de outra execução não libera a espera."""
    path = encounters_dir / f"rsu_status_enc_{encounter_id}.json"
    deadline = time.monotonic() + timeout
    expected = set(clients)
    while True:
        check_client_failures(encounters_dir, run_id, rsu_id, encounter_id, clients)
        if path.exists():
            with path.open() as stream:
                status = json.load(stream)
            if not isinstance(status, dict):
                raise ValueError("Status da RSU deve ser um objeto JSON.")
            if status.get("run_id") == run_id:
                if (
                    status.get("rsu_id") != rsu_id
                    or type(status.get("encounter_id")) is not int
                    or status["encounter_id"] != encounter_id
                ):
                    raise ValueError("Status da RSU contém identificadores incorretos.")
                terminal = status.get("status")
                if not isinstance(terminal, str) or terminal not in {
                    "completed",
                    "failed",
                    "timed_out",
                }:
                    raise ValueError("Status terminal da RSU desconhecido.")
                for key in ("clients", "received_clients", "applied_clients"):
                    values = status.get(key)
                    if (
                        not isinstance(values, list)
                        or any(type(value) is not int for value in values)
                        or len(values) != len(set(values))
                        or not set(values) <= expected
                    ):
                        raise ValueError(f"Participantes inválidos em {key} no status da RSU.")
                if set(status["clients"]) != expected or not set(status["applied_clients"]) <= set(
                    status["received_clients"]
                ):
                    raise ValueError(
                        "Participantes do status diferem dos participantes do encontro."
                    )
                if status["status"] == "completed" and (
                    set(status["received_clients"]) != expected
                    or set(status["applied_clients"]) != expected
                ):
                    raise ValueError(
                        "A RSU declarou sucesso sem confirmar todos os participantes."
                    )
                for key in ("upload_bytes", "download_bytes"):
                    if type(status.get(key)) is not int or status[key] < 0:
                        raise ValueError(f"Contagem inválida em {key} no status da RSU.")
                duration = status.get("exchange_time")
                if (
                    isinstance(duration, bool)
                    or not isinstance(duration, (int, float))
                    or not math.isfinite(duration)
                    or duration < 0
                ):
                    raise ValueError("Duração inválida no status da RSU.")
                return status
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Sem status da RSU {rsu_id} para o encontro {encounter_id}.")
        time.sleep(min(0.1, remaining))


def wait_for_client_evaluations(
    encounters_dir: Path,
    clients: list[int],
    encounter_id: int,
    timeout: float,
    *,
    run_id: str | None = None,
    rsu_id: str | None = None,
) -> None:
    """APPLIED confirma os pesos; .done confirma avaliação e persistência das métricas."""
    deadline = time.monotonic() + timeout
    while True:
        if run_id is not None and rsu_id is not None:
            check_client_failures(encounters_dir, run_id, rsu_id, encounter_id, clients)
        if run_id is not None and rsu_id is not None:
            summary = summarize_encounter_metrics(
                encounters_dir,
                {
                    "run_id": run_id,
                    "rsu_id": rsu_id,
                    "encounter_id": encounter_id,
                    "clients": clients,
                    "type": "v2i",
                },
            )
            completed = {
                r["client_id"]
                for r in summary["client_metrics"]
                if r.get("status") == "completed" and r.get("applied")
            }
            missing = [client_id for client_id in clients if client_id not in completed]
        else:
            missing = [
                client_id
                for client_id in clients
                if not (encounters_dir / f".done_enc_{encounter_id}_client_{client_id}").exists()
            ]
        if not missing:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Avaliação do encontro {encounter_id} pendente: clientes {missing}."
            )
        time.sleep(min(0.1, remaining))


def run_orchestrator(args: argparse.Namespace) -> None:
    """Loop principal do orquestrador V2I."""
    if args.total_clients < 2:
        raise ValueError("--total-clients deve ser pelo menos 2 para encontros V2I.")
    exchange_timeout = getattr(args, "exchange_timeout", DEFAULT_EXCHANGE_TIMEOUT)
    evaluation_timeout = getattr(args, "evaluation_timeout", DEFAULT_EVALUATION_TIMEOUT)

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

    with open(rsu_config_path) as f:
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
        "-c",
        str(sumo_cfg),
        "--step-length",
        str(args.step_length),
        "--seed",
        str(getattr(args, "seed", 42)),
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
    log.info("  Serviço RSU       : %s:%d", DEFAULT_RSU_HOST, DEFAULT_RSU_PORT)
    log.info("  Erro CSI OTA      : %.2f", args.ota_epsilon)
    log.info("  Banda OTA         : %.0f Hz", args.ota_bandwidth_hz)
    for rsu in rsus:
        log.info(
            "  RSU %s: (%.0f, %.0f) raio=%.0fm antenas=%d",
            rsu["id"],
            rsu["x"],
            rsu["y"],
            rsu["radius"],
            rsu.get("n_antennas", 16),
        )
    log.info("=" * 60)

    # Cada execução preserva os resultados e sinais das anteriores.
    run_id, run_dir, encounters_dir = prepare_run(PROJECT_ROOT, "v2i", args)
    log.info("Resultados desta execução: %s", run_dir)

    # ── PASSO 2: Gerar Docker Compose e iniciar containers ───────────────
    client_indices = list(range(args.total_clients))

    compose_file = compose_generate(
        client_indices=client_indices,
        dataset=args.dataset,
        rounds=3,
        prune=args.prune,
        ala=args.ala,
        exp_name="default_exp_v2i",
        max_epochs=args.max_epochs,
        model=args.model,
        config=args.config,
        strategy=args.strategy,
        rank=args.rank,
        paca=args.paca,
        mode="v2i",
        results_dir=run_dir / "clients",
        encounters_dir=encounters_dir,
        seed=getattr(args, "seed", 42),
        run_id=run_id,
        rsu_config=rsu_config_path,
        exchange_timeout=exchange_timeout,
    )

    # Contadores e estado da simulação
    encounter_count = 0
    rejected_count = 0
    event_log: list[V2IEvent] = []
    veh_to_client_idx: dict[str, int] = {}
    next_client_idx = 0

    # Cooldown POR RSU (cada RSU tem seu próprio timer)
    last_encounter_per_rsu: dict[str, float] = {rsu["id"]: -float("inf") for rsu in rsus}

    # Controle de duplicação: evitar re-sinalizar o mesmo grupo na mesma RSU
    prev_clients_per_rsu: dict[str, frozenset[int]] = {}

    step = 0
    sumo_started = False
    try:
        if not docker_compose_up(compose_file, PROJECT_ROOT, build=args.build):
            raise RuntimeError("Falha ao iniciar os veículos e o serviço de RSUs.")
        if args.warmup > 0:
            log.info("Aguardando %ds de warmup...", args.warmup)
            time.sleep(args.warmup)
        all_client_indices = list(range(args.total_clients))
        if not wait_all_clients_epoch_done(encounters_dir, all_client_indices, 0, timeout=300):
            raise TimeoutError("Os clientes não concluíram o treinamento inicial.")
        log.info("Iniciando SUMO...")
        traci.start(sumo_cmd)
        sumo_started = True
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
                        vid,
                        veh_to_client_idx[vid],
                    )

                if vid in veh_to_client_idx:
                    x, y = traci.vehicle.getPosition(vid)
                    speed = traci.vehicle.getSpeed(vid)
                    edge = traci.vehicle.getRoadID(vid)
                    angle = traci.vehicle.getAngle(vid)
                    vehicles.append(
                        VehicleState(
                            veh_id=vid,
                            x=x,
                            y=y,
                            speed=speed,
                            edge=edge,
                            angle=angle,
                        )
                    )

            if not vehicles:
                prev_clients_per_rsu.clear()
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

                c_indices = sorted({veh_to_client_idx[v.veh_id] for v, _, _ in viable})
                if len(c_indices) < 2:
                    rejected_count += 1
                    prev_clients_per_rsu.pop(rsu_id, None)
                    continue

                # Observar saídas mesmo durante o cooldown para liberar um novo encontro.
                if (sim_time - last_encounter_per_rsu[rsu_id]) < args.cooldown:
                    continue

                # ── Verificar se é um NOVO grupo (não repetição) ──────────
                current_clients_key = frozenset(c_indices)

                if prev_clients_per_rsu.get(rsu_id) == current_clients_key:
                    continue

                # ── ETC suficiente: gerar encontro V2I ────────────────────
                encounter_count += 1
                min_etc = min(etc_val for _, _, etc_val in viable)

                veh_names = [v.veh_id for v, _, _ in viable]
                log.info(
                    "Encontro V2I %d/%d! RSU %s ← Veículos %s (Clientes %s) | ETC=%.1fs",
                    encounter_count,
                    args.encounters,
                    rsu_id,
                    veh_names,
                    c_indices,
                    min_etc,
                )

                # ── Montar sinal de encontro V2I ──────────────────────────
                encounter_data = {
                    "encounter_id": encounter_count,
                    "type": "v2i",
                    "rsu_id": rsu_id,
                    "clients": c_indices,
                    "timestamp": sim_time,
                    "etc_seconds": min_etc,
                    "positions": {veh_to_client_idx[v.veh_id]: (v.x, v.y) for v, _, _ in viable},
                    "rsu_position": (rsu["x"], rsu["y"]),
                    "rsu_n_antennas": rsu.get("n_antennas", 16),
                    "run_id": run_id,
                    "rsu_host": DEFAULT_RSU_HOST,
                    "rsu_port": DEFAULT_RSU_PORT,
                    "exchange_timeout": exchange_timeout,
                    "ota_epsilon": args.ota_epsilon,
                    "ota_bandwidth_hz": args.ota_bandwidth_hz,
                }

                # O relógio total inclui controle, protocolo e avaliação dos clientes.
                exchange_start = time.monotonic()
                # Escrita atômica do JSON
                write_encounter_signal(encounters_dir, encounter_data, encounter_count)

                # Registrar evento
                event_log.append(
                    V2IEvent(
                        sim_time=sim_time,
                        event_type="encounter_formed",
                        rsu_id=rsu_id,
                        vehicles=veh_names,
                        details={
                            "encounter_id": encounter_count,
                            "run_id": run_id,
                            "rsu_host": DEFAULT_RSU_HOST,
                            "rsu_port": DEFAULT_RSU_PORT,
                            "status": "pending",
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
                    )
                )

                # Atualizar estado
                prev_clients_per_rsu[rsu_id] = current_clients_key
                last_encounter_per_rsu[rsu_id] = sim_time

                # O host espera apenas controle e métricas; os tensores passam por TCP.
                log.info("SUMO PAUSADO. Aguardando status da RSU %s...", rsu_id)
                details = event_log[-1].details
                status = None
                try:
                    status = wait_for_rsu_status(
                        encounters_dir,
                        run_id=run_id,
                        rsu_id=rsu_id,
                        encounter_id=encounter_count,
                        clients=c_indices,
                        # Dar tempo para a RSU encerrar conexões e publicar o status.
                        timeout=exchange_timeout + 5.0,
                    )
                    details.update(
                        {
                            "rsu_status": status["status"],
                            "status": status["status"],
                            "received_clients": status["received_clients"],
                            "applied_clients": status["applied_clients"],
                            "upload_bytes": status["upload_bytes"],
                            "download_bytes": status["download_bytes"],
                            "protocol_time": status["exchange_time"],
                        }
                    )
                    if status["status"] != "completed":
                        raise RuntimeError(status.get("error") or "A RSU não concluiu o encontro.")
                    details["status"] = "evaluating"
                    evaluation_start = time.monotonic()
                    wait_for_client_evaluations(
                        encounters_dir,
                        c_indices,
                        encounter_count,
                        evaluation_timeout,
                        run_id=run_id,
                        rsu_id=rsu_id,
                    )
                    details["evaluation_wait_time"] = round(
                        time.monotonic() - evaluation_start,
                        6,
                    )
                    details["status"] = "completed"
                except TimeoutError as exc:
                    details["status"] = (
                        "evaluation_timed_out"
                        if details["status"] == "evaluating"
                        else "timed_out"
                    )
                    details["error"] = str(exc)
                    raise
                except (OSError, ValueError, RuntimeError) as exc:
                    if details["status"] != "timed_out":
                        details["status"] = "failed"
                    details["error"] = str(exc)
                    raise
                finally:
                    details["exchange_time"] = round(time.monotonic() - exchange_start, 6)
                    details.update(
                        summarize_encounter_metrics(encounters_dir, encounter_data, status)
                    )
                    details["total_time"] = details["exchange_time"]
                    if status is not None:
                        details["server_protocol_time"] = status["exchange_time"]

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
                if not wait_all_clients_epoch_done(
                    encounters_dir,
                    all_client_indices,
                    round_num=encounter_count,
                    timeout=300,
                ):
                    raise TimeoutError(f"Treinamento da rodada {encounter_count} incompleto.")

            # Verificar limite após iterar todas as RSUs
            if encounter_count >= args.encounters:
                break

    except KeyboardInterrupt:
        if event_log and event_log[-1].details["status"] in {"pending", "evaluating"}:
            event_log[-1].details.update(
                status="failed",
                error="Execução interrompida pelo usuário.",
            )
        log.info("\nSimulação interrompida pelo usuário.")
    finally:
        # ── PASSO 5: Limpeza geral ───────────────────────────────────────
        log.info("Encerrando simulação V2I...")
        done_file = encounters_dir / ".simulation_done"
        done_file.touch()
        time.sleep(1)
        try:
            docker_compose_down(compose_file, PROJECT_ROOT)
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.error("Falha ao encerrar os contêineres: %s", exc)
        if sumo_started:
            try:
                traci.close()
            except Exception as exc:
                log.error("Falha ao encerrar SUMO: %s", exc)

        # Estatísticas finais
        log.info("=" * 60)
        log.info("ESTATÍSTICAS DA SIMULAÇÃO V2I")
        log.info("=" * 60)
        log.info("  Encontros V2I viáveis  : %d", encounter_count)
        log.info("  Encontros rejeitados   : %d (menos de 2 veículos elegíveis)", rejected_count)
        log.info("  Total de steps SUMO    : %d", step)
        for rsu in rsus:
            rsu_id = rsu["id"]
            rsu_encounters = sum(
                1 for e in event_log if e.event_type == "encounter_formed" and e.rsu_id == rsu_id
            )
            log.info("  %s: %d encontros OTA (RSU stateless)", rsu_id, rsu_encounters)
        log.info("=" * 60)

        # Salvar log de eventos
        log_path = run_dir / "v2i_events.jsonl"
        with open(log_path, "w") as f:
            for event in event_log:
                f.write(
                    json.dumps(
                        {
                            "sim_time": event.sim_time,
                            "type": event.event_type,
                            "rsu_id": event.rsu_id,
                            "vehicles": event.vehicles,
                            **event.details,
                        },
                        default=str,
                    )
                    + "\n"
                )
        log.info("Log de eventos salvo em: %s", log_path)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="V2I FL Orchestrator — OTA com RSUs stateless (antenas fixas)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sumo-cfg",
        required=True,
        help="Caminho para o arquivo .sumocfg do SUMO",
    )
    parser.add_argument(
        "--rsu-config",
        required=True,
        help="Caminho para o rsu_config.json com posições das RSUs",
    )
    parser.add_argument(
        "--total-clients",
        type=int,
        default=5,
        help="Total de veículos (containers) a criar no Docker (default: 5)",
    )
    parser.add_argument(
        "--encounters",
        type=int,
        default=6,
        help="Número máximo de encontros V2I viáveis (default: 6)",
    )
    parser.add_argument(
        "--min-contact-time",
        type=float,
        default=DEFAULT_MIN_CONTACT_TIME,
        help=f"Tempo mínimo de contato (ETC) em segundos (default: {DEFAULT_MIN_CONTACT_TIME})",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
        choices=["MNIST", "Cifar10", "Cifar100"],
        help="Dataset para o treinamento (default: MNIST)",
    )
    parser.add_argument(
        "--prune",
        type=int,
        default=1,
        choices=[0, 1],
        help="Habilitar pruning adaptativo (0=sim, 1=não, default: 1)",
    )
    parser.add_argument(
        "--ala",
        type=int,
        default=1,
        choices=[0, 1],
        help="Habilitar FedALA (0=sim, 1=não/FedAvg, default: 1)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Usar sumo-gui (interface gráfica)",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Forçar o rebuild das imagens Docker",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Segundos de warmup (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=DEFAULT_COOLDOWN,
        help=f"Segundos de cooldown por RSU (default: {DEFAULT_COOLDOWN})",
    )
    parser.add_argument(
        "--step-length",
        type=float,
        default=DEFAULT_STEP_LENGTH,
        help=f"Duração de cada step do SUMO (default: {DEFAULT_STEP_LENGTH})",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=0,
        help="Limite de épocas por cliente (0 = sem limite, default: 0)",
    )
    parser.add_argument(
        "--run-id",
        help="Identificador da execução (padrão: UUID gerado automaticamente)",
    )
    parser.add_argument(
        "--exchange-timeout",
        type=float,
        default=DEFAULT_EXCHANGE_TIMEOUT,
        help="Prazo do protocolo TCP até APPLIED de todos os participantes, em segundos",
    )
    parser.add_argument(
        "--evaluation-timeout",
        type=float,
        default=DEFAULT_EVALUATION_TIMEOUT,
        help="Prazo adicional para avaliações e métricas após APPLIED, em segundos",
    )
    parser.add_argument(
        "--ota-epsilon",
        type=float,
        default=DEFAULT_OTA_EPSILON,
        help="Erro de estimação do canal OTA no intervalo [0, 1) (default: 0.1)",
    )
    parser.add_argument(
        "--ota-bandwidth-hz",
        type=float,
        default=DEFAULT_OTA_BANDWIDTH_HZ,
        help="Largura de banda usada no ruído térmico OTA em Hz (default: 100000)",
    )
    # ── Parâmetros do Modelo ──────────────────────────────────────────────
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["cnn", "clip"],
        help="Modelo a ser utilizado (default: cnn)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="lora_clip/train_config.yml",
        help="Caminho do config YAML para o CLIP",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="lora",
        choices=["lora", "adalora", "sora_with_schedule", "sora_no_schedule"],
        help="Estratégia de PEFT",
    )
    parser.add_argument("--rank", type=int, default=8, help="Rank para LoRA/SoRA")
    parser.add_argument("--paca", type=int, default=12, help="Número de camadas upper para PaCA")

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.ota_epsilon < 1:
        parser.error("--ota-epsilon deve estar no intervalo [0, 1).")
    if not math.isfinite(args.ota_bandwidth_hz) or args.ota_bandwidth_hz <= 0:
        parser.error("--ota-bandwidth-hz deve ser positivo e finito.")
    if args.run_id is not None and not args.run_id.strip():
        parser.error("--run-id deve ser uma string não vazia.")
    for name in ("exchange_timeout", "evaluation_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} deve ser positivo e finito.")
    try:
        run_orchestrator(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Erro na execução V2I: {exc}\n")


if __name__ == "__main__":
    main()
