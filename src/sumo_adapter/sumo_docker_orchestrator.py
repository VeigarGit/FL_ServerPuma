#!/usr/bin/env python3
"""
sumo_docker_orchestrator.py
============================
Orquestrador V2X que integra o SUMO (mobilidade) com Docker (execução do FL).

Utiliza o TraCI para ler posições
de veículos e disparar sessões de treinamento federado On-Demand nos
contêineres Docker já existentes do FL_ServerPuma.

Modo de Rede: Estático (docker-tc aplica latência/perda fixa nos contêineres).

Fluxo:
  1. Inicia o SUMO via TraCI.
  2. A cada step, lê as coordenadas (x, y) de todos os veículos.
  3. Calcula a matriz de distância entre todos os pares.
  4. Se um grupo de veículos está dentro do RAIO_COMUNICACAO, e o número
     mínimo de clientes é atingido, dispara o docker compose up dos
     contêineres envolvidos.
  5. Se os veículos se afastam (saem do raio), dispara docker compose down.

Uso:
  python3 sumo_docker_orchestrator.py --sumo-cfg maps/grid.sumocfg --clients 5
  python3 sumo_docker_orchestrator.py --sumo-cfg maps/grid.sumocfg --clients 3 --radius 300 --gui
"""

import argparse
import itertools
import json
import logging
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from sumo_adapter.generate_v2x_compose import compose

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("v2x_orchestrator")

# ── Constantes ────────────────────────────────────────────────────────────────

DEFAULT_COMM_RADIUS = 300.0        # metros — alcance V2X (DSRC / C-V2X)
DEFAULT_MIN_CLIENTS = 2            # mínimo de clientes para iniciar uma rodada FL
DEFAULT_STEP_LENGTH = 1.0          # segundos por step do SUMO
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # FL_ServerPuma/


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class VehicleState:
    """Estado instantâneo de um veículo no SUMO."""
    veh_id: str
    x: float
    y: float
    speed: float
    edge: str


@dataclass
class FLSession:
    """Representa uma sessão de treinamento federado ativa."""
    session_id: str
    server_veh: str           # veículo que atua como servidor
    client_vehs: list[str]    # veículos que atuam como clientes
    client_indices: list[int] # índices dos clientes no docker-compose
    compose_file: str
    process: subprocess.Popen | None = None
    started_at: float = 0.0
    finished: bool = False


@dataclass 
class V2XEvent:
    """Evento de log da simulação V2X."""
    sim_time: float
    event_type: str   # "cluster_formed", "cluster_lost", "session_start", "session_end"
    vehicles: list[str]
    details: dict = field(default_factory=dict)


# ── Funções de Geometria ──────────────────────────────────────────────────────

def euclidean_distance(v1: VehicleState, v2: VehicleState) -> float:
    """Distância euclidiana 2D entre dois veículos."""
    return math.sqrt((v1.x - v2.x) ** 2 + (v1.y - v2.y) ** 2)


def find_clusters(
    vehicles: list[VehicleState],
    radius: float,
    min_size: int,
) -> list[list[VehicleState]]:
    """
    Encontra clusters de veículos onde TODOS os membros estão dentro
    do raio de comunicação uns dos outros (clique completa).
    
    Retorna apenas clusters com pelo menos `min_size` membros.
    Usa uma heurística gulosa: começar pelo veículo com mais vizinhos
    e expandir a clique.
    """
    if len(vehicles) < min_size:
        return []

    # Construir grafo de adjacência
    adj: dict[str, set[str]] = {v.veh_id: set() for v in vehicles}
    veh_map = {v.veh_id: v for v in vehicles}

    for v1, v2 in itertools.combinations(vehicles, 2):
        if euclidean_distance(v1, v2) <= radius:
            adj[v1.veh_id].add(v2.veh_id)
            adj[v2.veh_id].add(v1.veh_id)

    # Heurística gulosa para encontrar cliques
    used = set()
    clusters = []

    # Ordenar por grau decrescente (veículos mais conectados primeiro)
    sorted_vehs = sorted(vehicles, key=lambda v: len(adj[v.veh_id]), reverse=True)

    for seed in sorted_vehs:
        if seed.veh_id in used:
            continue
        if len(adj[seed.veh_id]) < min_size - 1:
            continue

        # Tentar expandir clique a partir do seed
        clique = {seed.veh_id}
        candidates = adj[seed.veh_id] - used

        for candidate_id in sorted(candidates, key=lambda c: len(adj[c]), reverse=True):
            # Verificar se o candidato é adjacente a TODOS os membros da clique
            if all(candidate_id in adj[member] for member in clique):
                clique.add(candidate_id)

        if len(clique) >= min_size:
            cluster = [veh_map[vid] for vid in clique]
            clusters.append(cluster)
            used.update(clique)

    return clusters


# ── Controle Docker ───────────────────────────────────────────────────────────

DOCKER_POLL_INTERVAL = 5  # segundos entre cada verificação de status

def docker_compose_up(compose_file: str, project_root: Path) -> bool:
    """
    Inicia os contêineres SINCRONAMENTE: builda, sobe, e só retorna quando
    todos os contêineres estiverem efetivamente rodando.
    Retorna True se o build+start teve sucesso.
    """
    log.info("docker compose up --build -d (aguardando build + start...)")
    result = subprocess.run(
        ["docker", "compose", "-f", compose_file, "up", "--build", "-d"],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        timeout=600,  # 10 min timeout para build
    )
    if result.returncode != 0:
        log.error("docker compose up falhou:\n%s", result.stderr[-500:])
        return False
    log.info("Contêineres buildados e iniciados com sucesso.")
    return True


def docker_compose_down(compose_file: str, project_root: Path) -> None:
    """Para e remove os contêineres."""
    log.info("docker compose down (%s)", compose_file)
    subprocess.run(
        ["docker", "compose", "-f", compose_file, "down", "--remove-orphans"],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        timeout=60,
    )


def get_compose_status(compose_file: str, project_root: Path) -> tuple[int, int]:
    """
    Retorna (running_count, total_count) dos contêineres do compose.
    """
    try:
        result = subprocess.run(
            ["docker", "compose", "-f", compose_file, "ps", "--format", "json", "-a"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return (0, 0)
        output = result.stdout.strip()
        if not output:
            return (0, 0)
        running = 0
        total = 0
        for line in output.splitlines():
            try:
                container = json.loads(line)
                total += 1
                state = container.get("State", "")
                if state == "running":
                    running += 1
            except json.JSONDecodeError:
                continue
        return (running, total)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return (0, 0)


def wait_for_training(compose_file: str, project_root: Path, session_id: str) -> str:
    """
    Bloqueia até que TODOS os contêineres do compose parem (treinamento terminou).
    Retorna 'completed' se todos pararam normalmente, ou 'error' se houve falha.
    """
    log.info("Aguardando treinamento FL terminar (sessão '%s')...", session_id)
    
    # Primeiro, esperar os contêineres aparecerem como 'running'
    for attempt in range(60):  # até 5 min esperando iniciar
        running, total = get_compose_status(compose_file, project_root)
        if running > 0:
            log.info("Contêineres ativos: %d/%d", running, total)
            break
        time.sleep(DOCKER_POLL_INTERVAL)
    else:
        log.error("Timeout: contêineres nunca iniciaram.")
        return "error"
    
    # Agora, esperar todos os contêineres pararem
    poll_count = 0
    while True:
        time.sleep(DOCKER_POLL_INTERVAL)
        running, total = get_compose_status(compose_file, project_root)
        poll_count += 1
        
        if poll_count % 12 == 0:  # a cada ~60s
            log.info("Status FL [%s]: %d/%d contêineres rodando", session_id, running, total)
        
        if total > 0 and running == 0:
            log.info("Todos os contêineres pararam — treinamento concluído!")
            return "completed"
        
        if total == 0:
            # Contêineres sumiram (alguém fez docker compose down externamente?)
            log.warning("Contêineres não encontrados.")
            return "error"

# ── Loop Principal ────────────────────────────────────────────────────────────

def run_orchestrator(args: argparse.Namespace) -> None:
    """Loop principal: SUMO step → proximidade → Docker on/off."""

    # ── Import TraCI (disponível apenas se SUMO estiver instalado) ─────────
    try:
        # pyrefly: ignore [missing-import]
        import traci
    except ImportError:
        log.error(
            "O pacote 'traci' não foi encontrado.\n"
            "   Instale o SUMO: sudo apt install sumo sumo-tools\n"
            "   Ou: pip install traci"
        )
        sys.exit(1)

    sumo_cfg = Path(args.sumo_cfg).resolve()
    if not sumo_cfg.exists():
        log.error("Arquivo de configuração SUMO não encontrado: %s", sumo_cfg)
        sys.exit(1)

    # Escolher binário do SUMO
    sumo_binary = "sumo-gui" if args.gui else "sumo"
    sumo_cmd = [sumo_binary, "-c", str(sumo_cfg), "--step-length", str(args.step_length)]

    log.info("═" * 60)
    log.info("V2X Federated Learning Orchestrator")
    log.info("═" * 60)
    log.info("  SUMO config : %s", sumo_cfg)
    log.info("  Comm Radius : %.0f m", args.radius)
    log.info("  Min Clients : %d", args.min_clients)
    log.info("  Dataset     : %s", args.dataset)
    log.info("  Rounds/sess : %d", args.rounds)
    log.info("  GUI         : %s", args.gui)
    log.info("═" * 60)

    # ── Estado do orquestrador ─────────────────────────────────────────────
    active_session: FLSession | None = None
    session_counter = 0
    event_log: list[V2XEvent] = []
    prev_cluster_key: frozenset[str] | None = None

    # Mapeamento veículo → índice de cliente
    # Os índices são atribuídos na ordem em que os veículos aparecem
    veh_to_client_idx: dict[str, int] = {}
    next_client_idx = 0

    # ── Iniciar SUMO ──────────────────────────────────────────────────────
    log.info("Iniciando SUMO...")
    traci.start(sumo_cmd)
    log.info("SUMO iniciado com sucesso")

    try:
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            sim_time = traci.simulation.getTime()
            step += 1

            # ── Coletar posições dos veículos ──────────────────────────────
            veh_ids = traci.vehicle.getIDList()
            vehicles: list[VehicleState] = []

            for vid in veh_ids:
                x, y = traci.vehicle.getPosition(vid)
                speed = traci.vehicle.getSpeed(vid)
                edge = traci.vehicle.getRoadID(vid)
                vehicles.append(VehicleState(vid, x, y, speed, edge))

                # Atribuir índice de cliente se é novo
                if vid not in veh_to_client_idx:
                    veh_to_client_idx[vid] = next_client_idx
                    next_client_idx += 1
                    log.info("Veículo %s entrou → client_idx=%d", vid, veh_to_client_idx[vid])

            if len(vehicles) < 2:
                if step % 10 == 0:
                    log.debug("t=%.1f: %d veículo(s) ativo(s), aguardando...", sim_time, len(vehicles))
                continue

            # ── Encontrar clusters V2X ─────────────────────────────────────
            clusters = find_clusters(vehicles, args.radius, args.min_clients)

            if clusters:
                # Usar o maior cluster encontrado
                best_cluster = max(clusters, key=len)
                cluster_key = frozenset(v.veh_id for v in best_cluster)

                # Logar mudanças no cluster
                if cluster_key != prev_cluster_key:
                    veh_names = [v.veh_id for v in best_cluster]
                    log.info(
                        "t=%.1f: Cluster V2X formado! %d veículos: %s",
                        sim_time, len(best_cluster), veh_names,
                    )
                    event_log.append(V2XEvent(
                        sim_time=sim_time,
                        event_type="cluster_formed",
                        vehicles=veh_names,
                        details={"radius": args.radius},
                    ))
                    prev_cluster_key = cluster_key

                # ── Se não há sessão ativa, iniciar uma ───────────────────
                if active_session is None:
                    session_counter += 1
                    session_id = f"v2x_session_{session_counter}"
                    
                    client_indices = [veh_to_client_idx[v.veh_id] for v in best_cluster]
                    server_veh = best_cluster[0].veh_id
                    client_vehs = [v.veh_id for v in best_cluster]

                    log.info(
                        "t=%.1f: Iniciando sessão FL '%s' (server=%s, clients=%s)",
                        sim_time, session_id, server_veh, client_vehs,
                    )

                    compose_file = compose(
                        client_indices=client_indices,
                        dataset=args.dataset,
                        rounds=args.rounds,
                        prune=args.prune,
                        ala=args.ala,
                        output_path=PROJECT_ROOT,
                    )

                    event_log.append(V2XEvent(
                        sim_time=sim_time,
                        event_type="session_start",
                        vehicles=client_vehs,
                        details={"session_id": session_id, "client_indices": client_indices},
                    ))

                    # ── BLOQUEANTE: build + start + aguardar treinamento ───
                    start_wall = time.time()
                    success = docker_compose_up(compose_file, PROJECT_ROOT)
                    
                    if success:
                        # Aguardar até todos os contêineres pararem (= treinamento concluído)
                        result = wait_for_training(compose_file, PROJECT_ROOT, session_id)
                        wall_duration = time.time() - start_wall
                        
                        log.info(
                            "Sessão '%s' finalizada (%s). Duração real: %.1fs",
                            session_id, result, wall_duration,
                        )
                        event_log.append(V2XEvent(
                            sim_time=sim_time,
                            event_type="session_end",
                            vehicles=client_vehs,
                            details={
                                "session_id": session_id,
                                "reason": result,
                                "wall_duration_s": round(wall_duration, 1),
                            },
                        ))
                    else:
                        log.error("Falha ao iniciar sessão '%s'.", session_id)
                        event_log.append(V2XEvent(
                            sim_time=sim_time,
                            event_type="session_end",
                            vehicles=client_vehs,
                            details={"session_id": session_id, "reason": "build_failed"},
                        ))
                    
                    # Limpar contêineres após cada sessão
                    docker_compose_down(compose_file, PROJECT_ROOT)
                    
                    # Resetar estado para permitir próxima sessão
                    active_session = None
                    prev_cluster_key = None
                    log.info("SUMO retomado — buscando próximo cluster V2X...")

            else:
                # Sem clusters válidos
                if prev_cluster_key is not None:
                    log.info("t=%.1f: Cluster V2X desfeito — veículos fora do raio.", sim_time)
                    event_log.append(V2XEvent(
                        sim_time=sim_time,
                        event_type="cluster_lost",
                        vehicles=list(prev_cluster_key),
                    ))
                    prev_cluster_key = None

            # ── Log periódico ──────────────────────────────────────────────
            if step % 30 == 0:
                log.info(
                    "📊 t=%.1f: %d veículos | %d cluster(s)",
                    sim_time, len(vehicles), len(clusters),
                )

    except KeyboardInterrupt:
        log.info("\nSimulação interrompida pelo usuário.")
    finally:
        # ── Cleanup ────────────────────────────────────────────────────────
        if active_session and not active_session.finished:
            log.info("Encerrando sessão ativa '%s'...", active_session.session_id)
            docker_compose_down(active_session.compose_file, PROJECT_ROOT)

        traci.close()
        log.info("SUMO encerrado.")

        # ── Salvar log de eventos ──────────────────────────────────────────
        log_path = PROJECT_ROOT /"results" / "v2x_events.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            for event in event_log:
                f.write(json.dumps({
                    "sim_time": event.sim_time,
                    "type": event.event_type,
                    "vehicles": event.vehicles,
                    **event.details,
                }, default=str) + "\n")
        log.info("Log de eventos salvo em: %s (%d eventos)", log_path, len(event_log))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="V2X Federated Learning Orchestrator — SUMO + Docker (sem OMNeT++)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sumo-cfg", required=True,
        help="Caminho para o arquivo .sumocfg do SUMO",
    )
    parser.add_argument(
        "--radius", type=float, default=DEFAULT_COMM_RADIUS,
        help=f"Raio de comunicação V2X em metros (default: {DEFAULT_COMM_RADIUS})",
    )
    parser.add_argument(
        "--min-clients", type=int, default=DEFAULT_MIN_CLIENTS,
        help=f"Número mínimo de clientes para iniciar uma sessão FL (default: {DEFAULT_MIN_CLIENTS})",
    )
    parser.add_argument(
        "--dataset", type=str, default="MNIST",
        choices=["MNIST", "Cifar10", "Cifar100"],
        help="Dataset para o treinamento (default: MNIST)",
    )
    parser.add_argument(
        "--rounds", type=int, default=3,
        help="Número de rodadas FL por sessão (default: 3)",
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
        help="Usar sumo-gui (interface gráfica) em vez do sumo headless",
    )
    parser.add_argument(
        "--step-length", type=float, default=DEFAULT_STEP_LENGTH,
        help=f"Duração de cada step do SUMO em segundos (default: {DEFAULT_STEP_LENGTH})",
    )

    args = parser.parse_args()
    run_orchestrator(args)


if __name__ == "__main__":
    main()
