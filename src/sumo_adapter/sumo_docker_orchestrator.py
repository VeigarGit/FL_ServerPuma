#!/usr/bin/env python3
"""
sumo_docker_orchestrator.py
============================
Orquestrador V2X para Treinamento Federado Oportunista (D-PSGD).

Inicia TODOS os contêineres dos clientes (veículos) no início.
Quando o SUMO detecta que veículos estão próximos (cluster), o orquestrador
cria um "Sinal de Encontro" (arquivo JSON) que os clientes leem para realizar
a troca descentralizada de pesos (P2P).
A simulação encerra após um número máximo de encontros estipulado.
"""

import argparse
import itertools
import json
import logging
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from sumo_adapter.generate_v2x_compose import generate as compose_generate

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("v2x_orchestrator")

# ── Constantes ────────────────────────────────────────────────────────────────
DEFAULT_COMM_RADIUS = 300.0        # metros — alcance V2X (DSRC / C-V2X)
DEFAULT_MIN_CLIENTS = 2            # mínimo de veículos para um encontro
DEFAULT_STEP_LENGTH = 1.0          # segundos por step do SUMO
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # FL_ServerPuma/

@dataclass
class VehicleState:
    """Estado instantâneo de um veículo no SUMO."""
    veh_id: str
    x: float
    y: float
    speed: float
    edge: str

@dataclass 
class V2XEvent:
    """Evento de log da simulação V2X."""
    sim_time: float
    event_type: str
    vehicles: list[str]
    details: dict = field(default_factory=dict)

# ── Funções de Geometria ──────────────────────────────────────────────────────
def euclidean_distance(v1: VehicleState, v2: VehicleState) -> float:
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
def docker_compose_up(compose_file: str, project_root: Path) -> bool:
    """Inicia todos os contêineres (veículos) de uma vez no início da simulação."""
    log.info("Iniciando contêineres Docker (aguarde o build)...")
    result = subprocess.run(
        ["docker", "compose", "-f", compose_file, "up", "--build", "-d"],
        cwd=str(project_root),
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        log.error("docker compose up falhou:\n%s", result.stderr[-500:])
        return False
    log.info("Contêineres iniciados com sucesso!")
    return True

def docker_compose_down(compose_file: str, project_root: Path) -> None:
    """Destrói os contêineres ao final da simulação."""
    log.info("Encerrando contêineres Docker...")
    subprocess.run(
        ["docker", "compose", "-f", compose_file, "down", "--remove-orphans", "-v"],
        cwd=str(project_root),
        capture_output=True, text=True, timeout=60,
    )
    log.info("Contêineres removidos.")

# ── Loop Principal ────────────────────────────────────────────────────────────
def run_orchestrator(args: argparse.Namespace) -> None:
    try:
        # pyrefly: ignore [missing-import]
        import traci
    except ImportError:
        log.error("O pacote 'traci' não foi encontrado. Instale o SUMO (sumo-tools) ou via pip install traci.")
        sys.exit(1)

    sumo_cfg = Path(args.sumo_cfg).resolve()
    if not sumo_cfg.exists():
        log.error("Arquivo SUMO config não encontrado: %s", sumo_cfg)
        sys.exit(1)

    sumo_cmd = ["sumo-gui" if args.gui else "sumo", "-c", str(sumo_cfg), "--step-length", str(args.step_length)]

    log.info("═" * 60)
    log.info("V2X Orchestrator - Treinamento Federado Oportunista (D-PSGD)")
    log.info("═" * 60)
    log.info("  Total Clientes: %d", args.total_clients)
    log.info("  Max Encontros : %d", args.encounters)
    log.info("  Comm Radius   : %.0f m", args.radius)
    log.info("  Dataset       : %s", args.dataset)
    log.info("═" * 60)

    # 1. Limpar e criar pasta de "Encontros" (sinalizador P2P)
    encounters_dir = PROJECT_ROOT / "src" / "results" / "encounters"
    encounters_dir.mkdir(parents=True, exist_ok=True)
    # Limpa encontros passados
    for f in encounters_dir.glob("*.json"):
        f.unlink()

    # 2. Gerar arquivo Docker Compose e Iniciar Contêineres
    client_indices = list(range(args.total_clients))
    compose_file = compose_generate(
        client_indices=client_indices,
        dataset=args.dataset,
        rounds=3, # Repassado, mas no modo descentralizado o cliente roda um loop infinito
        prune=args.prune,
        ala=args.ala,
        output_path=PROJECT_ROOT
    )
    
    if not docker_compose_up(compose_file, PROJECT_ROOT):
        sys.exit(1)

    # 3. Iniciar Simulação SUMO
    log.info("Iniciando SUMO...")
    traci.start(sumo_cmd)

    encounter_count = 0
    event_log: list[V2XEvent] = []
    prev_cluster_key: frozenset[str] | None = None
    veh_to_client_idx: dict[str, int] = {}
    next_client_idx = 0

    try:
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            sim_time = traci.simulation.getTime()
            step += 1

            # Mapear veículos
            veh_ids = traci.vehicle.getIDList()
            vehicles = []
            for vid in veh_ids:
                if vid not in veh_to_client_idx and next_client_idx < args.total_clients:
                    veh_to_client_idx[vid] = next_client_idx
                    next_client_idx += 1
                    log.info("Veículo SUMO %s vinculado ao container cliente %d", vid, veh_to_client_idx[vid])
                
                # Só processar veículos que estão vinculados a contêineres
                if vid in veh_to_client_idx:
                    x, y = traci.vehicle.getPosition(vid)
                    speed = traci.vehicle.getSpeed(vid)
                    edge = traci.vehicle.getRoadID(vid)
                    vehicles.append(VehicleState(vid, x, y, speed, edge))

            if len(vehicles) < 2:
                continue

            # Buscar clusters
            clusters = find_clusters(vehicles, args.radius, args.min_clients)

            if clusters:
                best_cluster = max(clusters, key=len)
                cluster_key = frozenset(v.veh_id for v in best_cluster)

                # Se for um NOVO agrupamento
                if cluster_key != prev_cluster_key:
                    encounter_count += 1
                    veh_names = [v.veh_id for v in best_cluster]
                    c_indices = [veh_to_client_idx[v] for v in veh_names]
                    
                    log.info("Encontro %d/%d! Veículos %s (Contêineres %s) estão próximos.", 
                             encounter_count, args.encounters, veh_names, c_indices)

                    # --- SINALIZAÇÃO P2P ---
                    # Escreve um arquivo JSON no volume compartilhado para os clientes lerem
                    encounter_file = encounters_dir / f"encounter_{encounter_count}.json"
                    encounter_data = {
                        "encounter_id": encounter_count,
                        "clients": c_indices,
                        "timestamp": sim_time
                    }
                    with open(encounter_file, "w") as f:
                        json.dump(encounter_data, f)
                    
                    event_log.append(V2XEvent(
                        sim_time=sim_time,
                        event_type="encounter_formed",
                        vehicles=veh_names,
                        details={"clients": c_indices}
                    ))
                    
                    prev_cluster_key = cluster_key

                    # Checar limite de encontros
                    if encounter_count >= args.encounters:
                        log.info("Limite de %d encontros atingido. Encerrando simulação...", args.encounters)
                        # Dá um tempo para os clientes lerem o último sinal e processarem a agregação
                        time.sleep(10)
                        break
            else:
                if prev_cluster_key is not None:
                    log.info("Encontro finalizado (veículos se afastaram).")
                    prev_cluster_key = None

    except KeyboardInterrupt:
        log.info("\nSimulação interrompida pelo usuário.")
    finally:
        # 4. Limpeza Geral
        docker_compose_down(compose_file, PROJECT_ROOT)
        traci.close()
        log.info("SUMO encerrado.")

        # Salvar logs
        log_path = PROJECT_ROOT / "src" / "results" / "v2x_events.jsonl"
        with open(log_path, "w") as f:
            for event in event_log:
                f.write(json.dumps({
                    "sim_time": event.sim_time,
                    "type": event.event_type,
                    "vehicles": event.vehicles,
                    **event.details,
                }, default=str) + "\n")
        log.info("Log de eventos salvo em: %s", log_path)

# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="V2X Opportunistic FL Orchestrator (Descentralizado)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sumo-cfg", required=True,
        help="Caminho para o arquivo .sumocfg do SUMO",
    )
    parser.add_argument(
        "--total-clients", type=int, default=5,
        help="Total de veículos (contêineres) a criar no Docker",
    )
    parser.add_argument(
        "--encounters", type=int, default=4,
        help="Número máximo de encontros para terminar a simulação",
    )
    parser.add_argument(
        "--radius", type=float, default=DEFAULT_COMM_RADIUS,
        help=f"Raio de comunicação V2X em metros (default: {DEFAULT_COMM_RADIUS})",
    )
    parser.add_argument(
        "--min-clients", type=int, default=DEFAULT_MIN_CLIENTS,
        help=f"Número mínimo de clientes para um encontro (default: {DEFAULT_MIN_CLIENTS})",
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
