#!/usr/bin/env python3
"""
sumo_docker_orchestrator.py
============================
Orquestrador V2X para Treinamento Federado.

Metodologia baseada no MobFedLS:
  1. Inicia TODOS os conteineres dos clientes (veiculos) no inicio.
  2. A cada step do SUMO, calcula o Tempo Estimado de Contato (ETC)
     entre veiculos proximos usando posicao, velocidade e angulo via TraCI.
  3. Somente gera um "Sinal de Encontro" (arquivo JSON atomico) se o ETC
     for superior ao tempo minimo de contato configurado (--min-contact-time).
  4. Os clientes leem os JSONs e fazem a troca descentralizada de pesos (P2P)
     via arquivos .pt no volume compartilhado Docker.
  5. A simulacao encerra apos um numero maximo de encontros viaveis.
"""

import argparse
import itertools
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
from sumo_adapter.generate_v2x_compose import generate as compose_generate

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("v2x_orchestrator")

# ── Constantes ────────────────────────────────────────────────────────────────

# Raio de comunicacao DSRC/C-V2X padrao (metros)
DEFAULT_COMM_RADIUS = 300.0

# Minimo de veiculos para formar um cluster valido
DEFAULT_MIN_CLIENTS = 2

# Duracao de cada step de simulacao do SUMO (segundos)
DEFAULT_STEP_LENGTH = 1.0

# Tempo minimo de contato (ETC) em segundos para que um encontro
# seja considerado viavel para troca de pesos P2P.
# Baseado no calculo: modelo MNIST CNN ~2.3MB, a 7Mbit/s (DSRC 802.11p)
# leva ~2.8s para transferir.
DEFAULT_MIN_CONTACT_TIME = 6.0

# Teto maximo para o ETC em segundos. Valores acima sao truncados.
# Evita ETCs irrealistas (ex: veiculos quase parados na mesma posicao).
MAX_ETC = 120.0

# Segundos de warmup antes de comecar a sinalizar encontros.
# Permite que os conteineres Docker iniciem, carreguem o modelo e
# completem pelo menos 1 epoca de treino local.
DEFAULT_WARMUP = 60

# Segundos de cooldown entre encontros consecutivos.
# Impede que mudancas incrementais na composicao do cluster
# (entra/sai 1 veiculo) consumam o contador de encontros.
DEFAULT_COOLDOWN = 30

# Raiz do projeto (FL_ServerPuma/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── Estruturas de Dados ──────────────────────────────────────────────────────

@dataclass
class VehicleState:
    """Estado instantaneo de um veiculo no SUMO.
    
    Atributos:
        veh_id: Identificador unico do veiculo no SUMO (ex: "veh_0")
        x: Coordenada X em metros (sistema de coordenadas do SUMO)
        y: Coordenada Y em metros
        speed: Velocidade instantanea em m/s
        edge: Identificador da via/aresta onde o veiculo esta
        angle: Angulo de direcao em graus (0=norte, sentido horario)
    """
    veh_id: str
    x: float
    y: float
    speed: float
    edge: str
    angle: float = 0.0


@dataclass
class V2XEvent:
    """Evento de log da simulacao V2X para analise posterior.
    
    Atributos:
        sim_time: Tempo de simulacao do SUMO em segundos
        event_type: Tipo do evento (ex: "encounter_formed", "encounter_rejected")
        vehicles: Lista de IDs dos veiculos envolvidos
        details: Dicionario com detalhes extras (clients, etc_seconds, etc)
    """
    sim_time: float
    event_type: str
    vehicles: list[str]
    details: dict = field(default_factory=dict)


# ── Funcoes de Geometria e Cinematica ────────────────────────────────────────

def euclidean_distance(v1: VehicleState, v2: VehicleState) -> float:
    """Calcula a distancia euclidiana entre dois veiculos em metros."""
    return math.sqrt((v1.x - v2.x) ** 2 + (v1.y - v2.y) ** 2)


def estimate_contact_time(
    v1: VehicleState,
    v2: VehicleState,
    radius: float,
) -> float:
    """Estima o Tempo de Contato restante (ETC) entre dois veiculos.
    
    Usa uma equacao quadratica de cinematica que modela as posicoes
    futuras como funcoes lineares do tempo (movimento retilineo uniforme):
    
      p1(t) = p1 + v1 * t
      p2(t) = p2 + v2 * t
    
    O ETC e o menor t > 0 tal que |p1(t) - p2(t)| = R.
    
    Expandindo:
      |dv|^2 * t^2 + 2*(dp . dv) * t + (|dp|^2 - R^2) = 0
    
    Onde:
      dp = posicao relativa (v2 - v1)
      dv = velocidade relativa (v2 - v1)
      A = |dv|^2, B = 2*(dp . dv), C = |dp|^2 - R^2
    
    Como |dp| < R (estao dentro do raio), C < 0.
    Com A > 0, o discriminante B^2 - 4AC > 0 sempre.
    A raiz positiva (-B + sqrt(D)) / 2A e o tempo ate saida.
    
    Vantagem sobre a formula linear: funciona corretamente tanto
    quando os veiculos se aproximam quanto quando se afastam.
    Nunca retorna infinito.
    
    Retorna:
      float: ETC em segundos, truncado em MAX_ETC.
             0.0 se ja estao fora do raio.
    
    Nota sobre o angulo do SUMO:
      Convencao nautica: 0 graus = Norte, sentido horario.
      vx = speed * sin(angulo)  (componente Leste)
      vy = speed * cos(angulo)  (componente Norte)
    """
    # Se ja estao fora do raio, tempo de contato e zero
    dist = euclidean_distance(v1, v2)
    if dist >= radius:
        return 0.0

    # Decompor velocidade de cada veiculo em componentes X (leste) e Y (norte)
    # usando o angulo do SUMO (convencao nautica: 0=Norte, sentido horario)
    a1_rad = math.radians(v1.angle)
    a2_rad = math.radians(v2.angle)

    v1_vx = v1.speed * math.sin(a1_rad)  # Componente Leste do veiculo 1
    v1_vy = v1.speed * math.cos(a1_rad)  # Componente Norte do veiculo 1
    v2_vx = v2.speed * math.sin(a2_rad)  # Componente Leste do veiculo 2
    v2_vy = v2.speed * math.cos(a2_rad)  # Componente Norte do veiculo 2

    # Posicao relativa (dp) e velocidade relativa (dv)
    dp_x = v2.x - v1.x
    dp_y = v2.y - v1.y
    dv_x = v2_vx - v1_vx
    dv_y = v2_vy - v1_vy

    # Coeficientes da equacao quadratica: A*t^2 + B*t + C = 0
    A = dv_x * dv_x + dv_y * dv_y         # |dv|^2
    B = 2.0 * (dp_x * dv_x + dp_y * dv_y) # 2 * (dp . dv)
    C = dp_x * dp_x + dp_y * dp_y - radius * radius  # |dp|^2 - R^2

    # Caso especial: A ≈ 0 significa que a velocidade relativa e
    # praticamente zero (veiculos parados ou com mesma velocidade).
    # Neste caso, a distancia nao muda e o ETC e o maximo possivel.
    if A < 1e-9:
        return MAX_ETC

    # Discriminante: sempre positivo quando C < 0 e A > 0
    discriminante = B * B - 4.0 * A * C

    if discriminante < 0:
        # Matematicamente impossivel quando estao dentro do raio,
        # mas tratamos por seguranca.
        return MAX_ETC

    # Raiz positiva: tempo ate a distancia atingir R
    # (-B + sqrt(D)) / 2A e a raiz positiva porque C < 0
    sqrt_d = math.sqrt(discriminante)
    t_exit = (-B + sqrt_d) / (2.0 * A)

    # Se t_exit <= 0, ja estao saindo (raro, mas possivel com
    # veiculos exatamente na borda do raio)
    if t_exit <= 0:
        return 0.0

    # Truncar no teto maximo para evitar valores irrealistas
    return min(t_exit, MAX_ETC)


def cluster_etc(
    cluster: list[VehicleState],
    radius: float,
) -> float:
    """Calcula o ETC de um cluster inteiro.
    
    O ETC do cluster e o MINIMO entre todos os pares de veiculos,
    pois basta um par se separar para quebrar a conectividade.
    
    Args:
        cluster: Lista de estados dos veiculos no cluster
        radius: Raio de comunicacao em metros
    
    Returns:
        float: ETC minimo do cluster em segundos
    """
    min_etc = float('inf')
    for v1, v2 in itertools.combinations(cluster, 2):
        etc = estimate_contact_time(v1, v2, radius)
        min_etc = min(min_etc, etc)
    return min_etc


def find_clusters(
    vehicles: list[VehicleState],
    radius: float,
    min_size: int,
) -> list[list[VehicleState]]:
    """Encontra clusters de veiculos onde TODOS os membros estao dentro
    do raio de comunicacao uns dos outros (clique completa).
    
    Usa uma heuristica gulosa: comeca pelo veiculo com mais vizinhos
    e expande a clique adicionando candidatos que sao adjacentes a
    todos os membros existentes.
    
    Args:
        vehicles: Lista de estados dos veiculos ativos na simulacao
        radius: Raio de comunicacao V2X em metros
        min_size: Tamanho minimo do cluster para ser considerado valido
    
    Returns:
        Lista de clusters, cada um sendo uma lista de VehicleState
    """
    if len(vehicles) < min_size:
        return []

    # Construir grafo de adjacencia: cada veiculo mapeia para o conjunto
    # de veiculos que estao dentro do raio de comunicacao
    adj: dict[str, set[str]] = {v.veh_id: set() for v in vehicles}
    veh_map = {v.veh_id: v for v in vehicles}

    for v1, v2 in itertools.combinations(vehicles, 2):
        if euclidean_distance(v1, v2) <= radius:
            adj[v1.veh_id].add(v2.veh_id)
            adj[v2.veh_id].add(v1.veh_id)

    # Heuristica gulosa para encontrar cliques
    used = set()
    clusters = []

    # Ordenar por grau decrescente (veiculos mais conectados primeiro)
    sorted_vehs = sorted(
        vehicles,
        key=lambda v: len(adj[v.veh_id]),
        reverse=True,
    )

    for seed in sorted_vehs:
        if seed.veh_id in used:
            continue
        if len(adj[seed.veh_id]) < min_size - 1:
            continue

        # Expandir clique a partir do seed
        clique = {seed.veh_id}
        candidates = adj[seed.veh_id] - used

        for candidate_id in sorted(candidates, key=lambda c: len(adj[c]), reverse=True):
            # O candidato so entra se for adjacente a TODOS os membros atuais
            if all(candidate_id in adj[member] for member in clique):
                clique.add(candidate_id)

        if len(clique) >= min_size:
            cluster = [veh_map[vid] for vid in clique]
            clusters.append(cluster)
            used.update(clique)

    return clusters


# ── Escrita Atomica de Sinais de Encontro ────────────────────────────────────

def write_encounter_signal(
    encounters_dir: Path,
    encounter_data: dict,
    encounter_id: int,
) -> Path:
    """Escreve o sinal de encontro (JSON) de forma ATOMICA no volume compartilhado.
    
    Usa a tecnica write-to-temp + rename para garantir que os clientes
    nunca leiam um arquivo parcialmente escrito:
      1. Cria um arquivo temporario no MESMO diretorio (mesmo filesystem)
      2. Escreve o JSON completo no temporario
      3. Faz fsync para garantir que os dados estao no disco
      4. Renomeia atomicamente para o nome final (operacao atomica no Linux)
    
    Args:
        encounters_dir: Diretorio do volume compartilhado (src/results/encounters/)
        encounter_data: Dicionario com os dados do encontro
        encounter_id: ID numerico do encontro (usado no nome do arquivo)
    
    Returns:
        Path do arquivo final criado
    """
    final_path = encounters_dir / f"encounter_{encounter_id}.json"

    # Criar arquivo temporario no MESMO diretorio para garantir
    # que o rename seja atomico (mesmo filesystem)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(encounters_dir),
        prefix=".encounter_",
        suffix=".tmp",
    )
    try:
        # Escrever JSON completo no temporario
        with os.fdopen(fd, "w") as tmp_file:
            json.dump(encounter_data, tmp_file)
            # Flush do buffer Python para o buffer do SO
            tmp_file.flush()
            # Fsync para garantir que os dados foram gravados no disco
            os.fsync(tmp_file.fileno())

        # Rename atomico: os clientes nunca verao o arquivo parcial
        os.rename(tmp_path, final_path)

    except Exception:
        # Em caso de erro, limpar o arquivo temporario
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return final_path


# ── Controle Docker ──────────────────────────────────────────────────────────

def docker_compose_up(compose_file: str, project_root: Path, build: bool = False) -> bool:
    """Inicia todos os conteineres (veiculos) de uma vez no inicio da simulacao.
    
    Args:
        compose_file: Caminho para o docker-compose.v2x.yml
        project_root: Raiz do projeto (FL_ServerPuma/)
        build: Se True, forca o rebuild das imagens (adiciona --build)
    
    Returns:
        True se o comando foi bem sucedido, False caso contrario
    """
    cmd = ["docker", "compose", "-f", compose_file, "up", "-d"]
    if build:
        cmd.insert(-1, "--build")
        log.info("Iniciando conteineres Docker...")
    else:
        log.info("Iniciando conteineres Docker...")
        
    result = subprocess.run(
        cmd,
        cwd=str(project_root),
    )
    if result.returncode != 0:
        log.error("docker compose up falhou (exit code %d)", result.returncode)
        return False
    log.info("Conteineres iniciados com sucesso!")
    return True


def docker_compose_down(compose_file: str, project_root: Path) -> None:
    """Destroi os conteineres ao final da simulacao.
    
    Usa --remove-orphans para limpar conteineres de execucoes anteriores
    e -v para remover volumes anonimos.
    """
    log.info("Encerrando conteineres Docker...")
    subprocess.run(
        ["docker", "compose", "-f", compose_file, "down", "--remove-orphans", "-v"],
        cwd=str(project_root),
        capture_output=True, text=True, timeout=120,
    )
    log.info("Conteineres removidos.")


# ── Loop Principal ───────────────────────────────────────────────────────────

def run_orchestrator(args: argparse.Namespace) -> None:
    """Loop principal do orquestrador V2X.
    
    Fluxo:
      1. Limpa pasta de encontros e pesos anteriores
      2. Gera docker-compose.v2x.yml e inicia conteineres
      3. Inicia SUMO e conecta via TraCI
      4. A cada step do SUMO:
         a) Coleta posicao, velocidade e angulo de cada veiculo
         b) Detecta clusters (veiculos dentro do raio de comunicacao)
         c) Calcula o ETC (Estimated Contact Time) do cluster
         d) Se ETC >= min_contact_time: gera sinal de encontro (JSON atomico)
         e) Se ETC insuficiente: ignora e loga o motivo
      5. Encerra quando o numero maximo de encontros viaveis e atingido
      6. Limpa conteineres e salva logs
    """
    # Importar TraCI (dependencia do SUMO, nao esta no pyproject.toml)
    try:
        import traci  # pyright: ignore [reportMissingImports]
    except ImportError:
        log.error(
            "O pacote 'traci' nao foi encontrado. "
            "Instale o SUMO (sumo-tools) ou via: pip install traci"
        )
        sys.exit(1)

    # Validar arquivo de configuracao do SUMO
    sumo_cfg = Path(args.sumo_cfg).resolve()
    if not sumo_cfg.exists():
        log.error("Arquivo SUMO config nao encontrado: %s", sumo_cfg)
        sys.exit(1)

    # Montar comando do SUMO (com ou sem interface grafica)
    sumo_cmd = [
        "sumo-gui" if args.gui else "sumo",
        "-c", str(sumo_cfg),
        "--step-length", str(args.step_length),
    ]

    # ── Banner de configuracao ────────────────────────────────────────────
    log.info("=" * 60)
    log.info("V2X Orchestrator - FL")
    log.info("=" * 60)
    log.info("  Total Clientes    : %d", args.total_clients)
    log.info("  Max Encontros     : %d", args.encounters)
    log.info("  Raio Comunicacao  : %.0f m", args.radius)
    log.info("  Min Tempo Contato : %.0f s (ETC)", args.min_contact_time)
    log.info("  Warmup            : %d s", args.warmup)
    log.info("  Cooldown          : %d s", args.cooldown)
    log.info("  Dataset           : %s", args.dataset)
    log.info("=" * 60)

    # ── PASSO 1: Limpar e criar pasta de sinalizacao P2P ─────────────────
    encounters_dir = PROJECT_ROOT / "src" / "results" / "encounters"
    encounters_dir.mkdir(parents=True, exist_ok=True)

    # Limpar encontros e pesos de execucoes anteriores
    for f in encounters_dir.glob("*.json"):
        f.unlink()
    for f in encounters_dir.glob("*.pt"):
        f.unlink()
    # Limpar arquivos temporarios orfaos de escritas atomicas interrompidas
    for f in encounters_dir.glob(".encounter_*.tmp"):
        f.unlink()

    # ── PASSO 2: Gerar Docker Compose e iniciar conteineres ──────────────
    client_indices = list(range(args.total_clients))
    compose_file = compose_generate(
        client_indices=client_indices,
        dataset=args.dataset,
        rounds=3,  # No modo descentralizado, o cliente roda em loop infinito
        prune=args.prune,
        ala=args.ala,
        exp_name='default_exp_v2x',
        output_path=PROJECT_ROOT,
    )

    if not docker_compose_up(compose_file, PROJECT_ROOT, build=args.build):
        sys.exit(1)

    # ── PASSO 3: Warmup ──────────────────────────────────────────────────
    # Esperar os conteineres Docker inicializarem completamente antes de
    # iniciar o SUMO. Isso garante que os clientes ja estejam treinando
    # (com pesos prontos para exportar) quando o primeiro encontro chegar.
    if args.warmup > 0:
        log.info(
            "Aguardando %ds de warmup (conteineres inicializando, "
            "clientes treinando primeira epoca)...",
            args.warmup,
        )
        time.sleep(args.warmup)
        log.info("Warmup concluido!")

    # ── PASSO 4: Iniciar simulacao SUMO ──────────────────────────────────
    log.info("Iniciando SUMO...")
    traci.start(sumo_cmd)

    # Contadores e estado da simulacao
    encounter_count = 0                            # Encontros viaveis gerados
    rejected_count = 0                             # Encontros rejeitados por ETC insuficiente
    event_log: list[V2XEvent] = []                 # Log de todos os eventos V2X
    prev_cluster_key: frozenset[str] | None = None # Chave do cluster anterior (para detectar mudanca)
    veh_to_client_idx: dict[str, int] = {}         # Mapa veiculo SUMO -> indice do conteiner
    next_client_idx = 0                            # Proximo indice de conteiner disponivel
    last_encounter_time = -float('inf')            # Timestamp SUMO do ultimo encontro (para cooldown)

    try:
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            # Avancar um step na simulacao
            traci.simulationStep()
            sim_time = traci.simulation.getTime()
            step += 1

            # ── Coletar estado dos veiculos ───────────────────────────────
            veh_ids = traci.vehicle.getIDList()
            vehicles = []

            for vid in veh_ids:
                # Vincular cada veiculo novo a um indice de conteiner
                if vid not in veh_to_client_idx and next_client_idx < args.total_clients:
                    veh_to_client_idx[vid] = next_client_idx
                    next_client_idx += 1
                    log.info(
                        "Veiculo SUMO '%s' vinculado ao conteiner cliente %d",
                        vid, veh_to_client_idx[vid],
                    )

                # So processar veiculos que tem conteiner associado
                if vid in veh_to_client_idx:
                    x, y = traci.vehicle.getPosition(vid)
                    speed = traci.vehicle.getSpeed(vid)
                    edge = traci.vehicle.getRoadID(vid)
                    # O angulo do SUMO: 0=Norte, sentido horario
                    angle = traci.vehicle.getAngle(vid)

                    vehicles.append(VehicleState(
                        veh_id=vid,
                        x=x, y=y,
                        speed=speed,
                        edge=edge,
                        angle=angle,
                    ))

            # Precisamos de pelo menos 2 veiculos para formar um cluster
            if len(vehicles) < 2:
                continue

            # ── Verificar cooldown ────────────────────────────────────────
            # Se estamos dentro do periodo de cooldown apos o ultimo
            # encontro, pular a deteccao para dar tempo a troca de pesos
            # e evitar que mudancas incrementais no cluster consumam o
            # contador de encontros.
            if (sim_time - last_encounter_time) < args.cooldown:
                continue

            # ── Detectar clusters ─────────────────────────────────────────
            clusters = find_clusters(vehicles, args.radius, args.min_clients)

            if clusters:
                # Pegar o maior cluster encontrado
                best_cluster = max(clusters, key=len)
                cluster_key = frozenset(v.veh_id for v in best_cluster)

                # Verificar se e um NOVO agrupamento (diferente do anterior)
                if cluster_key != prev_cluster_key:
                    veh_names = [v.veh_id for v in best_cluster]
                    c_indices = [veh_to_client_idx[v] for v in veh_names]

                    # ── Calcular ETC do cluster ───────────────────────────
                    etc = cluster_etc(best_cluster, args.radius)

                    # ── Filtro de viabilidade baseado no ETC ──────────────
                    if etc < args.min_contact_time:
                        # ETC insuficiente: ignorar este cluster
                        rejected_count += 1
                        log.info(
                            "Cluster %s (Clientes %s): ETC=%.1fs < minimo (%.1fs). "
                            "Encontro REJEITADO (%d rejeitados ate agora).",
                            veh_names, c_indices, etc,
                            args.min_contact_time, rejected_count,
                        )

                        # Registrar evento de rejeicao para analise
                        event_log.append(V2XEvent(
                            sim_time=sim_time,
                            event_type="encounter_rejected",
                            vehicles=veh_names,
                            details={
                                "clients": c_indices,
                                "etc_seconds": round(etc, 1),
                                "min_required": args.min_contact_time,
                            },
                        ))

                        # Atualizar chave para nao reprocessar o mesmo cluster
                        prev_cluster_key = cluster_key
                        continue

                    # ── ETC suficiente: gerar sinal de encontro ───────────
                    encounter_count += 1

                    log.info(
                        "Encontro %d/%d! Veiculos %s (Clientes %s) | ETC=%.1fs",
                        encounter_count, args.encounters,
                        veh_names, c_indices, etc,
                    )

                    # Montar dados do encontro com ETC para os clientes usarem
                    encounter_data = {
                        "encounter_id": encounter_count,
                        "clients": c_indices,
                        "timestamp": sim_time,
                        # O ETC permite que os clientes ajustem seu timeout
                        # de polling pela janela de contato real
                        "etc_seconds": etc,
                    }

                    # Escrita ATOMICA do JSON (write-to-temp + rename)
                    write_encounter_signal(encounters_dir, encounter_data, encounter_count)

                    # Registrar evento para log de analise
                    event_log.append(V2XEvent(
                        sim_time=sim_time,
                        event_type="encounter_formed",
                        vehicles=veh_names,
                        details={
                            "clients": c_indices,
                            "etc_seconds": round(etc, 1),
                        },
                    ))

                    # Atualizar estado
                    prev_cluster_key = cluster_key
                    last_encounter_time = sim_time  # Iniciar cooldown

                    # ── PAUSAR SUMO e esperar troca P2P ───────────────────
                    # O SUMO roda em tempo de simulacao (300 steps em segundos),
                    # mas os clientes precisam de tempo real (minutos) para:
                    #   1. Completar a epoca de treino local atual
                    #   2. Detectar o JSON do encontro
                    #   3. Salvar seus pesos (.pt)
                    #   4. Carregar pesos dos vizinhos
                    #   5. Agregar (media)
                    #
                    # Sem esta pausa, o SUMO avanca centenas de steps e termina
                    # antes dos clientes sequer lerem o JSON.
                    expected_pt_files = [
                        encounters_dir / f"client_{c}_enc_{encounter_count}.pt"
                        for c in c_indices
                    ]
                    log.info(
                        "SUMO PAUSADO. Aguardando %d clientes completarem "
                        "troca P2P do encontro %d...",
                        len(c_indices), encounter_count,
                    )

                    exchange_start = time.time()
                    exchange_timeout = 300  # Maximo 5 minutos por encontro

                    for tick in range(exchange_timeout):
                        received = [f.exists() for f in expected_pt_files]
                        received_count = sum(received)

                        # Log de progresso a cada 15 segundos
                        if tick > 0 and tick % 15 == 0:
                            log.info(
                                "  ... %d/%d .pt recebidos (%ds)",
                                received_count, len(expected_pt_files), tick,
                            )

                        if all(received):
                            exchange_time = time.time() - exchange_start
                            log.info(
                                "Todos os %d .pt recebidos em %.0fs! "
                                "Aguardando 5s para agregacao...",
                                len(expected_pt_files), exchange_time,
                            )
                            # Dar tempo extra para os clientes lerem os .pt
                            # dos vizinhos e completarem a agregacao D-PSGD
                            time.sleep(5)
                            break
                        time.sleep(1)
                    else:
                        exchange_time = time.time() - exchange_start
                        received_count = sum(f.exists() for f in expected_pt_files)
                        log.warning(
                            "Timeout de %ds esperando .pt do encontro %d "
                            "(%d/%d recebidos). Continuando...",
                            exchange_timeout, encounter_count,
                            received_count, len(expected_pt_files),
                        )

                    log.info("SUMO RETOMADO.")

                    # Verificar se atingimos o limite de encontros
                    if encounter_count >= args.encounters:
                        log.info(
                            "Limite de %d encontros atingido. "
                            "Encerrando simulacao.",
                            args.encounters,
                        )
                        break
            else:
                # Nenhum cluster detectado neste step
                if prev_cluster_key is not None:
                    log.info("Encontro finalizado (veiculos se afastaram).")
                    prev_cluster_key = None

    except KeyboardInterrupt:
        log.info("\nSimulacao interrompida pelo usuario.")
    finally:
        # ── PASSO 4: Limpeza geral ───────────────────────────────────────
        docker_compose_down(compose_file, PROJECT_ROOT)
        traci.close()
        log.info("SUMO encerrado.")

        # Exibir estatisticas finais
        log.info("=" * 60)
        log.info("ESTATISTICAS DA SIMULACAO")
        log.info("=" * 60)
        log.info("  Encontros viaveis  : %d", encounter_count)
        log.info("  Encontros rejeitados: %d (ETC insuficiente)", rejected_count)
        log.info("  Total de steps SUMO: %d", step)
        log.info("=" * 60)

        # Salvar log de eventos em formato JSONL para analise posterior
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


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="V2X Opportunistic FL Orchestrator (ETC + D-PSGD)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sumo-cfg", required=True,
        help="Caminho para o arquivo .sumocfg do SUMO",
    )
    parser.add_argument(
        "--total-clients", type=int, default=5,
        help="Total de veiculos (conteineres) a criar no Docker (default: 5)",
    )
    parser.add_argument(
        "--encounters", type=int, default=4,
        help="Numero maximo de encontros VIAVEIS para terminar a simulacao (default: 4)",
    )
    parser.add_argument(
        "--radius", type=float, default=DEFAULT_COMM_RADIUS,
        help=f"Raio de comunicacao V2X em metros (default: {DEFAULT_COMM_RADIUS})",
    )
    parser.add_argument(
        "--min-clients", type=int, default=DEFAULT_MIN_CLIENTS,
        help=f"Numero minimo de clientes para um encontro (default: {DEFAULT_MIN_CLIENTS})",
    )
    parser.add_argument(
        "--min-contact-time", type=float, default=DEFAULT_MIN_CONTACT_TIME,
        help=(
            f"Tempo minimo de contato (ETC) em segundos para disparar um encontro P2P. "
            f"Encontros com ETC abaixo deste valor sao ignorados (default: {DEFAULT_MIN_CONTACT_TIME})"
        ),
    )
    parser.add_argument(
        "--dataset", type=str, default="MNIST",
        choices=["MNIST", "Cifar10", "Cifar100"],
        help="Dataset para o treinamento (default: MNIST)",
    )
    parser.add_argument(
        "--prune", type=int, default=1, choices=[0, 1],
        help="Habilitar pruning adaptativo (0=sim, 1=nao, default: 1)",
    )
    parser.add_argument(
        "--ala", type=int, default=1, choices=[0, 1],
        help="Habilitar FedALA (0=sim, 1=nao/FedAvg, default: 1)",
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Usar sumo-gui (interface grafica) em vez do sumo headless",
    )
    parser.add_argument(
        "--build", action="store_true",
        help="Forcar o rebuild das imagens Docker antes de iniciar a simulacao",
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP,
        help=(
            f"Segundos de warmup antes de sinalizar encontros. "
            f"Permite os conteineres bootarem e completarem 1 epoca "
            f"de treino local (default: {DEFAULT_WARMUP})"
        ),
    )
    parser.add_argument(
        "--cooldown", type=int, default=DEFAULT_COOLDOWN,
        help=(
            f"Segundos de espera entre encontros consecutivos. "
            f"Evita que mudancas incrementais no cluster consumam "
            f"o contador de encontros (default: {DEFAULT_COOLDOWN})"
        ),
    )
    parser.add_argument(
        "--step-length", type=float, default=DEFAULT_STEP_LENGTH,
        help=f"Duracao de cada step do SUMO em segundos (default: {DEFAULT_STEP_LENGTH})",
    )

    args = parser.parse_args()
    run_orchestrator(args)


if __name__ == "__main__":
    main()
