# Responsabilidade: Avança a simulação TraCI e coordena encontros, sincronização, contêineres
# e registros de eventos.
#
# Principais funções: run_orchestrator.

"""SUMO federation — simulation_loop."""

import argparse
import json
import sys
import time
from pathlib import Path
from fl_serverpuma.simulation.sumo.compose import generate as compose_generate
from fl_serverpuma.simulation.sumo.mobility import VehicleState
from fl_serverpuma.simulation.sumo.mobility import V2VEvent
from fl_serverpuma.simulation.sumo.mobility import cluster_etc
from fl_serverpuma.simulation.sumo.mobility import find_clusters
from fl_serverpuma.simulation.sumo.signals import write_encounter_signal
from fl_serverpuma.simulation.sumo.signals import wait_all_clients_epoch_done
from fl_serverpuma.simulation.sumo.containers import docker_compose_up
from fl_serverpuma.simulation.sumo.containers import docker_compose_down
from fl_serverpuma.simulation.sumo.settings import PROJECT_ROOT
from fl_serverpuma.simulation.sumo.settings import log

def run_orchestrator(args: argparse.Namespace) -> None:
    """Loop principal do orquestrador V2V.

    Fluxo:
      1. Limpa pasta de encontros e pesos anteriores
      2. Gera docker-compose.v2v.yml e inicia conteineres
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
    for f in encounters_dir.glob(".done_enc_*"):
        f.unlink()
    # Limpar arquivos temporarios orfaos de escritas atomicas interrompidas
    for f in encounters_dir.glob(".encounter_*.tmp"):
        f.unlink()

    # ── PASSO 2: Gerar Docker Compose e iniciar conteineres ──────────────
    client_indices = list(range(args.total_clients))

    compose_file = compose_generate(
        client_indices=client_indices,
        dataset=args.dataset,
        rounds=3,
        prune=args.prune,
        ala=args.ala,
        exp_name='default_exp_v2v',
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
    # Esperar os conteineres Docker inicializarem completamente.
    if args.warmup > 0:
        log.info(
            "Aguardando %ds de warmup (conteineres inicializando)...",
            args.warmup,
        )
        time.sleep(args.warmup)
        log.info("Warmup concluido!")

    # ── PASSO 3.5: Iniciar Rodada 0 (Treino Inicial) ─────────────────────
    # Limpar qualquer arquivo antigo
    for f in encounters_dir.glob(".start_round_*"): f.unlink()
    for f in encounters_dir.glob(".epoch_done_*"): f.unlink()
    for f in encounters_dir.glob(".simulation_done"): f.unlink()

    all_client_indices = list(range(args.total_clients))
    log.info("--- Iniciando Treino (Rodada 0) ---")

    wait_all_clients_epoch_done(encounters_dir, all_client_indices, round_num=0, timeout=300, log=log)

    # ── PASSO 4: Iniciar simulacao SUMO ──────────────────────────────────
    log.info("Iniciando SUMO...")
    traci.start(sumo_cmd)

    # Contadores e estado da simulacao
    encounter_count = 0                            # Encontros viaveis gerados
    rejected_count = 0                             # Encontros rejeitados por ETC insuficiente
    event_log: list[V2VEvent] = []                 # Log de todos os eventos V2V
    prev_cluster_key: frozenset[str] | None = None # Chave do cluster anterior (para detectar mudanca)
    pair_encounter_counts: dict[frozenset[str], int] = {} # Contador de encontros por cluster
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

                    # ── Verificar Limite de Encontros por Par ─────────────
                    if args.max_encounters_per_pair > 0:
                        count = pair_encounter_counts.get(cluster_key, 0)
                        if count >= args.max_encounters_per_pair:
                            # Ignorar este cluster, pois ja se encontraram o maximo permitido
                            prev_cluster_key = cluster_key
                            continue

                    # ── Calcular ETC do cluster ───────────────────────────
                    etc = cluster_etc(best_cluster, args.radius)

                    # ── Filtro de viabilidade baseado no ETC ──────────────
                    if not isinstance(etc, str) and etc < args.min_contact_time:
                        # ETC insuficiente: ignorar este cluster
                        rejected_count += 1
                        log.info(
                            "Cluster %s (Clientes %s): ETC=%.1fs < minimo (%.1fs). "
                            "Encontro REJEITADO (%d rejeitados ate agora).",
                            veh_names, c_indices, etc,
                            args.min_contact_time, rejected_count,
                        )

                        # Registrar evento de rejeicao para analise
                        event_log.append(V2VEvent(
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
                    pair_encounter_counts[cluster_key] = pair_encounter_counts.get(cluster_key, 0) + 1

                    etc_display = f"{etc}" if isinstance(etc, str) else f"{etc:.1f}s"
                    log.info(
                        "Encontro %d/%d! Veiculos %s (Clientes %s) | ETC=%s | (Vezes: %d)",
                        encounter_count, args.encounters,
                        veh_names, c_indices, etc_display,
                        pair_encounter_counts[cluster_key],
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
                    event_log.append(V2VEvent(
                        sim_time=sim_time,
                        event_type="encounter_formed",
                        vehicles=veh_names,
                        details={
                            "encounter_id": encounter_count,
                            "clients": c_indices,
                            "etc_seconds": etc if isinstance(etc, str) else round(etc, 1),
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
                    expected_done_files = [
                        encounters_dir / f".done_enc_{encounter_count}_client_{c}"
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
                        received = [f.exists() for f in expected_done_files]
                        received_count = sum(received)

                        # Log de progresso a cada 15 segundos
                        if tick > 0 and tick % 15 == 0:
                            log.info(
                                "  ... %d/%d confirmacoes recebidas (%ds)",
                                received_count, len(expected_done_files), tick,
                            )

                        if all(received):
                            exchange_time = time.time() - exchange_start
                            log.info(
                                "Todas as %d confirmacoes recebidas em %.0fs! "
                                "Aguardando 5s para retomada...",
                                len(expected_done_files), exchange_time,
                            )
                            event_log[-1].details["exchange_time"] = round(exchange_time, 1)
                            # Dar tempo extra para garantia
                            time.sleep(5)
                            break
                        time.sleep(1)
                    else:
                        exchange_time = time.time() - exchange_start
                        received_count = sum(f.exists() for f in expected_done_files)
                        log.warning(
                            "Timeout de %ds esperando confirmacoes do encontro %d "
                            "(%d/%d recebidas). Continuando...",
                            exchange_timeout, encounter_count,
                            received_count, len(expected_done_files),
                        )
                        event_log[-1].details["exchange_time"] = round(exchange_time, 1)

                    log.info("SUMO RETOMADO.")

                    # ── PASSO 3: Iniciar Nova Rodada ──────────────────────
                    # Agora que o encontro terminou e o SUMO retomou, sinalizamos
                    # para TODOS os clientes treinarem mais 1 época
                    log.info(f"--- Iniciando Treino (Rodada {encounter_count}) ---")
                    all_client_indices = list(range(args.total_clients))
                    wait_all_clients_epoch_done(encounters_dir, all_client_indices, round_num=encounter_count, timeout=300, log=log)

                    # Atualizar tempo do ultimo encontro (para cooldown)
                    last_encounter_time = sim_time
                    prev_cluster_key = cluster_key

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
        log.info("Encerrando simulacao V2V...")
        done_file = encounters_dir / ".simulation_done"
        done_file.touch()
        time.sleep(5)  # Dar tempo para os clientes lerem o sinal
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
        log_path = PROJECT_ROOT / "src" / "results" / "v2v_events.jsonl"
        with open(log_path, "w") as f:
            for event in event_log:
                f.write(json.dumps({
                    "sim_time": event.sim_time,
                    "type": event.event_type,
                    "vehicles": event.vehicles,
                    **event.details,
                }, default=str) + "\n")
        log.info("Log de eventos salvo em: %s", log_path)
