#!/usr/bin/env python3
"""
createTrips.py
==============
Gera rotas longas e aleatorias para um numero fixo de veiculos em uma rede SUMO.

Diferente do randomTrips.py do SUMO (que cria muitos veiculos curtos),
este script cria poucos veiculos com rotas longas, garantindo que
eles permanecam no mapa durante toda a simulacao. Isso e ideal para
cenarios de Federated Learning V2V onde cada veiculo esta vinculado
a um container Docker fixo.

Funcionalidades:
  - Analisa o .net.xml para extrair o grafo de conectividade real
  - Filtra arestas por permissao de classe de veiculo (vclass)
  - Calcula duracao da rota por length/speed real (nao por contagem de arestas)
  - Backtracking em becos sem saida para evitar rotas truncadas
  - Filtra micro-arestas (< --min-start-length) como ponto de partida
  - Estima o tempo de simulacao necessario com base nos encontros desejados
  - Gera rotas por random walk no grafo, com vies para evitar retornos
  - Calcula cores distintas para visualizacao no sumo-gui
  - Atualiza automaticamente o .sumocfg com o tempo de simulacao calculado
  - Permite escalonar os departs para que os veiculos entrem gradualmente
  - Funciona com qualquer mapa SUMO (sintetico, OSM, LuST, etc.)

Uso (executar de dentro da pasta maps/):
  uv run python createTrips.py --net-file grid.net.xml \\
      --total-vehicles 5 --encounters 5

  uv run python createTrips.py --net-file manhattan.net.xml \\
      --total-vehicles 10 --total-steps 1000 --depart-spread 15

  uv run python createTrips.py --net-file osm_city.net.xml \\
      --total-vehicles 8 --vclass passenger --min-start-length 20
"""

import argparse
import math
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


# ── Constantes ────────────────────────────────────────────────────────────────

# Velocidade padrao (m/s) quando a aresta nao tem atributo 'speed'.
# 13.89 m/s ≈ 50 km/h (padrao urbano do SUMO).
DEFAULT_SPEED = 13.89

# Penalidade de tempo (segundos) adicionada por aresta para modelar
# desaceleracao, parada em semaforos/cruzamentos e reaceleracao.
INTERSECTION_PENALTY = 5.0

# Cooldown padrao entre encontros do orquestrador (deve bater com o
# argumento --cooldown do sumo_docker_orchestrator.py)
DEFAULT_COOLDOWN = 5

# Margem de seguranca adicionada ao tempo total para evitar que os
# veiculos saiam do mapa antes da simulacao acabar.
SAFETY_MARGIN = 1.3

# Comprimento minimo padrao (metros) para uma aresta ser usada como
# ponto de partida. Garante que o veiculo (~5m) caiba e tenha espaco
# para acelerar. Micro-arestas continuam no grafo para travessia.
DEFAULT_MIN_START_LENGTH = 15.0

# Numero maximo de tentativas de backtracking por rota antes de
# reiniciar de outra aresta inicial.
MAX_BACKTRACK_ATTEMPTS = 50

# Numero maximo de tentativas de gerar uma rota valida para um veiculo
# (cada tentativa usa uma aresta inicial diferente).
MAX_ROUTE_RETRIES = 10

# Cores para os veiculos no sumo-gui, otimizadas para distinguir
# visualmente ate 20 veiculos.
VEHICLE_COLORS = [
    "1,0,0",       # Vermelho
    "0,0.8,0",     # Verde
    "0,0,1",       # Azul
    "1,1,0",       # Amarelo
    "1,0,1",       # Magenta
    "0,1,1",       # Ciano
    "1,0.5,0",     # Laranja
    "0.5,0,1",     # Roxo
    "0,0.5,0",     # Verde escuro
    "1,0.5,0.5",   # Salmao
]


# ── Parsing da Rede SUMO ─────────────────────────────────────────────────────

def parse_network(
    net_file: Path,
    vclass: str = "passenger",
) -> tuple[dict[str, list[str]], dict[str, float], dict[str, float]]:
    """Analisa o arquivo .net.xml do SUMO e extrai o grafo de conectividade.

    Filtra arestas por permissao de classe de veiculo: apenas arestas
    que possuem pelo menos uma lane que permita a vclass especificada
    sao incluidas no grafo. Isso impede que rotas passem por ciclovias,
    calcadas, ferrovias ou outras vias restritas.

    Args:
        net_file: Caminho para o arquivo .net.xml
        vclass: Classe de veiculo para filtrar permissoes (default: "passenger")

    Returns:
        adjacency: dicionario {edge_id: [edges alcancaveis]}
        edge_lengths: dicionario {edge_id: comprimento em metros}
        edge_speeds: dicionario {edge_id: velocidade maxima em m/s}
    """
    tree = ET.parse(net_file)
    root = tree.getroot()

    # Extrair comprimentos e velocidades das arestas (ignorando internas :...)
    # Filtrando por permissoes de vclass nas lanes
    edge_lengths: dict[str, float] = {}
    edge_speeds: dict[str, float] = {}
    skipped_vclass = 0

    for edge in root.findall(".//edge"):
        eid = edge.get("id", "")
        if eid.startswith(":"):
            continue

        # Verificar se PELO MENOS uma lane permite a vclass
        has_permitted_lane = False
        best_length = 0.0
        best_speed = DEFAULT_SPEED

        for lane in edge.findall("lane"):
            allow = lane.get("allow")
            disallow = lane.get("disallow")

            lane_permitted = False
            if allow is not None:
                # Lista explicita de classes permitidas
                lane_permitted = vclass in allow.split()
            elif disallow is not None:
                # Lista explicita de classes proibidas
                lane_permitted = vclass not in disallow.split()
            else:
                # Sem restricoes = todas as classes permitidas
                lane_permitted = True

            if lane_permitted:
                has_permitted_lane = True
                lane_length = float(lane.get("length", "0"))
                lane_speed = float(lane.get("speed", str(DEFAULT_SPEED)))
                # Usar a lane mais longa (normalmente sao iguais por aresta)
                if lane_length > best_length:
                    best_length = lane_length
                    best_speed = lane_speed

        if not has_permitted_lane:
            skipped_vclass += 1
            continue

        if best_length > 0:
            edge_lengths[eid] = best_length
            edge_speeds[eid] = best_speed

    if skipped_vclass > 0:
        print(f"  {skipped_vclass} arestas ignoradas (sem permissao para '{vclass}')")

    # Extrair conectividade a partir das <connection>
    # Apenas incluir conexoes entre arestas que passaram no filtro
    adjacency: dict[str, list[str]] = {eid: [] for eid in edge_lengths}
    for conn in root.findall(".//connection"):
        fr = conn.get("from", "")
        to = conn.get("to", "")
        if fr.startswith(":") or to.startswith(":"):
            continue
        if fr in adjacency and to in adjacency and to not in adjacency[fr]:
            adjacency[fr].append(to)

    return adjacency, edge_lengths, edge_speeds


# ── Geracao de Rotas ─────────────────────────────────────────────────────────

def generate_route(
    adjacency: dict[str, list[str]],
    edge_lengths: dict[str, float],
    edge_speeds: dict[str, float],
    start_edge: str,
    target_duration: float,
    rng: random.Random,
) -> tuple[list[str], float]:
    """Gera uma rota por random walk no grafo da rede.

    Usa um vies para evitar retornos imediatos (U-turns), tornando
    as rotas mais realistas e espacialmente distribuidas.

    O comprimento da rota e determinado pelo tempo acumulado real
    (length / speed + penalidade de intersecao), e nao por contagem
    fixa de arestas. Isso garante rotas adequadas tanto para redes
    com arestas longas (rodovias) quanto curtas (cruzamentos OSM).

    Se a rota atinge um beco sem saida, faz backtracking e tenta
    outro caminho ate esgotar as opcoes.

    Args:
        adjacency: Grafo de conectividade da rede
        edge_lengths: Comprimento de cada aresta em metros
        edge_speeds: Velocidade maxima de cada aresta em m/s
        start_edge: Aresta inicial da rota
        target_duration: Duracao alvo da rota em segundos
        rng: Gerador de numeros aleatorios (para reprodutibilidade)

    Returns:
        Tupla (lista de edge IDs formando a rota, tempo acumulado estimado)
    """
    route = [start_edge]
    current = start_edge
    prev = None
    accumulated_time = _edge_traverse_time(start_edge, edge_lengths, edge_speeds)

    # Historico de tentativas por posicao para backtracking inteligente
    # Mapeia (posicao_na_rota, aresta_atual) -> set de vizinhos ja tentados
    tried: dict[tuple[int, str], set[str]] = {}
    backtrack_count = 0

    while accumulated_time < target_duration:
        neighbors = adjacency.get(current, [])

        if not neighbors:
            # Beco sem saida: tentar backtracking
            if len(route) > 1 and backtrack_count < MAX_BACKTRACK_ATTEMPTS:
                route.pop()
                # Descontar o tempo da aresta removida
                accumulated_time -= _edge_traverse_time(
                    current, edge_lengths, edge_speeds
                )
                current = route[-1]
                prev = route[-2] if len(route) > 1 else None
                backtrack_count += 1
                continue
            else:
                break  # Rede desconectada ou backtracking esgotado

        # Vies contra retorno: se temos mais de uma opcao, evitar voltar
        # para a aresta anterior (o que seria equivalente a um U-turn)
        if prev and len(neighbors) > 1:
            candidates = [n for n in neighbors if n != prev]
        else:
            candidates = neighbors

        if not candidates:
            candidates = neighbors

        # Remover candidatos ja tentados nesta posicao (para backtracking)
        pos_key = (len(route), current)
        already_tried = tried.get(pos_key, set())
        fresh_candidates = [c for c in candidates if c not in already_tried]

        if not fresh_candidates:
            # Todos os caminhos nesta posicao ja foram tentados: backtrack
            if len(route) > 1 and backtrack_count < MAX_BACKTRACK_ATTEMPTS:
                route.pop()
                accumulated_time -= _edge_traverse_time(
                    current, edge_lengths, edge_speeds
                )
                current = route[-1]
                prev = route[-2] if len(route) > 1 else None
                backtrack_count += 1
                continue
            else:
                break

        next_edge = rng.choice(fresh_candidates)
        tried.setdefault(pos_key, set()).add(next_edge)

        route.append(next_edge)
        traverse_time = _edge_traverse_time(next_edge, edge_lengths, edge_speeds)
        accumulated_time += traverse_time

        prev = current
        current = next_edge

    return route, accumulated_time


def _edge_traverse_time(
    edge_id: str,
    edge_lengths: dict[str, float],
    edge_speeds: dict[str, float],
) -> float:
    """Calcula o tempo estimado para percorrer uma aresta.

    Usa length/speed + penalidade de intersecao.
    """
    length = edge_lengths.get(edge_id, 100.0)
    speed = edge_speeds.get(edge_id, DEFAULT_SPEED)
    return (length / speed) + INTERSECTION_PENALTY


def estimate_simulation_time(
    encounters: int,
    cooldown: float,
    total_edges: int,
    total_vehicles: int,
) -> float:
    """Estima o tempo de simulacao SUMO necessario para N encontros.

    Leva em conta a densidade de veiculos na rede: quanto mais dispersos
    os veiculos (mais arestas por veiculo), mais tempo SUMO e necessario
    para que eles se cruzem organicamente.

    Em redes pequenas (ex: grid 3x3 com 24 arestas e 5 veiculos), os
    carros se encontram a cada ~15-25s. Em redes grandes (ex: mapa OSM
    com 5000+ arestas e 5 veiculos), podem levar minutos entre encontros.

    Args:
        encounters: Numero de encontros desejados
        cooldown: Segundos de cooldown entre encontros
        total_edges: Numero total de arestas na rede
        total_vehicles: Numero de veiculos na simulacao

    Returns:
        Tempo total estimado de simulacao em segundos (tempo SUMO)
    """
    # Tempo base por encontro: cooldown + margem para deteccao
    base_time_per_encounter = cooldown + 10

    # Fator de densidade: quanto mais arestas por veiculo, mais tempo
    # entre encontros. Em uma rede de 24 arestas com 5 carros (4.8
    # arestas/veiculo), o fator e ~1. Em uma rede de 5000 arestas
    # com 5 carros (1000 arestas/veiculo), o fator escala para ~14.
    edges_per_vehicle = total_edges / max(total_vehicles, 1)
    # Referencia: grid 3x3 = ~5 arestas/veiculo = fator 1.0
    density_factor = max(1.0, math.sqrt(edges_per_vehicle / 5.0))

    total_sumo_time = encounters * base_time_per_encounter * density_factor

    return total_sumo_time * SAFETY_MARGIN


def generate_routes_xml(
    adjacency: dict[str, list[str]],
    edge_lengths: dict[str, float],
    edge_speeds: dict[str, float],
    total_vehicles: int,
    total_steps: int,
    depart_spread: float,
    min_start_length: float,
    seed: int,
) -> tuple[str, float]:
    """Gera o conteudo XML do arquivo de rotas.

    Args:
        adjacency: Grafo de conectividade
        edge_lengths: Comprimentos das arestas em metros
        edge_speeds: Velocidades maximas das arestas em m/s
        total_vehicles: Numero de veiculos a criar
        total_steps: Duracao total da simulacao em steps (segundos)
        depart_spread: Intervalo entre departs de veiculos consecutivos
        min_start_length: Comprimento minimo de aresta para ponto de partida
        seed: Semente para reprodutibilidade

    Returns:
        Tupla (xml_string, tempo_medio_estimado_por_rota)
    """
    rng = random.Random(seed)
    all_edges = list(adjacency.keys())

    if not all_edges:
        print("ERRO: Nenhuma aresta encontrada na rede!", file=sys.stderr)
        sys.exit(1)

    # Filtrar arestas validas para ponto de partida:
    # - Comprimento >= min_start_length (para o veiculo caber e acelerar)
    # - Pelo menos um vizinho (para nao nascer em beco sem saida)
    spawnable_edges = [
        eid for eid in all_edges
        if edge_lengths.get(eid, 0) >= min_start_length
        and len(adjacency.get(eid, [])) > 0
    ]

    if not spawnable_edges:
        print(
            f"ERRO: Nenhuma aresta com comprimento >= {min_start_length}m "
            f"encontrada! Tente reduzir --min-start-length.",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(spawnable_edges) < total_vehicles:
        print(
            f"  AVISO: Apenas {len(spawnable_edges)} arestas de partida disponiveis "
            f"para {total_vehicles} veiculos (algumas arestas serao reutilizadas)."
        )

    # Duracao alvo por rota = tempo total + margem de seguranca
    target_duration = total_steps * SAFETY_MARGIN

    # Selecionar arestas iniciais espalhadas pelo mapa
    # Embaralhar e distribuir para maximizar diversidade espacial
    start_edges = []
    shuffled = spawnable_edges.copy()
    rng.shuffle(shuffled)
    for i in range(total_vehicles):
        start_edges.append(shuffled[i % len(shuffled)])

    # Gerar XML
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "",
        f"<!-- Gerado por createTrips.py | veiculos={total_vehicles} "
        f"steps={total_steps} seed={seed} -->",
        "<routes>",
    ]

    total_est_time = 0.0
    for i in range(total_vehicles):
        depart_time = round(i * depart_spread, 1)
        color = VEHICLE_COLORS[i % len(VEHICLE_COLORS)]

        # Tentar gerar rota a partir da aresta selecionada.
        # Se falhar (rota muito curta), tentar outra aresta inicial.
        best_route = None
        best_time = 0.0
        for attempt in range(MAX_ROUTE_RETRIES):
            if attempt == 0:
                start = start_edges[i]
            else:
                start = rng.choice(spawnable_edges)

            route, est_time = generate_route(
                adjacency, edge_lengths, edge_speeds,
                start_edge=start,
                target_duration=target_duration,
                rng=rng,
            )

            if best_route is None or est_time > best_time:
                best_route = route
                best_time = est_time

            # Se a rota cobre pelo menos 80% do tempo alvo, esta boa
            if est_time >= target_duration * 0.8:
                break

        if best_time < target_duration * 0.3:
            print(
                f"  AVISO: veh_{i} tem rota curta ({best_time:.0f}s estimados "
                f"de {target_duration:.0f}s alvo). A rede pode ser pequena demais.",
            )

        total_est_time += best_time
        edges_str = " ".join(best_route)

        lines.append(f"")
        lines.append(
            f'    <!-- veh_{i} | depart={depart_time}s | '
            f'{len(best_route)} edges | ~{best_time:.0f}s estimados -->'
        )
        lines.append(
            f'    <vehicle id="veh_{i}" depart="{depart_time}" color="{color}">'
        )
        lines.append(f'        <route edges="{edges_str}"/>')
        lines.append(f"    </vehicle>")

    lines.append("</routes>")
    lines.append("")

    avg_time = total_est_time / total_vehicles if total_vehicles > 0 else 0
    return "\n".join(lines), avg_time


def update_sumocfg(sumocfg_path: Path, end_time: int) -> None:
    """Atualiza o tempo final no arquivo .sumocfg.

    Modifica a tag <end value="..."/> para refletir o novo tempo
    de simulacao calculado.

    Args:
        sumocfg_path: Caminho para o arquivo .sumocfg
        end_time: Novo tempo final em segundos
    """
    if not sumocfg_path.exists():
        print(f"AVISO: .sumocfg nao encontrado: {sumocfg_path}", file=sys.stderr)
        return

    tree = ET.parse(sumocfg_path)
    root = tree.getroot()

    time_elem = root.find(".//time/end")
    if time_elem is not None:
        old_val = time_elem.get("value")
        time_elem.set("value", str(end_time))
        print(f"  .sumocfg atualizado: end {old_val} -> {end_time}")
    else:
        print("  AVISO: Tag <time><end> nao encontrada no .sumocfg", file=sys.stderr)
        return

    # Preservar declaracao XML e formatar bonito
    tree.write(sumocfg_path, xml_declaration=True, encoding="UTF-8")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Gerador de rotas longas para simulacao V2V-FL com SUMO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--net-file", type=str, required=True,
        help="Caminho para o arquivo .net.xml da rede SUMO",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help=(
            "Caminho de saida para o arquivo .rou.xml gerado. "
            "Por padrao, grava no mesmo diretorio do .net.xml "
            "como '<nome_da_rede>.rou.xml'"
        ),
    )
    parser.add_argument(
        "--total-vehicles", type=int, default=5,
        help="Numero de veiculos (containers Docker) a criar (default: 5)",
    )
    parser.add_argument(
        "--total-steps", type=int, default=100,
        help=(
            "Duracao total da simulacao em steps/segundos. "
            "Se 100 (padrao), calcula automaticamente a partir de --encounters"
        ),
    )
    parser.add_argument(
        "--encounters", type=int, default=5,
        help=(
            "Numero de encontros P2P desejados. Usado para estimar "
            "o tempo de simulacao quando --total-steps nao e fornecido (default: 5)"
        ),
    )
    parser.add_argument(
        "--depart-spread", type=float, default=2.0,
        help=(
            "Intervalo em segundos entre o depart de veiculos "
            "consecutivos. Espalha a entrada para evitar congestionamento "
            "inicial (default: 2.0)"
        ),
    )
    parser.add_argument(
        "--cooldown", type=float, default=DEFAULT_COOLDOWN,
        help=(
            f"Cooldown entre encontros em segundos. Deve bater com o "
            f"--cooldown do orquestrador (default: {DEFAULT_COOLDOWN})"
        ),
    )

    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semente para reproducibilidade das rotas (default: 42)",
    )
    parser.add_argument(
        "--update-sumocfg", type=str, default=None,
        help=(
            "Caminho para o .sumocfg a ser atualizado com o novo tempo "
            "de simulacao. Se nao fornecido, tenta encontrar um .sumocfg "
            "no mesmo diretorio do .net.xml"
        ),
    )
    parser.add_argument(
        "--vclass", type=str, default="passenger",
        help=(
            "Classe de veiculo para filtrar permissoes de arestas. "
            "Apenas arestas que permitem esta classe serao usadas nas rotas. "
            "Analogo ao --edge-permission do randomTrips.py (default: passenger)"
        ),
    )
    parser.add_argument(
        "--min-start-length", type=float, default=DEFAULT_MIN_START_LENGTH,
        help=(
            f"Comprimento minimo (metros) de uma aresta para ser usada como "
            f"ponto de partida de um veiculo. Arestas menores continuam "
            f"disponiveis para travessia durante a rota "
            f"(default: {DEFAULT_MIN_START_LENGTH})"
        ),
    )

    args = parser.parse_args()

    # ── Resolver caminhos ─────────────────────────────────────────────────
    net_file = Path(args.net_file).resolve()
    if not net_file.exists():
        print(f"ERRO: Arquivo de rede nao encontrado: {net_file}", file=sys.stderr)
        sys.exit(1)

    net_dir = net_file.parent

    if args.output:
        output_file = Path(args.output).resolve()
    else:
        # Derivar nome do .rou.xml a partir do nome da rede
        output_file = net_dir / f"{net_file.stem}.rou.xml"

    # ── Parsear rede ──────────────────────────────────────────────────────
    print(f"Parseando rede: {net_file}")
    adjacency, edge_lengths, edge_speeds = parse_network(net_file, vclass=args.vclass)

    if not edge_lengths:
        print(
            f"ERRO: Nenhuma aresta com permissao para '{args.vclass}' encontrada!",
            file=sys.stderr,
        )
        sys.exit(1)

    avg_length = sum(edge_lengths.values()) / len(edge_lengths)
    avg_speed = sum(edge_speeds.values()) / len(edge_speeds)
    print(f"  {len(edge_lengths)} arestas disponiveis para '{args.vclass}'")
    print(f"  Comprimento medio: {avg_length:.1f}m")
    print(f"  Velocidade media: {avg_speed:.1f} m/s ({avg_speed * 3.6:.0f} km/h)")

    # Contar arestas validas para spawn
    spawnable_count = sum(
        1 for eid in edge_lengths
        if edge_lengths[eid] >= args.min_start_length
        and len(adjacency.get(eid, [])) > 0
    )
    print(f"  Arestas de partida (>= {args.min_start_length}m): {spawnable_count}")

    # ── Calcular tempo de simulacao ───────────────────────────────────────
    if args.total_steps != 100 and args.total_steps > 0:
        total_steps = args.total_steps
        print(f"  Tempo de simulacao: {total_steps}s (definido manualmente)")
    else:
        total_steps = int(estimate_simulation_time(
            encounters=args.encounters,
            cooldown=args.cooldown,
            total_edges=len(edge_lengths),
            total_vehicles=args.total_vehicles,
        ))
        # Minimo razoavel
        total_steps = max(total_steps, 300)
        print(
            f"  Tempo estimado para {args.encounters} encontros: "
            f"{total_steps}s (~{total_steps // 60}min)"
        )

    # ── Gerar rotas ───────────────────────────────────────────────────────
    print(f"\nGerando {args.total_vehicles} rotas (seed={args.seed})...")
    xml_content, avg_route_time = generate_routes_xml(
        adjacency=adjacency,
        edge_lengths=edge_lengths,
        edge_speeds=edge_speeds,
        total_vehicles=args.total_vehicles,
        total_steps=total_steps,
        depart_spread=args.depart_spread,
        min_start_length=args.min_start_length,
        seed=args.seed,
    )

    # ── Salvar arquivo de rotas ───────────────────────────────────────────
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(xml_content, encoding="utf-8")
    print(f"  Rotas salvas em: {output_file}")
    print(f"  Tempo medio estimado por rota: ~{avg_route_time:.0f}s")

    # ── Atualizar .sumocfg ────────────────────────────────────────────────
    if args.update_sumocfg:
        sumocfg = Path(args.update_sumocfg).resolve()
    else:
        # Tentar encontrar um .sumocfg no mesmo diretorio
        candidates = list(net_dir.glob("*.sumocfg"))
        sumocfg = candidates[0] if candidates else None

    if sumocfg:
        print(f"\nAtualizando {sumocfg.name}...")
        update_sumocfg(sumocfg, total_steps)
    else:
        print(
            f"\n  AVISO: Nenhum .sumocfg encontrado. Lembre-se de atualizar "
            f"o <end value=\"...\"/> manualmente para {total_steps}."
        )

    # ── Resumo ────────────────────────────────────────────────────────────
    # Derivar caminho relativo do .sumocfg para a mensagem final
    if sumocfg:
        try:
            sumocfg_relative = f"maps/{sumocfg.name}"
        except Exception:
            sumocfg_relative = str(sumocfg)
    else:
        sumocfg_relative = f"maps/{net_file.stem}.sumocfg"

    print(f"\n{'=' * 60}")
    print(f"  Veiculos         : {args.total_vehicles}")
    print(f"  vclass           : {args.vclass}")
    print(f"  Min start length : {args.min_start_length}m")
    print(f"  Tempo rota (med) : ~{avg_route_time:.0f}s")
    print(f"  Steps (tempo)    : {total_steps}s (~{total_steps // 60}min)")
    print(f"  Encounters       : {args.encounters}")
    print(f"  Seed             : {args.seed}")
    print(f"  Saida            : {output_file.name}")
    print(f"{'=' * 60}")
    print(f"\nPronto! Execute a simulacao com (de dentro de src/sumo_adapter/):")
    print(
        f'  uv run python sumo_docker_orchestrator.py '
        f'--sumo-cfg {sumocfg_relative} '
        f'--total-clients {args.total_vehicles} '
        f'--encounters {args.encounters}'
    )


if __name__ == "__main__":
    main()
