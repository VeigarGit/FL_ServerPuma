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
  - Estima o tempo de simulacao necessario com base nos encontros desejados
  - Gera rotas por random walk no grafo, com viés para evitar retornos
  - Calcula cores distintas para visualizacao no sumo-gui
  - Atualiza automaticamente o .sumocfg com o tempo de simulacao calculado
  - Permite escalonar os departs para que os veiculos entrem gradualmente

Uso (executar de dentro da pasta maps/):
  uv run python createTrips.py --net-file grid.net.xml \\
      --total-vehicles 5 --encounters 5

  uv run python createTrips.py --net-file grid.net.xml \\
      --total-vehicles 10 --total-steps 1000 --depart-spread 15
"""

import argparse
import math
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


# ── Constantes ────────────────────────────────────────────────────────────────

# Tempo medio estimado que um veiculo leva para percorrer uma aresta,
# considerando aceleracao, cruzamentos e semaforos.
# Usado para calcular quantas arestas sao necessarias na rota.
# (190m / 13.89 m/s ≈ 13.7s, mas com paradas ~20s e uma margem extra)
DEFAULT_EDGE_TRAVERSE_TIME = 20.0

# Cooldown padrao entre encontros do orquestrador (deve bater com o
# argumento --cooldown do sumo_docker_orchestrator.py)
DEFAULT_COOLDOWN = 5

# Margem de seguranca adicionada ao tempo total para evitar que os
# veiculos saiam do mapa antes da simulacao acabar.
SAFETY_MARGIN = 1.3

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

def parse_network(net_file: Path) -> tuple[dict[str, list[str]], dict[str, float]]:
    """Analisa o arquivo .net.xml do SUMO e extrai o grafo de conectividade.
    
    Retorna:
        adjacency: dicionario {edge_id: [edges alcancaveis]}
        edge_lengths: dicionario {edge_id: comprimento em metros}
    """
    tree = ET.parse(net_file)
    root = tree.getroot()

    # Extrair comprimentos das arestas (ignorando arestas internas :...)
    edge_lengths: dict[str, float] = {}
    for edge in root.findall(".//edge"):
        eid = edge.get("id", "")
        if eid.startswith(":"):
            continue
        lane = edge.find("lane")
        if lane is not None:
            edge_lengths[eid] = float(lane.get("length", "0"))

    # Extrair conectividade a partir das <connection>
    adjacency: dict[str, list[str]] = {eid: [] for eid in edge_lengths}
    for conn in root.findall(".//connection"):
        fr = conn.get("from", "")
        to = conn.get("to", "")
        if fr.startswith(":") or to.startswith(":"):
            continue
        if fr in adjacency and to not in adjacency[fr]:
            adjacency[fr].append(to)

    return adjacency, edge_lengths


# ── Geracao de Rotas ─────────────────────────────────────────────────────────

def generate_route(
    adjacency: dict[str, list[str]],
    edge_lengths: dict[str, float],
    start_edge: str,
    min_edges: int,
    rng: random.Random,
) -> list[str]:
    """Gera uma rota por random walk no grafo da rede.
    
    Usa um vies para evitar retornos imediatos (U-turns), tornando
    as rotas mais realistas e espacialmente distribuidas.
    
    Args:
        adjacency: Grafo de conectividade da rede
        edge_lengths: Comprimento de cada aresta
        start_edge: Aresta inicial da rota
        min_edges: Numero minimo de arestas na rota
        rng: Gerador de numeros aleatorios (para reprodutibilidade)
        
    Returns:
        Lista de edge IDs formando a rota
    """
    route = [start_edge]
    current = start_edge
    prev = None

    for _ in range(min_edges - 1):
        neighbors = adjacency.get(current, [])
        if not neighbors:
            break

        # Viés contra retorno: se temos mais de uma opcao, evitar voltar
        # para a aresta anterior (o que seria equivalente a um U-turn)
        if prev and len(neighbors) > 1:
            candidates = [n for n in neighbors if n != prev]
        else:
            candidates = neighbors

        if not candidates:
            candidates = neighbors

        next_edge = rng.choice(candidates)
        route.append(next_edge)
        prev = current
        current = next_edge

    return route


def estimate_simulation_time(
    encounters: int,
    cooldown: float,
) -> float:
    """Estima o tempo de simulacao SUMO necessario para N encontros.
    
    Args:
        encounters: Numero de encontros desejados
        cooldown: Segundos de cooldown entre encontros
        
    Returns:
        Tempo total estimado de simulacao em segundos (tempo SUMO)
    """
    sumo_time_per_encounter = cooldown + 10  # cooldown + margem para deteccao
    total_sumo_time = encounters * sumo_time_per_encounter
    
    return total_sumo_time * SAFETY_MARGIN


def generate_routes_xml(
    adjacency: dict[str, list[str]],
    edge_lengths: dict[str, float],
    total_vehicles: int,
    total_steps: int,
    depart_spread: float,
    seed: int,
) -> tuple[str, int]:
    """Gera o conteudo XML do arquivo de rotas.
    
    Args:
        adjacency: Grafo de conectividade
        edge_lengths: Comprimentos das arestas
        total_vehicles: Numero de veiculos a criar
        total_steps: Duracao total da simulacao em steps (segundos)
        depart_spread: Intervalo entre departs de veiculos consecutivos
        seed: Semente para reprodutibilidade
        
    Returns:
        Tupla (xml_string, edges_per_route)
    """
    rng = random.Random(seed)
    all_edges = list(adjacency.keys())

    if not all_edges:
        print("ERRO: Nenhuma aresta encontrada na rede!", file=sys.stderr)
        sys.exit(1)

    # Calcular quantas arestas sao necessarias para cobrir o tempo total.
    # Cada aresta leva ~DEFAULT_EDGE_TRAVERSE_TIME segundos para percorrer.
    avg_length = sum(edge_lengths.values()) / len(edge_lengths)
    edges_needed = int(math.ceil(total_steps / DEFAULT_EDGE_TRAVERSE_TIME))
    # Adicionar margem de seguranca
    edges_needed = int(edges_needed * SAFETY_MARGIN)
    # Minimo razoavel
    edges_needed = max(edges_needed, 20)

    # Selecionar arestas iniciais espalhadas pelo mapa
    # Embaralhar e distribuir para maximizar diversidade espacial
    start_edges = []
    shuffled = all_edges.copy()
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

    for i in range(total_vehicles):
        depart_time = round(i * depart_spread, 1)
        color = VEHICLE_COLORS[i % len(VEHICLE_COLORS)]
        route = generate_route(
            adjacency, edge_lengths,
            start_edge=start_edges[i],
            min_edges=edges_needed,
            rng=rng,
        )
        edges_str = " ".join(route)

        lines.append(f"")
        lines.append(f'    <!-- veh_{i} | depart={depart_time}s | {len(route)} edges -->')
        lines.append(
            f'    <vehicle id="veh_{i}" depart="{depart_time}" color="{color}">'
        )
        lines.append(f'        <route edges="{edges_str}"/>')
        lines.append(f"    </vehicle>")

    lines.append("</routes>")
    lines.append("")

    return "\n".join(lines), edges_needed


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
            "como 'grid.rou.xml'"
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
        output_file = net_dir / "grid.rou.xml"

    # ── Parsear rede ──────────────────────────────────────────────────────
    print(f"Parseando rede: {net_file}")
    adjacency, edge_lengths = parse_network(net_file)
    print(f"  {len(edge_lengths)} arestas encontradas")
    print(f"  Comprimento medio: {sum(edge_lengths.values()) / len(edge_lengths):.1f}m")

    # ── Calcular tempo de simulacao ───────────────────────────────────────
    if args.total_steps != 100 and args.total_steps > 0:
        total_steps = args.total_steps
        print(f"  Tempo de simulacao: {total_steps}s (definido manualmente)")
    else:
        total_steps = int(estimate_simulation_time(
            encounters=args.encounters,
            cooldown=args.cooldown,
        ))
        # Minimo razoavel
        total_steps = max(total_steps, 300)
        print(
            f"  Tempo estimado para {args.encounters} encontros: "
            f"{total_steps}s (~{total_steps // 60}min)"
        )

    # ── Gerar rotas ───────────────────────────────────────────────────────
    print(f"\nGerando {args.total_vehicles} rotas (seed={args.seed})...")
    xml_content, edges_per_route = generate_routes_xml(
        adjacency=adjacency,
        edge_lengths=edge_lengths,
        total_vehicles=args.total_vehicles,
        total_steps=total_steps,
        depart_spread=args.depart_spread,
        seed=args.seed,
    )

    # ── Salvar arquivo de rotas ───────────────────────────────────────────
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(xml_content, encoding="utf-8")
    print(f"  Rotas salvas em: {output_file}")
    print(f"  Arestas por rota: ~{edges_per_route}")

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
    print(f"\n{'=' * 50}")
    print(f"  Veiculos      : {args.total_vehicles}")
    print(f"  Arestas/rota  : ~{edges_per_route}")
    print(f"  Steps (tempo) : {total_steps}s (~{total_steps // 60}min)")
    print(f"  Encounters    : {args.encounters}")
    print(f"  Seed          : {args.seed}")
    print(f"{'=' * 50}")
    print(f"\nPronto! Execute a simulacao com (de dentro de src/sumo_adapter/):")
    print(
        f'  uv run python sumo_docker_orchestrator.py '
        f'--sumo-cfg maps/grid.sumocfg '
        f'--total-clients {args.total_vehicles} '
        f'--encounters {args.encounters}'
    )


if __name__ == "__main__":
    main()
