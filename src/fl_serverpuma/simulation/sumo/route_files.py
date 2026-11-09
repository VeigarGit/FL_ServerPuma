# Responsabilidade: Grava rotas XML e atualiza o arquivo de configuração SUMO.
#
# Principais funções: generate_routes_xml, update_sumocfg.

"""SUMO route generation — route_files."""

import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from fl_serverpuma.simulation.sumo.routing import generate_route
from fl_serverpuma.simulation.sumo.route_settings import VEHICLE_COLORS
from fl_serverpuma.simulation.sumo.route_settings import MAX_ROUTE_RETRIES
from fl_serverpuma.simulation.sumo.route_settings import SAFETY_MARGIN

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
