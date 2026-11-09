# Responsabilidade: Gera trajetos válidos e estima os tempos de percurso/simulação.
#
# Principais funções: generate_route, estimate_simulation_time.

"""SUMO route generation — routing."""

import math
import random
from fl_serverpuma.simulation.sumo.route_settings import MAX_BACKTRACK_ATTEMPTS
from fl_serverpuma.simulation.sumo.route_settings import SAFETY_MARGIN
from fl_serverpuma.simulation.sumo.route_settings import INTERSECTION_PENALTY
from fl_serverpuma.simulation.sumo.route_settings import DEFAULT_SPEED

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
