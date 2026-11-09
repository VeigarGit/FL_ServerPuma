# Responsabilidade: Define VehicleState/V2VEvent e calcula distâncias, tempo de contato e
# agrupamentos de veículos.
#
# Principais funções/classes: VehicleState, V2VEvent, euclidean_distance,
# estimate_contact_time, cluster_etc, find_clusters.

"""SUMO federation — mobility."""

import itertools
import math
from dataclasses import dataclass, field
from fl_serverpuma.simulation.sumo.settings import MAX_ETC

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
class V2VEvent:
    """Evento de log da simulacao V2V para analise posterior.

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
    if A < 1e-9:
        avg_speed = (v1.speed + v2.speed) / 2.0
        if avg_speed < 0.1:
            # Ambos realmente parados — ETC alto e correto
            return "Carros Parados"
        # Se tem velocidade mas e quase igual (movendo juntos),
        # estimar pela margem restante do raio dividida pela
        # velocidade media. Pessimista mas muito melhor que MAX_ETC.
        remaining = radius - dist
        return min(remaining / avg_speed, MAX_ETC)

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
) -> float | str:
    """Calcula o ETC de um cluster inteiro.

    O ETC do cluster e o MINIMO entre todos os pares de veiculos,
    pois basta um par se separar para quebrar a conectividade.

    Args:
        cluster: Lista de estados dos veiculos no agrupamento.
        radius: Raio de comunicacao (metros).

    Returns:
        float: ETC minimo do cluster em segundos
    """
    if len(cluster) < 2:
        return 0.0

    min_etc = float('inf')
    carros_parados = False
    for v1, v2 in itertools.combinations(cluster, 2):
        etc = estimate_contact_time(v1, v2, radius)
        if isinstance(etc, str) and etc == "Carros Parados":
            carros_parados = True
        else:
            min_etc = min(min_etc, etc)

    if min_etc == float('inf'):
        if carros_parados:
            return "Carros Parados"
        return 0.0
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
        radius: Raio de comunicacao V2V em metros
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
