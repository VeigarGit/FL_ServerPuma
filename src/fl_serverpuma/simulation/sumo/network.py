# Responsabilidade: Lê a rede SUMO e extrai arestas/conexões e seus metadados.
#
# Principais funções: parse_network.

"""SUMO route generation — network."""

import xml.etree.ElementTree as ET
from pathlib import Path
from fl_serverpuma.simulation.sumo.route_settings import DEFAULT_SPEED

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
