#!/usr/bin/env python3
"""
randomAntenna.py — Gerador de posições aleatórias para RSUs (antenas fixas).

Lê um arquivo .net.xml do SUMO, extrai os limites do mapa e as posições
dos cruzamentos (junctions), e gera N posições aleatórias para RSUs.

Dois modos de posicionamento:
  1. Aleatório puro: distribui as RSUs uniformemente dentro do bounding box.
  2. Próximo a cruzamentos (--near-junctions): escolhe N cruzamentos
     aleatórios e coloca as RSUs nessas coordenadas exatas.
     Cruzamentos são pontos de alto tráfego, o que maximiza a
     probabilidade de encontros V2I.

Uso:
    cd FL_ServerPuma/
    uv run python src/v2v_fl/maps/randomAntenna.py \\
        --net-file src/v2v_fl/maps/grid.net.xml \\
        --n-antennas 4 \\
        --radius 200 \\
        --seed 42 \\
        --output src/v2v_fl/maps/rsu_config.json

    # Modo próximo a cruzamentos:
    uv run python src/v2v_fl/maps/randomAntenna.py \\
        --net-file src/v2v_fl/maps/marambaia.net.xml \\
        --n-antennas 10 \\
        --near-junctions \\
        --seed 42
"""

import argparse
import json
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_net_file(net_path: str) -> tuple[tuple[float, float, float, float], list[dict]]:
    """Lê o .net.xml do SUMO e extrai o bounding box e os cruzamentos.

    O bounding box vem do atributo 'convBoundary' do elemento <location>.
    Os cruzamentos vêm dos elementos <junction> com type != "internal"
    (junctions internas são subdivisões internas das vias, não cruzamentos reais).

    Args:
        net_path: Caminho para o arquivo .net.xml do SUMO.

    Returns:
        Tupla (bounding_box, junctions):
          - bounding_box: (x_min, y_min, x_max, y_max) em metros
          - junctions: Lista de dicts {"id": str, "x": float, "y": float}
    """
    tree = ET.parse(net_path)
    root = tree.getroot()

    # ── Extrair bounding box do elemento <location> ──────────────────────
    location = root.find("location")
    if location is None:
        print("[ERRO] Elemento <location> não encontrado no .net.xml")
        sys.exit(1)

    conv_boundary = location.get("convBoundary", "")
    if not conv_boundary:
        print("[ERRO] Atributo 'convBoundary' não encontrado em <location>")
        sys.exit(1)

    # convBoundary="x_min,y_min,x_max,y_max"
    parts = conv_boundary.split(",")
    bbox = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))

    # ── Extrair cruzamentos reais (excluir junctions internas) ───────────
    # Junctions internas têm IDs que começam com ":" e type="internal"
    junctions = []
    for junc in root.findall("junction"):
        junc_type = junc.get("type", "")
        junc_id = junc.get("id", "")

        # Filtrar: ignorar junctions internas (subdivisões de vias)
        if junc_type == "internal" or junc_id.startswith(":"):
            continue

        x = float(junc.get("x", 0))
        y = float(junc.get("y", 0))
        junctions.append({"id": junc_id, "x": x, "y": y})

    return bbox, junctions


def generate_random_positions(
    bbox: tuple[float, float, float, float],
    n_antennas: int,
    seed: int | None = None,
) -> list[tuple[float, float]]:
    """Gera N posições aleatórias uniformemente dentro do bounding box.

    Usa uma margem interna de 5% para evitar que RSUs fiquem exatamente
    na borda do mapa (onde não há vias).

    Args:
        bbox: (x_min, y_min, x_max, y_max) em metros.
        n_antennas: Número de posições a gerar.
        seed: Seed para reprodutibilidade.

    Returns:
        Lista de tuplas (x, y).
    """
    if seed is not None:
        random.seed(seed)

    x_min, y_min, x_max, y_max = bbox

    # Margem de 5% para evitar bordas sem vias
    margin_x = (x_max - x_min) * 0.05
    margin_y = (y_max - y_min) * 0.05

    positions = []
    for _ in range(n_antennas):
        x = random.uniform(x_min + margin_x, x_max - margin_x)
        y = random.uniform(y_min + margin_y, y_max - margin_y)
        positions.append((round(x, 2), round(y, 2)))

    return positions


def generate_near_junction_positions(
    junctions: list[dict],
    n_antennas: int,
    seed: int | None = None,
) -> list[tuple[float, float]]:
    """Escolhe N cruzamentos aleatórios e usa suas coordenadas como posições.

    Se N > número de cruzamentos disponíveis, usa todos os cruzamentos
    e emite um aviso.

    Args:
        junctions: Lista de dicts com "id", "x", "y".
        n_antennas: Número de RSUs desejadas.
        seed: Seed para reprodutibilidade.

    Returns:
        Lista de tuplas (x, y).
    """
    if seed is not None:
        random.seed(seed)

    if n_antennas > len(junctions):
        print(
            f"  [WARN] Pedidas {n_antennas} RSUs, mas só existem "
            f"{len(junctions)} cruzamentos. Usando todos."
        )
        n_antennas = len(junctions)

    # Escolher N cruzamentos aleatórios sem repetição
    chosen = random.sample(junctions, n_antennas)
    return [(j["x"], j["y"]) for j in chosen]


def main():
    parser = argparse.ArgumentParser(
        description="Gera posições aleatórias para RSUs (antenas fixas) no mapa SUMO.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--net-file", required=True,
        help="Caminho para o arquivo .net.xml do SUMO",
    )
    parser.add_argument(
        "--n-antennas", type=int, default=5,
        help="Número de RSUs a espalhar no mapa (default: 5)",
    )
    parser.add_argument(
        "--radius", type=float, default=200.0,
        help="Raio de cobertura de cada RSU em metros (default: 200.0)",
    )
    parser.add_argument(
        "--n-antennas-per-antenna", type=int, default=16,
        help="Número de elementos MIMO por RSU (default: 16)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed para reprodutibilidade (default: aleatório)",
    )
    parser.add_argument(
        "--near-junctions", action="store_true",
        help="Posicionar RSUs em cruzamentos reais do mapa (em vez de aleatório puro)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Caminho de saída do rsu_config.json (default: mesmo diretório do net-file)",
    )
    args = parser.parse_args()

    # ── Resolver caminhos ─────────────────────────────────────────────────
    net_path = Path(args.net_file).resolve()
    if not net_path.exists():
        print(f"[ERRO] Arquivo não encontrado: {net_path}")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = net_path.parent / "rsu_config.json"

    # ── Parsear o mapa SUMO ───────────────────────────────────────────────
    print(f"Lendo mapa SUMO: {net_path.name}")
    bbox, junctions = parse_net_file(str(net_path))

    print(f"  Bounding box: x=[{bbox[0]:.0f}, {bbox[2]:.0f}] y=[{bbox[1]:.0f}, {bbox[3]:.0f}]")
    print(f"  Cruzamentos encontrados: {len(junctions)}")

    # ── Gerar posições ────────────────────────────────────────────────────
    if args.near_junctions:
        print(f"\nModo: Próximo a cruzamentos")
        if not junctions:
            print("[ERRO] Nenhum cruzamento encontrado no mapa.")
            sys.exit(1)
        positions = generate_near_junction_positions(
            junctions, args.n_antennas, args.seed,
        )
    else:
        print(f"\nModo: Aleatório dentro do bounding box")
        positions = generate_random_positions(
            bbox, args.n_antennas, args.seed,
        )

    # ── Montar JSON de saída ──────────────────────────────────────────────
    rsu_list = []
    for i, (x, y) in enumerate(positions):
        rsu_list.append({
            "id": f"rsu_{i}",
            "x": x,
            "y": y,
            "radius": args.radius,
            "n_antennas": args.n_antennas_per_antenna,
        })

    config = {
        "map_file": net_path.name,
        "seed": args.seed,
        "n_rsus": len(rsu_list),
        "rsus": rsu_list,
    }

    # ── Salvar ────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nRSUs geradas: {len(rsu_list)}")
    for rsu in rsu_list:
        print(f"  {rsu['id']}: ({rsu['x']:.1f}, {rsu['y']:.1f}) "
              f"raio={rsu['radius']}m, antenas={rsu['n_antennas']}")
    print(f"\nSalvo em: {output_path}")


if __name__ == "__main__":
    main()
