# Responsabilidade: CLI de geração de viagens; seleciona rede/opções e coordena geração e
# gravação das rotas.
#
# Principais funções: main.

"""SUMO route generation — create_trips."""

import argparse
import sys
from pathlib import Path
from fl_serverpuma.simulation.sumo.network import parse_network
from fl_serverpuma.simulation.sumo.routing import estimate_simulation_time
from fl_serverpuma.simulation.sumo.route_files import generate_routes_xml
from fl_serverpuma.simulation.sumo.route_files import update_sumocfg
from fl_serverpuma.simulation.sumo.route_settings import DEFAULT_COOLDOWN
from fl_serverpuma.simulation.sumo.route_settings import DEFAULT_MIN_START_LENGTH

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
        f'  uv run python -m fl_serverpuma.simulation.sumo.orchestrator '
        f'--sumo-cfg {sumocfg_relative} '
        f'--total-clients {args.total_vehicles} '
        f'--encounters {args.encounters}'
    )

if __name__ == '__main__':
    main()
