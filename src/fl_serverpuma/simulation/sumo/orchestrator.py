# Responsabilidade: CLI do orquestrador SUMO; valida opções e inicia o loop de simulação.
#
# Principais funções: main.

"""SUMO federation — orchestrator."""

import argparse
from fl_serverpuma.simulation.sumo.simulation_loop import run_orchestrator
from fl_serverpuma.simulation.sumo.settings import DEFAULT_COMM_RADIUS
from fl_serverpuma.simulation.sumo.settings import DEFAULT_MIN_CONTACT_TIME
from fl_serverpuma.simulation.sumo.settings import DEFAULT_STEP_LENGTH
from fl_serverpuma.simulation.sumo.settings import DEFAULT_WARMUP
from fl_serverpuma.simulation.sumo.settings import DEFAULT_MIN_CLIENTS
from fl_serverpuma.simulation.sumo.settings import DEFAULT_COOLDOWN

def main():
    parser = argparse.ArgumentParser(
        description="V2V Opportunistic FL Orchestrator",
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
        "--max-encounters-per-pair", type=int, default=0,
        help="Numero maximo de encontros permitidos entre o mesmo par/cluster de veiculos. 0 = sem limite (default: 0)",
    )
    parser.add_argument(
        "--radius", type=float, default=DEFAULT_COMM_RADIUS,
        help=f"Raio de comunicacao V2V em metros (default: {DEFAULT_COMM_RADIUS})",
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
    parser.add_argument(
        "--max-epochs", type=int, default=0,
        help="Limite de epocas por cliente no modo descentralizado (0 = sem limite, default: 0)",
    )

    # ── Parametros do Modelo (CNN / CLIP) ─────────────────────────────────
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "clip"], help="Modelo a ser utilizado (default: cnn)")
    parser.add_argument("--config", type=str, default="lora_clip/train_config.yml", help="Caminho do config YAML para o CLIP")
    parser.add_argument("--strategy", type=str, default="lora", choices=["lora", "adalora", "sora_with_schedule", "sora_no_schedule"], help="Estrategia de PEFT")
    parser.add_argument("--rank", type=int, default=8, help="Rank para LoRA/SoRA")
    parser.add_argument("--paca", type=int, default=12, help="Numero de camadas upper para a estrategia PaCA")

    args = parser.parse_args()
    run_orchestrator(args)

if __name__ == '__main__':
    main()
