"""Gera os Composes V2V (veículos) e V2I (veículos + serviço único de RSUs).

Na raiz do projeto:
    uv run python docker/generate_v2v_compose.py --clients 2 --dataset MNIST

Sem --mode, grava ambos os arquivos em docker/. Os orquestradores chamam
generate(mode="v2v" ou "v2i") para atualizar somente o Compose utilizado.
Use --project-directory apontando para a raiz ao executar docker compose.
"""

import argparse
import math
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILES = {mode: f"docker-compose.{mode}.yml" for mode in ("v2v", "v2i")}
CONTAINER_RESULTS = "/app/src/results"
CONTAINER_RSU_CONFIG = "/app/rsu_config.json"


def generate(
    client_indices: list[int],
    dataset: str,
    rounds: int,
    prune: int,
    ala: int,
    exp_name: str,
    max_epochs: int = 0,
    model: str = "cnn",
    config: str = "lora_clip/train_config.yml",
    strategy: str = "lora",
    rank: int = 8,
    paca: int = 12,
    *,
    mode: str = "v2v",
    run_id: str = "v2i",
    rsu_config: str | Path | None = None,
    exchange_timeout: float = 300.0,
    results_dir: Path | None = None,
    encounters_dir: Path | None = None,
    seed: int = 42,
    dataset_dir: Path | None = None,
) -> str:
    """Grava o Compose do modo escolhido e retorna seu caminho absoluto.

    Os clientes preservam as reservas de GPU e os parâmetros de treinamento.
    O V2I acrescenta um rsu-service em CPU, na mesma rede docker-tc. O serviço
    reutiliza a imagem dos clientes e monta a configuração das RSUs para leitura.
    A rede mantém os valores existentes: 100 Mbit/s, 5 ms e 3% de perda.
    """
    if mode not in COMPOSE_FILES:
        raise ValueError("mode deve ser 'v2v' ou 'v2i'.")
    if (
        not client_indices
        or any(type(index) is not int or index < 0 for index in client_indices)
        or len(set(client_indices)) != len(client_indices)
    ):
        raise ValueError("client_indices deve conter índices únicos e não negativos.")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id deve ser uma string não vazia.")
    if not math.isfinite(exchange_timeout) or exchange_timeout <= 0:
        raise ValueError("exchange_timeout deve ser positivo e finito.")

    def container_path(path: Path) -> str:
        relative = Path(path).resolve().relative_to(PROJECT_ROOT / "src/results")
        return f"{CONTAINER_RESULTS}/{relative.as_posix()}"

    container_encounters = (
        container_path(encounters_dir) if encounters_dir else f"{CONTAINER_RESULTS}/encounters"
    )
    container_outputs = container_path(results_dir) if results_dir else None
    container_dataset = None
    if dataset_dir is not None:
        selected = Path(dataset_dir)
        if not selected.is_absolute():
            selected = PROJECT_ROOT / selected
        relative = selected.resolve().relative_to((PROJECT_ROOT / 'src/dataset').resolve())
        if not selected.is_dir():
            raise ValueError(f'Diretório do dataset não encontrado: {selected}')
        container_dataset = f'/app/src/dataset/{relative.as_posix()}'

    build = {
        "context": ".",
        "dockerfile": "docker/dockerfile.client",
        "args": {"NO_CACHE": "true"},
    }
    environment = {
        "PYTHONUNBUFFERED": "1",
        "PYTHONHASHSEED": str(seed),
        "PYTHONPATH": "/app/src/system:/app/src",
    }
    labels = {
        "com.docker-tc.enabled": "1",
        "com.docker-tc.limit": "100mbit",
        "com.docker-tc.delay": "5ms",
        "com.docker-tc.loss": "3%",
    }
    services = {}
    for index, client_idx in enumerate(client_indices):
        command = [
            "python",
            "client.py",
            "--mode",
            "decentralized",
            "--client-idx",
            str(client_idx),
            "--dataset",
            dataset,
            "--rounds",
            str(rounds),
            "--prune",
            str(prune),
            "--ala",
            str(ala),
            "--exp-name",
            exp_name,
            "--model",
            model,
            "--config",
            config,
            "--strategy",
            strategy,
            "--rank",
            str(rank),
            "--paca",
            str(paca),
        ]
        command.extend(
            [
                "--execution-id",
                run_id,
                "--seed",
                str(seed),
                "--encounters-dir",
                container_encounters,
            ]
        )
        if container_outputs:
            command.extend(["--results-dir", container_outputs])
        if container_dataset:
            command.extend(["--dataset-dir", container_dataset])
        if max_epochs > 0:
            command.extend(["--max-epochs", str(max_epochs)])
        service = {
            "build": build,
            "image": "fl-client-image",
            "container_name": f"fl-client-{mode}-{index}",
            "restart": "no",
            "working_dir": "/app/src/system",
            "environment": environment,
            "volumes": [
                "./src/dataset:/app/src/dataset",
                f"./src/results:{CONTAINER_RESULTS}",
            ],
            "deploy": {
                "resources": {
                    "reservations": {
                        "devices": [{"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}]
                    }
                }
            },
            "networks": ["docker-tc"],
            "labels": labels,
            "command": command,
        }
        if mode == "v2i":
            service["depends_on"] = {"rsu-service": {"condition": "service_healthy"}}
        services[f"client-{mode}-{index}"] = service

    if mode == "v2i":
        rsu_path = Path(rsu_config or "src/v2v_fl/maps/rsu_config_marambaia.json")
        if not rsu_path.is_absolute():
            rsu_path = PROJECT_ROOT / rsu_path
        rsu_path = rsu_path.resolve()
        if not rsu_path.is_file():
            raise ValueError(f"Configuração das RSUs não encontrada: {rsu_path}")
        # Caminhos no repositório permanecem relativos para permitir mover o projeto.
        rsu_source = (
            f"./{rsu_path.relative_to(PROJECT_ROOT).as_posix()}"
            if rsu_path.is_relative_to(PROJECT_ROOT)
            else str(rsu_path)
        )
        services["rsu-service"] = {
            "build": build,
            "image": "fl-client-image",
            "container_name": "fl-rsu-service",
            "restart": "no",
            "working_dir": "/app",
            "environment": {**environment, "CUDA_VISIBLE_DEVICES": ""},
            "volumes": [
                f"./src/results:{CONTAINER_RESULTS}",
                {
                    "type": "bind",
                    "source": rsu_source,
                    "target": CONTAINER_RSU_CONFIG,
                    "read_only": True,
                },
            ],
            "networks": ["docker-tc"],
            "labels": labels,
            "command": [
                "python",
                "-m",
                "v2v_fl.rsu_server",
                "--run-id",
                run_id,
                "--host",
                "0.0.0.0",
                "--port",
                "9000",
                "--rsu-config",
                CONTAINER_RSU_CONFIG,
                "--encounters-dir",
                container_encounters,
                "--exchange-timeout",
                str(exchange_timeout),
                "--threads",
                "1",
                "--seed",
                str(seed),
            ],
            "healthcheck": {
                "test": [
                    "CMD",
                    "python",
                    "-c",
                    "import socket; "
                    "from utils.network_utils import "
                    "make_v2i_message, recv_v2i_message, send_v2i_message; "
                    "conn = socket.create_connection(('127.0.0.1', 9000), timeout=2); "
                    "send_v2i_message(conn, make_v2i_message('PING')); "
                    "response = recv_v2i_message(conn)[0]; "
                    "conn.close(); raise SystemExit(response['type'] != 'PONG')",
                ],
                "interval": "5s",
                "timeout": "3s",
                "retries": 12,
                "start_period": "10s",
            },
        }

    document = {
        "name": f"fl-{mode}",
        "services": services,
        "networks": {"docker-tc": {"driver": "bridge"}},
    }
    compose_path = PROJECT_ROOT / "docker" / COMPOSE_FILES[mode]
    compose_path.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True))
    return str(compose_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["v2v", "v2i", "both"], default="both")
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--dataset", default="Cifar10")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--prune", type=int, choices=[0, 1], default=1)
    parser.add_argument("--ala", type=int, choices=[0, 1], default=1)
    parser.add_argument("--exp-name", help="Padrão: default_exp_v2v ou default_exp_v2i")
    parser.add_argument("--max-epochs", type=int, default=0)
    parser.add_argument("--model", choices=["cnn", "clip"], default="cnn")
    parser.add_argument("--config", default="lora_clip/train_config.yml")
    parser.add_argument("--strategy", default="lora")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--paca", type=int, default=12)
    parser.add_argument("--run-id", default="v2i")
    parser.add_argument("--rsu-config", type=Path)
    parser.add_argument("--exchange-timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--encounters-dir", type=Path)
    args = parser.parse_args()
    if args.clients < 1:
        parser.error("--clients deve ser positivo.")
    modes = COMPOSE_FILES if args.mode == "both" else [args.mode]
    for mode in modes:
        try:
            path = generate(
                client_indices=list(range(args.clients)),
                dataset=args.dataset,
                dataset_dir=args.dataset_dir,
                rounds=args.rounds,
                prune=args.prune,
                ala=args.ala,
                exp_name=args.exp_name or f"default_exp_{mode}",
                max_epochs=args.max_epochs,
                model=args.model,
                config=args.config,
                strategy=args.strategy,
                rank=args.rank,
                paca=args.paca,
                mode=mode,
                run_id=args.run_id,
                rsu_config=args.rsu_config,
                exchange_timeout=args.exchange_timeout,
                seed=args.seed,
                results_dir=args.results_dir,
                encounters_dir=args.encounters_dir,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(path)


if __name__ == "__main__":
    main()
