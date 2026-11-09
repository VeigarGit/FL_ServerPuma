# Responsabilidade: Inicia e encerra os contêineres do experimento por Docker Compose.
#
# Principais funções: docker_compose_up, docker_compose_down.

"""SUMO federation — containers."""

import subprocess
from pathlib import Path
from fl_serverpuma.simulation.sumo.settings import log

def docker_compose_up(compose_file: Path, project_root: Path, build: bool = False) -> bool:
    """Inicia todos os conteineres (veiculos) de uma vez no inicio da simulacao.

    Args:
        compose_file: Caminho para o docker-compose.v2v.yml
        project_root: Raiz do projeto (FL_ServerPuma/)
        build: Se True, forca o rebuild das imagens (adiciona --build)

    Returns:
        True se o comando foi bem sucedido, False caso contrario
    """
    cmd = ["docker", "compose", "--project-directory", ".", "-f", str(compose_file), "up", "-d"]
    if build:
        cmd.insert(-1, "--build")
        log.info("Iniciando conteineres Docker...")
    else:
        log.info("Iniciando conteineres Docker...")

    result = subprocess.run(
        cmd,
        cwd=str(project_root),
    )
    if result.returncode != 0:
        log.error("docker compose up falhou (exit code %d)", result.returncode)
        return False
    log.info("Conteineres iniciados com sucesso!")
    return True


def docker_compose_down(compose_file: str, project_root: Path) -> None:
    """Destroi os conteineres ao final da simulacao.

    Usa --remove-orphans para limpar conteineres de execucoes anteriores
    e -v para remover volumes anonimos.
    """
    log.info("Encerrando conteineres Docker...")
    subprocess.run(
        ["docker", "compose", "--project-directory", ".", "-f", compose_file, "down", "--remove-orphans", "-v"],
        cwd=str(project_root),
        capture_output=True, text=True, timeout=120,
    )
    log.info("Conteineres removidos.")
