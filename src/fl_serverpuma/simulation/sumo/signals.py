# Responsabilidade: Escreve sinais atômicos de encontros e aguarda marcadores de conclusão de
# épocas dos clientes.
#
# Principais funções: write_encounter_signal, wait_all_clients_epoch_done.

"""SUMO federation — signals."""

import json
import os
import tempfile
import time
from pathlib import Path
from fl_serverpuma.simulation.sumo.settings import log

def write_encounter_signal(
    encounters_dir: Path,
    encounter_data: dict,
    encounter_id: int,
) -> Path:
    """Escreve o sinal de encontro (JSON) de forma ATOMICA no volume compartilhado.

    Usa a tecnica write-to-temp + rename para garantir que os clientes
    nunca leiam um arquivo parcialmente escrito:
      1. Cria um arquivo temporario no MESMO diretorio (mesmo filesystem)
      2. Escreve o JSON completo no temporario
      3. Faz fsync para garantir que os dados estao no disco
      4. Renomeia atomicamente para o nome final (operacao atomica no Linux)

    Args:
        encounters_dir: Diretorio do volume compartilhado (src/results/encounters/)
        encounter_data: Dicionario com os dados do encontro
        encounter_id: ID numerico do encontro (usado no nome do arquivo)

    Returns:
        Path do arquivo final criado
    """
    final_path = encounters_dir / f"encounter_{encounter_id}.json"

    # Criar arquivo temporario no MESMO diretorio para garantir
    # que o rename seja atomico (mesmo filesystem)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(encounters_dir),
        prefix=".encounter_",
        suffix=".tmp",
    )
    try:
        # Escrever JSON completo no temporario
        with os.fdopen(fd, "w") as tmp_file:
            json.dump(encounter_data, tmp_file)
            # Flush do buffer Python para o buffer do SO
            tmp_file.flush()
            # Fsync para garantir que os dados foram gravados no disco
            os.fsync(tmp_file.fileno())

        # Rename atomico: os clientes nunca verao o arquivo parcial
        os.rename(tmp_path, final_path)

    except Exception:
        # Em caso de erro, limpar o arquivo temporario
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return final_path


def wait_all_clients_epoch_done(encounters_dir: Path, client_indices: list[int], round_num: int, timeout: int = 300, log=None):
    """Espera todos os clientes sinalizarem que completaram a época da rodada atual."""
    expected = [encounters_dir / f".epoch_done_client_{c}" for c in client_indices]

    # Garantir que o sinal de start_round exista para que os clientes comecem
    start_round_file = encounters_dir / f".start_round_{round_num}"
    if not start_round_file.exists():
        start_round_file.touch()

    if log:
        log.info("Aguardando todos os %d clientes completarem a época da rodada %d...", len(client_indices), round_num)

    for tick in range(timeout):
        if all(f.exists() for f in expected):
            # Limpar sinais de conclusão para a próxima rodada
            for f in expected:
                f.unlink(missing_ok=True)
            if log:
                log.info("Todos os clientes completaram a época!")
            return True

        if log and tick > 0 and tick % 15 == 0:
            count = sum(f.exists() for f in expected)
            log.info("  ... %d/%d clientes concluídos (%ds)", count, len(expected), tick)

        time.sleep(1)

    if log:
        log.warning("Timeout (%ds) aguardando clientes completarem a época!", timeout)
    return False
