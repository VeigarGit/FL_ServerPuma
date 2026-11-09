# Responsabilidade: Centraliza caminhos, constantes e configuração de log da orquestração V2V.
#
# Contém configurações ou processamento no nível do módulo; não define funções próprias.

"""SUMO federation — settings."""

from fl_serverpuma.paths import SOURCE_ROOT
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger("v2v_orchestrator")

DEFAULT_COMM_RADIUS = 300.0

DEFAULT_MIN_CLIENTS = 2

DEFAULT_STEP_LENGTH = 1.0

DEFAULT_MIN_CONTACT_TIME = 6.0

MAX_ETC = 120.0

DEFAULT_WARMUP = 60

DEFAULT_COOLDOWN = 15

PROJECT_ROOT = SOURCE_ROOT.parent



