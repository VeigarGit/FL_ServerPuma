# Responsabilidade: Ponto de entrada do cliente: cria a sessão e seleciona o fluxo
# centralizado ou P2P.
#
# Principais funções: main.

"""Client entrypoint: initialize and select the execution mode."""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import time
import logging
from fl_serverpuma.execution.client_cli import parse_args
from fl_serverpuma.execution.client_initialization import initialize_client
from fl_serverpuma.execution.centralized_client import run_centralized
from fl_serverpuma.execution.peer_client import run_decentralized

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG, format='%(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def main():
    args = parse_args()
    session = initialize_client(args)
    time.sleep(10)
    if args.mode == 'centralized':
        run_centralized(session)
    elif args.mode == 'decentralized':
        run_decentralized(session)

if __name__ == '__main__':
    main()
