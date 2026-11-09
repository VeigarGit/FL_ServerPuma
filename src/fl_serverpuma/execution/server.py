# Responsabilidade: Ponto de entrada do servidor: lê a CLI, seleciona dispositivo e inicia os
# experimentos.
#
# Principais funções: main.

"""Server entrypoint: device selection and experiment lifecycle."""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import logging
import torch
from fl_serverpuma.execution.server_cli import parse_args
from fl_serverpuma.execution.round_coordinator import run_server
from fl_serverpuma.execution.server_state import ServerState

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG, format='%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def main():
    args = parse_args()

    # Configuração inteligente de device: MPS > CUDA > CPU
    if args.device == "mps":
        if torch.backends.mps.is_available():
            logger.info("Using Metal Performance Shaders (MPS) for MacBook GPU acceleration.")
        else:
            logger.warning("MPS is not available. Falling back to CUDA if available, otherwise CPU.")
            if torch.cuda.is_available():
                args.device = "cuda"
            else:
                args.device = "cpu"
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Checking for MPS...")
            if torch.backends.mps.is_available():
                logger.info("Falling back to Metal Performance Shaders (MPS).")
                args.device = "mps"
            else:
                logger.warning("CUDA and MPS not available. Using CPU.")
                args.device = "cpu"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    logger.info(f"Device set to: {args.device}")

    for i in range(args.experiments):
        logger.info(f"\n=== Iniciando Experimento {i+1}/{args.experiments} ===")
        server = ServerState(args)
        run_server(server, i)

if __name__ == '__main__':
    main()
