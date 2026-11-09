# Responsabilidade: Seleciona dispositivo/PaCA, constrói modelo e dados, configura ALA e
# entrega a sessão pronta.
#
# Principais funções: initialize_client.

"""Initialize the client model, data, personalization and session."""

import os
import sys
import logging
import random
import torch
import torch.nn as nn
from fl_serverpuma.models.clip.config import load_config
from fl_serverpuma.models.clip.config import resolve_run_modes
from fl_serverpuma.models.clip.config import build_run_config
from fl_serverpuma.models.clip.model import build_model
from fl_serverpuma.algorithms.personalization.ala import ALA
from fl_serverpuma.algorithms.personalization.ala_lora import ALA_LoRA
from fl_serverpuma.models.cnn import SimpleModel
from fl_serverpuma.data.client_loaders import load_data
from fl_serverpuma.execution.client_session import ClientSession

logger = logging.getLogger(__name__)

def initialize_client(args):
    run_config = None

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

    device = torch.device(args.device)

    # --- Resolução do PaCA Heterogêneo ---
    if args.paca_list is not None:
        # Modo lista: valores pré-definidos por cliente (determinístico)
        paca_values = [int(x.strip()) for x in args.paca_list.split(',')]
        args.paca = paca_values[args.client_idx % len(paca_values)]
        logger.info(f"PaCA Heterogêneo (lista): Cliente {args.client_idx} usando PaCA = {args.paca}")
    elif args.random_paca:
        # Modo aleatório: sorteia uma vez no início (seed baseada no client_idx + run_id para reprodutibilidade)
        rng = random.Random(args.client_idx * 1000 + args.run_id)
        args.paca = rng.randint(args.paca_min, args.paca_max)
        logger.info(f"PaCA Heterogêneo (aleatório): Cliente {args.client_idx} sorteou PaCA = {args.paca} (intervalo [{args.paca_min}, {args.paca_max}])")

    logger.info("=== Federated Learning Client ===")
    logger.info(f"Host: {args.host}:{args.port}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Client: {args.client_idx} | PaCA: {args.paca}")
    logger.info(f"Rounds: {args.rounds}")
    logger.info(f"Batch size: {args.batch_size} | LR: {args.learning_rate}")
    logger.info("=================================")

    # Adoção de Structural Pattern Matching
    match args.model:
        case 'cnn':
            match args.dataset:
                case 'MNIST':
                    model = SimpleModel(in_features=1, num_classes=10, dim=1024)
                case 'Cifar10':
                    model = SimpleModel(in_features=args.in_features, num_classes=10, dim=args.dim)
                case 'Cifar100':
                    model = SimpleModel(in_features=args.in_features, num_classes=args.num_classes, dim=args.dim)
                case _:
                    logger.error(f"Dataset inválido: {args.dataset}")
                    sys.exit(1)
        case 'clip' | 'slm':
            if args.model == 'clip':
                config = load_config(args.config)
            else:
                from fl_serverpuma.models.slm.config import load_config as load_config_slm
                config = load_config_slm(args.config)

            if args.strategy == 'lora':
                config["model"]["lora"]["mode"] = "with_lora"
            elif args.strategy == 'adalora':
                config["model"]["lora"]["mode"] = "with_adalora"
            elif args.strategy == 'sora_with_schedule':
                config["model"]["lora"]["mode"] = "with_sora_schedule"
            elif args.strategy == 'sora_no_schedule':
                config["model"]["lora"]["mode"] = "with_sora_no_schedule"


            if "paca" not in config["model"]:
                config["model"]["paca"] = {}

            if "lora" in config["model"]:
                config["model"]["lora"]["r"] = args.rank

            if args.paca is not None and args.paca > 0:
                config["model"]["paca"]["enabled"] = True
                config["model"]["paca"]["upper_layers"] = args.paca
            else:
                config["model"]["paca"]["enabled"] = False
                config["model"]["paca"]["upper_layers"] = None

            if args.model == 'clip':
                run_mode = resolve_run_modes(config)[0]
                run_config = build_run_config(config, run_mode=run_mode)
                model = build_model(config=run_config, num_classes=args.num_classes, device=device)
            else:
                from fl_serverpuma.models.slm.config import resolve_run_modes as resolve_run_modes_slm
                from fl_serverpuma.models.slm.config import build_run_config as build_run_config_slm
                from fl_serverpuma.models.slm.model import build_model as build_model_slm
                run_mode = resolve_run_modes_slm(config)[0]
                run_config = build_run_config_slm(config, run_mode=run_mode)
                model = build_model_slm(config=run_config, num_classes=args.num_classes, device=device)

    model = model.to(device)
    loss = nn.CrossEntropyLoss()

    try:
        is_clip_flag = (args.model == 'clip')
        is_slm_flag = (args.model == 'slm')
        train_loader = load_data(args.dataset, args.client_idx, device, is_train=True, batch_size=args.batch_size, is_clip=is_clip_flag, is_slm=is_slm_flag, slm_config=run_config if is_slm_flag else None)
        test_loader = load_data(args.dataset, args.client_idx, device, is_train=False, batch_size=args.batch_size, is_clip=is_clip_flag, is_slm=is_slm_flag, slm_config=run_config if is_slm_flag else None)
        logger.info(f"Data loaded successfully - Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    except Exception as e:
        logger.exception("Error loading data")
        sys.exit(1)

    if args.model == 'clip':
        ala = ALA_LoRA(args.client_idx, loss, train_loader, 32, 80, 1.0, device)
    else:
        ala = ALA(args.client_idx, loss, train_loader, 32, 80, 2, 1.0, device)




    return ClientSession(args, device, model, train_loader, test_loader, ala, run_config)
