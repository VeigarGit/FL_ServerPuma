
import time
import os
import sys
import argparse
import logging
import random
from pathlib import Path

# Desativa checagem online redundante do Hugging Face Hub (usa cache local instantaneamente)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

import torch
import torch.nn as nn

from lora_clip.clip_setup import (
    load_config, resolve_run_modes, build_run_config, build_model
)

# --- HACK PERMITIDO: Inserindo a pasta 'src' no path do Python ---
# Nota: Esta é uma mitigação aceita enquanto o projeto não for
# completamente empacotado como biblioteca via pyproject.toml
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))
# -----------------------------------------------------------------

from ALA import ALA
from ALA_LoRA import ALA_LoRA
from model import SimpleModel

# --- CONFIGURAÇÃO DE OBSERVABILIDADE (LOGGING ENXUTO) ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)
# --------------------------------------------------------

from client_workload import (
    map_sequential_to_simplemodel as map_sequential_to_simplemodel,
    local_training as local_training,
    load_data as load_data,
    local_initialization as local_initialization,
)
from client_results import ClientMetrics, save_results as save_results
from client_centralized import run_centralized
from client_v2v import run_decentralized

def parse_args():
    # Colocar um argumento -t para servir como um for 
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='Cifar100')
    parser.add_argument('--client-idx', type=int, default=0)
    parser.add_argument('--in-features', type=int, default=3)
    parser.add_argument('--num-classes', type=int, default=100)
    parser.add_argument('--dim', type=int, default=1600)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--random-client', action='store_true')
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument("--ala", type=int, default=0)
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument('--config', type=str, default="lora_clip/train_config.yml")
    parser.add_argument('--experiments', type=int, default=1)
    parser.add_argument('--run-id', type=int, default=1)
    parser.add_argument('--exp-name', type=str, default='default_exp', help='Nome da sessão')
    parser.add_argument('--timestamp', type=str, default='', help='Timestamp para diferenciar rodadas de simulação')
    parser.add_argument('--strategy', type=str, default='lora', choices=['lora', 'adalora', 'sora_with_schedule', 'sora_no_schedule'])
    parser.add_argument('--rank', type=int, default=8, help='Rank para o SoRA/LoRA') 
    parser.add_argument('--paca', type=int, default=12, help='Número de camadas para a estratégia PaCA')
    
    # --- Parâmetros de Sparsidade (SoRA) ---
    parser.add_argument('--sparse-lr', type=float, default=None, help='Taxa de aprendizado para a poda (Softshrink)')
    parser.add_argument('--max-lambda', type=float, default=None, help='Força máxima da penalidade L1 (max_lambda)')
    
    # --- PaCA Heterogêneo ---
    parser.add_argument('--random-paca', action='store_true', help='Sorteia um valor de PaCA aleatório para este cliente')
    parser.add_argument('--paca-min', type=int, default=1, help='Valor mínimo de PaCA no sorteio aleatório')
    parser.add_argument('--paca-max', type=int, default=12, help='Valor máximo de PaCA no sorteio aleatório')
    parser.add_argument('--paca-list', type=str, default=None, help='Lista de PaCAs pré-definidos por cliente (ex: "4,6,8,12"). O cliente usa o valor na posição client_idx %% len(lista)')
    parser.add_argument('--prune', type=int, default=1, choices=[0, 1], help='Habilitar pruning adaptativo (0=sim, 1=não, default: 1)')
    parser.add_argument('--mode', type=str, default='centralized', choices=['centralized', 'decentralized'])
    parser.add_argument('--max-epochs', type=int, default=0, help='Limite de epocas no modo descentralizado (0 = infinito)')
    parser.add_argument('--delta-coding', action='store_true', help='Ativa LHDQ (Low Huffman-coded Delta Quantization) em vez de int8')
    
    # --- Otimização de Avaliação ---
    parser.add_argument('--skip-post-eval', action='store_true', help='Pula todas as avaliações pós-treino no cliente (local_test_acc e train_acc) para máxima velocidade, mantendo apenas Global Model Test Acc')
    parser.add_argument('--skip-train-eval', action='store_true', help='Pula apenas a avaliação demorada no conjunto de treino pós-treino')
    
    return parser.parse_args()
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
                from lora_slm.slm_setup import load_config as load_config_slm
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
                from lora_slm.slm_setup import resolve_run_modes as resolve_run_modes_slm
                from lora_slm.slm_setup import build_run_config as build_run_config_slm
                from lora_slm.slm_setup import build_model as build_model_slm
                run_mode = resolve_run_modes_slm(config)[0]
                run_config = build_run_config_slm(config, run_mode=run_mode)
                model = build_model_slm(config=run_config, num_classes=args.num_classes, device=device)
            
    model = model.to(device)
    loss = nn.CrossEntropyLoss()
    rs_global_acc = []
    rs_global_loss = []
    rs_ala_acc = []
    rs_local_acc = []
    rs_train_acc = []
    rs_post_p2p_acc = []
    rs_post_p2p_encounter_ids = []
    rs_p2p_data_sent_mb = []
    rs_p2p_data_sent_enc_ids = []
    
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
        
    time.sleep(10)
    metrics = ClientMetrics(
        rs_global_acc=rs_global_acc,
        rs_global_loss=rs_global_loss,
        rs_ala_acc=rs_ala_acc,
        rs_local_acc=rs_local_acc,
        rs_train_acc=rs_train_acc,
        rs_post_p2p_acc=rs_post_p2p_acc,
        rs_post_p2p_encounter_ids=rs_post_p2p_encounter_ids,
        rs_p2p_data_sent_mb=rs_p2p_data_sent_mb,
        rs_p2p_data_sent_enc_ids=rs_p2p_data_sent_enc_ids,
    )
    if args.mode == 'centralized':
        run_centralized(
            args, model, train_loader, test_loader, ala,
            run_config if args.model in ['clip', 'slm'] else None, metrics,
        )
    elif args.mode == 'decentralized':
        run_decentralized(args, model, train_loader, test_loader, ala, metrics)

if __name__ == '__main__':
    main()
