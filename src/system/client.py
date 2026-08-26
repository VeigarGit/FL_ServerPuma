
import socket
import pickle
import struct
import copy
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

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import h5py

from lora_clip.clip_setup import (
    load_config, resolve_run_modes, build_run_config, build_model
)

from lora_clip.clip_setup import build_optimizer
from lora_clip.sora import get_trainable_state_dict

# --- HACK PERMITIDO: Inserindo a pasta 'src' no path do Python ---
# Nota: Esta é uma mitigação aceita enquanto o projeto não for
# completamente empacotado como biblioteca via pyproject.toml
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))
# -----------------------------------------------------------------

from utils.data_utils import read_client_data
from prunning import prune_and_restructure
from ALA import ALA
from ALA_LoRA import ALA_LoRA
from model import SimpleModel
from utils.network_utils import send_data, recv_data
from utils.fl_math import quantization, dequantization, evaluate_model, set_parameters
from utils.fl_math import resize_model_to_pruned
from utils.lhdq import lhdq_encode, lhdq_decode, is_lhdq_encoded

# --- CONFIGURAÇÃO DE OBSERVABILIDADE (LOGGING ENXUTO) ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)
# --------------------------------------------------------

def map_sequential_to_simplemodel(state_dict):
    mapped_dict = {}
    mapping = {
        '0.weight': 'conv1.0.weight',
        '0.bias': 'conv1.0.bias',
        '3.weight': 'conv2.0.weight', 
        '3.bias': 'conv2.0.bias',
        '7.weight': 'fc1.0.weight',
        '7.bias': 'fc1.0.bias',
        '9.weight': 'fc.weight',
        '9.bias': 'fc.bias'
    }
    for sequential_key, simple_key in mapping.items():
        if sequential_key in state_dict:
            mapped_dict[simple_key] = state_dict[sequential_key]
    return mapped_dict

# def dequantization(global_state):
#     dequantized_state_dict = {}
#     for k, v in global_state.items():
#         if isinstance(v, dict) and v.get('dtype') == 'quantized_int8':
#             scale = v['scale']
#             dequantized_state_dict[k] = v['weights'].float() * scale
#         else:
#             dequantized_state_dict[k] = v
#     return dequantized_state_dict

# def quantization(state_dict):
#     quantized_state_dict = {}
#     for k, v in state_dict.items():
#         if isinstance(v, torch.Tensor):
#             scale = torch.max(torch.abs(v)) / 127.0
#             quantized_weights = torch.clamp((v / scale).round(), -128, 127).to(torch.int8)
#             quantized_state_dict[k] = {
#                 'dtype': 'quantized_int8',
#                 'scale': scale,
#                 'weights': quantized_weights
#             }
#         else:
#             quantized_state_dict[k] = v
#     return quantized_state_dict

def local_training(model, state_dict, prune, train_loader, test_loader, learning_rate=0.01, round_num=2, alaarg=1, ala=None, model_type='cnn', run_config=None, model_lock=None):
    # Verifica se os pesos precisam ser desquantizados (se não forem tensores)
    if state_dict and not isinstance(next(iter(state_dict.values())), torch.Tensor):
        state_dict = dequantization(state_dict)
    
    personalized_acc = None  # Inicializa a variável que guardará a acurácia pós-ALA
    
    if round_num >= 2 and prune == 1:
        if model_type == 'cnn':
            state_dict = map_sequential_to_simplemodel(state_dict)
    # === CORREÇÃO ESTRUTURAL PARA CLIP E SLM ===
    if model_type in ['clip', 'slm']:
        # FedALA_LoRA só se aplica a CLIP (model_type != 'slm').
        # Evita deepcopy desnecessário do SLM inteiro (~4-6 GB).
        needs_ala = (alaarg == 0 and round_num >= 2 and model_type != 'slm')
        
        if needs_ala:
            global_model = copy.deepcopy(model)
            global_model = resize_model_to_pruned(global_model, state_dict)
            logger.info(f"Client {ala.cid}: Applying FedALA_LoRA (Adaptive Local Aggregation)...")
            local_initialization(ala, global_model, model)
            personalized_acc, _ = evaluate_model(model, test_loader)
            logger.info(f"Client {ala.cid}: Post-ALA Test Accuracy: {personalized_acc:.2f}%")
            del global_model
        else:
            resize_model_to_pruned(model, state_dict)
        
    else:
        # --- LÓGICA ANTIGA PARA CNN MANTIDA INTACTA ---
        local_model = resize_model_to_pruned(model, state_dict)
        state = copy.deepcopy(local_model)
        
        if alaarg == 0:
            # Aviso: Este código da CNN continua bugado (round_num == 2)
            if round_num == 2:
                logger.info(f"Client {ala.cid}: Applying FedALA (Adaptive Local Aggregation)...")
                local_initialization(ala, state, model)
            
        set_parameters(model, state)
        
        if alaarg == 0 and round_num >= 2:
            personalized_acc, _ = evaluate_model(model, test_loader)
            logger.info(f"Client {ala.cid}: Post-ALA Test Accuracy: {personalized_acc:.2f}%")
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    device = next(model.parameters()).device
    
    # --- 1. SELEÇÃO DE OTIMIZADOR ---
    if model_type in ['clip', 'slm']:
        if model_type == 'clip':
            from lora_clip.clip_setup import build_optimizer, train_epoch, update_paca_runtime
        else:
            from lora_slm.slm_setup import build_optimizer, train_epoch, update_paca_runtime
        
        # --- PaCA RUNTIME UPDATE ---
        paca_config = run_config["model"].get("paca", {})
        if paca_config.get("enabled"):
            update_paca_runtime(model, paca_config.get("upper_layers"))
        
        # --- DIAGNÓSTICO: Contagem de parâmetros treináveis (após PaCA update) ---
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if not hasattr(model, '_last_trainable_count') or model._last_trainable_count != trainable_count:
            logger.info(f"[DIAG] Parâmetros treináveis neste round: {trainable_count:,}")
            model._last_trainable_count = trainable_count
            
        # VRAM OPTIMIZATION: Habilita Gradient Checkpointing para SLM (Ação 1)
        if model_type == 'slm':
            if hasattr(model, "slm_model") and hasattr(model.slm_model, "gradient_checkpointing_enable"):
                model.slm_model.gradient_checkpointing_enable()
                # logger.debug("Gradient Checkpointing ativado no SLM (Redução de VRAM).")
        
        optimizer, sparse_optimizer = build_optimizer(model, run_config)
        is_sora = run_config["model"]["lora"]["mode"] in ["with_sora_no_schedule", "with_sora_schedule"]
        sparse_lambda = run_config["model"].get("sora", {}).get("sparse_lambda", 0.0) if is_sora else 0.0
        
        if model_type == 'slm':
            accumulation_steps = run_config.get("training", {}).get("gradient_accumulation_steps", 1)
            metrics = train_epoch(model, train_loader, optimizer, sparse_optimizer, sparse_lambda, accumulation_steps)
        else:
            metrics = train_epoch(model, train_loader, optimizer, sparse_optimizer, sparse_lambda)
        
        logger.info(f"Métricas do Treino Local - CE Loss: {metrics['ce_loss']:.4f} | Sparse Loss: {metrics['sparse_loss']:.4f}")
    else:
        # --- DIAGNÓSTICO: Contagem de parâmetros treináveis ---
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if not hasattr(model, '_last_trainable_count') or model._last_trainable_count != trainable_count:
            logger.info(f"[DIAG] Parâmetros treináveis neste round: {trainable_count:,}")
            model._last_trainable_count = trainable_count
        
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        sparse_optimizer = None

        model_dtype = next(model.parameters()).dtype
        
        for x, y in train_loader:
            if x.is_floating_point():
                x = x.to(device, dtype=model_dtype)
            else:
                x = x.to(device)
            y = y.to(device)
            
            # Adquirir lock por batch para permitir interrupcao imediata
            # pela thread de encontro V2V. O lock e liberado apos cada
            # batch, dando uma janela para a thread watcher reagir.
            if model_lock:
                model_lock.acquire()
            try:
                optimizer.zero_grad()
                if sparse_optimizer is not None:
                    sparse_optimizer.zero_grad()
                    
                output = model(x)
                if isinstance(output, dict):
                    output = output["logits"]
                    
                loss = loss_fn(output, y)
                loss.backward()
                optimizer.step()
                
                if sparse_optimizer is not None:
                    sparse_optimizer.step()
            finally:
                if model_lock:
                    model_lock.release()
    
    # --- 2. SELEÇÃO DOS PESOS QUE SERÃO ENVIADOS PARA O SERVIDOR ---
    if model_type in ['clip', 'slm']:
        if model_type == 'clip':
            from lora_clip.sora import get_trainable_state_dict
        else:
            from lora_slm.sora import get_trainable_state_dict
        return get_trainable_state_dict(model), personalized_acc 
    else:
        return model.state_dict(), personalized_acc 



import torchvision.transforms as T
from PIL import Image

def load_data(dataset, client_idx, device, is_train=True, batch_size=32, is_clip=False, is_slm=False, slm_config=None):
    train_data = read_client_data(dataset, client_idx, is_train)
    X, y = zip(*train_data)
    X = torch.stack(X)
    
    if is_slm:
        from lora_slm.slm_setup import CustomCollator
        from transformers import AutoProcessor
        
        processor = AutoProcessor.from_pretrained(slm_config["model"]["name"])
        label_to_idx = {i: i for i in range(1000)}
        label_to_idx.update({str(i): i for i in range(1000)})
        collate_fn = CustomCollator(processor, label_to_idx)
        
        slm_data = []
        for x_tensor, y_tensor in zip(X, y):
            x_tensor = x_tensor.cpu().float()
            
            # Se o tensor vier com normalização do CLIP (valores negativos < -0.1)
            if x_tensor.min() < -0.1:
                clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1)
                clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1)
                if x_tensor.shape[0] == 3:
                    x_tensor = x_tensor * clip_std + clip_mean
                else:
                    x_tensor = x_tensor * 0.5 + 0.5
            elif x_tensor.min() < 0:
                # Normalização padrão [-1, 1]
                x_tensor = x_tensor * 0.5 + 0.5
                
            if x_tensor.max() <= 1.0:
                x_tensor = x_tensor * 255.0
                
            # Clamp de segurança para garantir [0, 255]
            x_tensor = torch.clamp(x_tensor, 0, 255)
            x_numpy = x_tensor.byte().numpy()
            
            if x_numpy.shape[0] in [1, 3]: # (C, H, W) -> (H, W, C)
                x_numpy = np.transpose(x_numpy, (1, 2, 0))
            if x_numpy.shape[-1] == 1:
                x_numpy = np.squeeze(x_numpy, axis=-1)
                
            pil_img = Image.fromarray(x_numpy)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            
            slm_data.append({"image": pil_img, "label": y_tensor.item()})
            
        pin_mem = True if hasattr(device, 'type') and device.type == 'cuda' else False
        return DataLoader(slm_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=pin_mem)

    # Redimensionamento e normalização vitais para o CLIP
    if is_clip:
        if X.shape[-2:] != (224, 224):
            resize = T.Resize((224, 224), antialias=True)
            X = resize(X)
        if not X.is_floating_point():
            X = X.float() / 255.0
        elif X.max() > 1.0:
            X = X / 255.0
        # Se os dados não tiverem normalização do CLIP, aplica a normalização oficial do CLIP
        if X.min() >= 0.0:
            normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
            X = normalize(X)
        
    # Garante que o rótulo é inteiro longo e move os dados INTEIROS para a GPU
    X = X.to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)
    
    dataset = torch.utils.data.TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)



def save_results(args, rs_global_acc, rs_global_loss, rs_ala_acc, rs_local_acc, rs_train_acc, idx=0, argalgo=0):
    b = "FedALA" if argalgo == 0 else "FedAVG"
    
    paca_val = args.paca if (isinstance(args.paca, (int, float)) and not isinstance(args.paca, bool) and args.paca > 0) else 0
    rank_val = args.rank if (isinstance(args.rank, (int, float)) and not isinstance(args.rank, bool) and args.rank > 0) else 8
    ts_suffix = f"_{args.timestamp}" if hasattr(args, 'timestamp') and args.timestamp else ""
    algo = f"client_{idx}_{args.dataset}_{args.strategy}_rank{rank_val}_paca{paca_val}_{b}_run{args.run_id}{ts_suffix}"
    
    current_dir = Path(__file__).resolve().parent
    result_path = current_dir.parent / "results" / args.exp_name
    result_path.mkdir(parents=True, exist_ok=True)
    
    file_path = result_path / f"{algo}.h5"
    logger.info(f"Salvando resultados em: {file_path}")
    
    with h5py.File(file_path, 'w') as hf:
        hf.create_dataset('rs_global_acc', data=rs_global_acc)
        hf.create_dataset('rs_global_loss', data=rs_global_loss)
        if rs_ala_acc is not None and len(rs_ala_acc) > 0:
            hf.create_dataset('rs_ala_acc', data=rs_ala_acc)
        hf.create_dataset('rs_local_acc', data=rs_local_acc)
        hf.create_dataset('rs_train_acc', data=rs_train_acc)
        hf.attrs['paca_used'] = paca_val

def local_initialization(ala, received_global_model, model, mask=None):
    ala.adaptive_local_aggregation(received_global_model, model, mask=mask)

# def resize_model_to_pruned(model, pruned_dict):
#     """ Redimensiona o modelo existente para as dimensões podadas """
#     with torch.no_grad():
#         for name, param in model.named_parameters():
#             if name in pruned_dict:
#                 pruned_weight = pruned_dict[name]
                
#                 if param.shape != pruned_weight.shape:
#                     logger.debug(f"Redimensionando {name}: {param.shape} -> {pruned_weight.shape}")
#                     new_param = nn.Parameter(pruned_weight.to(param.device))
                    
#                     if '.' in name:
#                         parts = name.split('.')
#                         module = model
#                         for part in parts[:-1]:
#                             module = getattr(module, part)
#                         setattr(module, parts[-1], new_param)
#                         if hasattr(module, 'r') and 'lora_A' in name:
#                             module.r = pruned_weight.shape[0]
#                     else:
#                         setattr(model, name, new_param)
#                 else:
#                     param.data.copy_(pruned_weight.to(param.device))
#     return model

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
    
    if args.mode == 'centralized':
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            connected = False
            for attempt in range(10):
                try:
                    s.connect((args.host, args.port))
                    connected = True
                    break
                except OSError as e:
                    logger.warning(f"Connection failed, retrying in 5 seconds... (Attempt {attempt+1}/10)")
                    time.sleep(5)
                    
            if not connected:
                logger.exception("Connection failed after multiple attempts")
                sys.exit(1)
                
            # Timeout de segurança: evita que o client fique preso
            # indefinidamente caso o server feche a conexão (race condition
            # no último round)
            if args.model == 'slm':
                s.settimeout(1800)
            else:
                s.settimeout(120)
            
            # LHDQ: cache do estado anterior (para delta coding)
            previous_recv_state = {}  # estado global recebido na rodada anterior
            previous_sent_state = {}  # update enviado na rodada anterior
            
            send_data(s, args.client_idx)
            logger.info(f"Connected to server {args.host}:{args.port}")
            for round_num in range(args.rounds):
                logger.info(f"\n--- Round {round_num + 1}/{args.rounds} ---")
                
                # --- Recebe PaCA e Rank dinâmicos do servidor (CLIP e SLM) ---
                if args.model in ['clip', 'slm']:
                    server_paca, _ = recv_data(s)
                    if server_paca is None or server_paca == "end":
                        logger.warning(f"Sinal de término ou desconexão recebido ao aguardar PaCA ({server_paca}). Encerrando cliente.")
                        break
                    if isinstance(server_paca, int) and not isinstance(server_paca, bool):
                        if server_paca != args.paca:
                            logger.info(f"PaCA adaptativo: servidor ajustou {args.paca} -> {server_paca}")
                            args.paca = server_paca
                            if "paca" in run_config.get("model", {}):
                                run_config["model"]["paca"]["upper_layers"] = server_paca
                    else:
                        logger.warning(f"Valor inesperado de PaCA recebido: {server_paca} ({type(server_paca)}). Mantendo PaCA={args.paca}")

                    server_r, _ = recv_data(s)
                    if server_r is None or server_r == "end":
                        logger.warning(f"Sinal de término ou desconexão recebido ao aguardar Rank ({server_r}). Encerrando cliente.")
                        break
                    if isinstance(server_r, int) and not isinstance(server_r, bool):
                        if server_r != args.rank:
                            logger.info(f"r adaptativo: servidor ajustou {args.rank} -> {server_r}")
                            args.rank = server_r
                            if "lora" in run_config.get("model", {}):
                                run_config["model"]["lora"]["r"] = server_r
                    else:
                        logger.warning(f"Valor inesperado de Rank recebido: {server_r} ({type(server_r)}). Mantendo Rank={args.rank}")

                global_state, _ = recv_data(s)
                if global_state is None or global_state == "end" or not isinstance(global_state, dict):
                    logger.warning(f"Sinal de término ou payload inesperado recebido ({type(global_state)}): {global_state}. Encerrando cliente com segurança.")
                    break

                prune, _ = recv_data(s)
                if prune is None or prune == "end":
                    logger.warning(f"Sinal de término ou desconexão recebido ao aguardar Prune ({prune}). Encerrando cliente.")
                    break
                    
                logger.info("Received global model.")
                client_dequant_start = time.time()
                if is_lhdq_encoded(global_state):
                    global_state = lhdq_decode(global_state, previous_recv_state)
                else:
                    global_state = dequantization(global_state)
                client_dequant_time = time.time() - client_dequant_start
                
                # LHDQ: cachear o estado global recebido (já decodificado) para a próxima rodada
                if args.delta_coding:
                    previous_recv_state = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in global_state.items()}
                
                
                if round_num + 1 >= 2 and prune == 1 and args.model == 'cnn':
                    ammount, _ = recv_data(s)
                
                eval_start = time.time()
                if round_num + 1 >= 2:
                    if args.model in ['clip', 'slm']:
                        old_local_weights = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

                    local = resize_model_to_pruned(model, global_state)
                    test_accuracy, test_loss = evaluate_model(local, test_loader)
                    
                    if args.model in ['clip', 'slm']:
                        for n, p in model.named_parameters():
                            if p.requires_grad and n in old_local_weights:
                                if p.data.shape == old_local_weights[n].shape:
                                    p.data.copy_(old_local_weights[n])
                else:
                    test_accuracy, test_loss = evaluate_model(model, test_loader)
                client_pre_eval_time = time.time() - eval_start
                    
                logger.info(f"Client {args.client_idx}: Global Model Test Accuracy: {test_accuracy:.2f}% | Test Loss: {test_loss:.4f}")
                rs_global_acc.append(test_accuracy)
                rs_global_loss.append(test_loss)

            
                
                # --- Problema #1: Mede o tempo puro de treino do cliente ---
                train_start = time.time()
                updated_state, personalized_acc = local_training(
                    model=model, 
                    state_dict=global_state, 
                    prune=prune, 
                    train_loader=train_loader, 
                    test_loader=test_loader,
                    learning_rate=args.learning_rate, 
                    round_num=round_num + 1, 
                    alaarg=args.ala, 
                    ala=ala,
                    model_type=args.model, 
                    run_config=run_config if args.model in ['clip', 'slm'] else None
                )
                client_training_time = time.time() - train_start
                
                # Se o FedALA calculou a acurácia, nós salvamos ela; se não (ex: round 1 ou ala desativado), salvamos 0.0 para manter o array com mesmo tamanho
                if personalized_acc is not None:
                    rs_ala_acc.append(personalized_acc)
                else:
                    # Se o ALA não rodou, salvamos a acurácia global para a linha do gráfico não cair
                    rs_ala_acc.append(test_accuracy)

                logger.info(f"Local training completed in {client_training_time:.2f}s.")

                # Avaliações pós-treino (opcionalmente puladas para acelerar SLM/LLM)
                post_eval_start = time.time()
                if not args.skip_post_eval:
                    local_test_acc, local_test_loss = evaluate_model(model, test_loader)
                    rs_local_acc.append(local_test_acc)
                    
                    if not args.skip_train_eval:
                        train_accuracy, train_loss = evaluate_model(model, train_loader)
                        rs_train_acc.append(train_accuracy)
                    else:
                        train_accuracy = 0.0
                        rs_train_acc.append(0.0)
                        
                    logger.info(f"Client {args.client_idx}: Post-Training Test Accuracy: {local_test_acc:.2f}% | Training Accuracy: {train_accuracy:.2f}%")
                else:
                    local_test_acc = test_accuracy
                    train_accuracy = 0.0
                    rs_local_acc.append(test_accuracy)
                    rs_train_acc.append(0.0)
                    logger.info(f"Client {args.client_idx}: Avaliação pós-treino pulada (--skip-post-eval). Usando Global Test Acc={test_accuracy:.2f}%")
                client_post_eval_time = time.time() - post_eval_start
                
                quant_start = time.time()
                use_lhdq = (args.delta_coding and round_num >= 1 and previous_sent_state)
                if use_lhdq:
                    # LHDQ: cachear antes de codificar
                    current_plain_state = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in updated_state.items()}
                    updated_state = lhdq_encode(updated_state, previous_sent_state)
                    previous_sent_state = current_plain_state
                else:
                    if args.delta_coding:
                        previous_sent_state = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in updated_state.items()}
                    updated_state = quantization(updated_state)
                client_quant_time = time.time() - quant_start
                
                # Tempo total de avaliação (pré + pós treinamento) e dequantização no cliente
                client_eval_time = client_pre_eval_time + client_post_eval_time
                client_processing_overhead = client_eval_time + client_quant_time + client_dequant_time
                
                logger.info(f"Client {args.client_idx}: Tempos - treino={client_training_time:.2f}s, eval={client_eval_time:.2f}s, quant={client_quant_time:.2f}s, dequant={client_dequant_time:.2f}s")
                
                try:
                    send_data(s, updated_state)
                    send_data(s, len(train_loader.dataset))
                    send_data(s, args.ala)
                    send_data(s, client_training_time)
                    
                    # Envia as métricas do modelo global calculadas no cliente
                    send_data(s, test_accuracy)
                    send_data(s, test_loss)
                    
                    # Envia tempos de processamento do cliente para medição correta de comm_time
                    send_data(s, client_processing_overhead)
                    
                    logger.info("Client update sent.")
                    
                    if args.model == 'slm':
                        round_ack, _ = recv_data(s)
                        if round_ack is None:
                            logger.warning("Conexão fechada pelo servidor ou timeout ao aguardar confirmação de round. Finalizando.")
                            break
                    else:
                        s.recv(3)
                    logger.info("Ready for next round...")
                except (OSError, BrokenPipeError, ConnectionResetError, socket.timeout) as e:
                    logger.warning(f"Conexão fechada pelo servidor ou timeout ({type(e).__name__}). Finalizando.")
                    break
                    
        save_results(args, rs_global_acc, rs_global_loss, rs_ala_acc, rs_local_acc, rs_train_acc, idx=args.client_idx, argalgo=args.ala)
        logger.info("\nTraining completed!")

    elif args.mode == 'decentralized':
        import json
        import threading

        logger.info(f"Client {args.client_idx}: Modo Descentralizado (D-PSGD) Ativado.")

        # Caminho para a pasta de sinalizacao P2P (volume compartilhado Docker).
        encounters_dir = Path(__file__).resolve().parents[1] / "results" / "encounters"

        # ── Estado compartilhado entre threads ────────────────────────────
        # Lock para sincronizar acesso ao modelo entre treino e troca P2P.
        # No local_training da CNN, e adquirido/liberado a cada batch,
        # permitindo que a thread watcher interrompa o treino quase
        # instantaneamente (espera no maximo a duracao de 1 batch).
        model_lock = threading.Lock()

        # Dicionario que armazena pesos recebidos via TCP, indexado por
        # (encounter_id, sender_client_id). Protegido pelo in_memory_lock.
        in_memory_weights = {}
        in_memory_lock = threading.Lock()

        # Contador thread-safe do ultimo encontro processado.
        last_processed_encounter = 0
        encounter_counter_lock = threading.Lock()

        # ── Thread P2P Server (TCP Listener) ──────────────────────────────
        # Roda em background aceitando conexoes TCP dos vizinhos.
        # Armazena os pesos recebidos em in_memory_weights para a thread
        # watcher consumir. NAO toca no modelo diretamente.
        def p2p_server_thread():
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(('0.0.0.0', 9000))
            server_socket.listen(50)
            logger.info(f"Client {args.client_idx}: P2P Server listening on port 9000")

            while True:
                try:
                    conn, addr = server_socket.accept()
                    data, _ = recv_data(conn)
                    if data is not None:
                        sender_id, enc_id, state = data
                        with in_memory_lock:
                            if enc_id not in in_memory_weights:
                                in_memory_weights[enc_id] = {}
                            in_memory_weights[enc_id][sender_id] = state
                        logger.info(f"Client {args.client_idx}: Recebeu pesos do Cliente {sender_id} para encontro {enc_id}")
                    conn.close()
                except Exception as e:
                    logger.error(f"Erro no P2P Server: {e}")

        threading.Thread(target=p2p_server_thread, daemon=True).start()

        # ── Thread Encounter Watcher ──────────────────────────────────────
        # Roda em background fazendo polling da pasta de encontros a cada 1s.
        # Quando detecta um encontro novo para este cliente:
        #   1. Adquire model_lock (espera no maximo 1 batch do treino)
        #   2. Faz snapshot dos pesos atuais do modelo
        #   3. Libera model_lock (treino retoma)
        #   4. Envia snapshot para vizinhos via TCP
        #   5. Aguarda pesos dos vizinhos (poll_timeout = ETC - margem)
        #   6. Agrega via media D-PSGD
        #   7. Adquire model_lock e aplica os pesos agregados no modelo
        #   8. Escreve .done_enc_* para destravar o orquestrador
        def encounter_watcher_thread():
            nonlocal last_processed_encounter

            while True:
                time.sleep(1)  # Polling de 1 segundo

                if not encounters_dir.exists():
                    continue

                encounter_files = list(encounters_dir.glob("encounter_*.json"))
                if not encounter_files:
                    continue

                # Funcao auxiliar para extrair o ID do encontro do JSON
                def extract_id(filepath):
                    try:
                        with open(filepath, "r") as json_f:
                            return json.load(json_f).get("encounter_id", 0)
                    except Exception:
                        return 0

                encounter_files.sort(key=extract_id)

                for ef in encounter_files:
                    try:
                        with open(ef, "r") as f:
                            data = json.load(f)

                        enc_id = data.get("encounter_id", 0)
                        clients_in_encounter = data.get("clients", [])
                        etc_seconds = data.get("etc_seconds", 60)

                        with encounter_counter_lock:
                            if enc_id <= last_processed_encounter:
                                continue
                            if args.client_idx not in clients_in_encounter:
                                continue

                        if isinstance(etc_seconds, str) and etc_seconds == "Carros Parados":
                            poll_timeout = 115 # MAX_ETC (120) - 5
                        else:
                            poll_timeout = max(5, int(etc_seconds - 5))

                        etc_display = f"{etc_seconds}" if isinstance(etc_seconds, str) else f"{etc_seconds:.1f}"
                        logger.info(
                            f"Client {args.client_idx}: [WATCHER] Encontro {enc_id} detectado! "
                            f"ETC={etc_display}s, timeout={poll_timeout}s. "
                            f"Pausando treino para D-PSGD."
                        )

                        # ── SNAPSHOT: Adquirir lock e copiar pesos ─────────
                        # O treino libera o lock a cada batch, entao esta
                        # aquisicao espera no maximo a duracao de 1 batch.
                        with model_lock:
                            if args.model == 'clip':
                                snapshot = get_trainable_state_dict(model)
                            else:
                                snapshot = copy.deepcopy(model.state_dict())
                        # Lock liberado: treino pode continuar enquanto
                        # fazemos a troca P2P pela rede.

                        logger.info(f"Client {args.client_idx}: [WATCHER] Snapshot dos pesos capturado. Treino retomado.")

                        # ── ENVIO DE PESOS (TCP) ──────────────────────────
                        other_clients = [c for c in clients_in_encounter if c != args.client_idx]

                        for c in other_clients:
                            target_host = f"fl-client-v2v-{c}"
                            sent = False
                            for attempt in range(poll_timeout):
                                try:
                                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                                        s.settimeout(2.0)
                                        s.connect((target_host, 9000))
                                        send_data(s, (args.client_idx, enc_id, snapshot))
                                        sent = True
                                        break
                                except Exception as e:
                                    if attempt % 10 == 0:
                                        logger.warning(
                                            f"Client {args.client_idx}: Tentativa {attempt+1}/{poll_timeout} "
                                            f"para Cliente {c} falhou: {e}"
                                        )
                                    time.sleep(1)
                            if not sent:
                                logger.warning(f"Client {args.client_idx}: Falha ao enviar para Cliente {c} no encontro {enc_id}")

                        # ── POLLING dos pesos dos vizinhos na MEMÓRIA ─────
                        logger.info(f"Client {args.client_idx}: [WATCHER] Aguardando vizinhos {other_clients} (timeout={poll_timeout}s)...")

                        all_received = False
                        other_states = []

                        for tick in range(poll_timeout):
                            with in_memory_lock:
                                enc_weights = in_memory_weights.get(enc_id, {})
                                if all(c in enc_weights for c in other_clients):
                                    all_received = True
                                    other_states = [enc_weights[c] for c in other_clients]
                                    break
                            time.sleep(1)

                        # ── AGREGACAO D-PSGD (Media Simples) ──────────────
                        if all_received:
                            logger.info(
                                f"Client {args.client_idx}: [WATCHER] Pesos de {len(other_states)} "
                                f"vizinhos recebidos! Aplicando consenso (Media)."
                            )

                            avg_state = {}
                            for key in snapshot.keys():
                                avg_state[key] = snapshot[key].clone()
                                for neighbor_state in other_states:
                                    avg_state[key] += neighbor_state[key].to(device)
                                avg_state[key] = avg_state[key] / (1 + len(other_states))

                            # Adquirir lock para aplicar pesos agregados no modelo
                            with model_lock:
                                if args.model == 'clip':
                                    resize_model_to_pruned(model, avg_state)
                                else:
                                    model.load_state_dict(avg_state, strict=False)

                            logger.info(f"Client {args.client_idx}: [WATCHER] Agregacao P2P do encontro {enc_id} concluida!")

                            # Limpar memoria
                            with in_memory_lock:
                                if enc_id in in_memory_weights:
                                    del in_memory_weights[enc_id]

                            # Sinalizar ao orquestrador
                            done_file = encounters_dir / f".done_enc_{enc_id}_client_{args.client_idx}"
                            done_file.touch()
                        else:
                            # ── FALLBACK: timeout expirado ────────────────
                            logger.warning(
                                f"Client {args.client_idx}: [WATCHER] Timeout no encontro {enc_id}. "
                                f"Descartando rodada P2P."
                            )
                            # Sinalizar ao orquestrador mesmo no timeout
                            # para nao travar a simulacao por 300s
                            done_file = encounters_dir / f".done_enc_{enc_id}_client_{args.client_idx}"
                            done_file.touch()

                        # Marcar encontro como processado
                        with encounter_counter_lock:
                            last_processed_encounter = enc_id

                    except Exception as e:
                        logger.error(f"Erro processando encontro {ef}: {e}")

        threading.Thread(target=encounter_watcher_thread, daemon=True).start()

        # ── LOOP PRINCIPAL: TREINO CONTÍNUO ───────────────────────────────
        # O treino roda continuamente. A thread watcher cuida dos encontros
        # em paralelo, interrompendo o treino por batches conforme necessario.
        epoch = 1
        max_epochs = args.max_epochs if args.max_epochs > 0 else float('inf')
        while epoch <= max_epochs:
            logger.info(f"\n--- Iniciando Epoca {epoch}{f'/{args.max_epochs}' if args.max_epochs > 0 else ''} ---")

            # Obter o state_dict atual do modelo para usar como base do treino.
            # Para CLIP/SLM, usa apenas os parametros treinaveis (LoRA/SoRA).
            if args.model in ['clip', 'slm']:
                if args.model == 'slm':
                    from lora_slm.sora import get_trainable_state_dict as get_trainable_state_dict_slm
                    dummy_state = get_trainable_state_dict_slm(model)
                else:
                    from lora_clip.sora import get_trainable_state_dict as get_trainable_state_dict_clip
                    dummy_state = get_trainable_state_dict_clip(model)
            else:
                dummy_state = model.state_dict()

            # ── TREINO LOCAL ──────────────────────────────────────────────
            # Acontece SEMPRE, independente de haver encontro ou nao.
            # O model_lock e passado para que a CNN libere a trava a cada
            # batch, permitindo a thread watcher interromper rapidamente.
            train_start = time.time()
            updated_state, personalized_acc = local_training(
                model=model,
                state_dict=dummy_state,
                prune=args.prune,
                train_loader=train_loader,
                test_loader=test_loader,
                learning_rate=args.learning_rate,
                round_num=1,
                alaarg=args.ala,
                ala=ala,
                model_type=args.model,
                run_config=run_config if args.model in ['clip', 'slm'] else None,
                model_lock=model_lock if args.model == 'cnn' else None,
            )
            client_training_time = time.time() - train_start
            logger.info(
                f"Client {args.client_idx}: Treino local finalizado em "
                f"{client_training_time:.2f}s."
            )

            # ── AVALIACAO POS-EPOCA ───────────────────────────────────────
            # Avaliar o modelo (com ou sem agregacao P2P) nos dados locais
            local_test_acc, local_test_loss = evaluate_model(model, test_loader)
            train_accuracy, train_loss = evaluate_model(model, train_loader)

            rs_local_acc.append(local_test_acc)
            rs_train_acc.append(train_accuracy)

            logger.info(
                f"Client {args.client_idx} | Pos-Epoca {epoch} | "
                f"Test Acc: {local_test_acc:.2f}% | Train Acc: {train_accuracy:.2f}%"
            )

            # Salvar metricas incrementalmente (sobrescreve o arquivo a cada epoca)
            save_results(
                args, rs_global_acc, rs_global_loss, rs_ala_acc,
                rs_local_acc, rs_train_acc,
                idx=args.client_idx, argalgo=args.ala,
            )

            epoch += 1

if __name__ == '__main__':
    main()
