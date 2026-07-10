
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

from data_utils import read_client_data
from prunning import prune_and_restructure
from ALA import ALA
from ALA_LoRA import ALA_LoRA
from model import SimpleModel
from utils.network_utils import send_data, recv_data
from utils.fl_math import quantization, dequantization, evaluate_model, set_parameters
from utils.fl_math import resize_model_to_pruned

# --- CONFIGURAÇÃO DE OBSERVABILIDADE (LOGGING ENXUTO) ---
logging.basicConfig(
    level=logging.INFO,
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

def local_training(model, state_dict, prune, train_loader, test_loader, learning_rate=0.01, round_num=2, alaarg=1, ala=None, model_type='cnn', run_config=None):
    # Verifica se os pesos precisam ser desquantizados (se não forem tensores)
    if state_dict and not isinstance(next(iter(state_dict.values())), torch.Tensor):
        state_dict = dequantization(state_dict)
    
    personalized_acc = None  # Inicializa a variável que guardará a acurácia pós-ALA
    
    if round_num >= 2 and prune == 0:
        if model_type == 'cnn':
            state_dict = map_sequential_to_simplemodel(state_dict)
    # === CORREÇÃO ESTRUTURAL APENAS PARA O CLIP ===
    if model_type == 'clip':
        # 1. 'model' contém nossos pesos locais. 
        # Criamos o 'global_model' copiando a estrutura atual.
        global_model = copy.deepcopy(model)
        
        # 2. Injetamos os pesos do servidor (state_dict) APENAS no global_model
        global_model = resize_model_to_pruned(global_model, state_dict)
        
        # 3. Agora temos os mundos perfeitamente isolados:
        # model -> Possui os Pesos Locais
        # global_model -> Possui os Pesos Globais
        
        if alaarg == 0 and round_num >= 2:
            logger.info(f"Client {ala.cid}: Applying FedALA_LoRA (Adaptive Local Aggregation)...")
            
            # O FedALA lê os dois modelos, calcula a proporção ideal 
            # e salva o resultado da mistura DENTRO do 'model'
            local_initialization(ala, global_model, model)
            
            # Avaliação imediata pós-ALA para ver se a mistura foi boa
            personalized_acc, _ = evaluate_model(model, test_loader)
            logger.info(f"Client {ala.cid}: Post-ALA Test Accuracy: {personalized_acc:.2f}%")
        else:
            # Fluxo FedAvg Clássico (alaarg != 0) ou Rodada 1: 
            # ATENÇÃO: Nunca use set_parameters(model, global_model) aqui se o shape puder mudar (SoRA).
            # set_parameters troca o ponteiro de .data e corrompe o grafo do otimizador/CUDA.
            # O resize_model_to_pruned recria o nn.Parameter com segurança via setattr se o shape mudar.
            resize_model_to_pruned(model, state_dict)
            
        # Limpa o modelo global da memória para evitar Out Of Memory (OOM) na GPU
        del global_model
        
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
    if model_type == 'clip':
        from lora_clip.clip_setup import build_optimizer, train_epoch
        # O build_optimizer já sabe se é LoRA ou SoRA lendo o run_config. 
        # Como é só LoRA, sparse_optimizer será None.
        optimizer, sparse_optimizer = build_optimizer(model, run_config)
        # Verifica se o SoRA está ativo e resgata o lambda de esparsidade do YAML
        is_sora = run_config["model"]["lora"]["mode"] in ["with_sora_no_schedule", "with_sora_schedule"]
        sparse_lambda = run_config["model"].get("sora", {}).get("sparse_lambda", 0.0) if is_sora else 0.0
        
        # O train_epoch já faz o loop no train_loader, calcula as perdas e atualiza os pesos
        metrics = train_epoch(model, train_loader, optimizer, sparse_optimizer, sparse_lambda)
        
        logger.info(f"Métricas do Treino Local - CE Loss: {metrics['ce_loss']:.4f} | Sparse Loss: {metrics['sparse_loss']:.4f}")
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        sparse_optimizer = None

        model_dtype = next(model.parameters()).dtype
        
        for x, y in train_loader:
            if x.is_floating_point():
                x = x.to(device, dtype=model_dtype)
            else:
                x = x.to(device)
            y = y.to(device)
            
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
    
    # --- 2. SELEÇÃO DOS PESOS QUE SERÃO ENVIADOS PARA O SERVIDOR ---
    if model_type == 'clip':
        from lora_clip.sora import get_trainable_state_dict
        return get_trainable_state_dict(model), personalized_acc # Envia APENAS o LoRA e Head, e a acuracia pos-ALA
    else:
        return model.state_dict(), personalized_acc # Envia tudo do CNN, e a acuracia pos-ALA



import torchvision.transforms as T # Adicione isto no topo dos imports

def load_data(dataset, client_idx, device, is_train=True, batch_size=32, is_clip=False):
    train_data = read_client_data(dataset, client_idx, is_train)
    X, y = zip(*train_data)
    X = torch.stack(X)
    
    # Redimensionamento vital para o CLIP não quebrar
    if is_clip:
        resize = T.Resize((224, 224), antialias=True)
        X = resize(X)
        
    # Garante que o rótulo é inteiro longo e move os dados INTEIROS para a GPU
    X = X.to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)
    
    dataset = torch.utils.data.TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)



def save_results(args, rs_global_acc, rs_global_loss, rs_ala_acc, rs_local_acc, rs_train_acc, idx=0, argalgo=0):
    b = "FedALA" if argalgo == 0 else "FedAVG"
    
    paca_val = args.paca if (args.paca is not None and args.paca > 0) else 0
    algo = f"client_{idx}_{args.dataset}_{args.strategy}_rank{args.rank}_paca{paca_val}_{b}_run{args.run_id}"
    
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
    parser.add_argument('--dataset', type=str, default='Cifar100', choices=['Cifar10', 
                                                                            'MNIST', 
                                                                            'FashionMNIST',
                                                                            'Cifar100',
                                                                            "OxfordPets"])
    parser.add_argument('--client-idx', type=int, default=0)
    parser.add_argument('--in-features', type=int, default=3)
    parser.add_argument('--num-classes', type=int, default=100)
    parser.add_argument('--dim', type=int, default=1600)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--random-client', action='store_true')
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument("--ala", type=int, default=0)
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument('--config', type=str, default="lora_clip/config.yaml")
    parser.add_argument('--experiments', type=int, default=1)
    parser.add_argument('--run-id', type=int, default=1)
    parser.add_argument('--exp-name', type=str, default='default_exp', help='Nome da sessão com timestamp')
    parser.add_argument('--strategy', type=str, default='lora', choices=['lora', 'sora_with_schedule', 'sora_no_schedule'])
    parser.add_argument('--rank', type=int, default=8, help='Rank para o SoRA/LoRA') 
    parser.add_argument('--paca', type=int, default=12, help='Número de camadas para a estratégia PaCA')
    
    # --- PaCA Heterogêneo ---
    parser.add_argument('--random-paca', action='store_true', help='Sorteia um valor de PaCA aleatório para este cliente')
    parser.add_argument('--paca-min', type=int, default=1, help='Valor mínimo de PaCA no sorteio aleatório')
    parser.add_argument('--paca-max', type=int, default=12, help='Valor máximo de PaCA no sorteio aleatório')
    parser.add_argument('--paca-list', type=str, default=None, help='Lista de PaCAs pré-definidos por cliente (ex: "4,6,8,12"). O cliente usa o valor na posição client_idx %% len(lista)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Controle contextual (evitar import os espalhado)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # Desativado: causa alto uso de CPU. Usar apenas para debug de erros CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available. Falling back to CPU.")
        args.device = "cpu"
    
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
        case 'clip':
            config = load_config(args.config)
            
            if args.strategy == 'lora':
                config["model"]["lora"]["mode"] = "with_lora"
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
            
            # Resolve o modo de execução a partir do YAML modificado
            run_mode = resolve_run_modes(config)[0]
            run_config = build_run_config(config, run_mode=run_mode)
            
            match config["dataset"]["name"]:
                case "enterprise-explorers/oxford-pets":
                    model = build_model(config=run_config, num_classes=37, device=device)
                case _:
                    # Fallback para caso utilize Cifar10 ou Cifar100 via bash
                    model = build_model(config=run_config, num_classes=args.num_classes, device=device)
            
    model = model.to(device)
    loss = nn.CrossEntropyLoss()
    rs_global_acc = []
    rs_global_loss = []
    rs_ala_acc = []
    rs_local_acc = []
    rs_train_acc = []
    
    try:
        is_clip_flag = (args.model == 'clip')
        train_loader = load_data(args.dataset, args.client_idx, device, is_train=True, batch_size=args.batch_size, is_clip=is_clip_flag)
        test_loader = load_data(args.dataset, args.client_idx, device, is_train=False, batch_size=args.batch_size, is_clip=is_clip_flag)
        logger.info(f"Data loaded successfully - Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    except Exception as e:
        logger.exception("Error loading data")
        sys.exit(1)
    
    if args.model == 'clip':
        ala = ALA_LoRA(args.client_idx, loss, train_loader, 32, 80, 1.0, device)
    else:
        ala = ALA(args.client_idx, loss, train_loader, 32, 80, 2, 1.0, device)
        
    time.sleep(10)
    
    # Colocar variavel de iteração para colocar o resultado médio de varias simulações e tirar média
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((args.host, args.port))
            send_data(s, args.client_idx)
            logger.info(f"Connected to server {args.host}:{args.port}")
        except OSError as e:
            logger.exception("Connection failed")
            sys.exit(1)
        
        for round_num in range(args.rounds):
            logger.info(f"\n--- Round {round_num + 1}/{args.rounds} ---")
            
            global_state, _ = recv_data(s)
            prune, _ = recv_data(s)
            
            if global_state is None:
                logger.warning("Failed to receive global model. Connection may be closed.")
                break
                
            logger.info("Received global model.")
            global_state = dequantization(global_state)
            
            
            if round_num + 1 >= 2 and prune == 0 and args.model != 'clip':
                ammount, _ = recv_data(s)
            
            if round_num + 1 >= 2:
                if args.model == 'clip':
                    old_local_weights = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

                local = resize_model_to_pruned(model, global_state)
                test_accuracy, test_loss = evaluate_model(local, test_loader)
                
                if args.model == 'clip':
                    for n, p in model.named_parameters():
                        if p.requires_grad and n in old_local_weights:
                            if p.data.shape == old_local_weights[n].shape:
                                p.data.copy_(old_local_weights[n])
            else:
                test_accuracy, test_loss = evaluate_model(model, test_loader)
                
            logger.info(f"Client {args.client_idx}: Global Model Test Accuracy: {test_accuracy:.2f}% | Test Loss: {test_loss:.4f}")
            rs_global_acc.append(test_accuracy)
            rs_global_loss.append(test_loss)

         
            
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
                run_config=run_config if args.model == 'clip' else None
            )
            
            # Se o FedALA calculou a acurácia, nós salvamos ela; se não (ex: round 1 ou ala desativado), salvamos 0.0 para manter o array com mesmo tamanho
            if personalized_acc is not None:
                rs_ala_acc.append(personalized_acc)
            else:
                # Se o ALA não rodou, salvamos a acurácia global para a linha do gráfico não cair
                rs_ala_acc.append(test_accuracy)

            logger.info("Local training completed.")

            # Acurácia logo após o treinamento local (avaliado no conjunto de teste para comparação justa)
            local_test_acc, local_test_loss = evaluate_model(model, test_loader)
            rs_local_acc.append(local_test_acc)
            
            train_accuracy, train_loss = evaluate_model(model, train_loader)
            rs_train_acc.append(train_accuracy)

            logger.info(f"Client {args.client_idx}: Post-Training Test Accuracy: {local_test_acc:.2f}% | Training Accuracy: {train_accuracy:.2f}%")
            
            updated_state = quantization(updated_state)
            send_data(s, updated_state)
            send_data(s, len(train_loader))
            send_data(s, args.ala)
            logger.info("Client update sent.")
            
            try:
                s.recv(3)
                logger.info("Ready for next round...")
            except OSError as e:
                logger.exception("Error waiting for server")
                break
                
    save_results(args, rs_global_acc, rs_global_loss, rs_ala_acc, rs_local_acc, rs_train_acc, idx=args.client_idx, argalgo=args.ala)
    logger.info("\nTraining completed!")

if __name__ == '__main__':
    main()