import socket
import pickle
import struct
import copy
import time
import os
import sys
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import h5py

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
from model import SimpleModel
from utils.network_utils import send_data, recv_data
from utils.fl_math import quantization, dequantization, evaluate_model, set_parameters

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

def local_training(model, state_dict, prune, train_loader, learning_rate=0.01, round_num=2, alaarg=1, ala=None):
    state_dict = dequantization(state_dict)
    
    if round_num >= 2 and prune == 0:
        state_dict = map_sequential_to_simplemodel(state_dict)

    local_model = resize_model_to_pruned(model, state_dict)
    state = copy.deepcopy(local_model)
    
    if alaarg == 0 and round_num == 2:
        logger.info(f"Client {ala.cid}: Applying FedALA (Adaptive Local Aggregation)...")
        local_initialization(ala, state, model)
        
    set_parameters(model, state)
    
    size_before = sys.getsizeof(pickle.dumps(model)) / (1024 * 1024)    
    logger.debug(f"Tamanho antes: {size_before:.2f} MB")    
    
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    device = next(model.parameters()).device
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
    
    return model.state_dict()

def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(output, y)
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = 100 * correct / total
    average_loss = total_loss / len(data_loader)
    return accuracy, average_loss

def load_data(dataset, client_idx, is_train=True, batch_size=32):
    train_data = read_client_data(dataset, client_idx, is_train)
    X, y = zip(*train_data)
    X = torch.stack(X)
    y = torch.tensor(y)
    dataset = torch.utils.data.TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def set_parameters(model, state_new):
    for new_param, old_param in zip(state_new.parameters(), model.parameters()):
        old_param.data = new_param.data.clone()

def save_results(args, rs_test_acc, rs_test_loss, idx=0, argalgo=0):
    b = "FedALA" if argalgo == 0 else "FedAVG"
    algo = f"{args.dataset}_{b}_client_{idx}"
    
    current_dir = Path(__file__).resolve().parent
    result_path = current_dir / "dados_compartilhados"
    result_path.mkdir(parents=True, exist_ok=True)
    
    file_path = result_path / f"{algo}.h5"
    logger.info(f"Salvando resultados em: {file_path}")
    
    with h5py.File(file_path, 'w') as hf:
        hf.create_dataset('rs_test_acc', data=rs_test_acc)
        hf.create_dataset('rs_train_loss', data=rs_test_loss)

def local_initialization(ala, received_global_model, model, mask=None):
    ala.adaptive_local_aggregation(received_global_model, model, mask=mask)

def resize_model_to_pruned(model, pruned_dict):
    """ Redimensiona o modelo existente para as dimensões podadas """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in pruned_dict:
                pruned_weight = pruned_dict[name]
                
                if param.shape != pruned_weight.shape:
                    logger.debug(f"Redimensionando {name}: {param.shape} -> {pruned_weight.shape}")
                    new_param = nn.Parameter(pruned_weight)
                    
                    if '.' in name:
                        parts = name.split('.')
                        module = model
                        for part in parts[:-1]:
                            module = getattr(module, part)
                        setattr(module, parts[-1], new_param)
                    else:
                        setattr(model, name, new_param)
                else:
                    param.data.copy_(pruned_weight)
    return model

def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='Cifar100', choices=['Cifar10', 'MNIST', 'FashionMNIST', 'Cifar100'])
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
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Controle contextual (evitar import os espalhado)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available. Falling back to CPU.")
        args.device = "cpu"
    
    device = torch.device(args.device)
    
    logger.info("=== Federated Learning Client ===")
    logger.info(f"Host: {args.host}:{args.port}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Client: {args.client_idx}")
    logger.info(f"Rounds: {args.rounds}")
    logger.info(f"Batch size: {args.batch_size} | LR: {args.learning_rate}")
    logger.info("=================================")
    
    # Adoção de Structural Pattern Matching
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
            
    model = model.to(device)
    loss = nn.CrossEntropyLoss()
    acc = []
    losses = []
    
    try:
        train_loader = load_data(args.dataset, args.client_idx, is_train=True, batch_size=args.batch_size)
        test_loader = load_data(args.dataset, args.client_idx, is_train=False, batch_size=args.batch_size)
        logger.info(f"Data loaded successfully - Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    except Exception as e:
        logger.exception("Error loading data")
        sys.exit(1)
    
    ala = ALA(args.client_idx, loss, train_loader, 32, 80, 2, 1.0, device)
    time.sleep(10)
    
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
            
            if round_num + 1 >= 2 and prune == 0:
                ammount, _ = recv_data(s)
            
            if round_num + 1 >= 2:
                local = resize_model_to_pruned(model, global_state)
                test_accuracy, test_loss = evaluate_model(local, test_loader)
            else:
                test_accuracy, test_loss = evaluate_model(model, test_loader)
                
            logger.info(f"Client {args.client_idx}: Test Accuracy: {test_accuracy:.2f}% | Test Loss: {test_loss:.4f}")
            acc.append(test_accuracy)
            losses.append(test_loss)
            
            updated_state = local_training(model, global_state, prune, train_loader, args.learning_rate, round_num + 1, args.ala, ala)
            logger.info("Local training completed.")

            train_accuracy, train_loss = evaluate_model(model, train_loader)
            logger.info(f"Client {args.client_idx}: Training Accuracy: {train_accuracy:.2f}% | Training Loss: {train_loss:.4f}")
            
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
                
    save_results(args, acc, losses, idx=args.client_idx, argalgo=args.ala)
    logger.info("\nTraining completed!")

if __name__ == '__main__':
    main()