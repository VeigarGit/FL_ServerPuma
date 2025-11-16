import socket
import pickle
import struct
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .data_utils import read_client_data
import argparse
import sys
import copy
import time
import os
from .prunning import prune_and_restructure
from .ALA import ALA
from .model import SimpleModel
import builtins

from ..utils.network_utils import send_data, recv_data, recvall
from ..utils.model_utils import quantization, dequantization

def print(*args, **kwargs):
    builtins.print(*args, **kwargs, flush=True)

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

def local_training(model, state_dict, prune, train_loader, learning_rate=0.01, round=2, alaarg=1, ala=None):
    state_dict = dequantization(state_dict)
    if round==2 and prune==0:
        state_dict = map_sequential_to_simplemodel(state_dict)
    
    state = copy.deepcopy(model)
    state.load_state_dict(state_dict)
    if alaarg==0 and round==2:
        print(f"Client {ala.cid}: Applying FedALA (Adaptive Local Aggregation)...")
        local_initialization(ala, state, model)
    set_parameters(model, state)
    
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    for x, y in train_loader:
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

    with torch.no_grad():
        for x, y in data_loader:
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

def local_initialization(ala, received_global_model, model, mask=None):
    ala.adaptive_local_aggregation(received_global_model, model, mask=mask)

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"
    device = torch.device(args.device)
    #if args.random_client:
        #args.client_idx = random.randint(0, 5)
    
    print("=== Federated Learning Client ===")
    print(f"Host: {args.host}:{args.port}")
    print(f"Dataset: {args.dataset}")
    print(f"Client: {args.client_idx}")
    print(f"Rounds: {args.rounds}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 40)
    
    if args.dataset =='MNIST':
        model = SimpleModel(in_features=1, num_classes=10, dim=1024)
    if args.dataset =='Cifar10':
        model = SimpleModel(in_features=args.in_features, num_classes=10, dim=args.dim)
    if args.dataset =='Cifar100':
        model = SimpleModel(in_features=args.in_features, num_classes=args.num_classes, dim=args.dim)
    
    loss = nn.CrossEntropyLoss()
    
    try:
        train_loader = load_data(args.dataset, args.client_idx, is_train=True, batch_size=args.batch_size)
        test_loader = load_data(args.dataset, args.client_idx, is_train=False, batch_size=args.batch_size)
        print(f"Data loaded successfully - Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    ala = ALA(args.client_idx, loss, train_loader, 32, 80, 2, 1.0, args.device)
    time.sleep(10)
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((args.host, args.port))
            send_data(s, args.client_idx)
            print(f"Connected to server {args.host}:{args.port}")
        except Exception as e:
            print(f"Connection failed: {e}")
            sys.exit(1)
        
        for round_num in range(args.rounds):
            print(f"\n--- Round {round_num + 1}/{args.rounds} ---")
            
            global_state = recv_data(s)
            prune = recv_data(s)
            if round_num+1 ==2 and prune ==0:
                ammount = recv_data(s)
                print(f"Client {args.client_idx}: Received pruning rate {ammount:.4f}. Pruning local model...")
                local_model, _ = prune_and_restructure(model=model, pruning_rate=ammount, size_fc=25, data=args.dataset)
                set_parameters(model, local_model)
            if global_state is None:
                print("Failed to receive global model. Connection may be closed.")
                break
            print("Received global model.")
            
            test_accuracy, test_loss = evaluate_model(model, test_loader)
            print(f"Client {args.client_idx}: Test Accuracy: {test_accuracy:.2f}% | Test Loss: {test_loss:.4f}")
            
            updated_state = local_training(model, global_state, prune, train_loader, args.learning_rate, round_num+1, args.ala, ala)
            print("Local training completed.")

            train_accuracy, train_loss = evaluate_model(model, train_loader)
            print(f"Client {args.client_idx}: Training Accuracy: {train_accuracy:.2f}% | Training Loss: {train_loss:.4f}")
            updated_state  = quantization(updated_state)
            send_data(s, updated_state)
            send_data(s, len(train_loader))
            send_data(s, args.ala)
            print("Client update sent.")
            
            try:
                s.recv(3)
                print("Ready for next round...")
            except Exception as e:
                print(f"Error waiting for server: {e}")
                break

    print("\nTraining completed!")

if __name__ == '__main__':
    main()