import socket
import pickle
import struct
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import read_client_data  # Importing the data reading utility
import argparse
import sys
import copy
import time
from prunning import restore_to_original_size, prune_and_restructure
from ALA import ALA
# Simple CNN model for MNIST or other datasets
class SimpleModel(nn.Module):
    def __init__(self, in_features=3, num_classes=10, dim=1600):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, 32, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512), 
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out

def send_data(conn, data):
    data_bytes = pickle.dumps(data)
    conn.sendall(struct.pack('!I', len(data_bytes)))
    conn.sendall(data_bytes)

def recv_data(conn):
    raw_msglen = recvall(conn, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('!I', raw_msglen)[0]
    data_bytes = recvall(conn, msglen)
    return pickle.loads(data_bytes)

def recvall(conn, n):
    data = b''
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data
def map_sequential_to_simplemodel(state_dict):
    """
    Mapeia state_dict de modelo Sequential para estrutura SimpleModel
    """
    mapped_dict = {}
    
    # Mapeamento baseado na estrutura Sequential típica
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
# Update local training to use data loaded via read_client_data
def local_training(model, state_dict, train_loader, learning_rate=0.01, round=2, alaarg=1, ala=None):
    #model.load_state_dict(state_dict)
    if round==2:
        state_dict = map_sequential_to_simplemodel(state_dict)
    
    state = copy.deepcopy(model)
    state.load_state_dict(state_dict)
    if alaarg==0 and round==2:
        local_initialization(ala, state, model)
    set_parameters(model, state)
    
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()  # appropriate for classification
    
    for x, y in train_loader:  # Train on batches from the loaded data
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
    # Convert list of (x, y) pairs into a DataLoader for batch processing
    X, y = zip(*train_data)
    X = torch.stack(X)  # Stack images into a tensor
    y = torch.tensor(y)  # Convert labels into a tensor
    dataset = torch.utils.data.TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
def set_parameters(model, state_new):
        for new_param, old_param in zip(state_new.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()
def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    
    # Connection arguments
    parser.add_argument('--host', type=str, default='localhost', 
                       help='Server IP address (default: 10.0.23.189)')
    parser.add_argument('--port', type=int, default=9090, 
                       help='Server port (default: 9090)')
    
    # Training arguments
    parser.add_argument('--rounds', type=int, default=10, 
                       help='Number of training rounds (default: 4)')
    parser.add_argument('--dataset', type=str, default='Cifar100', 
                       choices=['Cifar10', 'MNIST', 'FashionMNIST', 'Cifar100'], 
                       help='Dataset name (default: Cifar10)')
    parser.add_argument('--client-idx', type=int, default=0, 
                       help='Client index (default: 0)')
    
    # Model arguments
    parser.add_argument('--in-features', type=int, default=3, 
                       help='Input features/channels (default: 3)')
    parser.add_argument('--num-classes', type=int, default=100, 
                       help='Number of classes (default: 10)')
    parser.add_argument('--dim', type=int, default=1600, 
                       help='Dimension for first linear layer (default: 1600)')
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='Batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.01, 
                       help='Learning rate (default: 0.01)')
    
    # Other options
    parser.add_argument('--random-client', action='store_true', 
                       help='Use random client index instead of fixed')
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--ala", type=int, default=0)
    
    return parser.parse_args()
def local_initialization(ala, received_global_model, model, mask = None):
        ala.adaptive_local_aggregation(received_global_model, model, mask = mask)

def main():
    args = parse_args()
    
    # Set random client if requested
    if args.random_client:
        args.client_idx = random.randint(0, 5)
    
    print("=== Federated Learning Client ===")
    print(f"Host: {args.host}:{args.port}")
    print(f"Dataset: {args.dataset}")
    print(f"Client: {args.client_idx}")
    print(f"Rounds: {args.rounds}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 40)
    
    # Initialize model with arguments
    if args.dataset  =='MNIST':
        model = SimpleModel(
                in_features=1,
                num_classes=10,
                dim=1024
            )
    if args.dataset  =='Cifar10':
        model= SimpleModel(
                in_features=args.in_features,
                num_classes=10,
                dim=args.dim
            )
    if args.dataset  =='Cifar100':
        model = SimpleModel(
            in_features=args.in_features,
            num_classes=args.num_classes,
            dim=args.dim
        )
    loss = nn.CrossEntropyLoss()
    eta = 1
    rand_percent = 80
    layer_idx = 2
    # Load the dataset using the custom data loader
    try:
        train_loader = load_data(args.dataset, args.client_idx, is_train=True, batch_size=args.batch_size)
        test_loader = load_data(args.dataset, args.client_idx, is_train=False, batch_size=args.batch_size)
        print(f"Data loaded successfully - Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    ala = ALA(args.client_idx, loss, train_loader, 32, 
                    80, 2, 1.0, args.device)
    time.sleep(15)
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
            
            # Receive the global model from the server
            global_state = recv_data(s)
            if round_num+1 ==2:
                ammount = recv_data(s)
                local_model, _ = prune_and_restructure(model=model, 
                                                           pruning_rate=ammount, 
                                                           size_fc=25, data=args.dataset)
                set_parameters(model, local_model)
            if global_state is None:
                print("Failed to receive global model. Connection may be closed.")
                break
            print("Received global model.")
            # Evaluate test performance
            test_accuracy, test_loss = evaluate_model(model, test_loader)
            print(f"Client {args.client_idx}: Test Accuracy: {test_accuracy:.2f}% | Test Loss: {test_loss:.4f}")
            #set_parameters(local_model)
            # Perform local training using the received global model
            updated_state = local_training(model, global_state, train_loader, args.learning_rate, round_num+1, args.ala, ala)
            print("Local training completed.")

            # Evaluate training performance
            train_accuracy, train_loss = evaluate_model(model, train_loader)
            print(f"Client {args.client_idx}: Training Accuracy: {train_accuracy:.2f}% | Training Loss: {train_loss:.4f}")
            
            
            
            # Send the updated model state back to the server
            send_data(s, updated_state)
            
            send_data(s, len(train_loader))
            send_data(s, args.ala)
            print("Client update sent.")
            
            # Wait for the server to finish the round
            try:
                s.recv(3)
                print("Ready for next round...")
            except Exception as e:
                print(f"Error waiting for server: {e}")
                break

    print("\nTraining completed!")

if __name__ == '__main__':
    main()