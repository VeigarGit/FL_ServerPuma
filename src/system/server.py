import socket
import pickle
import struct
import torch
import threading
import torch.nn as nn
import copy
from torch.utils.data import DataLoader
import time
import argparse
import sys
# Assuming you already have a test dataset available on the server side
from data_utils import read_client_data  # Utility to read the server's dataset

# A simple model for demonstration; replace with your actual model (e.g., from your FedAvg code)
class SimpleModel(nn.Module):
    def __init__(self, in_features=3, num_classes=10, dim=1600):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
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

# Aggregates a list of state_dicts by averaging their values
def aggregate_models(model_list):
    agg_state = {}
    for key in model_list[0].keys():
        agg_state[key] = sum([m[key] for m in model_list]) / len(model_list)
    return agg_state

# Assuming you have a method to evaluate the model on test data
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

# Helper to send pickled data with a header indicating its length
def send_data(conn, data):
    data_bytes = pickle.dumps(data)
    conn.sendall(struct.pack('!I', len(data_bytes)))
    conn.sendall(data_bytes)

# Helper to receive data given the 4-byte length header
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

# Thread function to handle a single client connection:
def handle_client(conn, client_updates, lock, round_num, client_id):
    try:
        start_time = time.time()
        print(f"Round {round_num}: Handling client {client_id}")
        
        # Send the current global model state
        with lock:
            current_global_state = global_state.copy()
        
        send_data(conn, current_global_state)
        print(f"Round {round_num}: Sent global model to client {client_id}")
        
        # Receive the updated model from the client
        updated_state = recv_data(conn)
        
        end_time = time.time()
        
        if updated_state is not None:
            with lock:
                client_updates.append(updated_state)
            training_time = end_time - start_time
            print(f"Round {round_num}: Client {client_id} training completed in {training_time:.2f} seconds")
        else:
            print(f"Round {round_num}: No update received from client {client_id}")
            
    except Exception as e:
        print(f"Round {round_num}: Error handling client {client_id}: {e}")

def load_test_data(dataset, client_idx, batch_size=32):
    try:
        test_data = read_client_data(dataset, client_idx, is_train=False)
        X, y = zip(*test_data)
        X = torch.stack(X)
        y = torch.tensor(y)
        dataset = torch.utils.data.TensorDataset(X, y)
        return DataLoader(dataset, batch_size=batch_size)
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None

def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning Server')
    
    # Server configuration
    parser.add_argument('--host', type=str, default='0.0.0.0', 
                       help='Server host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=9090, 
                       help='Server port (default: 9090)')
    
    # Federated learning parameters
    parser.add_argument('--clients-per-round', type=int, default=2, 
                       help='Number of clients per round (default: 2)')
    parser.add_argument('--rounds', type=int, default=4, 
                       help='Number of training rounds (default: 4)')
    
    # Dataset and model parameters
    parser.add_argument('--dataset', type=str, default='Cifar10', 
                       choices=['Cifar10', 'MNIST', 'FashionMNIST'],
                       help='Dataset name (default: Cifar10)')
    parser.add_argument('--test-client-idx', type=int, default=100, 
                       help='Client index for test data (default: 100)')
    
    # Model architecture
    parser.add_argument('--in-features', type=int, default=3, 
                       help='Input features/channels (default: 3)')
    parser.add_argument('--num-classes', type=int, default=10, 
                       help='Number of classes (default: 10)')
    parser.add_argument('--dim', type=int, default=1600, 
                       help='Dimension for first linear layer (default: 1600)')
    
    # Evaluation parameters
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='Batch size for evaluation (default: 32)')
    
    # Server options
    parser.add_argument('--max-clients', type=int, default=10, 
                       help='Maximum number of client connections (default: 10)')
    
    return parser.parse_args()

def main():
    global global_state
    
    args = parse_args()
    
    print("=== Federated Learning Server ===")
    print(f"Host: {args.host}:{args.port}")
    print(f"Dataset: {args.dataset}")
    print(f"Clients per round: {args.clients_per_round}")
    print(f"Total rounds: {args.rounds}")
    print(f"Test client index: {args.test_client_idx}")
    print("=" * 40)
    
    # Initialize global model with arguments
    global_model = SimpleModel(
        in_features=args.in_features,
        num_classes=args.num_classes,
        dim=args.dim
    )
    global_state = global_model.state_dict()
    
    lock = threading.Lock()
    
    # Load test data
    test_loader = load_test_data(args.dataset, args.test_client_idx, args.batch_size)
    if test_loader is None:
        print("Warning: Could not load test data. Evaluation will be skipped.")
        test_loader = None
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((args.host, args.port))
        s.listen(args.max_clients)
        print(f"Server listening on {args.host}:{args.port}")
        print(f"Waiting for {args.clients_per_round} clients to connect...")
        
        # Wait for initial client connections
        client_connections = []
        client_addresses = []
        
        while len(client_connections) < args.clients_per_round:
            conn, addr = s.accept()
            print(f"Client {len(client_connections) + 1} connected: {addr}")
            client_connections.append(conn)
            client_addresses.append(addr)
        
        print(f"All {args.clients_per_round} clients connected. Starting training...")
        
        for round_num in range(args.rounds):
            print(f"\n--- Round {round_num + 1}/{args.rounds} ---")
            client_updates = []
            threads = []
            
            # Handle each client in a separate thread
            for i, conn in enumerate(client_connections):
                t = threading.Thread(
                    target=handle_client, 
                    args=(conn, client_updates, lock, round_num + 1, i + 1)
                )
                t.start()
                threads.append(t)
            
            # Wait for all threads to complete
            for t in threads:
                t.join()
            
            # Aggregate the client updates
            if client_updates:
                print(f"Round {round_num + 1}: Aggregating {len(client_updates)} client updates")
                aggregated_state = aggregate_models(client_updates)
                
                with lock:
                    global_state = aggregated_state
                    global_model.load_state_dict(global_state)
                
                # Evaluate model on the test dataset after aggregation
                if test_loader is not None:
                    accuracy, avg_loss = evaluate_model(global_model, test_loader)
                    print(f"Round {round_num + 1}: Test Accuracy: {accuracy:.2f}% | Test Loss: {avg_loss:.4f}")
                else:
                    print(f"Round {round_num + 1}: Model aggregated (no test data for evaluation)")
                
                # Notify clients that the round ended
                successful_notifications = 0
                for conn in client_connections:
                    try:
                        conn.send('end'.encode('utf-8'))
                        successful_notifications += 1
                    except Exception as e:
                        print(f"Error notifying client: {e}")
                
                print(f"Round {round_num + 1}: Global model updated. Notified {successful_notifications} clients.")
            else:
                print(f"Round {round_num + 1}: No client updates received this round.")
        
        print(f"\nTraining completed after {args.rounds} rounds!")
        
        # Close all client connections
        for conn in client_connections:
            try:
                conn.close()
            except:
                pass
        print("All client connections closed.")

if __name__ == '__main__':
    main()