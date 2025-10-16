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
import traceback
# Assuming you already have a test dataset available on the server side
from data_utils import read_client_data  # Utility to read the server's dataset
from prunning import restore_to_original_size, prune_and_restructure
from size_mode import get_model_size
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

class FederatedLearningServer:
    def __init__(self, args):
        self.args = args
        if args.dataset  =='MNIST':
            self.global_model = SimpleModel(
                in_features=1,
                num_classes=10,
                dim=1024
            )
        else:
            self.global_model = SimpleModel(
                in_features=args.in_features,
                num_classes=args.num_classes,
                dim=args.dim
            )
        self.global_state = self.global_model.state_dict()
        self.lock = threading.Lock()
        self.client_data = {}
        self.client_connections = []
        self.client_addresses = []
        self.size_fc = 25
        self.client_idx =[]
        self.prune = args.prune
        # Load test data
        self.test_loader = self.load_test_data(args.dataset, args.test_client_idx, args.batch_size)
        if self.test_loader is None:
            print("Warning: Could not load test data. Evaluation will be skipped.")
            self.test_loader = None

    # Aggregates a list of state_dicts by averaging their values
    def aggregate_models(self, model_list):
        agg_state = {}
        for key in model_list[0].keys():
            agg_state[key] = sum([m[key] for m in model_list]) / len(model_list)
        return agg_state

    # Assuming you have a method to evaluate the model on test data
    def evaluate_model(self, model, data_loader):
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
    def send_data(self, conn, data):
        data_bytes = pickle.dumps(data)
        conn.sendall(struct.pack('!I', len(data_bytes)))
        conn.sendall(data_bytes)

    # Helper to receive data given the 4-byte length header
    def recv_data(self, conn):
        raw_msglen = self.recvall(conn, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('!I', raw_msglen)[0]
        data_bytes = self.recvall(conn, msglen)
        return pickle.loads(data_bytes)

    def recvall(self, conn, n):
        data = b'' 
        while len(data) < n:
            packet = conn.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data
    
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.global_model.parameters()):
            old_param.data = new_param.data.clone()

    # Thread function to handle a single client connection:
    def handle_client(self, conn, client_updates, round_num, client_id):
        try:
            start_time = time.time()
            print(f"Round {round_num}: Handling client {client_id}")
            
            # Send the current global model state
            with self.lock:
                current_global_state = self.global_state.copy()
            if round_num == 2 and self.prune==0:
                max_amount = self.set_amount_prune()
                print(max_amount)
                g_model_pruned = copy.deepcopy(self.global_model)
                g_model_pruned, _ = prune_and_restructure(model=self.global_model,
                                                        pruning_rate=max_amount, 
                                                        size_fc=self.size_fc, data= self.args.dataset)
                self.set_parameters(g_model_pruned)
                g_model_pruned = g_model_pruned.state_dict()
                
            if round_num == 2 and self.prune==0:
                self.send_data(conn, g_model_pruned)
                self.send_data(conn, max_amount)
            else:
                self.send_data(conn, current_global_state)
            print(f"Round {round_num}: Sent global model to client {client_id}")
            
            # Receive the updated model from the client
            updated_state = self.recv_data(conn)
            self.client_data[client_id] = self.recv_data(conn)
            end_time = time.time()
            
            if updated_state is not None:
                with self.lock:
                    client_updates.append(updated_state)
                training_time = end_time - start_time
                print(f"Round {round_num}: Client {client_id} training completed in {training_time:.2f} seconds")
            else:
                print(f"Round {round_num}: No update received from client {client_id}")
                
        except Exception as e:
            print(f"Round {round_num}: Error handling client {client_id}: {e}")
            print("Traceback:")
            traceback.print_exc()

    def load_test_data(self, dataset, client_idx, batch_size=32):
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
    
    def set_amount_prune(self):
        total = sum(self.client_data.values())
        max_amount = 0
        
        for client_key, client_value in self.client_data.items():
            # Usando client_value (valor) em vez de client_key (chave)
            amount = 1 - (total / (2 * client_value)) if client_value != 0 else 0
            amount = max(0, min(amount, 0.9))

            if amount > max_amount:
                max_amount = amount
        
        return max_amount

    def run_server(self):
        print("=== Federated Learning Server ===")
        print(f"Host: {self.args.host}:{self.args.port}")
        print(f"Dataset: {self.args.dataset}")
        print(f"Clients per round: {self.args.clients_per_round}")
        print(f"Total rounds: {self.args.rounds}")
        print(f"Test client index: {self.args.test_client_idx}")
        print("=" * 40)
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.args.host, self.args.port))
            s.listen(self.args.max_clients)
            print(f"Server listening on {self.args.host}:{self.args.port}")
            print(f"Waiting for {self.args.clients_per_round} clients to connect...")
            
            # Wait for initial client connections
            self.client_data = {index: None for index in range(1, self.args.clients_per_round+1)}
            while len(self.client_connections) < self.args.clients_per_round:
                conn, addr = s.accept()
                idx = self.recv_data(conn)
                print("client idx:", idx) 

                print(f"Client {len(self.client_connections) + 1} connected: {addr}")
                self.client_idx.append(idx)
                self.client_connections.append(conn)
                self.client_addresses.append(addr)
            
            print(f"All {self.args.clients_per_round} clients connected. Starting training...")
            
            for round_num in range(self.args.rounds):
                print(f"\n--- Round {round_num + 1}/{self.args.rounds} ---")
                client_updates = []
                threads = []
                
                # Handle each client in a separate thread
                for i, conn in enumerate(self.client_connections):
                    t = threading.Thread(
                        target=self.handle_client, 
                        args=(conn, client_updates, round_num + 1, i + 1)
                    )
                    t.start()
                    threads.append(t)
                
                # Wait for all threads to complete
                for t in threads:
                    t.join()
                
                # Aggregate the client updates
                if client_updates:
                    print(f"Round {round_num + 1}: Aggregating {len(client_updates)} client updates")
                    aggregated_state = self.aggregate_models(client_updates)
                    
                    with self.lock:
                        self.global_state = aggregated_state
                        self.global_model.load_state_dict(self.global_state)
                    
                    # Evaluate model on the test dataset after aggregation
                    if self.test_loader is not None:
                        accuracy, avg_loss = self.evaluate_model(self.global_model, self.test_loader)
                        print(f"Round {round_num + 1}: Test Accuracy: {accuracy:.2f}% | Test Loss: {avg_loss:.4f}")
                    else:
                        print(f"Round {round_num + 1}: Model aggregated (no test data for evaluation)")
                    size_global_model = get_model_size(self.global_model)
                    print(f'Size Global Model: {size_global_model:.2f}MB')
                    
                    # Notify clients that the round ended
                    successful_notifications = 0
                    for conn in self.client_connections:
                        try:
                            conn.send('end'.encode('utf-8'))
                            successful_notifications += 1
                        except Exception as e:
                            print(f"Error notifying client: {e}")
                    
                    print(f"Round {round_num + 1}: Global model updated. Notified {successful_notifications} clients.")
                else:
                    print(f"Round {round_num + 1}: No client updates received this round.")
            
            print(f"\nTraining completed after {self.args.rounds} rounds!")
            
            # Close all client connections
            for conn in self.client_connections:
                try:
                    conn.close()
                except:
                    pass
            print("All client connections closed.")

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
    parser.add_argument('--rounds', type=int, default=10, 
                       help='Number of training rounds (default: 4)')
    
    # Dataset and model parameters
    parser.add_argument('--dataset', type=str, default='Cifar100', 
                       choices=['Cifar10', 'MNIST', 'FashionMNIST'],
                       help='Dataset name (default: Cifar10)')
    parser.add_argument('--test-client-idx', type=int, default=0, 
                       help='Client index for test data (default: 100)')
    
    # Model architecture
    parser.add_argument('--in-features', type=int, default=3, 
                       help='Input features/channels (default: 3)')
    parser.add_argument('--num-classes', type=int, default=100, 
                       help='Number of classes (default: 10)')
    parser.add_argument('--dim', type=int, default=1600, 
                       help='Dimension for first linear layer (default: 1600)')
    
    # Evaluation parameters
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='Batch size for evaluation (default: 32)')
    
    # Server options
    parser.add_argument('--max-clients', type=int, default=10, 
                       help='Maximum number of client connections (default: 10)')
    parser.add_argument('--prune', type=int, default=0, 
                       help='Maximum number of client connections (default: 10)')
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    
    return parser.parse_args()

def main():
    args = parse_args()
    server = FederatedLearningServer(args)
    server.run_server()

if __name__ == '__main__':
    main()