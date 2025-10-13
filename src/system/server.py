import socket
import pickle
import struct
import torch
import threading
import torch.nn as nn
import copy
from torch.utils.data import DataLoader
import time
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

HOST = '0.0.0.0'
PORT = 9090
NUM_CLIENTS_PER_ROUND = 2  # how many client updates to wait for each round
NUM_ROUNDS = 4

global_model = SimpleModel()
global_state = global_model.state_dict()

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
def handle_client(conn, client_updates, lock):
    # Send the current global model state
    start_time = time.time()
    send_data(conn, global_state)
    # Receive the updated model from the client
    updated_state = recv_data(conn)
    
    end_time = time.time()
    if updated_state is not None:
        with lock:
            client_updates.append(updated_state)
    training_time = end_time - start_time  # Calculate the training time
    print(f"Client training time: {training_time:.2f} seconds")  # Log the training time

def load_test_data(dataset, client_idx, is_train=True, batch_size=32):
    # Placeholder for loading test data, you can customize it according to your needs
    test_data = read_client_data(dataset, client_idx, is_train)  # Update this based on actual test dataset
    X, y = zip(*test_data)
    X = torch.stack(X)
    y = torch.tensor(y)
    dataset = torch.utils.data.TensorDataset(X, y)
    return DataLoader(dataset, batch_size=32)

def main():
    global global_state
    lock = threading.Lock()
    dataset = "Cifar10"
    client_idx=100
    test_loader = load_test_data(dataset, client_idx, is_train=False, batch_size=32)  # Load test data for evaluation after aggregation
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {(HOST, PORT)}")
        client_connections = []
        
        while len(client_connections) < NUM_CLIENTS_PER_ROUND:
            conn, addr = s.accept()
            print(f"Client connected: {addr}")
            client_connections.append(conn)
        
        for round_num in range(NUM_ROUNDS):
            print(f"\n--- Round {round_num} ---")
            client_updates = []
            threads = []
            
            for conn in client_connections:
                t = threading.Thread(target=handle_client, args=(conn, client_updates, lock))
                t.start()
                threads.append(t)
            
            for t in threads:
                t.join()
            
            # Aggregate the client updates
            if client_updates:
                aggregated_state = aggregate_models(client_updates)
                global_state = aggregated_state
                global_model.load_state_dict(global_state)
                
                # Evaluate model on the test dataset after aggregation
                
                accuracy, avg_loss = evaluate_model(global_model, test_loader)
                print(f"Round {round_num}: Test Accuracy: {accuracy:.2f}% | Test Loss: {avg_loss:.4f}")
                
                # Notify clients that the round ended
                for conn in client_connections:
                    conn.send('end'.encode('utf-8'))
                print("Global model updated and evaluated after aggregation.")
            else:
                print("No client updates received this round.")

if __name__ == '__main__':
    main()
