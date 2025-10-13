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

# Update local training to use data loaded via read_client_data
def local_training(model, state_dict, train_loader):
    model.load_state_dict(state_dict)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
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

def main():
    HOST = '10.0.23.189'  # Change to the server's IP if needed.
    PORT = 9090
    NUM_ROUNDS = 4
    dataset = "Cifar10"  # Specify the dataset you are working with
    client_idx =0 #random.randint(0, 5)  # Specify the client index
    model = SimpleModel()

    # Load the dataset using the custom data loader
    train_loader = load_data(dataset, client_idx, is_train=True, batch_size=32)
    test_loader = load_data(dataset, client_idx, is_train=False, batch_size=32)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        
        for round_num in range(NUM_ROUNDS):
            print(f"\n--- Round {round_num} ---")
            # Receive the global model from the server
            global_state = recv_data(s)
            print("Received global model.")
            # Perform local training using the received global model
            updated_state = local_training(model, global_state, train_loader)
            print("Client update sent.")

            # Evaluate training performance
            train_accuracy, train_loss = evaluate_model(model, train_loader)
            print(f"Client {client_idx}: Training Accuracy: {train_accuracy:.2f}% | Training Loss: {train_loss:.4f}")
            
            # Evaluate test performance
            test_accuracy, test_loss = evaluate_model(model, test_loader)
            print(f"Client {client_idx}: Test Accuracy: {test_accuracy:.2f}% | Test Loss: {test_loss:.4f}")
            
            # Send the updated model state back to the server
            send_data(s, updated_state)
            # Wait for the server to finish the round
            s.recv(3)

if __name__ == '__main__':
    main()
