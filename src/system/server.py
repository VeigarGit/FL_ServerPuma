import socket
import pickle
import struct
import torch
import threading
import torch.nn as nn
import copy
from torch.utils.data import DataLoader
from .model import SimpleModel
import time
import argparse
import sys
import traceback
import os
import h5py
from .data_utils import read_client_data
from .prunning import prune_and_restructure
from .size_mode import get_model_size
import logging

# Import utilities
from ..utils.network_utils import send_data, recv_data, recvall
from ..utils.model_utils import quantization, dequantization

# Define Logger Globally
logger = logging.getLogger("Server")

# --- FIX: MAPPING FUNCTION ---
def map_simplemodel_to_sequential(state_dict):
    """
    Maps keys from SimpleModel (named) to nn.Sequential (indexed).
    This is required because the server converts the model to Sequential 
    after pruning, but clients still send back SimpleModel keys.
    """
    mapped_dict = {}
    mapping = {
        'conv1.0.weight': '0.weight',
        'conv1.0.bias': '0.bias',
        'conv2.0.weight': '3.weight', 
        'conv2.0.bias': '3.bias',
        'fc1.0.weight': '7.weight',
        'fc1.0.bias': '7.bias',
        'fc.weight': '9.weight',
        'fc.bias': '9.bias'
    }
    for simple_key, sequential_key in mapping.items():
        if simple_key in state_dict:
            mapped_dict[sequential_key] = state_dict[simple_key]
    return mapped_dict

class FederatedLearningServer:
    def __init__(self, args):
        self.args = args
        if args.dataset =='MNIST':
            self.global_model = SimpleModel(in_features=1, num_classes=10, dim=1024)
        if args.dataset =='Cifar10':
            self.global_model = SimpleModel(in_features=args.in_features, num_classes=10, dim=args.dim)
        if args.dataset =='Cifar100':
            self.global_model = SimpleModel(in_features=args.in_features, num_classes=args.num_classes, dim=args.dim)
        
        self.rs_test_acc = []
        self.rs_test_loss = []
        self.global_state = self.global_model.state_dict()
        self.lock = threading.Lock()
        self.client_data = {}
        self.client_connections = []
        self.client_addresses = []
        self.size_fc = 25
        self.client_idx = []
        self.clients_info = {}
        self.prune = args.prune
        self.pruning_rate = 0.0 
        
        self.test_loader = self.load_test_data(args.dataset, args.test_client_idx, args.batch_size)
        if self.test_loader is None:
            logger.warning("Warning: Could not load test data. Evaluation will be skipped.")
            self.test_loader = None

    def aggregate_models(self, model_list):
        
        if isinstance(self.global_model, nn.Sequential):
             new_list = []
             for m in model_list:
                 # Check if the update has 'conv1.0.weight' (SimpleModel key)
                 if 'conv1.0.weight' in m:
                     new_list.append(map_simplemodel_to_sequential(m))
                 else:
                     new_list.append(m)
             model_list = new_list

        agg_state = {}
        # Use the first model in the list to determine keys
        for key in model_list[0].keys():
            agg_state[key] = sum([m[key] for m in model_list]) / len(model_list)
        return agg_state

    def evaluate_model(self, model, data_loader):
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

    def set_threthold(self):
        tot_time = 0
        clients_with_time = 0
        for client_id, info in self.clients_info.items():
            if info['training_time'] is not None:
                tot_time += info['training_time']
                clients_with_time += 1
        
        if clients_with_time == 0:
            logger.warning("Warning: No clients reported training time. Sticking to default threshold.")
            return

        mean_time = tot_time / clients_with_time
        self.time_threthold = 0.9 * mean_time
        logger.info(f'time_threthold: {self.time_threthold}s')

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.global_model.parameters()):
            old_param.data = new_param.data.clone()

    def handle_client(self, conn, client_updates, round_num, client_id):
        try:
            start_time = time.time()
            logger.info(f"Round {round_num}: Handling client {client_id}")
            
            with self.lock:
                current_state_to_send = self.global_state.copy()
            
            size_before = sys.getsizeof(pickle.dumps(current_state_to_send))/ (1024 * 1024)
            current_state_to_send = quantization(current_state_to_send)
            size_after = sys.getsizeof(pickle.dumps(current_state_to_send)) / (1024 * 1024)
            size_saved = size_before - size_after

            logger.info(f"Round {round_num}: Sending model to client {client_id} | Size: {size_after:.2f} MB | Saved: {size_saved:.2f} MB")

            send_data(conn, current_state_to_send)
            send_data(conn, self.prune)
            
            if round_num == 2 and self.prune == 0:
                send_data(conn, self.pruning_rate)

            updated_state = recv_data(conn)
            
            if updated_state is None:
                logger.warning(f"Round {round_num}: Client {client_id} disconnected unexpectedly.")
                return
                
            updated_state = dequantization(updated_state)
            
            client_data = recv_data(conn)
            if client_data is not None:
                self.client_data[client_id] = client_data
            
            algo_arg = recv_data(conn)
            if algo_arg is not None:
                self.argalgo = algo_arg
            
            end_time = time.time()
            
            with self.lock:
                client_updates.append(updated_state)
            
            training_time = end_time - start_time
            self.clients_info[client_id]['training_time'] = training_time
            logger.info(f"Round {round_num}: Client {client_id} training completed in {training_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Round {round_num}: Error handling client {client_id}: {e}", exc_info=True)

    def load_test_data(self, dataset, client_idx, batch_size=32):
        try:
            test_data = read_client_data(dataset, client_idx, is_train=False)
            X, y = zip(*test_data)
            X = torch.stack(X)
            y = torch.tensor(y)
            dataset = torch.utils.data.TensorDataset(X, y)
            return DataLoader(dataset, batch_size=batch_size)
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            return None
    
    def set_amount_prune(self):
        values = [v for v in self.client_data.values() if v != None]
    
        if not values:
            return 0
        if len(values) == 1:
            data = values[0]
            non_null_times = [info['training_time'] for info in self.clients_info.values() if info.get('training_time') is not None and info['training_time'] != 0]       
            training_time = non_null_times[0]
            ammount = training_time/data
            ammount = ammount * 10
            return ammount
        else:
            min_val = min(values)
            max_val = max(values)
        
        if max_val == min_val:
            return 0.9
        
        max_amount = []
        for client_key, client_value in self.client_data.items():
            if client_value == 0:
                continue
                
            amount = 0.9 * (1 - (client_value - min_val) / (max_val - min_val))
            amount = max(0, min(amount, 0.9))
            max_amount.append(amount)
        
        if max_amount:
            average_amount = sum(max_amount) / len(max_amount)
            closest_to_average = min(max_amount, key=lambda x: abs(x - average_amount))
            
            logger.info(f"Média dos amounts: {average_amount:.4f}")
            logger.info(f"Valor mais próximo da média: {closest_to_average:.4f}")
            if closest_to_average == 0:
                closest_to_average = average_amount
            return closest_to_average
        else:
            return 0

    def save_results(self):
        if self.args.prune == 0:
            a = "prune"
        else:
            a = "withou_Prune"
        b = self.argalgo
        if b == 0:
            b = "FedALA"
        else:
            b = "FedAVG"
        algo = self.args.dataset + "_" + a + "_" + b
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        file_path = result_path + "{}.h5".format(algo)
        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
            hf.create_dataset('rs_train_loss', data=self.rs_test_loss)

    def run_server(self):
        logger.info("=== Federated Learning Server ===")
        logger.info(f"Host: {self.args.host}:{self.args.port}")
        logger.info(f"Dataset: {self.args.dataset}")
        logger.info(f"Clients per round: {self.args.clients_per_round}")
        logger.info(f"Total rounds: {self.args.rounds}")
        logger.info(f"Test client index: {self.args.test_client_idx}")
        logger.info("=" * 40)
        
        self.time_threthold = 7
        self.time= []
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.args.host, self.args.port))
            s.listen(self.args.max_clients)
            logger.info(f"Server listening on {self.args.host}:{self.args.port}")
            logger.info(f"Waiting for {self.args.clients_per_round} clients to connect...")
            
            self.client_data = {index: None for index in range(1, self.args.clients_per_round+1)}
            while len(self.client_connections) < self.args.clients_per_round:
                conn, addr = s.accept()
                idx = recv_data(conn)
                logger.info(f"client idx: {idx}") 
                self.clients_info[idx+1] = {'training_time': None}
                logger.info(f"Client {len(self.client_connections) + 1} connected: {addr}")
                self.client_idx.append(idx)
                self.client_connections.append(conn)
                self.client_addresses.append(addr)
            
            logger.info(f"All {self.args.clients_per_round} clients connected. Starting training...")
            
            for round_num in range(self.args.rounds):
                time.sleep(5)
                logger.info(f"\n--- Round {round_num + 1}/{self.args.rounds} ---")

                if round_num == 1 and self.prune == 0: # Round 2
                    logger.info("--- SERVER: INITIATING GLOBAL PRUNING ---")
                    self.pruning_rate = self.set_amount_prune()
                    logger.info(f"--- SERVER: Calculated pruning rate: {self.pruning_rate:.4f}")
                    
                    self.global_model, _ = prune_and_restructure(
                        model=self.global_model, 
                        pruning_rate=self.pruning_rate, 
                        size_fc=self.size_fc, 
                        data=self.args.dataset
                    )
                    
                    device = next(self.global_model.parameters()).device
                    self.global_model.to(device)
                    
                    self.global_state = self.global_model.state_dict()
                    logger.info("--- SERVER: GLOBAL MODEL PRUNED SUCCESSFULLY ---")

                client_updates = []
                threads = []

                self.stop_event = threading.Event()
                for i, conn in enumerate(self.client_connections):
                    t = threading.Thread(target=self.handle_client, daemon=True, args=(conn, client_updates, round_num + 1, i + 1))
                    t.start()
                    threads.append(t)
                
                logger.info(f"threshold: {self.time_threthold}")
                
                init_time = time.time()
                logger.info(f"Round {round_num + 1}: Waiting for {len(threads)} clients to finish training...")
                for t in threads:
                    t.join()
                logger.info(f"Round {round_num + 1}: All client threads completed.")

                if round_num == 0:
                    self.set_threthold()
                
                end_time = time.time()
                round_duration = end_time - init_time
                logger.info(f"Round {round_num + 1} duration: {round_duration:.2f} seconds")
                logger.info(f"client_updates length: {len(client_updates)}")
                
                if client_updates:
                    logger.info(f"Round {round_num + 1}: Aggregating {len(client_updates)} client updates")
                    
                    # Call aggregate (now with auto-mapping)
                    aggregated_state = self.aggregate_models(client_updates)
                    
                    with self.lock:
                        self.global_state = aggregated_state
                        # This load should now work because keys match
                        self.global_model.load_state_dict(self.global_state)
                        device = next(self.global_model.parameters()).device
                        self.global_model.to(device)
                    
                    if self.test_loader is not None:
                        acc = []
                        loss = []
                        for i in self.client_idx:
                            self.test_loader = self.load_test_data(self.args.dataset, i, self.args.batch_size)
                            accuracy, avg_loss = self.evaluate_model(self.global_model, self.test_loader)
                            acc.append(accuracy)
                            loss.append(avg_loss)
                        logger.info(f'acc: {acc}')
                        logger.info(f'len acc {len(acc)}')
                        accuracy = sum(acc)/len(acc)
                        logger.info(f'avg acc: {accuracy}')
                        avg_loss = sum(loss)/len(loss)
                        self.rs_test_acc.append(accuracy)
                        self.rs_test_loss.append(avg_loss)
                        logger.info(f"Round {round_num + 1}: Test Accuracy: {accuracy:.2f}% | Test Loss: {avg_loss:.4f}")
                    else:
                        logger.info(f"Round {round_num + 1}: Model aggregated (no test data for evaluation)")
                    
                    size_global_model = get_model_size(self.global_model)
                    logger.info(f'Size Global Model: {size_global_model:.2f}MB')
                    
                    successful_notifications = 0
                    for conn in self.client_connections:
                        try:
                            conn.send(b'end') 
                            successful_notifications += 1
                        except Exception as e:
                            logger.error(f"Error notifying client: {e}")
                    
                    logger.info(f"Round {round_num + 1}: Global model updated. Notified {successful_notifications} clients.")
                else:
                    logger.info(f"Round {round_num + 1}: No client updates received this round.")
            
            logger.info(f"\nTraining completed after {self.args.rounds} rounds!")
            
            for conn in self.client_connections:
                try:
                    conn.close()
                except:
                    pass
            logger.info("All client connections closed.")
        self.save_results()

def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning Server')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--clients-per-round', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='Cifar10', choices=['Cifar10', 'MNIST', 'FashionMNIST', 'Cifar100'])
    parser.add_argument('--test-client-idx', type=int, default=0)
    parser.add_argument('--in-features', type=int, default=3)
    parser.add_argument('--num-classes', type=int, default=100)
    parser.add_argument('--dim', type=int, default=1600)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-clients', type=int, default=10)
    parser.add_argument('--prune', type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    return parser.parse_args()

def main():
    args = parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/server.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("\ncuda is not avaiable.\n")
        args.device = "cpu"
    device = torch.device(args.device)
    server = FederatedLearningServer(args)
    server.run_server()

if __name__ == '__main__':
    main()