import socket
import pickle
import struct
import torch
import threading
import torch.nn as nn
import copy
from torch.utils.data import DataLoader
from model import SimpleModel
import time
import argparse
import sys
import traceback
import os
import h5py
from data_utils import read_client_data
from prunning import prune_and_restructure
from size_mode import get_model_size
import builtins
from synexp import LayerComplexityCalculator
import numpy as np
def print(*args, **kwargs):
    builtins.print(*args, **kwargs, flush=True)

class FederatedLearningServer:
    def __init__(self, args):
        self.args = args
        if args.dataset =='MNIST':
            self.global_model = SimpleModel(in_features=1, num_classes=10, dim=1024)
            self.input_size = (1, 1, 28, 28)
        if args.dataset =='Cifar10':
            self.global_model = SimpleModel(in_features=args.in_features, num_classes=10, dim=args.dim)
            self.input_size = (1, 3, 32, 32)
        if args.dataset =='Cifar100':
            self.global_model = SimpleModel(in_features=args.in_features, num_classes=args.num_classes, dim=args.dim)
            self.input_size = (1, 3, 32, 32)
        

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
        self.sended_ammount=[0]
        self.sended_withouquant=[0]
        self.aggregated_clients=[]
        self.round_time=[]
        self.bit=[]
        self.complexity_calculated = False
        self.alpha_list = None
        self.beta_list = None
        self.densities = None
        calculator = LayerComplexityCalculator(self.global_model, self.input_size)
        
        # Calcular
        alpha_list, beta_list = calculator.calculate_alpha_beta()
        
        self.alpha_list = alpha_list
        self.beta_list = beta_list
        #self.layer_names = layer_names
        self.complexity_calculated = True
        print(f"Calculado: {len(alpha_list)} camadas")
        print(f"Total α: {sum(alpha_list):,} parâmetros")
        print(f"Total β: {sum(beta_list):,} FLOPs")
        
        self.test_loader = self.load_test_data(args.dataset, args.test_client_idx, args.batch_size)
        if self.test_loader is None:
            print("Warning: Could not load test data. Evaluation will be skipped.")
            self.test_loader = None

    def aggregate_models(self, model_list, global_model):
        agg_state = global_model.copy()  # Começa com uma cópia do modelo global
    
        for key in global_model.keys():
            # Coleta apenas valores de modelos que têm esta chave COM O MESMO TAMANHO
            compatible_values = []
            
            for model in model_list:
                if key in model:
                    local_val = model[key]
                    global_val = global_model[key]
                    
                    # Verifica compatibilidade de tamanho
                    if self._are_compatible(local_val, global_val):
                        compatible_values.append(local_val)
            
            # Se encontrou valores compatíveis, calcula a média
            if compatible_values:
                # Para numpy arrays ou tensores
                if hasattr(compatible_values[0], 'shape'):
                    agg_state[key] = sum(compatible_values) / len(compatible_values)
                else:
                    # Para escalares
                    agg_state[key] = np.mean(compatible_values)
            # Se não encontrou, mantém o valor original do global_model
        
        return agg_state
    def _are_compatible(self, local_val, global_val):
        """Verifica se dois valores são compatíveis para agregação."""
        # Se ambos são arrays/tensores
        if hasattr(local_val, 'shape') and hasattr(global_val, 'shape'):
            return local_val.shape == global_val.shape
        
        # Se ambos são escalares ou listas do mesmo comprimento
        try:
            len_local = len(local_val) if hasattr(local_val, '__len__') else 1
            len_global = len(global_val) if hasattr(global_val, '__len__') else 1
            return len_local == len_global
        except:
            return True
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

    def send_data(self, conn, data):
        data_bytes = pickle.dumps(data)
        conn.sendall(struct.pack('!I', len(data_bytes)))
        conn.sendall(data_bytes)
    '''
    def recv_data(self, conn):
        raw_msglen = self.recvall(conn, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('!I', raw_msglen)[0]
        data_bytes = self.recvall(conn, msglen)
        return pickle.loads(data_bytes)
    '''
    def recv_data(self, conn):
        # Record start time
        start_time = time.time()
        
        # Receive the data length (4 bytes)
        raw_msglen = self.recvall(conn, 4)
        if not raw_msglen:
            return None, 0
        
        msglen = struct.unpack('!I', raw_msglen)[0]
        
        # Record time after receiving length header
        header_time = time.time()
        
        # Receive the actual data
        data_bytes = self.recvall(conn, msglen)
        
        # Record end time
        end_time = time.time()
        
        if not data_bytes:
            return None, 0
        
        # Calculate bit rate
        total_bits = (msglen + 4) * 8  # +4 for the length header
        total_time = end_time - start_time
        
        # Avoid division by zero
        if total_time > 0:
            bit_rate = total_bits / total_time  # bits per second
        else:
            bit_rate = 0
        
        return pickle.loads(data_bytes), bit_rate
    
    def recvall(self, conn, n):
        data = b'' 
        while len(data) < n:
            packet = conn.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data
    
    def set_trashold(self):
        tot_time = 0
        clients_with_time = 0
        for client_id, info in self.clients_info.items():
            if info['training_time'] is not None:
                tot_time += info['training_time']
                clients_with_time += 1
        
        # --- This is the check we added ---
        if clients_with_time == 0:
            print("Warning: No clients reported training time. Sticking to default threshold.")
            return # Exit the function early
        # --- End of added check ---

        mean_time = tot_time / clients_with_time
        self.time_trashold = 0.9 * mean_time
        print(f'time_trashold: {self.time_trashold}s')

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.global_model.parameters()):
            old_param.data = new_param.data.clone()
    def dequantization(self, global_state):
        dequantized_state_dict = {}
        for k, v in global_state.items():
            if isinstance(v, dict) and v.get('dtype') == 'quantized_int8':
                # Recupera tensores quantizados
                scale = v['scale']
                dequantized_state_dict[k] = v['weights'].float() * scale
            else:
                # Mantém tensores normais
                dequantized_state_dict[k] = v
        return dequantized_state_dict

    def quantization(self, state_dict):
        quantized_state_dict = {}
        keys = list(state_dict.keys())
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                scale = torch.max(torch.abs(v)) / 127.0
                quantized_weights = torch.clamp((v / scale).round(), -128, 127).to(torch.int8)
                quantized_state_dict[k] = {
                    'dtype': 'quantized_int8',
                    'scale': scale,
                    'weights': quantized_weights
                }
            else:
                quantized_state_dict[k] = v
        return quantized_state_dict
    
    def handle_client(self, conn, client_updates, round_num, client_id):
        bit_rate = []
        self.masks = []
        try:
            start_time = time.time()
            print(f"Round {round_num}: Handling client {client_id}")
            
            with self.lock:
                current_global_state = self.global_state.copy()
            
            if round_num >= 2 and self.prune == 0:
                print(f"--- SERVER: PRUNING START (Round 2) ---")
                max_amount = self.set_amount_prune()
                #max_amount = self.calculate_adaptive_densities(client_id, t_target=30)
                print(f"max_amount before check: {max_amount}")
                if max_amount ==0:
                    max_amount = 0.8
                print(f"--- SERVER: Calculated pruning rate: {max_amount:.4f}")
                g_model_pruned = copy.deepcopy(self.global_model)
                g_model_pruned, mask = prune_and_restructure(model=g_model_pruned, pruning_rate=max_amount, size_fc=self.size_fc, data=self.args.dataset)
                self.masks.append(mask)
                #self.set_parameters(g_model_pruned)
                g_model_pruned = g_model_pruned.state_dict()
                print(f"--- SERVER: PRUNING COMPLETE ---")

            if round_num >= 2 and self.prune == 0:
                size_before = sys.getsizeof(pickle.dumps(g_model_pruned))/ (1024 * 1024)
                novo_val = self.sended_withouquant[-1] + size_before
                self.sended_withouquant.append(novo_val)
                g_model_pruned = self.quantization(g_model_pruned)
                size_after = sys.getsizeof(pickle.dumps(g_model_pruned)) / (1024 * 1024)
                novo_val = self.sended_ammount[-1] + size_after
                self.sended_ammount.append(novo_val)
                self.send_data(conn, g_model_pruned)
                self.send_data(conn, self.prune)
                self.send_data(conn, max_amount)
            else:
                size_before = sys.getsizeof(pickle.dumps(current_global_state))/ (1024 * 1024)
                novo_val = self.sended_withouquant[-1] + size_before
                self.sended_withouquant.append(novo_val)
                current_global_state = self.quantization(current_global_state)
                size_after = sys.getsizeof(pickle.dumps(current_global_state)) / (1024 * 1024)
                novo_val = self.sended_ammount[-1] + size_after
                self.sended_ammount.append(novo_val)
                self.send_data(conn, current_global_state)
                self.send_data(conn, self.prune)

            size_saved = size_before - size_after
            
            print("-------------------------------------")
            print(f"Tamanho antes: {size_before:.2f} MB")
            print(f"Tamanho depois: {size_after:.2f} MB")
            print(f"Economia: {size_saved:.2f} MB")
            print("-------------------------------------")
            print(f"Round {round_num}: Sent global model to client {client_id}")
            
            updated_state, rate = self.recv_data(conn)
            bit_rate.append(rate)
            updated_state = self.dequantization(updated_state)
            data ,rate = self.recv_data(conn)
            self.client_data[client_id] = data
            print("client_data:", self.client_data[client_id])
            #bit_rate.append(rate)
            self.argalgo, rate = self.recv_data(conn)
            #bit_rate.append(rate)
            media_rate = sum(bit_rate)/ len(bit_rate)
            media_rate = media_rate / 8 / (1024 * 1024)
            #media_rate = str(media_rate) + '.' + str(client_id) + "." + str(self.client_data[client_id])
            self.clients_info[client_id]['bandwidth'] = media_rate / 8 / (1024 * 1024)  # Convert to MB/s
            self.clients_info[client_id][''] = self.client_data[client_id]
            self.bit.append(media_rate)
            end_time = time.time()
            training_time = end_time - start_time
            if updated_state is not None:
                with self.lock:
                    client_updates.append(updated_state)
                
                self.clients_info[client_id]['training_time'] = training_time
                print(f"Round {round_num}: Client {client_id} training completed in {training_time:.2f} seconds")
            else:
                print(f"Round {round_num}: No update received from client {client_id}")
                self.clients_info[client_id]['training_time'] = training_time
                
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
        values = [v for v in self.client_data.values() if v != None]
    
        if not values:
            return 0
        if len(values) == 1:
            data = values[0]
            non_null_times = [info['training_time'] for info in self.clients_info.values() if info.get('training_time') is not None and info['training_time'] != 0]       
            training_time = non_null_times[0]
            ammount = training_time/data
            ammount = ammount 
            return ammount
        else:
            non_null_times = [info['training_time'] for info in self.clients_info.values() if info.get('training_time') is not None and info['training_time'] != 0]       
            sorted_values = sorted(values, reverse=True)
            maior_valor = sorted_values[0]
            penultimo_maior = sorted_values[1]
            
            # Diferentes formas de calcular a distância:
            
            # 1. Diferença absoluta
            distancia_absoluta = maior_valor - penultimo_maior
            
            # 2. Diferença relativa (percentual)
            distancia_relativa = (maior_valor - penultimo_maior) / penultimo_maior if penultimo_maior != 0 else float('inf')
            
            # 3. Razão entre os valores
            razao = maior_valor / penultimo_maior if penultimo_maior != 0 else float('inf')
            
            # 4. Distância normalizada (entre 0 e 1)
            if maior_valor != 0:
                distancia_normalizada = (maior_valor - penultimo_maior) / maior_valor
                distancia_normalizada = 1 - distancia_normalizada
            else:
                distancia_normalizada = 0
            ammount = distancia_normalizada
            
            if ammount>0.9:
                ammount = 0.85
            return ammount
            '''
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
            print("ammount:", amount)
            max_amount.append(amount)
        
        if max_amount:
            average_amount = sum(max_amount) / len(max_amount)
            closest_to_average = min(max_amount, key=lambda x: abs(x - average_amount))
            
            print(f"Média dos amounts: {average_amount:.4f}")
            print(f"Valor mais próximo da média: {closest_to_average:.4f}")
            if closest_to_average == 0:
                closest_to_average = average_amount
            return 0.9
        else:
            return 0
        '''
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
            hf.create_dataset('sended_model_Mb', data=self.sended_ammount)
            hf.create_dataset('Sended_without_quant', data=self.sended_withouquant)
            hf.create_dataset('Aggregated_clients', data=self.aggregated_clients)
            hf.create_dataset('Round_time', data=self.round_time)
    
    
    def run_server(self):
        print("=== Federated Learning Server ===")
        print(f"Host: {self.args.host}:{self.args.port}")
        print(f"Dataset: {self.args.dataset}")
        print(f"Clients per round: {self.args.clients_per_round}")
        print(f"Total rounds: {self.args.rounds}")
        print(f"Test client index: {self.args.test_client_idx}")
        print("=" * 40)
        
        self.time_trashold = 100
        self.time= []
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.args.host, self.args.port))
            s.listen(self.args.max_clients)
            print(f"Server listening on {self.args.host}:{self.args.port}")
            print(f"Waiting for {self.args.clients_per_round} clients to connect...")
            clientsid =[]
            self.client_data = {index: None for index in range(1, self.args.clients_per_round+1)}
            while len(self.client_connections) < self.args.clients_per_round:
                conn, addr = s.accept()
                idx, rate = self.recv_data(conn)
                print("client idx:", idx) 
                clientsid.append(idx)
                self.clients_info[idx] = {'training_time': None, 'bandwidth': None,
            'data_size': None,  # default
            'last_flops': None,  # Para armazenar FLOPs do submodelo
            'last_params': None  # Para armazenar parâmetros do submodelo
        }
                print(f"Client {len(self.client_connections) + 1} connected: {addr}")
                self.client_idx.append(idx)
                self.client_connections.append(conn)
                self.client_addresses.append(addr)
            
            print(f"All {self.args.clients_per_round} clients connected. Starting training...")
            
            for round_num in range(self.args.rounds):
                time.sleep(5)
                print("bit_rate medio:", self.bit)
                self.bit=[]
                print(f"\n--- Round {round_num + 1}/{self.args.rounds} ---")
                client_updates = []
                threads = []

                self.stop_event = threading.Event()
                for i, conn in enumerate(self.client_connections):
                    t = threading.Thread(target=self.handle_client, daemon=True, args=(conn, client_updates, round_num + 1, clientsid[i]))
                    t.start()
                    threads.append(t)
                
                print("trashold:", self.time_trashold)
                if round_num == 1:
                    for t in threads:
                        t.join()
                else:
                    init_time = time.time()
                    for t in threads:
                        t.join(timeout=self.time_trashold)
                if round_num ==0:
                    self.set_trashold()
                end_time = time.time()
                round_duration = end_time - init_time
                print(f"Round {round_num + 1} duration: {round_duration:.2f} seconds")
                self.round_time.append(round_duration)
                print("client_updates length:", len(client_updates))
                if client_updates:
                    print(f"Round {round_num + 1}: Aggregating {len(client_updates)} client updates")
                    self.aggregated_clients.append(len(client_updates))
                    #client_updates.append(self.global_state)
                    aggregated_state = self.aggregate_models(client_updates, self.global_state)
                    
                    with self.lock:
                        self.global_state = aggregated_state
                        self.global_model.load_state_dict(self.global_state, strict=False)
                    
                    if self.test_loader is not None:
                        acc = []
                        loss = []
                        for i in self.client_idx:
                            self.test_loader = self.load_test_data(self.args.dataset, i, self.args.batch_size)
                            accuracy, avg_loss = self.evaluate_model(self.global_model, self.test_loader)
                            acc.append(accuracy)
                            loss.append(avg_loss)
                        print('acc: ', acc)
                        print('len acc', len(acc))
                        accuracy = sum(acc)/len(acc)
                        print('avg acc: ', accuracy)
                        avg_loss = sum(loss)/len(loss)
                        self.rs_test_acc.append(accuracy)
                        self.rs_test_loss.append(avg_loss)
                        print(f"Round {round_num + 1}: Test Accuracy: {accuracy:.2f}% | Test Loss: {avg_loss:.4f}")
                    else:
                        print(f"Round {round_num + 1}: Model aggregated (no test data for evaluation)")
                    
                    size_global_model = get_model_size(self.global_model)
                    print(f'Size Global Model: {size_global_model:.2f}MB')
                    
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
            
            for conn in self.client_connections:
                try:
                    conn.close()
                except:
                    pass
            print("All client connections closed.")
        self.save_results()

def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning Server')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--clients-per-round', type=int, default=3)
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
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"
    device = torch.device(args.device)
    server = FederatedLearningServer(args)
    server.run_server()

if __name__ == '__main__':
    main()
