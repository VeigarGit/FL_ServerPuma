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
import random
from prunning_nisp import prune_fc1
from prunning_snip import snip_pruning, apply_mask

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
        
        # Dicionário para armazenar máscaras por cliente
        self.client_masks = {}
        
        self.test_loader = self.load_test_data(args.dataset, args.test_client_idx, args.batch_size)
        if self.test_loader is None:
            print("Warning: Could not load test data. Evaluation will be skipped.")
            self.test_loader = None

    def aggregate_models(self, model_list, global_model, client_weights=None):
        """
        Agrega modelos com suporte a:
        1. Pruning estrutural (arquiteturas diferentes)
        2. Peso por dataset size dos clientes
        3. Alinhamento de formas diferentes
        """
        agg_state = global_model.copy()
        
        # Se não fornecer pesos, usa média simples
        if client_weights is None:
            client_weights = [1.0] * len(model_list)
        
        # Normalizar pesos para soma = 1
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]
        
        # Para cada camada no modelo global
        for key in global_model.keys():
            weighted_sum = None
            total_weight_for_key = 0.0
            
            for idx, model_state in enumerate(model_list):
                if key in model_state:
                    local_val = model_state[key]
                    global_val = global_model[key]
                    
                    # Verificar compatibilidade
                    if self._are_compatible_with_pruning(local_val, global_val, key):
                        # Converter para numpy para consistência
                        if isinstance(local_val, torch.Tensor):
                            local_np = local_val.detach().cpu().numpy()
                        elif isinstance(local_val, np.ndarray):
                            local_np = local_val
                        else:
                            local_np = np.array(local_val)
                        
                        # Alinhar valores se necessário
                        aligned_val = self._align_to_global_structure(
                            local_np, global_val, key, model_state, idx
                        )
                        
                        weight = normalized_weights[idx]
                        
                        if weighted_sum is None:
                            weighted_sum = np.zeros_like(aligned_val)
                        
                        weighted_sum += aligned_val * weight
                        total_weight_for_key += weight
            
            # Calcular média ponderada
            if weighted_sum is not None and total_weight_for_key > 0:
                agg_state[key] = weighted_sum / total_weight_for_key
                
                # Converter de volta para torch.Tensor se necessário
                if isinstance(global_model[key], torch.Tensor):
                    agg_state[key] = torch.from_numpy(agg_state[key]).to(global_model[key].device)
        
        return agg_state
    
    def _are_compatible_with_pruning(self, local_val, global_val, key):
        """
        Verifica compatibilidade considerando pruning estrutural.
        Para pruning estrutural, permitimos formas diferentes mas com estrutura similar.
        """
        # Se não são tensores/arrays, verificar comprimento básico
        if not hasattr(local_val, 'shape') or not hasattr(global_val, 'shape'):
            try:
                len_local = len(local_val) if hasattr(local_val, '__len__') else 1
                len_global = len(global_val) if hasattr(global_val, '__len__') else 1
                return len_local == len_global
            except:
                return True
        
        # Para pruning estrutural, as formas podem ser diferentes
        # Mas verificamos se têm o mesmo número de dimensões
        if local_val.ndim != global_val.ndim:
            return False
        
        # Para camadas específicas, verificar regras específicas
        if 'conv' in key and 'weight' in key:
            # Para convoluções: kernel size deve ser igual, canais podem variar
            if local_val.ndim == 4:  # [out_c, in_c, H, W]
                # Altura e largura do kernel devem ser iguais
                return local_val.shape[2:] == global_val.shape[2:]
        
        elif 'weight' in key and local_val.ndim == 2:
            # Para camadas lineares: dimensões podem variar
            return True
        
        # Para bias e outros parâmetros
        return local_val.ndim == global_val.ndim
    
    def _align_to_global_structure(self, local_val, global_val, key, model_state, client_idx):
        """
        Alinha valores locais à estrutura global para agregação.
        Se houver máscara armazenada para este cliente, usa para alinhar.
        Caso contrário, faz alinhamento padrão.
        """
        # Se as formas já são iguais, não precisa alinhar
        if local_val.shape == global_val.shape:
            return local_val
        
        # Verificar se temos máscara para este cliente
        if client_idx in self.client_masks and key in self.client_masks[client_idx]:
            return self._align_with_mask(local_val, global_val, self.client_masks[client_idx][key])
        
        # Alinhamento padrão baseado no tipo de camada
        if 'conv' in key and 'weight' in key and local_val.ndim == 4:
            return self._align_conv_weights(local_val, global_val)
        
        elif 'weight' in key and local_val.ndim == 2:
            return self._align_linear_weights(local_val, global_val)
        
        elif 'bias' in key and local_val.ndim == 1:
            return self._align_bias(local_val, global_val)
        
        # Se não conseguir alinhar, retorna o valor local (pode causar problemas)
        print(f"Aviso: Não foi possível alinhar {key}. Usando valor local.")
        return local_val
    
    def _align_with_mask(self, local_val, global_val, mask_info):
        """
        Alinha usando informações da máscara de pruning.
        """
        # Implementação simplificada - assumindo que a máscara indica quais neurônios foram mantidos
        aligned = np.zeros_like(global_val)
        
        if 'indices' in mask_info:
            # Se a máscara contém índices de mapeamento
            indices = mask_info['indices']
            if len(indices) == len(local_val):
                for i, idx in enumerate(indices):
                    if idx is not None:
                        aligned[idx] = local_val[i]
        else:
            # Alinhamento padrão por tamanho mínimo
            slices = tuple(slice(0, min(l, g)) for l, g in zip(local_val.shape, global_val.shape))
            aligned[slices] = local_val[slices]
        
        return aligned
    
    def _align_conv_weights(self, local_val, global_val):
        """
        Alinha pesos de convolução para a forma global.
        """
        if local_val.ndim != 4 or global_val.ndim != 4:
            return local_val
        
        local_out, local_in, kh, kw = local_val.shape
        global_out, global_in, kh_global, kw_global = global_val.shape
        
        # Verificar se kernel size é compatível
        if kh != kh_global or kw != kw_global:
            print(f"Aviso: Kernel size incompatível para convolução")
            return local_val
        
        # Criar tensor alinhado
        aligned = np.zeros_like(global_val)
        
        # Copiar valores onde possível
        min_out = min(local_out, global_out)
        min_in = min(local_in, global_in)
        
        aligned[:min_out, :min_in, :, :] = local_val[:min_out, :min_in, :, :]
        
        return aligned
    
    def _align_linear_weights(self, local_val, global_val):
        """
        Alinha pesos de camadas lineares para a forma global.
        """
        if local_val.ndim != 2 or global_val.ndim != 2:
            return local_val
        
        local_out, local_in = local_val.shape
        global_out, global_in = global_val.shape
        
        aligned = np.zeros_like(global_val)
        
        # Copiar valores onde possível
        min_out = min(local_out, global_out)
        min_in = min(local_in, global_in)
        
        aligned[:min_out, :min_in] = local_val[:min_out, :min_in]
        
        return aligned
    
    def _align_bias(self, local_val, global_val):
        """
        Alinha bias para a forma global.
        """
        if local_val.ndim != 1 or global_val.ndim != 1:
            return local_val
        
        local_len = local_val.shape[0]
        global_len = global_val.shape[0]
        
        aligned = np.zeros_like(global_val)
        
        min_len = min(local_len, global_len)
        aligned[:min_len] = local_val[:min_len]
        
        return aligned
    
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
    
    def new_set_amount_prune(self, client_id):
        """
        Calculate adaptive pruning rate based on client's characteristics using the latency formula:
        latency(p^c) = |D^c| · T(FLOPs^c) + B^c_params / B^c
        
        where:
        - |D^c| = dataset size of client c
        - T(FLOPs^c) = training time proportional to FLOPs
        - B^c_params = Σ α_l · p_l^c (total parameters after pruning)
        - B^c = client bandwidth
        
        Returns pruning rate between 0.2 and 0.85
        """
        client_info = self.clients_info.get(client_id)
        
        if client_info is None:
            return 0.5  # Default if no client info
        
        # Get client characteristics
        bandwidth = client_info.get('bandwidth', 1.0)  # MB/s
        training_time = client_info.get('training_time', 60.0)  # seconds
        dataset_size = client_info.get('dataset_size', 1.0)  # |D^c|
        
        if dataset_size <= 0:
            dataset_size = 1.0
        
        # Calculate FLOPs and parameters for the current model
        # Use the global model's complexity (alpha_list, beta_list)
        if not hasattr(self, 'alpha_list') or not hasattr(self, 'beta_list'):
            # Fallback to simpler calculation if complexity not calculated
            return self._fallback_prune_calculation(client_id)
        
        total_alpha = sum(self.alpha_list)  # Total parameters in original model
        total_beta = sum(self.beta_list)    # Total FLOPs in original model
        
        # Store client's computational complexity for future rounds
        if 'last_flops' not in client_info or client_info['last_flops'] is None:
            client_info['last_flops'] = total_beta
        if 'last_params' not in client_info or client_info['last_params'] is None:
            client_info['last_params'] = total_alpha
        
        # Get target latency (using server's time threshold)
        target_latency = self.time_trashold if hasattr(self, 'time_trashold') else 60.0
        
        # Calculate the relationship between pruning rate and latency
        # We solve for p (pruning rate) that satisfies: latency(p) ≤ target_latency
        
        # 1. Computational component: |D^c| · T(FLOPs^c)
        # T(FLOPs^c) is proportional to FLOPs after pruning
        # FLOPs after pruning = total_beta * (1 - p)
        
        # Estimate computation time per FLOP from previous training
        if training_time > 0 and total_beta > 0:
            # This assumes training_time includes computation for full model
            time_per_flop = training_time / (dataset_size * total_beta)
        else:
            time_per_flop = 1e-6  # Default: 1 microsecond per FLOP
        
        # 2. Communication component: B^c_params / B^c
        # B^c_params = total_alpha * (1 - p) * bytes_per_param
        # After quantization: 1 byte per parameter (int8)
        bytes_per_param = 1
        
        # Convert bandwidth from MB/s to bytes/s
        bandwidth_bytes = bandwidth * 1024 * 1024
        
        # Solve for p using the latency formula
        # latency(p) = dataset_size * (total_beta * (1-p)) * time_per_flop + 
        #              (total_alpha * (1-p) * bytes_per_param) / bandwidth_bytes
        
        # Rearrange to find p that makes latency(p) = target_latency
        # latency(p) = (1-p) * [dataset_size * total_beta * time_per_flop + total_alpha * bytes_per_param / bandwidth_bytes]
        
        computational_component = dataset_size * total_beta * time_per_flop
        communication_component = total_alpha * bytes_per_param / bandwidth_bytes
        
        total_factor = computational_component + communication_component
        
        if total_factor <= 0:
            return 0.5  # Default if calculation fails
        
        # Calculate required (1-p) to meet target latency
        required_one_minus_p = target_latency / total_factor
        
        # Ensure reasonable bounds
        if required_one_minus_p >= 1.0:
            # No pruning needed to meet latency target
            pruning_rate = 0.0
        elif required_one_minus_p <= 0.15:
            # Need significant pruning
            pruning_rate = 0.85
        else:
            pruning_rate = 1.0 - required_one_minus_p
        
        # Apply bounds
        pruning_rate = max(0.2, min(0.85, pruning_rate))
        
        # Additional adjustment based on client's past performance
        pruning_rate = self._adjust_pruning_with_history(client_id, pruning_rate)
        
        # Log detailed calculation
        print(f"\n--- Pruning Calculation for Client {client_id} ---")
        print(f"Dataset size: {dataset_size}")
        print(f"Training time: {training_time:.2f}s")
        print(f"Bandwidth: {bandwidth:.2f} MB/s ({bandwidth_bytes:.0f} bytes/s)")
        print(f"Total params (α): {total_alpha:,}")
        print(f"Total FLOPs (β): {total_beta:,}")
        print(f"Time per FLOP: {time_per_flop:.2e}s")
        print(f"Computational component: {computational_component:.2f}s")
        print(f"Communication component: {communication_component:.2f}s")
        print(f"Target latency: {target_latency:.2f}s")
        print(f"Calculated pruning rate: {pruning_rate:.3f}")
        print("-" * 50)
        
        return pruning_rate

    def _fallback_prune_calculation(self, client_id):
        """Fallback calculation when complexity data is not available"""
        client_info = self.clients_info.get(client_id)
        
        if client_info is None:
            return 0.5
        
        bandwidth = client_info.get('bandwidth', 1.0)
        training_time = client_info.get('training_time', 60.0)
        
        # Simple heuristic based on bandwidth and training time
        if bandwidth < 0.5:
            bw_factor = 0.8
        elif bandwidth < 5.0:
            bw_factor = 0.6
        elif bandwidth < 20.0:
            bw_factor = 0.4
        else:
            bw_factor = 0.2
        
        if training_time < 30:
            time_factor = 0.2
        elif training_time < 60:
            time_factor = 0.4
        elif training_time < 120:
            time_factor = 0.6
        else:
            time_factor = 0.8
        
        pruning_rate = (bw_factor + time_factor) / 2
        pruning_rate = max(0.2, min(0.85, pruning_rate))
        
        return pruning_rate

    def _adjust_pruning_with_history(self, client_id, base_pruning_rate):
        """Adjust pruning rate based on client's historical performance"""
        client_info = self.clients_info.get(client_id)
        
        if client_info is None:
            return base_pruning_rate
        
        # Check if client has consistently been slow
        training_time = client_info.get('training_time', 0)
        last_training_time = client_info.get('last_training_time', training_time)
        
        # Store current training time for next round
        client_info['last_training_time'] = training_time
        
        # If client is consistently slow, increase pruning slightly
        if training_time > last_training_time * 1.5:  # Slower than before
            adjustment = 0.05
        elif training_time > self.time_trashold * 1.2:  # Above threshold
            adjustment = 0.03
        else:
            adjustment = 0.0
        
        adjusted_rate = base_pruning_rate + adjustment
        return max(0.2, min(0.85, adjusted_rate))
    
    def handle_client(self, conn, client_updates, client_weights, round_num, client_id):
        bit_rate = []
        self.masks = []
        try:
            start_time = time.time()
            print(f"Round {round_num}: Handling client {client_id}")
            
            with self.lock:
                current_global_state = self.global_state.copy()
            
            if round_num >= 2 and self.prune == 0:
                print(f"--- SERVER: PRUNING START (Round 2) ---")
                if self.clients_info[client_id]['original_model_size'] is None:
                    #max_amount = 0.45
                    max_amount = self.new_set_amount_prune(client_id)
                    self.clients_info[client_id]['pruning_rate'] = max_amount
                else:
                    max_amount = self.clients_info[client_id]['pruning_rate']
                
                self.clients_info[client_id]['original_model_size'] = sys.getsizeof(pickle.dumps(self.global_model))/ (1024 * 1024)
                #max_amount = 0.0 m
                print(f"--- SERVER: Calculated pruning rate: {max_amount:.4f}")
                g_model_pruned = copy.deepcopy(self.global_model)
                mask = None
                if self.args.pm == 'OPALA':
                    g_model_pruned, mask = prune_and_restructure(model=g_model_pruned, pruning_rate=max_amount, size_fc=self.size_fc, data=self.args.dataset)
                elif self.args.pm == 'NISP':
                    trainloader = self.load_test_data("Cifar10", client_id, 32)
                    
                    g_model_pruned, _ = prune_fc1(model=g_model_pruned, 
                                                       dataloader=trainloader, 
                                                       pruning_ratio=max_amount,
                                                       device=self.args.device)
                    g_model_pruned, mask = prune_and_restructure(model=g_model_pruned, pruning_rate=0.0, size_fc=self.size_fc, data=self.args.dataset)
                elif self.args.pm == 'SNIP':
                    trainloader = self.load_test_data("Cifar10", client_id, 32)
                    
                    self.mask = snip_pruning(model=g_model_pruned, 
                                                  dataloader=trainloader,
                                                  criterion=nn.CrossEntropyLoss(), 
                                                  pruning_ratio=max_amount,
                                                  device=self.args.device)
                    g_model_pruned = apply_mask(g_model_pruned, self.mask)
                    g_model_pruned, mask = prune_and_restructure(model=g_model_pruned, pruning_rate=0.0, size_fc=self.size_fc, data=self.args.dataset)
                    #mask = self.mask
                self.masks.append(mask)
                
                # Armazenar máscara para este cliente para usar na agregação
                self.client_masks[client_id] = mask
                
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
            
            # Armazenar tamanho do dataset para agregação ponderada
            dataset_size = data if isinstance(data, (int, float)) else 1.0
            self.clients_info[client_id]['dataset_size'] = dataset_size
            
            print(f"Client {client_id} dataset size: {dataset_size}")
            
            self.argalgo, rate = self.recv_data(conn)
            media_rate = sum(bit_rate)/ len(bit_rate)
            media_rate = media_rate / 8 / (1024 * 1024)
            self.clients_info[client_id]['bandwidth'] = media_rate
            self.clients_info[client_id]['data'] = self.client_data[client_id]
            self.bit.append(media_rate)
            
            end_time = time.time()
            training_time = end_time - start_time
            
            if updated_state is not None:
                with self.lock:
                    client_updates.append(updated_state)
                    # Adicionar peso proporcional ao tamanho do dataset
                    client_weights.append(dataset_size)
                
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
            
            distancia_absoluta = maior_valor - penultimo_maior
            
            distancia_relativa = (maior_valor - penultimo_maior) / penultimo_maior if penultimo_maior != 0 else float('inf')
            
            razao = maior_valor / penultimo_maior if penultimo_maior != 0 else float('inf')
            
            if maior_valor != 0:
                distancia_normalizada = (maior_valor - penultimo_maior) / maior_valor
                distancia_normalizada = 1 - distancia_normalizada
            else:
                distancia_normalizada = 0
            ammount = distancia_normalizada
            
            if ammount>0.9:
                ammount = 0.85
            return ammount
    
    def save_results(self, i):
        i =str(i)
        if self.args.prune == 0:
            a = "prune"
        else:
            a = "withou_Prune"
        b = self.argalgo
        if b == 0:
            b = "FedALA"
        else:
            b = "FedAVG"
        algo = self.args.dataset + "_" + a + "_" + b + "_" + i
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
    
    
    def run_server(self, times):
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
                self.clients_info[idx] = {
                    'training_time': None, 
                    'bandwidth': None, 
                    'original_model_size': None, 
                    'pruning_rate': None,
                    'data_size': None,  # default
                    'last_flops': None,
                    'last_params': None,
                    'dataset_size': 1.0  # Inicializar com 1.0 para média simples caso não receba
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
                client_weights = []  # Lista para armazenar pesos dos clientes
                threads = []

                self.stop_event = threading.Event()
                for i, conn in enumerate(self.client_connections):
                    t = threading.Thread(target=self.handle_client, daemon=True, 
                                         args=(conn, client_updates, client_weights, round_num + 1, clientsid[i]))
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
                print("client_weights:", client_weights)
                
                if client_updates:
                    print(f"Round {round_num + 1}: Aggregating {len(client_updates)} client updates")
                    self.aggregated_clients.append(len(client_updates))
                    
                    # Agregar com pesos proporcionais ao tamanho do dataset
                    aggregated_state = self.aggregate_models(client_updates, self.global_state, client_weights)
                    
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
        self.save_results(times)

def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning Server')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--clients-per-round', type=int, default=8)
    parser.add_argument('--rounds', type=int, default=25)
    parser.add_argument('--dataset', type=str, default='Cifar10', choices=['Cifar10', 'MNIST', 'FashionMNIST', 'Cifar100'])
    parser.add_argument('--test-client-idx', type=int, default=0)
    parser.add_argument('--in-features', type=int, default=3)
    parser.add_argument('--num-classes', type=int, default=100)
    parser.add_argument('--dim', type=int, default=1600)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-clients', type=int, default=10)
    parser.add_argument('--prune', type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument('--pm', type=str, default='OPALA', choices=['OPALA', 'SNIP', 'NISP'])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    return parser.parse_args()

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"
    device = torch.device(args.device)
    for i in range(0, 11):
        server = FederatedLearningServer(args)
        server.run_server(i)

if __name__ == '__main__':
    main()