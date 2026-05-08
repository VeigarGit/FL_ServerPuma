import socket
import pickle
import struct
import threading
import copy
import time
import argparse
import sys
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import h5py

#Para o Lora
from lora_clip.clip_setup import (
    parse_args, load_config, get_device, resolve_run_modes, 
    build_run_config, build_output_path, build_dataloaders,
    build_model, build_optimizer, build_scheduler, 
    benchmark_attention, quantize_weights
)

# --- HACK PERMITIDO: Inserindo a pasta 'src' no path do Python ---
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))
# -----------------------------------------------------------------

from model import SimpleModel
from data_utils import read_client_data
from prunning import prune_and_restructure
from size_mode import get_model_size
from synexp import LayerComplexityCalculator
from prunning_nisp import prune_fc1
from prunning_snip import snip_pruning, apply_mask
from utils.network_utils import send_data, recv_data
from utils.fl_math import quantization, dequantization, evaluate_model, set_parameters

# --- CONFIGURAÇÃO DE OBSERVABILIDADE (LOGGING) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
# -------------------------------------------------

class FederatedLearningServer:
    def __init__(self, args):
        self.args = args
        
        # Uso do Structural Pattern Matching (Substituindo a cadeia de IFs)
        match args.model:
            case 'cnn':
                match args.dataset:
                    case 'MNIST':
                        self.global_model = SimpleModel(in_features=1, num_classes=10, dim=1024)
                        self.input_size = (1, 1, 28, 28)
                    case 'Cifar10':
                        self.global_model = SimpleModel(in_features=args.in_features, num_classes=10, dim=args.dim)
                        self.input_size = (1, 3, 32, 32)
                    case 'Cifar100':
                        self.global_model = SimpleModel(in_features=args.in_features, num_classes=args.num_classes, dim=args.dim)
                        self.input_size = (1, 3, 32, 32)
                    case _:
                        logger.error(f"Dataset não reconhecido: {args.dataset}")
                        raise ValueError(f"Dataset inválido: {args.dataset}")
            case 'clip':
                config = load_config(args.config)
                match config["dataset"]["name"]:
                    case "enterprise-explorers/oxford-pets":
                        # 1. Carrega o YAML passado no argumento --config
                        
                        # 2. Descobre o modo (ex: "with_lora") configurado no YAML
                        run_mode = resolve_run_modes(config)[0]
                        run_config = build_run_config(config, run_mode=run_mode)
                        
                        # 3. Instancia o CLIP (Oxford Pets tem 37 classes)
                        self.global_model = build_model(
                            config=run_config, 
                            num_classes=args.num_classes, 
                            device=args.device
                        )
                        
                        # 4. Input size padrão do processador do CLIP (1 batch, 3 canais, 224x224)
                        # Necessário para o LayerComplexityCalculator não quebrar logo abaixo
                        self.input_size = (1, 3, 224, 224)
                        
                

        self.global_model = self.global_model.to(args.device)
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
        self.sended_ammount = [0]
        self.sended_withouquant = [0]
        self.aggregated_clients = []
        self.round_time = []
        self.bit = []
        self.complexity_calculated = False
        
        calculator = LayerComplexityCalculator(self.global_model, self.input_size)
        
        # Calcular complexidade
        alpha_list, beta_list = calculator.calculate_alpha_beta()
        self.alpha_list = alpha_list
        self.beta_list = beta_list
        self.complexity_calculated = True
        
        logger.info(f"Calculado: {len(alpha_list)} camadas")
        logger.info(f"Total α: {sum(alpha_list):,} parâmetros")
        logger.info(f"Total β: {sum(beta_list):,} FLOPs")
        
        # Dicionário para armazenar máscaras por cliente
        self.client_masks = {}
        
        self.test_loader = self.load_test_data(args.dataset, args.test_client_idx, args.batch_size)
        if self.test_loader is None:
            logger.warning("Could not load test data. Evaluation will be skipped.")

    def aggregate_models(self, model_list, global_model, client_weights=None):
        agg_state = global_model.copy()
        
        if client_weights is None:
            client_weights = [1.0] * len(model_list)
        
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]
        
        for key in global_model.keys():
            weighted_sum = None
            total_weight_for_key = 0.0
            
            for idx, model_state in enumerate(model_list):
                if key in model_state:
                    local_val = model_state[key]
                    global_val = global_model[key]
                    
                    if self._are_compatible_with_pruning(local_val, global_val, key):
                        # Conversões para Numpy
                        if isinstance(local_val, torch.Tensor):
                            local_np = local_val.detach().cpu().numpy()
                        else:
                            local_np = np.array(local_val)
                            
                        if isinstance(global_val, torch.Tensor):
                            global_np = global_val.detach().cpu().numpy()
                        else:
                            global_np = np.array(global_val)

                        aligned_val = self._align_to_global_structure(
                            local_np, global_np, key, model_state, idx
                        )
                        
                        weight = normalized_weights[idx]
                        
                        if weighted_sum is None:
                            weighted_sum = np.zeros_like(aligned_val)
                        
                        weighted_sum += aligned_val * weight
                        total_weight_for_key += weight
            
            if weighted_sum is not None and total_weight_for_key > 0:
                agg_state[key] = weighted_sum / total_weight_for_key
                if isinstance(global_model[key], torch.Tensor):
                    agg_state[key] = torch.from_numpy(agg_state[key]).to(global_model[key].device)
        
        return agg_state
    
    def _are_compatible_with_pruning(self, local_val, global_val, key):
        if not hasattr(local_val, 'shape') or not hasattr(global_val, 'shape'):
            try:
                len_local = len(local_val) if hasattr(local_val, '__len__') else 1
                len_global = len(global_val) if hasattr(global_val, '__len__') else 1
                return len_local == len_global
            except TypeError:
                return True
        
        if local_val.ndim != global_val.ndim:
            return False
        
        if 'conv' in key and 'weight' in key:
            if local_val.ndim == 4:
                return local_val.shape[2:] == global_val.shape[2:]
        elif 'weight' in key and local_val.ndim == 2:
            return True
            
        return local_val.ndim == global_val.ndim
    
    def _align_to_global_structure(self, local_val, global_val, key, model_state, client_idx):
        if local_val.shape == global_val.shape:
            return local_val
        
        if client_idx in self.client_masks and key in self.client_masks[client_idx]:
            return self._align_with_mask(local_val, global_val, self.client_masks[client_idx][key])
        
        if 'conv' in key and 'weight' in key and local_val.ndim == 4:
            return self._align_conv_weights(local_val, global_val)
        elif 'weight' in key and local_val.ndim == 2:
            return self._align_linear_weights(local_val, global_val)
        elif 'bias' in key and local_val.ndim == 1:
            return self._align_bias(local_val, global_val)
        
        logger.warning(f"Não foi possível alinhar {key}. Usando valor local.")
        return local_val
    
    def _align_with_mask(self, local_val, global_val, mask_info):
        aligned = np.zeros_like(global_val)
        if 'indices' in mask_info:
            indices = mask_info['indices']
            if len(indices) == len(local_val):
                for i, idx in enumerate(indices):
                    if idx is not None:
                        aligned[idx] = local_val[i]
        else:
            slices = tuple(slice(0, min(l, g)) for l, g in zip(local_val.shape, global_val.shape))
            aligned[slices] = local_val[slices]
        return aligned
    
    def _align_conv_weights(self, local_val, global_val):
        if local_val.ndim != 4 or global_val.ndim != 4:
            return local_val
        
        local_out, local_in, kh, kw = local_val.shape
        global_out, global_in, kh_global, kw_global = global_val.shape
        
        if kh != kh_global or kw != kw_global:
            logger.warning("Kernel size incompatível para convolução")
            return local_val
        
        aligned = np.zeros_like(global_val)
        min_out = min(local_out, global_out)
        min_in = min(local_in, global_in)
        aligned[:min_out, :min_in, :, :] = local_val[:min_out, :min_in, :, :]
        return aligned
    
    def _align_linear_weights(self, local_val, global_val):
        if local_val.ndim != 2 or global_val.ndim != 2:
            return local_val
        
        local_out, local_in = local_val.shape
        global_out, global_in = global_val.shape
        
        aligned = np.zeros_like(global_val)
        min_out = min(local_out, global_out)
        min_in = min(local_in, global_in)
        aligned[:min_out, :min_in] = local_val[:min_out, :min_in]
        return aligned
    
    def _align_bias(self, local_val, global_val):
        if local_val.ndim != 1 or global_val.ndim != 1:
            return local_val
        
        local_len = local_val.shape[0]
        global_len = global_val.shape[0]
        
        aligned = np.zeros_like(global_val)
        min_len = min(local_len, global_len)
        aligned[:min_len] = local_val[:min_len]
        return aligned
    
    
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
        for info in self.clients_info.values():
            if info.get('training_time') is not None:
                tot_time += info['training_time']
                clients_with_time += 1
        
        if clients_with_time == 0:
            logger.warning("No clients reported training time. Sticking to default threshold.")
            return

        mean_time = tot_time / clients_with_time
        self.time_trashold = 0.9 * mean_time
        logger.debug(f'time_trashold: {self.time_trashold}s')
    
    def new_set_amount_prune(self, client_id):
        # Filosofia EAFP: Tenta acessar direto e falha limpo
        try:
            client_info = self.clients_info[client_id]
        except KeyError:
            logger.warning(f"Informações do cliente {client_id} ausentes. Usando pruning rate padrão.")
            return 0.5
        
        bandwidth = client_info.get('bandwidth', 1.0)
        training_time = client_info.get('training_time', 60.0)
        dataset_size = max(client_info.get('dataset_size', 1.0), 1.0)
        
        if not hasattr(self, 'alpha_list') or not hasattr(self, 'beta_list'):
            return self._fallback_prune_calculation(client_id)
        
        total_alpha = sum(self.alpha_list)
        total_beta = sum(self.beta_list)
        
        client_info.setdefault('last_flops', total_beta)
        client_info.setdefault('last_params', total_alpha)
        
        target_latency = getattr(self, 'time_trashold', 60.0)
        
        time_per_flop = (training_time / (dataset_size * total_beta)) if (training_time > 0 and total_beta > 0) else 1e-6
        bytes_per_param = 1
        bandwidth_bytes = bandwidth * 1024 * 1024
        
        computational_component = dataset_size * total_beta * time_per_flop
        communication_component = total_alpha * bytes_per_param / bandwidth_bytes
        total_factor = computational_component + communication_component
        
        if total_factor <= 0:
            return 0.5
        
        required_one_minus_p = target_latency / total_factor
        
        if required_one_minus_p >= 1.0:
            pruning_rate = 0.0
        elif required_one_minus_p <= 0.15:
            pruning_rate = 0.85
        else:
            pruning_rate = 1.0 - required_one_minus_p
        
        pruning_rate = max(0.2, min(0.85, pruning_rate))
        pruning_rate = self._adjust_pruning_with_history(client_id, pruning_rate)
        
        logger.debug(f"Calculated pruning rate para {client_id}: {pruning_rate:.3f}")
        return pruning_rate

    def _fallback_prune_calculation(self, client_id):
        try:
            client_info = self.clients_info[client_id]
        except KeyError:
            return 0.5
        
        bandwidth = client_info.get('bandwidth', 1.0)
        training_time = client_info.get('training_time', 60.0)
        
        bw_factor = 0.8 if bandwidth < 0.5 else 0.6 if bandwidth < 5.0 else 0.4 if bandwidth < 20.0 else 0.2
        time_factor = 0.2 if training_time < 30 else 0.4 if training_time < 60 else 0.6 if training_time < 120 else 0.8
        
        pruning_rate = (bw_factor + time_factor) / 2
        return max(0.2, min(0.85, pruning_rate))

    def _adjust_pruning_with_history(self, client_id, base_pruning_rate):
        try:
            client_info = self.clients_info[client_id]
        except KeyError:
            return base_pruning_rate
        
        training_time = client_info.get('training_time', 0)
        last_training_time = client_info.get('last_training_time', training_time)
        client_info['last_training_time'] = training_time
        
        adjustment = 0.05 if training_time > last_training_time * 1.5 else (0.03 if training_time > getattr(self, 'time_trashold', 0) * 1.2 else 0.0)
        return max(0.2, min(0.85, base_pruning_rate + adjustment))
    
    def handle_client(self, conn, client_updates, client_weights, round_num, client_id):
        bit_rate = []
        self.masks = []
        try:
            start_time = time.time()
            logger.info(f"Round {round_num}: Handling client {client_id}")
            
            with self.lock:
                if self.args.model == 'clip':
                    from lora_clip.sora import get_trainable_state_dict
                    current_global_state = get_trainable_state_dict(self.global_model)
                else:
                    current_global_state = self.global_state.copy()
            
            if round_num >= 2 and self.prune == 0:
                logger.info("--- SERVER: PRUNING START (Round 2) ---")
                if self.clients_info[client_id]['original_model_size'] is None:
                    max_amount = self.new_set_amount_prune(client_id)
                    self.clients_info[client_id]['pruning_rate'] = max_amount
                else:
                    max_amount = self.clients_info[client_id]['pruning_rate']
                
                self.clients_info[client_id]['original_model_size'] = sys.getsizeof(pickle.dumps(self.global_model)) / (1024 * 1024)
                logger.info(f"--- SERVER: Calculated pruning rate: {max_amount:.4f}")
                
                g_model_pruned = copy.deepcopy(self.global_model)
                mask = None
                
                if self.args.pm == 'OPALA':
                    g_model_pruned, mask = prune_and_restructure(model=g_model_pruned, pruning_rate=max_amount, size_fc=self.size_fc, data=self.args.dataset)
                elif self.args.pm == 'NISP':
                    trainloader = self.load_test_data("Cifar10", client_id, 32)
                    g_model_pruned, _ = prune_fc1(model=g_model_pruned, dataloader=trainloader, pruning_ratio=max_amount, device=self.args.device)
                    g_model_pruned, mask = prune_and_restructure(model=g_model_pruned, pruning_rate=0.0, size_fc=self.size_fc, data=self.args.dataset)
                elif self.args.pm == 'SNIP':
                    trainloader = self.load_test_data("Cifar10", client_id, 32)
                    self.mask = snip_pruning(model=g_model_pruned, dataloader=trainloader, criterion=nn.CrossEntropyLoss(), pruning_ratio=max_amount, device=self.args.device)
                    g_model_pruned = apply_mask(g_model_pruned, self.mask)
                    g_model_pruned, mask = prune_and_restructure(model=g_model_pruned, pruning_rate=0.0, size_fc=self.size_fc, data=self.args.dataset)
                
                self.masks.append(mask)
                self.client_masks[client_id] = mask
                g_model_pruned = g_model_pruned.state_dict()
                logger.info("--- SERVER: PRUNING COMPLETE ---")

            if round_num >= 2 and self.prune == 0:
                size_before = sys.getsizeof(pickle.dumps(g_model_pruned)) / (1024 * 1024)
                self.sended_withouquant.append(self.sended_withouquant[-1] + size_before)
                g_model_pruned = quantization(g_model_pruned)
                size_after = sys.getsizeof(pickle.dumps(g_model_pruned)) / (1024 * 1024)
                self.sended_ammount.append(self.sended_ammount[-1] + size_after)
                
                send_data(conn, g_model_pruned)
                send_data(conn, self.prune)
                send_data(conn, max_amount)
            else:
                size_before = sys.getsizeof(pickle.dumps(current_global_state)) / (1024 * 1024)
                self.sended_withouquant.append(self.sended_withouquant[-1] + size_before)
                current_global_state = quantization(current_global_state)
                size_after = sys.getsizeof(pickle.dumps(current_global_state)) / (1024 * 1024)
                self.sended_ammount.append(self.sended_ammount[-1] + size_after)
                
                send_data(conn, current_global_state)
                send_data(conn, self.prune)

            size_saved = size_before - size_after
            logger.info(f"Tamanho antes: {size_before:.2f} MB | Tamanho depois: {size_after:.2f} MB | Economia: {size_saved:.2f} MB")
            logger.info(f"Round {round_num}: Sent global model to client {client_id}")
            
            updated_state, rate = recv_data(conn)
            bit_rate.append(rate)
            
            if updated_state is None:
                logger.warning(f"Round {round_num}: Cliente {client_id} falhou/desconectou. Abortando thread.")
                self.clients_info[client_id]['training_time'] = time.time() - start_time
                return
            
            updated_state = dequantization(updated_state)
            data, rate = recv_data(conn)
            self.client_data[client_id] = data
            
            dataset_size = data if isinstance(data, (int, float)) else 1.0
            self.clients_info[client_id]['dataset_size'] = dataset_size
            
            logger.debug(f"Client {client_id} dataset size: {dataset_size}")
            
            self.argalgo, rate = recv_data(conn)
            media_rate = (sum(bit_rate) / len(bit_rate)) / 8 / (1024 * 1024)
            self.clients_info[client_id]['bandwidth'] = media_rate
            self.clients_info[client_id]['data'] = self.client_data[client_id]
            self.bit.append(media_rate)
            
            training_time = time.time() - start_time
            
            if updated_state is not None:
                with self.lock:
                    client_updates.append(updated_state)
                    client_weights.append(dataset_size)
                
                self.clients_info[client_id]['training_time'] = training_time
                logger.info(f"Round {round_num}: Client {client_id} training completed in {training_time:.2f} seconds")
            else:
                logger.warning(f"Round {round_num}: No update received from client {client_id}")
                self.clients_info[client_id]['training_time'] = training_time
                
        # Tratamento Exato de Exceções de Rede e Pickle 
        except (OSError, pickle.PickleError, struct.error) as e:
            logger.exception(f"Round {round_num}: Network/Serialization error handling client {client_id}")
        except Exception as e:
            logger.exception(f"Round {round_num}: Critical error handling client {client_id}")

    def load_test_data(self, dataset, client_idx, batch_size=32):
        try:
            test_data = read_client_data(dataset, client_idx, is_train=False)
            X, y = zip(*test_data)
            X = torch.stack(X)
            y = torch.tensor(y)
            dataset = torch.utils.data.TensorDataset(X, y)
            return DataLoader(dataset, batch_size=batch_size)
        except Exception as e:
            logger.exception("Error loading test data")
            return None
    
    def set_amount_prune(self):
        values = [v for v in self.client_data.values() if v is not None]
        if not values:
            return 0
        
        if len(values) == 1:
            data = values[0]
            non_null_times = [info['training_time'] for info in self.clients_info.values() if info.get('training_time')]
            return non_null_times[0] / data if non_null_times else 0.0
        else:
            sorted_values = sorted(values, reverse=True)
            maior_valor = sorted_values[0]
            penultimo_maior = sorted_values[1]
            
            distancia_normalizada = (1 - ((maior_valor - penultimo_maior) / maior_valor)) if maior_valor != 0 else 0
            return 0.85 if distancia_normalizada > 0.9 else distancia_normalizada
    
    def save_results(self, i: int):
        i_str = str(i)
        a = "prune" if self.args.prune == 0 else "withou_Prune"
        b = "FedALA" if getattr(self, 'argalgo', 0) == 0 else "FedAVG"
        algo = f"{self.args.dataset}_{a}_{b}_{i_str}"
        
        # O Descarte Definitivo da Biblioteca os.path na Gestão Relacional 
        result_path = Path("..") / "results"
        result_path.mkdir(parents=True, exist_ok=True)
        
        file_path = result_path / f"{algo}.h5"
        
        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
            hf.create_dataset('rs_train_loss', data=self.rs_test_loss)
            hf.create_dataset('sended_model_Mb', data=self.sended_ammount)
            hf.create_dataset('Sended_without_quant', data=self.sended_withouquant)
            hf.create_dataset('Aggregated_clients', data=self.aggregated_clients)
            hf.create_dataset('Round_time', data=self.round_time)
            
    def run_server(self, times):
        logger.info("=== Federated Learning Server ===")
        logger.info(f"Host: {self.args.host}:{self.args.port}")
        logger.info(f"Dataset: {self.args.dataset}")
        logger.info(f"Clients per round: {self.args.clients_per_round}")
        logger.info(f"Total rounds: {self.args.rounds}")
        logger.info("=======================================")
        
        self.time_trashold = 100
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.args.host, self.args.port))
            s.listen(self.args.max_clients)
            
            logger.info(f"Server listening on {self.args.host}:{self.args.port}")
            logger.info(f"Waiting for {self.args.clients_per_round} clients to connect...")
            
            clientsid = []
            self.client_data = {index: None for index in range(1, self.args.clients_per_round + 1)}
            
            while len(self.client_connections) < self.args.clients_per_round:
                conn, addr = s.accept()
                idx, rate = recv_data(conn)
                logger.info(f"Client idx: {idx}") 
                clientsid.append(idx)
                
                self.clients_info[idx] = {
                    'training_time': None, 
                    'bandwidth': None, 
                    'original_model_size': None, 
                    'pruning_rate': None,
                    'data_size': None,
                    'last_flops': None,
                    'last_params': None,
                    'dataset_size': 1.0 
                }
                
                logger.info(f"Client {len(self.client_connections) + 1} connected: {addr}")
                self.client_idx.append(idx)
                self.client_connections.append(conn)
                self.client_addresses.append(addr)
            
            logger.info(f"All {self.args.clients_per_round} clients connected. Starting training...")
            
            for round_num in range(self.args.rounds):
                time.sleep(5)
                self.bit = []
                logger.info(f"\n--- Round {round_num + 1}/{self.args.rounds} ---")
                
                client_updates = []
                client_weights = []
                threads = []
                self.stop_event = threading.Event()
                
                for i, conn in enumerate(self.client_connections):
                    t = threading.Thread(
                        target=self.handle_client, 
                        daemon=True, 
                        args=(conn, client_updates, client_weights, round_num + 1, clientsid[i])
                    )
                    t.start()
                    threads.append(t)
                
                if round_num == 1:
                    for t in threads: t.join()
                else:
                    init_time = time.time()
                    for t in threads: t.join(timeout=self.time_trashold)
                
                if round_num == 0:
                    self.set_trashold()
                    
                round_duration = time.time() - (init_time if round_num != 1 else time.time())
                self.round_time.append(round_duration)
                
                logger.info(f"Round {round_num + 1} duration: {round_duration:.2f} seconds")
                logger.debug(f"Client updates length: {len(client_updates)}")
                
                if client_updates:
                    logger.info(f"Round {round_num + 1}: Aggregating {len(client_updates)} updates")
                    self.aggregated_clients.append(len(client_updates))
                    
                    aggregated_state = self.aggregate_models(client_updates, self.global_state, client_weights)
                    
                    with self.lock:
                        self.global_state = aggregated_state
                        self.global_model.load_state_dict(self.global_state, strict=False)
                    
                    if self.test_loader is not None:
                        acc, loss = [], []
                        for i in self.client_idx:
                            self.test_loader = self.load_test_data(self.args.dataset, i, self.args.batch_size)
                            accuracy, avg_loss = evaluate_model(self.global_model, self.test_loader)
                            acc.append(accuracy)
                            loss.append(avg_loss)
                            
                        final_accuracy = sum(acc) / len(acc)
                        final_loss = sum(loss) / len(loss)
                        
                        self.rs_test_acc.append(final_accuracy)
                        self.rs_test_loss.append(final_loss)
                        logger.info(f"Round {round_num + 1}: Test Acc: {final_accuracy:.2f}% | Loss: {final_loss:.4f}")
                    else:
                        logger.info(f"Round {round_num + 1}: Model aggregated (no test data)")
                    
                    size_global_model = get_model_size(self.global_model)
                    logger.info(f'Size Global Model: {size_global_model:.2f}MB')
                    
                    successful_notifications = 0
                    for conn in self.client_connections:
                        try:
                            conn.send('end'.encode('utf-8'))
                            successful_notifications += 1
                        except OSError as e:
                            logger.error(f"Error notifying client: {e}")
                    
                    logger.info(f"Round {round_num + 1}: Notified {successful_notifications} clients.")
                else:
                    logger.warning(f"Round {round_num + 1}: No client updates received.")
            
            logger.info(f"\nTraining completed after {self.args.rounds} rounds!")
            
            # Tratamento tolerante para fechamento residual dos soquetes TCP 
            for conn in self.client_connections:
                try:
                    conn.close()
                except OSError as e:
                    logger.debug(f"Socket close exception (safe to ignore): {e}")
            logger.info("All client connections closed.")
            
        self.save_results(times)

def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning Server')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--clients-per-round', type=int, default=8)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='Cifar10', choices=['Cifar10', 
                                                                           'MNIST', 'FashionMNIST', 
                                                                           'Cifar100', 
                                                                           "OxfordPets"])
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
    parser.add_argument('--experiments', type=int, default=1)
    parser.add_argument('--model', type=str, default="cnn", choices=["cnn", "clip"])
    parser.add_argument('--config', type=str, default="lora_clip/config.yaml")
    
    parser.add_argument('--prune-freq', type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Substituir os.environ por um controle interno (caso se aplique futuramente, mas mantido torch contextually)
    import os 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available. Falling back to CPU.")
        args.device = "cpu"
    
    for i in range(args.experiments):
        logger.info(f"\n=== Iniciando Experimento {i+1}/{args.experiments} ===")
        server = FederatedLearningServer(args)
        server.run_server(i)

if __name__ == '__main__':
    main()