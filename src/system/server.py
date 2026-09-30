import socket
import threading
import time
import argparse
import sys
import os
import logging
from pathlib import Path

# Desativa checagem online redundante do Hugging Face Hub (usa cache local instantaneamente)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

import torch

#Para o Lora
from lora_clip.clip_setup import (
    load_config, resolve_run_modes, build_run_config, build_model
)
from lora_clip.sora import get_trainable_state_dict

from utils.fl_math import resize_model_to_pruned

from lora_clip.clip_sora_utils import reduce_sora_state_dict_rank

# Imports SLM
from lora_slm.slm_setup import (
    load_config as load_config_slm, resolve_run_modes as resolve_run_modes_slm, 
    build_run_config as build_run_config_slm, build_model as build_model_slm
)
from lora_slm.sora import get_trainable_state_dict as get_trainable_state_dict_slm


# --- HACK PERMITIDO: Inserindo a pasta 'src' no path do Python ---
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))
# -----------------------------------------------------------------

from model import SimpleModel
from size_mode import get_trainable_size_and_params
from synexp import LayerComplexityCalculator
from utils.network_utils import send_data, recv_data
from utils.fl_math import evaluate_model

# --- CONFIGURAÇÃO DE OBSERVABILIDADE (LOGGING) ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
# -------------------------------------------------

from server_aggregation import ServerAggregationMixin
from server_adaptation import ServerAdaptationMixin
from server_exchange import ServerExchangeMixin, get_state_dict_size_mb as get_state_dict_size_mb
from server_results import ServerResultsMixin

class FederatedLearningServer(
    ServerExchangeMixin, ServerAdaptationMixin, ServerAggregationMixin, ServerResultsMixin
):
    def __init__(self, args):
        self.args = args
        self.config = None
        self.slm_collator = None
        self.test_loaders_cache = {}
        
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
            case 'clip' | 'slm':
                if args.model == 'clip':
                    config = load_config(args.config)
                else:
                    config = load_config_slm(args.config)
                self.config = config
                
                # --- INÍCIO DA SOBRESCRITA DA ESTRATÉGIA ---
                # Pega a flag do terminal (--strategy) e força o modo no dicionário do YAML
                if args.strategy == 'lora':
                    print("Linha 82 =========================================")
                    config["model"]["lora"]["mode"] = "with_lora"
                elif args.strategy == 'sora_no_schedule':
                    print("Linha 85 =========================================")
                    config["model"]["lora"]["mode"] = "with_sora_no_schedule" 
                elif args.strategy == 'sora_with_schedule':
                    print("Linha 88 =========================================")
                    config["model"]["lora"]["mode"] = "with_sora_schedule" 
                elif args.strategy == 'full_ft':
                    print("Linha 91 =========================================")
                    config["model"]["lora"]["mode"] = "without_lora"
                elif args.strategy == 'adalora':
                    print("Linha AdaLoRA =========================================")
                    config["model"]["lora"]["mode"] = "with_adalora"
                    
                if "paca" not in config["model"]:
                    config["model"]["paca"] = {}
                    
                if "lora" in config["model"]:
                    config["model"]["lora"]["r"] = args.rank
                
                if args.paca is not None and args.paca > 0:
                    config["model"]["paca"]["enabled"] = True
                    config["model"]["paca"]["upper_layers"] = args.paca
                else:
                    config["model"]["paca"]["enabled"] = False
                    config["model"]["paca"]["upper_layers"] = None

                if args.model == 'clip':
                    run_mode = resolve_run_modes(config)[0]
                    run_config = build_run_config(config, run_mode=run_mode)
                    
                    self.global_model = build_model(
                        config=run_config, 
                        num_classes=args.num_classes, 
                        device=args.device
                    )
                    self.input_size = (1, 3, 224, 224)
                else:
                    run_mode = resolve_run_modes_slm(config)[0]
                    run_config = build_run_config_slm(config, run_mode=run_mode)
                    
                    self.global_model = build_model_slm(
                        config=run_config, 
                        num_classes=args.num_classes, 
                        device=args.device
                    )
                    self.input_size = None # SLM usa o processor
                    
                    try:
                        from lora_slm.slm_setup import CustomCollator
                        from transformers import AutoProcessor
                        model_name = self.config.get("model", {}).get("name", "Qwen/Qwen3-VL-2B-Instruct") if self.config else "Qwen/Qwen3-VL-2B-Instruct"
                        processor = AutoProcessor.from_pretrained(model_name)
                        label_to_idx = {i: i for i in range(1000)}
                        label_to_idx.update({str(i): i for i in range(1000)})
                        self.slm_collator = CustomCollator(processor, label_to_idx)
                    except Exception as e:
                        logger.warning(f"Não foi possível pré-carregar CustomCollator para SLM na inicialização: {e}")
                        
                        
        if args.load_model:
            logger.info(f"Carregando pesos salvos de: {args.load_model}")
            state_dict = torch.load(args.load_model, map_location=args.device)
            
            # 1. Redimensiona a arquitetura do modelo em memória para o tamanho podado
            self.global_model = resize_model_to_pruned(self.global_model, state_dict)
            
            # 2. Agora os formatos coincidem, podemos carregar os pesos com segurança
            self.global_model.load_state_dict(state_dict, strict=False)
            self.global_state = self.global_model.state_dict()
                

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
        self.received_ammount = [0]
        self.total_transmitted_per_round = []
        self.rs_client_paca = []
        self.rs_client_training_time = []
        self.rs_client_comm_time = []
        self.rs_client_eval_time = []
        self.rs_server_processing_time = []
        self.server_pruning_time = []
        self.aggregated_clients = []
        self.round_time = []
        self.model_size_per_round = []
        self.trainable_params_per_round = []
        self.bit = []
        self.complexity_calculated = False
        
        # Performance tracking variables
        self.time_threshold = 0.0
        self.mean_training_time = 0.0
        
        # LHDQ: cache do estado anterior por cliente (para delta coding)
        self.previous_sent_state = {}   # client_id → state_dict enviado na rodada anterior
        self.previous_recv_state = {}   # client_id → state_dict recebido na rodada anterior
        
        if self.args.model == 'slm':
            alpha_list, beta_list = [0], [0]
            logger.info("Cálculo de complexidade ignorado para SLM (forward multimodal complexo).")
        else:
            calculator = LayerComplexityCalculator(self.global_model, self.input_size)
            alpha_list, beta_list = calculator.calculate_alpha_beta()
            
        self.alpha_list = alpha_list
        self.beta_list = beta_list
        self.complexity_calculated = True
        
        logger.info(f"Calculado: {len(alpha_list)} camadas")
        logger.info(f"Total α: {sum(alpha_list):,} parâmetros")
        logger.info(f"Total β: {sum(beta_list):,} FLOPs")
        
        # Detectar total de encoder layers dinamicamente (para PaCA)
        if hasattr(self.global_model, 'vision_model') and hasattr(self.global_model.vision_model, 'encoder'):
            self.total_encoder_layers = len(self.global_model.vision_model.encoder.layers)
        elif self.args.model == 'slm':
            from lora_slm.slm_setup import _get_encoder_layers
            layers = _get_encoder_layers(self.global_model)
            self.total_encoder_layers = len(layers) if layers else self.args.paca
        else:
            self.total_encoder_layers = self.args.paca  # fallback
        logger.info(f"Total encoder layers detectado: {self.total_encoder_layers}")
        
        # Dicionário para armazenar máscaras por cliente
        self.client_masks = {}
        
        self.test_loader = self.load_test_data(args.dataset, args.test_client_idx, args.batch_size)
        if self.test_loader is not None:
            self.test_loaders_cache[args.test_client_idx if args.test_client_idx is not None else 0] = self.test_loader
        else:
            logger.warning("Could not load test data. Evaluation will be skipped.")
    def run_server(self, times):
        logger.info("=== Federated Learning Server ===")
        logger.info(f"Host: {self.args.host}:{self.args.port}")
        logger.info(f"Dataset: {self.args.dataset}")
        logger.info(f"Clients per round: {self.args.clients_per_round}")
        logger.info(f"Total rounds: {self.args.rounds}")
        logger.info("=======================================")
        
        if self.args.model == 'slm':
            self.time_threshold = 1800.0
        elif self.args.model == 'clip':
            self.time_threshold = 600.0
        else:
            self.time_threshold = 120.0
        
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
                    'dataset_size': 1.0,
                    'current_paca': self.args.paca,  # PaCA atual do cliente (dinâmico)
                    'current_rank': self.args.rank,  # Rank atual do cliente (dinâmico)
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
                client_accuracies = []
                client_losses = []
                threads = []
                self.stop_event = threading.Event()
                round_pruning_time = 0.0
                
                for i, conn in enumerate(self.client_connections):
                    t = threading.Thread(
                        target=self.handle_client, 
                        daemon=True, 
                        args=(conn, client_updates, client_weights, round_num + 1, clientsid[i], client_accuracies, client_losses)
                    )
                    t.start()
                    threads.append(t)
                
                init_time = time.time()
                for t in threads:
                    remaining_time = max(0.0, self.time_threshold - (time.time() - init_time))
                    t.join(timeout=remaining_time)
                
                if round_num == 0 or (self.args.model == 'slm' and client_updates):
                    self.set_threshold()
                    
                round_duration = time.time() - init_time
                self.round_time.append(round_duration)
                
                logger.info(f"Round {round_num + 1} duration: {round_duration:.2f} seconds")
                logger.debug(f"Client updates length: {len(client_updates)}")
                
                if client_updates:
                    logger.info(f"Round {round_num + 1}: Aggregating {len(client_updates)} updates")
                    self.aggregated_clients.append(len(client_updates))
                    
                    # Garantir acesso seguro no momento de ler os updates!
                    with self.lock:
                        safe_client_updates = list(client_updates)
                        safe_client_weights = list(client_weights)
                    
                    aggregated_state = self.aggregate_models(safe_client_updates, self.global_state, safe_client_weights)
                    
                    if self.args.model in ['clip', 'slm'] and 'sora' in self.args.strategy and self.args.prune_freq > 0:
                        # Verifica se a rodada atual é múltipla da frequência desejada
                        if (round_num + 1) % self.args.prune_freq == 0:
                            logger.info(f"--- SERVER: Iniciando Poda Iterativa do SoRA (Round {round_num + 1}) ---")
                            
                            sora_prune_start = time.time()
                            # Threshold adaptativo: sobe progressivamente ao longo das rodadas
                            # Limitado a 5x para não ser destrutivo
                            progress = (round_num + 1) / self.args.rounds  # 0.0 → 1.0
                            decay_factor = 1.0 + progress * 9.0  # 1x → 10x
                            adaptive_threshold = 1e-3 * decay_factor
                            logger.info(f"   Adaptive threshold: {adaptive_threshold:.6f} (progress={progress:.2f})")
                            
                            if self.args.model == 'slm':
                                from lora_slm.slm_sora_utils import reduce_sora_state_dict_rank as reduce_sora_state_dict_rank_slm
                                prune_fn = reduce_sora_state_dict_rank_slm
                            else:
                                prune_fn = reduce_sora_state_dict_rank

                            pruned_state, before, after, ranks_info = prune_fn(
                                aggregated_state,
                                threshold=adaptive_threshold,
                                min_rank=self.args.min_rank
                            )
                            
                            sora_prune_time = time.time() - sora_prune_start
                            round_pruning_time += sora_prune_time
                            
                            if before > 0:
                                reduction = (1 - after/before) * 100
                                logger.info(f"   SoRA Params: {before:,} -> {after:,} ({reduction:.2f}% reduzido) em {sora_prune_time:.2f}s")
                                if after < before:
                                    for info in ranks_info[:5]: # Mostra apenas as 5 primeiras para não poluir
                                        logger.info(f"   - {info}")
                                
                                # Atualiza o estado agregado com as matrizes reduzidas
                                aggregated_state = pruned_state
                                
                                # Redimensiona o modelo global localmente no servidor
                                self.global_model = resize_model_to_pruned(self.global_model, aggregated_state)
                    
                    with self.lock:
                        self.global_state = aggregated_state
                        self.global_model.load_state_dict(self.global_state, strict=False)
                    
                    if self.test_loader is not None or self.client_idx:
                        try:
                            total_correct_weight = 0.0
                            total_loss_weight = 0.0
                            total_samples = 0
                            eval_clients = self.client_idx if self.client_idx else ([self.args.test_client_idx] if self.args.test_client_idx is not None else [0])
                            for i in eval_clients:
                                if i not in self.test_loaders_cache:
                                    self.test_loaders_cache[i] = self.load_test_data(self.args.dataset, i, self.args.batch_size)
                                client_loader = self.test_loaders_cache[i]
                                if client_loader is not None:
                                    accuracy, avg_loss = evaluate_model(self.global_model, client_loader)
                                    n_samples = len(client_loader.dataset)
                                    total_correct_weight += accuracy * n_samples
                                    total_loss_weight += avg_loss * n_samples
                                    total_samples += n_samples
                            
                            final_accuracy = total_correct_weight / total_samples if total_samples > 0 else 0.0
                            final_loss = total_loss_weight / total_samples if total_samples > 0 else 0.0
                            self.rs_test_acc.append(final_accuracy)
                            self.rs_test_loss.append(final_loss)
                            logger.info(f"Round {round_num + 1}: Test Acc: {final_accuracy:.2f}% | Loss: {final_loss:.4f}")
                        except Exception as e:
                            logger.exception(f"Round {round_num + 1}: Erro durante avaliação global no servidor: {e}")
                            if client_accuracies:
                                fallback_acc = sum(client_accuracies) / len(client_accuracies)
                                fallback_loss = sum(client_losses) / len(client_losses) if client_losses else 0.0
                                self.rs_test_acc.append(fallback_acc)
                                self.rs_test_loss.append(fallback_loss)
                                logger.warning(f"Round {round_num + 1}: Usando fallback de acurácia reportada pelos clientes: {fallback_acc:.2f}%")
                            else:
                                self.rs_test_acc.append(0.0)
                                self.rs_test_loss.append(0.0)
                    else:
                        logger.info(f"Round {round_num + 1}: Model aggregated (no test data)")
                    
                    size_trainable_mb, num_trainable_params = get_trainable_size_and_params(self.global_model)
                    logger.info(f'Size Trainable Adapters: {size_trainable_mb:.2f} MB | Trainable Params: {num_trainable_params:,}')
                    self.model_size_per_round.append(size_trainable_mb)
                    self.trainable_params_per_round.append(num_trainable_params)
                    
                    # Total transmitido neste round: (server→clients) + (clients→server)
                    sent_this_round = self.sended_ammount[-1] - (self.sended_ammount[-len(client_updates)-1] if len(self.sended_ammount) > len(client_updates) else 0)
                    recv_this_round = self.received_ammount[-1] - (self.received_ammount[-len(client_updates)-1] if len(self.received_ammount) > len(client_updates) else 0)
                    self.total_transmitted_per_round.append(sent_this_round + recv_this_round)
                    
                    paca_list = [self.clients_info[cid].get('current_paca', self.args.paca) for cid in sorted(self.clients_info.keys())]
                    self.rs_client_paca.append(paca_list)
                    
                    training_time_list = [self.clients_info[cid].get('training_time', 0.0) for cid in sorted(self.clients_info.keys())]
                    self.rs_client_training_time.append(training_time_list)
                    
                    comm_time_list = [self.clients_info[cid].get('comm_time', 0.0) for cid in sorted(self.clients_info.keys())]
                    self.rs_client_comm_time.append(comm_time_list)
                    
                    eval_time_list = [self.clients_info[cid].get('client_eval_time', 0.0) for cid in sorted(self.clients_info.keys())]
                    self.rs_client_eval_time.append(eval_time_list)
                    
                    server_proc_list = [self.clients_info[cid].get('server_processing_time', 0.0) for cid in sorted(self.clients_info.keys())]
                    self.rs_server_processing_time.append(server_proc_list)
                    
                    self.server_pruning_time.append(round_pruning_time)
                    
                    successful_notifications = 0
                    for conn in self.client_connections:
                        try:
                            if self.args.model == 'slm':
                                send_data(conn, "end")
                            else:
                                conn.send('end'.encode('utf-8'))
                            successful_notifications += 1
                        except (OSError, BrokenPipeError, ConnectionResetError) as e:
                            logger.error(f"Error notifying client: {e}")
                    
                    logger.info(f"Round {round_num + 1}: Notified {successful_notifications} clients.")
                else:
                    logger.warning(f"Round {round_num + 1}: No client updates received (timeout or client failure).")
            
            logger.info(f"\nTraining completed after {self.args.rounds} rounds!")
            
            # Tratamento tolerante para fechamento residual dos soquetes TCP 
            for conn in self.client_connections:
                try:
                    conn.close()
                except OSError as e:
                    logger.debug(f"Socket close exception (safe to ignore): {e}")
            logger.info("All client connections closed.")
            
        if self.args.save_model:
            logger.info(f"Salvando adaptadores e gates treinados...")
            save_path = Path(self.args.save_model)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            if self.args.model == 'clip':
                # Salva APENAS o LoRA/SoRA e a Head de classificação para economizar espaço
                weights_to_save = get_trainable_state_dict(self.global_model)
            elif self.args.model == 'slm':
                weights_to_save = get_trainable_state_dict_slm(self.global_model)
            else:
                weights_to_save = self.global_model.state_dict()

            torch.save(weights_to_save, save_path)
            logger.info(f"Modelo salvo com sucesso em: {save_path}")
            
        self.save_results(times)

def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning Server')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--clients-per-round', type=int, default=8)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='Cifar10')
    parser.add_argument('--test-client-idx', type=int, default=0)
    parser.add_argument('--in-features', type=int, default=3)
    parser.add_argument('--num-classes', type=int, default=100)
    parser.add_argument('--dim', type=int, default=1600)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-clients', type=int, default=10)
    parser.add_argument('--prune', type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument('--pm', type=str, default='OPALA', choices=['OPALA', 'SNIP', 'NISP'])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('--experiments', type=int, default=1)
    parser.add_argument('--model', type=str, default="cnn", choices=["cnn", "clip", "slm"])
    parser.add_argument('--config', type=str, default="lora_clip/train_config.yml")
    parser.add_argument('--prune-freq', type=int, default=0)
    
    parser.add_argument('--save-model', type=str, default=None, help='Caminho para salvar os pesos treinados (.pt)')
    parser.add_argument('--load-model', type=str, default=None, help='Caminho para carregar pesos pré-treinados (.pt)')
    
    parser.add_argument('--run-id', type=int, default=1, help='ID da simulação atual')
    parser.add_argument('--exp-name', type=str, default='default_exp', help='Nome da sessão')
    parser.add_argument('--timestamp', type=str, default='', help='Timestamp para diferenciar rodadas de simulação')
    parser.add_argument('--strategy', type=str, default='lora', choices=['lora', 'adalora', 'sora_with_schedule', 'sora_no_schedule'])
    parser.add_argument('--rank', type=int, default=8, help='Rank para o SoRA/LoRA')
    parser.add_argument('--paca', type=int, default=12, help='Número de camadas do modelo base para injetar adaptadores (PaCA)')
    parser.add_argument('--adaptive-paca', action='store_true', help='Ativa PaCA dinâmico: servidor ajusta automaticamente o número de camadas por cliente baseado em latência')
    parser.add_argument('--adaptive-paca-min', type=int, default=2, help='Número mínimo de camadas que o PaCA adaptativo pode alcançar')
    parser.add_argument('--adaptive-rank', action='store_true', help='Ativa rank adaptativo: servidor ajusta automaticamente o rank por cliente baseado em latência')
    parser.add_argument('--adaptive-rank-min', type=int, default=2, help='Rank mínimo para o rank adaptativo')
    parser.add_argument('--adaptive-rank-max', type=int, default=8, help='Rank máximo para o rank adaptativo')
    parser.add_argument('--allow-paca-upscale', action='store_true', help='Permite a subida de PaCA (por padrão apenas desce)')
    parser.add_argument('--allow-rank-upscale', action='store_true', help='Permite a subida de Rank (por padrão apenas desce)')
    parser.add_argument('--min-rank', type=int, default=2, help='Rank mínimo permitido por módulo SoRA durante poda iterativa')
    parser.add_argument('--sora-prune', action='store_true', help='Ativa poda iterativa de rank do SoRA (independente do --prune CNN)')
    parser.add_argument('--delta-coding', action='store_true', help='Ativa LHDQ (Low Huffman-coded Delta Quantization) em vez de int8')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Configuração inteligente de device: MPS > CUDA > CPU
    if args.device == "mps":
        if torch.backends.mps.is_available():
            logger.info("Using Metal Performance Shaders (MPS) for MacBook GPU acceleration.")
        else:
            logger.warning("MPS is not available. Falling back to CUDA if available, otherwise CPU.")
            if torch.cuda.is_available():
                args.device = "cuda"
            else:
                args.device = "cpu"
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Checking for MPS...")
            if torch.backends.mps.is_available():
                logger.info("Falling back to Metal Performance Shaders (MPS).")
                args.device = "mps"
            else:
                logger.warning("CUDA and MPS not available. Using CPU.")
                args.device = "cpu"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    
    logger.info(f"Device set to: {args.device}")
    
    for i in range(args.experiments):
        logger.info(f"\n=== Iniciando Experimento {i+1}/{args.experiments} ===")
        server = FederatedLearningServer(args)
        server.run_server(i)

if __name__ == '__main__':
    main()
