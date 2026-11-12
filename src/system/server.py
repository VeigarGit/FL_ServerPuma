import socket
import pickle
import struct
import threading
import copy
import time
import argparse
import sys
import os
import logging
import json
from contextlib import suppress
from pathlib import Path

# Desativa checagem online redundante do Hugging Face Hub (usa cache local instantaneamente)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

import torch
import torch.nn as nn
import numpy as np

#Para o Lora
from lora_clip.clip_setup import (
    parse_args, load_config, get_device, resolve_run_modes, 
    build_run_config, build_output_path, build_dataloaders,
    build_model, build_optimizer, build_scheduler, 
    benchmark_attention, quantize_weights
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
from lora_slm.qlora import enable_qlora_in_config, get_qlora_state_dict, set_qlora_state_dict


# --- HACK PERMITIDO: Inserindo a pasta 'src' no path do Python ---
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))
# -----------------------------------------------------------------

from model import SimpleModel
from prunning import prune_and_restructure
from size_mode import get_trainable_size_and_params
from synexp import LayerComplexityCalculator
from prunning_nisp import prune_fc1
from prunning_snip import snip_pruning, apply_mask
from utils.network_utils import send_data, recv_data
from utils.fl_math import quantization, dequantization, evaluate_model, set_parameters
from utils.lhdq import lhdq_encode, lhdq_decode, is_lhdq_encoded
from server_puma import PumaEngineMixin
from server_alignment import ServerAlignmentMixin
from server_data import ServerDataMixin, get_state_dict_size_mb

# --- CONFIGURAÇÃO DE OBSERVABILIDADE (LOGGING) ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
# -------------------------------------------------

class FederatedLearningServer(PumaEngineMixin, ServerAlignmentMixin, ServerDataMixin):
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

                if getattr(args, 'qlora', False):
                    if args.model != 'slm':
                        raise ValueError("--qlora é suportado apenas com --model slm")
                    enable_qlora_in_config(config, args.strategy)
                    logger.info("QLoRA ATIVADO: backbone 4-bit NF4 (não agregado), FedAvg só nos adaptadores LoRA 16 bits.")
                    
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
            case 'lfm':
                import yaml
                from lora_lfm.lfm_setup import LFMCollator, setup_lfm_model

                config_path = (
                    Path(args.config) if args.config else Path("src/system/lora_lfm/train_config.yml")
                )
                if not config_path.exists():
                    config_path = Path("lora_lfm/train_config.yml")
                with config_path.open("r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                self.config = config

                if args.rank:
                    config["model"]["lora"]["r"] = args.rank
                    config["model"]["lora"]["alpha"] = 2 * args.rank

                if args.paca is not None and args.paca > 0:
                    config["model"]["paca"]["enabled"] = True
                    config["model"]["paca"]["upper_layers"] = args.paca
                else:
                    config["model"]["paca"]["enabled"] = False
                    config["model"]["paca"]["upper_layers"] = None

                config["is_qlora"] = bool(getattr(args, "qlora", False))
                self.global_model, tokenizer = setup_lfm_model(
                    config, args.device, is_qlora=bool(getattr(args, "qlora", False))
                )
                self.input_size = None
                self.lfm_collator = LFMCollator(tokenizer.pad_token_id)
                self.lfm_tokenizer = tokenizer
                        
                        
        self.qlora = getattr(args, 'qlora', False)

        if args.load_model:
            logger.info(f"Carregando pesos salvos de: {args.load_model}")
            state_dict = torch.load(args.load_model, map_location=args.device)
            
            if self.args.model == 'lfm':
                from lora_lfm.lfm_setup import set_lfm_state_dict
                set_lfm_state_dict(self.global_model, state_dict)
            elif self.qlora:
                # QLoRA: o checkpoint contém apenas adaptadores + head
                set_qlora_state_dict(self.global_model, state_dict)
            else:
                # 1. Redimensiona a arquitetura do modelo em memória para o tamanho podado
                self.global_model = resize_model_to_pruned(self.global_model, state_dict)
                
                # 2. Agora os formatos coincidem, podemos carregar os pesos com segurança
                self.global_model.load_state_dict(state_dict, strict=False)
                self.global_state = self.global_model.state_dict()
                

        if not self.qlora:  # QLoRA: backbone 4-bit já alocado via device_map
            self.global_model = self.global_model.to(args.device)
        self.rs_test_acc = []
        self.rs_test_loss = []
        self.rs_aux_simple_avg_acc = []  # Média simples das acurácias reportadas pelos clientes (usada pelo PaCA/Rank adaptativo)
        # QLoRA / PEFT: o estado agregável contém SOMENTE adaptadores LoRA + head (backbone nunca é agregado)
        if self.args.model == 'lfm':
            self.global_state = self._lfm_trainable_state(trainable_only=False)
        elif self.qlora:
            self.global_state = get_qlora_state_dict(self.global_model, trainable_only=False)
        else:
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
        self.rs_client_rank = []
        self.rs_client_gamma = []  # Índice de Capacidade γ^c por cliente (Eq. analítica do PUMA)
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
        self.current_server_round = 0
        
        # LHDQ: cache do estado anterior por cliente (para delta coding)
        self.previous_sent_state = {}   # client_id → state_dict enviado na rodada anterior
        self.previous_recv_state = {}   # client_id → state_dict recebido na rodada anterior
        
        if self.args.model in ['slm', 'lfm']:
            alpha_list, beta_list = [0], [0]
            logger.info("Cálculo de complexidade ignorado para SLM/LFM.")
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
        elif self.args.model == 'lfm':
            from lora_lfm.lfm_setup import _get_encoder_layers
            layers = _get_encoder_layers(self.global_model)
            self.total_encoder_layers = len(layers) if layers else (self.args.paca or 24)
        else:
            self.total_encoder_layers = self.args.paca  # fallback
        logger.info(f"Total encoder layers detectado: {self.total_encoder_layers}")
        
        # Dicionário para armazenar máscaras por cliente
        self.client_masks = {}
        
        if args.model in ['slm', 'lfm']:
            self.test_loader = None
            logger.info(f"Modelo {args.model}: Avaliação global configurada via agregação federada ponderada dos clientes (Zero Data Leakage).")
        else:
            self.test_loader = self.load_test_data(args.dataset, args.test_client_idx, args.batch_size)
            if self.test_loader is not None:
                self.test_loaders_cache[args.test_client_idx if args.test_client_idx is not None else 0] = self.test_loader
            else:
                logger.warning("Could not load test data. Evaluation will be skipped.")

    def _slm_trainable_state(self):
        """Estado do SLM que trafega/é salvo. No QLoRA: apenas adaptadores LoRA 16 bits + head."""
        if self.qlora:
            return get_qlora_state_dict(self.global_model, trainable_only=False)
        return get_trainable_state_dict_slm(self.global_model)

    def _lfm_trainable_state(self, trainable_only: bool = False):
        """Estado do LFM que trafega/é salvo (apenas adaptadores LoRA + DecisionHead)."""
        from lora_lfm.lfm_setup import get_lfm_state_dict
        return get_lfm_state_dict(self.global_model, trainable_only=trainable_only)

    def aggregate_models(self, model_list, global_model, client_weights=None):
        agg_state = global_model.copy()
        
        if client_weights is None:
            client_weights = [1.0] * len(model_list)
        
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]
        
        for key in global_model.keys():
            weighted_sum = None
            total_weight_for_key = 0.0
            global_val = global_model[key]
            
            for idx, model_state in enumerate(model_list):
                if key in model_state:
                    local_val = model_state[key]
                    
                    if self._are_compatible_with_pruning(local_val, global_val, key):
                        if not isinstance(global_val, torch.Tensor):
                            global_val = torch.tensor(global_val, device=self.args.device)
                        
                        if not isinstance(local_val, torch.Tensor):
                            local_val = torch.tensor(local_val, device=global_val.device)
                        else:
                            local_val = local_val.to(global_val.device)

                        aligned_val = self._align_to_global_structure(
                            local_val, global_val, key, model_state, idx
                        )
                        
                        weight = normalized_weights[idx]
                        
                        if weighted_sum is None:
                            # Acumula em fp32 quando os tensores são 16 bits (ex.: adaptadores QLoRA bf16)
                            out_dtype = aligned_val.dtype
                            acc_dtype = torch.float32 if out_dtype in (torch.float16, torch.bfloat16) else out_dtype
                            weighted_sum = torch.zeros_like(aligned_val, dtype=acc_dtype)
                        
                        weighted_sum += aligned_val.to(weighted_sum.dtype) * weight
                        total_weight_for_key += weight
            
            if weighted_sum is not None and total_weight_for_key > 0:
                agg_state[key] = (weighted_sum / total_weight_for_key).to(out_dtype)
        
        return agg_state

    def _deactivate_client(self, client_id, conn, *, shutdown: bool = True) -> None:
        """Marca um cliente como inativo e encerra seu socket com tolerância a erros."""
        self.clients_info[client_id]['active'] = False
        if shutdown:
            with suppress(OSError):
                conn.shutdown(socket.SHUT_RDWR)
        with suppress(OSError):
            conn.close()

    def _cleanup_pending_round_threads(
        self, round_num, active_round_threads, round_cancel_event, *, log_timeout: bool = True
    ) -> None:
        """Sinaliza cancelamento e encerra threads de clientes bloqueadas ao fim da rodada."""
        round_cancel_event.set()
        for cid, conn, thread in active_round_threads:
            if thread.is_alive():
                if log_timeout:
                    logger.warning(
                        f"Round {round_num}: Cliente {cid} não respondeu até o fim da "
                        "rodada. Encerrando conexão e thread."
                    )
                self._deactivate_client(cid, conn)
                thread.join(timeout=2.0)

    def set_threshold(self):
        tot_time = 0
        clients_with_time = 0
        max_client_time = 0.0
        for info in self.clients_info.values():
            if not info.get('active', True):
                continue
            t_train = info.get('training_time', 0.0) or 0.0
            t_eval = info.get('client_eval_time', 0.0) or 0.0
            t_proc = info.get('server_processing_time', 0.0) or 0.0
            t_comm = info.get('comm_time', 0.0) or 0.0
            total_client_time = t_train + t_eval + t_proc + t_comm
            
            if total_client_time > 0:
                tot_time += total_client_time
                clients_with_time += 1
                if total_client_time > max_client_time:
                    max_client_time = total_client_time
        
        if clients_with_time == 0:
            logger.warning("No clients reported training time. Sticking to default threshold.")
            return

        mean_time = tot_time / clients_with_time
        self.mean_training_time = mean_time
        
        # Meta de latência adaptativa do cluster: média dos clientes + 15% de tolerância
        # Utilizada pelo PaCA e Rank adaptativos para identificar stragglers relativos
        self.target_latency = mean_time * 1.15
        
        # Pisos físicos realistas por tipo de modelo para evitar colapso de timeout
        if self.args.model in ['slm', 'lfm']:
            min_floor = 240.0
        elif self.args.model == 'clip':
            min_floor = 60.0
        else:
            min_floor = 15.0

        # Limiar realista alinhado ao PUMA-GT Plus: 1.5x a meta de latência (ou 1.25x o pior)
        calculated_threshold = max(
            min_floor, 1.50 * self.target_latency, 1.25 * max_client_time
        )

        # Amortecimento ágil: impede quedas bruscas, permitindo redução de até 40% por rodada
        if getattr(self, 'time_threshold', 0) and self.time_threshold > 0:
            self.time_threshold = max(
                calculated_threshold, self.time_threshold * 0.60, min_floor
            )
        else:
            self.time_threshold = calculated_threshold

        logger.info(
            f"Limiares definidos: timeout_socket={self.time_threshold:.2f}s | "
            f"target_latency={self.target_latency:.2f}s (mean_time={mean_time:.2f}s)"
        )
    
    def handle_client(
        self,
        conn,
        client_updates,
        client_weights,
        round_num,
        client_id,
        client_accuracies,
        client_losses,
        round_closed_event=None,
        round_cancel_event=None,
        completed_clients=None,
    ):
        bit_rate = []
        masks = []
        if 'socket_lock' not in self.clients_info[client_id]:
            self.clients_info[client_id]['socket_lock'] = threading.Lock()
            
        lock_timeout = max(5.0, self.time_threshold if self.time_threshold > 0 else 60.0)
        acquired = self.clients_info[client_id]['socket_lock'].acquire(timeout=lock_timeout)
        if not acquired:
            logger.warning(
                f"Round {round_num}: Cliente {client_id} com socket_lock retido por "
                "thread anterior. Desativando cliente."
            )
            self._deactivate_client(client_id, conn)
            return

        cancelled_fn = (
            round_cancel_event.is_set if round_cancel_event is not None else None
        )
        base_timeout = self.time_threshold if self.time_threshold > 0 else 60.0
        deadline = time.monotonic() + max(15.0, base_timeout * 1.5)
        try:
            start_time = time.time()
            logger.info(f"Round {round_num}: Handling client {client_id}")
            
            if hasattr(self, 'current_round_state') and self.current_round_state is not None:
                current_global_state = self.current_round_state.copy()
            else:
                with self.lock:
                    if self.args.model == 'clip':
                        current_global_state = get_trainable_state_dict(self.global_model)
                    elif self.args.model == 'slm':
                        current_global_state = self._slm_trainable_state()
                    elif self.args.model == 'lfm':
                        current_global_state = self._lfm_trainable_state()
                    else:
                        current_global_state = self.global_state.copy()
            
            # --- PaCA DINÂMICO (para CLIP, SLM e LFM, a partir do round 2, se ativado) ---
            if self.args.model in ['clip', 'slm', 'lfm'] and round_num >= 1 and getattr(self.args, 'adaptive_paca', False):
                ideal_paca = self.calculate_adaptive_paca(client_id)
                self.clients_info[client_id]['current_paca'] = ideal_paca
                
                # Filtra: remove chaves de camadas fora do PaCA ideal
                min_layer = self.total_encoder_layers - ideal_paca
                if min_layer > 0:
                    keys_before = len(current_global_state)
                    current_global_state = {
                        k: v for k, v in current_global_state.items()
                        if not self._is_excluded_by_paca(k, min_layer)
                    }
                    keys_after = len(current_global_state)
                    if keys_before != keys_after:
                        logger.info(f"PaCA filtro {client_id}: {keys_before} -> {keys_after} chaves "
                                    f"(PaCA={ideal_paca}/{self.total_encoder_layers})")

            # --- RANK ADAPTATIVO (para CLIP, SLM e LFM, a partir do round 2, se ativado) ---
            if self.args.model in ['clip', 'slm', 'lfm'] and round_num >= 1 and getattr(self.args, 'adaptive_rank', False):
                ideal_rank = self.calculate_adaptive_rank(client_id)
                self.clients_info[client_id]['current_rank'] = ideal_rank
                
                # Slicing de Rank
                for k, v in current_global_state.items():
                    if '.lora_A' in k:
                        current_global_state[k] = v[:ideal_rank, :].clone()
                    elif '.lora_B' in k:
                        current_global_state[k] = v[:, :ideal_rank].clone()
                    elif '.gate' in k:
                        current_global_state[k] = v[:, :ideal_rank].clone()
                logger.info(f"Rank filtro {client_id}: rank={ideal_rank}/{self.args.rank}")

            
            # --- Problema #4: Envia PaCA dinâmico para que o cliente treine apenas as camadas necessárias ---
            if self.args.model in ['clip', 'slm', 'lfm']:
                client_paca = self.clients_info[client_id].get('current_paca', self.args.paca) if getattr(self.args, 'adaptive_paca', False) else self.args.paca
                send_data(conn, client_paca)

                client_rank = self.clients_info[client_id].get('current_rank', self.args.rank) if getattr(self.args, 'adaptive_rank', False) else self.args.rank
                send_data(conn, client_rank)

            if round_num >= 2 and self.prune == 1 and self.args.model == 'cnn':
                logger.info("--- SERVER: PRUNING START (Round 2) ---")
                cnn_prune_start = time.time()
                if self.clients_info[client_id]['original_model_size'] is None:
                    max_amount = self.new_set_amount_prune(client_id)
                    self.clients_info[client_id]['pruning_rate'] = max_amount
                else:
                    max_amount = self.clients_info[client_id]['pruning_rate']
                
                self.clients_info[client_id]['original_model_size'] = get_state_dict_size_mb(self.global_model)
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
                
                masks.append(mask)
                self.client_masks[client_id] = mask
                g_model_pruned = g_model_pruned.state_dict()
                cnn_prune_time = time.time() - cnn_prune_start
                self.clients_info[client_id]['server_pruning_time'] = cnn_prune_time
                logger.info(f"--- SERVER: PRUNING COMPLETE ({cnn_prune_time:.2f}s) ---")

            # --- Medição granular: quantização do servidor ---
            server_quant_start = time.time()
            comm_start = time.time()
            
            # Decide se usa LHDQ ou int8
            use_lhdq = (getattr(self.args, 'delta_coding', False) 
                        and round_num >= 2 
                        and client_id in self.previous_sent_state)
            
            if round_num >= 2 and self.prune == 1 and self.args.model == 'cnn':
                size_before = (len(pickle.dumps(g_model_pruned)) + 4) / (1024 * 1024)
                if use_lhdq:
                    prev_state = self.previous_sent_state.get(client_id, {})
                    g_model_pruned = lhdq_encode(g_model_pruned, prev_state)
                    # Re-decode to ensure identical reference state with the client.
                    # This guarantees both sides use the same base reference for future deltas.
                    self.previous_sent_state[client_id] = lhdq_decode(g_model_pruned, prev_state)
                else:
                    if getattr(self.args, 'delta_coding', False):
                        self.previous_sent_state[client_id] = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in g_model_pruned.items()}
                    if not getattr(self.args, 'delta_coding', False):
                        g_model_pruned = quantization(g_model_pruned)
                server_quant_time = time.time() - server_quant_start
                
                bytes_sent = send_data(conn, g_model_pruned)
                send_data(conn, self.prune)
                send_data(conn, max_amount)
                size_after = bytes_sent / (1024 * 1024)
                with self.lock:
                    self.sended_withouquant.append(self.sended_withouquant[-1] + size_before)
                    self.sended_ammount.append(self.sended_ammount[-1] + size_after)
            else:
                ideal_paca = self.clients_info[client_id].get('current_paca', self.args.paca)
                ideal_rank = self.clients_info[client_id].get('current_rank', self.args.rank)
                cache_key = (ideal_paca, ideal_rank)
                with self.lock:
                    if not hasattr(self, 'size_before_cache'):
                        self.size_before_cache = {}
                    if cache_key not in self.size_before_cache:
                        self.size_before_cache[cache_key] = (len(pickle.dumps(current_global_state)) + 4) / (1024 * 1024)
                    size_before = self.size_before_cache[cache_key]
                if use_lhdq:
                    prev_state = self.previous_sent_state.get(client_id, {})
                    current_global_state = lhdq_encode(current_global_state, prev_state)
                    # Re-decode to ensure identical reference state with the client.
                    # This guarantees both sides use the same base reference for future deltas.
                    self.previous_sent_state[client_id] = lhdq_decode(current_global_state, prev_state)
                else:
                    if getattr(self.args, 'delta_coding', False):
                        self.previous_sent_state[client_id] = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in current_global_state.items()}
                    if not getattr(self.args, 'delta_coding', False) and not self.qlora:  # QLoRA: envia em 16 bits
                        current_global_state = quantization(current_global_state)
                server_quant_time = time.time() - server_quant_start
                
                bytes_sent = send_data(conn, current_global_state)
                send_data(conn, self.prune)
                size_after = bytes_sent / (1024 * 1024)
                with self.lock:
                    self.sended_withouquant.append(self.sended_withouquant[-1] + size_before)
                    self.sended_ammount.append(self.sended_ammount[-1] + size_after)

            size_saved = size_before - size_after
            logger.debug(f"Tamanho antes: {size_before:.2f} MB | Tamanho depois: {size_after:.2f} MB | Economia: {size_saved:.2f} MB")
            logger.info(f"Round {round_num}: Sent global model to client {client_id}")
            
            updated_state, rate, recv_bytes = recv_data(
                conn, return_bytes=True, deadline=deadline, cancelled=cancelled_fn
            )
            bit_rate.append(rate)
            if updated_state is not None:
                recv_size_mb = recv_bytes / (1024 * 1024)
                with self.lock:
                    self.received_ammount.append(self.received_ammount[-1] + recv_size_mb)
            
            if updated_state is None:
                logger.warning(
                    f"Round {round_num}: Cliente {client_id} falhou/desconectou. Abortando thread."
                )
                self.clients_info[client_id]['training_time'] = time.time() - start_time
                self._deactivate_client(client_id, conn)
                return
            
            # --- Medição granular: dequantização do servidor ---
            server_dequant_start = time.time()
            if is_lhdq_encoded(updated_state):
                prev_recv = self.previous_recv_state.get(client_id, {})
                updated_state = lhdq_decode(updated_state, prev_recv)
            else:
                if not getattr(self.args, 'delta_coding', False):
                    updated_state = dequantization(updated_state)
            server_dequant_time = time.time() - server_dequant_start
            
            # LHDQ: cachear o update recebido para a próxima rodada
            if getattr(self.args, 'delta_coding', False) and updated_state is not None:
                self.previous_recv_state[client_id] = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in updated_state.items()}
            
            data, rate = recv_data(conn, deadline=deadline, cancelled=cancelled_fn)
            self.client_data[client_id] = data
            
            dataset_size = data if isinstance(data, (int, float)) else 1.0
            self.clients_info[client_id]['dataset_size'] = dataset_size
            
            logger.debug(f"Client {client_id} dataset size: {dataset_size}")
            
            self.argalgo, rate = recv_data(conn, deadline=deadline, cancelled=cancelled_fn)
            
            # Recebe tempo de treino puro medido pelo cliente
            client_training_time, rate = recv_data(conn, deadline=deadline, cancelled=cancelled_fn)
            
            client_test_acc, _ = recv_data(conn, deadline=deadline, cancelled=cancelled_fn)
            client_test_loss, _ = recv_data(conn, deadline=deadline, cancelled=cancelled_fn)
            
            # Recebe overhead de processamento do cliente (eval + quant + dequant)
            client_processing_overhead, _ = recv_data(
                conn, deadline=deadline, cancelled=cancelled_fn
            )
            if not isinstance(client_processing_overhead, (int, float)):
                client_processing_overhead = 0.0
            
            is_round_open = (
                (round_closed_event is None or not round_closed_event.is_set())
                and getattr(self, 'current_server_round', round_num) == round_num
            )

            if client_test_acc is not None and is_round_open:
                with self.lock:
                    client_accuracies.append(client_test_acc)
                    client_losses.append(client_test_loss)
                self.clients_info[client_id]['last_loss'] = client_test_loss
                if 'acc_history' not in self.clients_info[client_id]:
                    self.clients_info[client_id]['acc_history'] = []
                self.clients_info[client_id]['acc_history'].append(client_test_acc)
            
            media_rate = (sum(bit_rate) / len(bit_rate)) / 8 / (1024 * 1024)
            self.clients_info[client_id]['bandwidth'] = media_rate
            self.clients_info[client_id]['data'] = self.client_data[client_id]
            
            # --- PaCA DINÂMICO: Remove chaves devolvidas que o cliente NÃO treinou ---
            if self.args.model in ['clip', 'slm', 'lfm'] and round_num >= 1 and getattr(self.args, 'adaptive_paca', False):
                ideal_paca = self.clients_info[client_id].get('current_paca', self.args.paca)
                min_layer = self.total_encoder_layers - ideal_paca
                if min_layer > 0 and updated_state is not None:
                    updated_state = {
                        k: v for k, v in updated_state.items()
                        if not self._is_excluded_by_paca(k, min_layer)
                    }

            handle_time = time.time() - start_time
            # Usa o tempo de treino reportado pelo cliente (mais preciso, exclui overhead de rede)
            training_time = client_training_time if isinstance(client_training_time, (int, float)) else handle_time
            
            # --- Cálculo CORRETO do comm_time ---
            # Tempo total desde comm_start até agora, menos:
            #   - treinamento do cliente (medido pelo cliente)
            #   - processamento do cliente (eval + quant + dequant, medido pelo cliente)
            #   - quantização do servidor (medido aqui)
            #   - dequantização do servidor (medido aqui)
            # O que sobra é o tempo real de rede (serialização + transferência)
            total_comm_window = time.time() - comm_start
            server_processing_time = server_quant_time + server_dequant_time
            comm_time = total_comm_window - training_time - client_processing_overhead - server_processing_time
            comm_time = max(0.001, comm_time)
            self.clients_info[client_id]['comm_time'] = comm_time
            self.clients_info[client_id]['client_eval_time'] = client_processing_overhead
            self.clients_info[client_id]['server_processing_time'] = server_processing_time
            
            # Banda de rede real (MB/s) calculada exclusivamente sobre o tempo puro de transmissão
            if 'recv_size_mb' in locals() and recv_size_mb > 0:
                self.clients_info[client_id]['bandwidth'] = recv_size_mb / comm_time
            
            logger.info(f"Round {round_num}: Client {client_id} tempos: "
                        f"treino={training_time:.2f}s, "
                        f"comm_rede={max(0.0, comm_time):.2f}s, "
                        f"client_overhead={client_processing_overhead:.2f}s, "
                        f"server_proc={server_processing_time:.2f}s")
            
            if updated_state is not None:
                if is_round_open:
                    with self.lock:
                        client_updates.append(updated_state)
                        client_weights.append(dataset_size)
                else:
                    logger.warning(
                        f"Round {round_num}: Update tardio do cliente {client_id} recebido após o "
                        f"fechamento da janela de agregação (descartado da agregação)."
                    )
                if completed_clients is not None:
                    with self.lock:
                        completed_clients.add(client_id)
                self.clients_info[client_id]['training_time'] = training_time
            else:
                logger.warning(f"Round {round_num}: No update received from client {client_id}")
                self.clients_info[client_id]['training_time'] = training_time
                
        # Tratamento Exato de Exceções de Rede e Pickle 
        except (OSError, pickle.PickleError, struct.error) as e:
            logger.warning(
                f"Round {round_num}: Cliente {client_id} excedeu o timeout "
                f"ou desconectou ({type(e).__name__})."
            )
            self._deactivate_client(client_id, conn)
        except Exception as e:
            logger.exception(f"Round {round_num}: Critical error handling client {client_id}")
        finally:
            self.clients_info[client_id]['socket_lock'].release()

    def run_server(self, times):
        logger.info("=== Federated Learning Server ===")
        logger.info(f"Host: {self.args.host}:{self.args.port}")
        logger.info(f"Dataset: {self.args.dataset}")
        logger.info(f"Clients per round: {self.args.clients_per_round}")
        logger.info(f"Total rounds: {self.args.rounds}")
        logger.info("=======================================")
        
        if self.args.model in ['slm', 'lfm']:
            self.time_threshold = 600.0
        elif self.args.model == 'clip':
            self.time_threshold = 180.0
        else:
            self.time_threshold = 60.0
        
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
                    'active': True,
                    'socket_lock': threading.Lock(),
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
                    'paca_before_change': None,
                    'paca_acc_at_reduction': None,
                    'last_paca_reduction_round': -999,
                    'rank_before_change': None,
                    'rank_acc_at_reduction': None,
                    'last_rank_reduction_round': -999,
                    'acc_history': [],
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
                self.current_server_round = round_num + 1
                round_closed_event = threading.Event()
                round_cancel_event = threading.Event()
                completed_clients = set()
                active_round_threads = []
                round_pruning_time = 0.0
                
                # Extrai o estado global treinável UMA ÚNICA VEZ para todas as threads deste round,
                # eliminando a contenção de lock e execução redundante de get_trainable_state_dict.
                with self.lock:
                    if self.args.model == 'clip':
                        self.current_round_state = get_trainable_state_dict(self.global_model)
                    elif self.args.model == 'slm':
                        self.current_round_state = self._slm_trainable_state()
                    elif self.args.model == 'lfm':
                        self.current_round_state = self._lfm_trainable_state()
                    else:
                        self.current_round_state = self.global_state.copy()
                self.size_before_cache = {}
                
                for i, conn in enumerate(self.client_connections):
                    cid = clientsid[i]
                    if not self.clients_info[cid].get('active', True):
                        continue
                    t = threading.Thread(
                        target=self.handle_client, 
                        daemon=True, 
                        args=(
                            conn,
                            client_updates,
                            client_weights,
                            round_num + 1,
                            cid,
                            client_accuracies,
                            client_losses,
                            round_closed_event,
                            round_cancel_event,
                            completed_clients,
                        ),
                    )
                    t.start()
                    threads.append(t)
                    active_round_threads.append((cid, conn, t))
                
                init_time = time.time()
                for _, _, t in active_round_threads:
                    remaining_time = max(0.0, self.time_threshold - (time.time() - init_time))
                    t.join(timeout=remaining_time)
                
                # Fecha a janela de agregação desta rodada (bloqueia updates tardios)
                round_closed_event.set()
                
                if round_num <= 2 or client_updates:
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
                        # Poda permanece ativa durante todas as rodadas (sem congelamento na reta final)
                        if (round_num + 1) % self.args.prune_freq == 0:
                            logger.info(f"--- SERVER: Iniciando Poda Iterativa do SoRA (Round {round_num + 1}) ---")
                            
                            sora_prune_start = time.time()
                            # Threshold adaptativo: sobe progressivamente ao longo das rodadas (commit b946cf6)
                            # 1x → 10x (1e-3 → 1e-2)
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
                        if self.args.model == 'lfm':
                            from lora_lfm.lfm_setup import set_lfm_state_dict
                            set_lfm_state_dict(self.global_model, self.global_state)
                        elif self.qlora:
                            set_qlora_state_dict(self.global_model, self.global_state)
                        else:
                            self.global_model.load_state_dict(self.global_state, strict=False)
                    
                    if self.args.model in ['slm', 'lfm']:
                        # Avaliação Federada Canônica: Média ponderada reportada pelos clientes
                        if client_accuracies and safe_client_weights and len(client_accuracies) == len(safe_client_weights):
                            tot_weight = sum(safe_client_weights)
                            final_accuracy = sum(w * a for w, a in zip(safe_client_weights, client_accuracies)) / tot_weight if tot_weight > 0 else 0.0
                            final_loss = sum(w * l for w, l in zip(safe_client_weights, client_losses)) / tot_weight if tot_weight > 0 else 0.0
                        elif client_accuracies:
                            final_accuracy = sum(client_accuracies) / len(client_accuracies)
                            final_loss = sum(client_losses) / len(client_losses) if client_losses else 0.0
                        else:
                            final_accuracy, final_loss = 0.0, 0.0
                        self.rs_test_acc.append(final_accuracy)
                        self.rs_test_loss.append(final_loss)
                        logger.info(f"Round {round_num + 1}: Federated Weighted Test Acc (from clients): {final_accuracy:.2f}% | Loss: {final_loss:.4f}")
                    elif self.test_loader is not None or self.client_idx:
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
                    
                    # Acurácia auxiliar (média simples dos clientes) — usada pelo PaCA/Rank adaptativo
                    if client_accuracies:
                        aux_simple_avg = sum(client_accuracies) / len(client_accuracies)
                        self.rs_aux_simple_avg_acc.append(aux_simple_avg)
                        logger.info(f"Round {round_num + 1}: Aux Simple Avg Acc (from clients): {aux_simple_avg:.2f}%")
                    else:
                        self.rs_aux_simple_avg_acc.append(0.0)
                    
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
                    if getattr(self.args, 'adaptive_paca', False) and paca_list:
                        paca_counts = {p: paca_list.count(p) for p in sorted(set(paca_list))}
                        logger.info(f"Round {round_num + 1}: PaCA distribution: {paca_counts} (min={min(paca_list)}, max={max(paca_list)}, avg={sum(paca_list)/len(paca_list):.1f})")
                    
                    rank_list = [self.clients_info[cid].get('current_rank', self.args.rank) for cid in sorted(self.clients_info.keys())]
                    self.rs_client_rank.append(rank_list)
                    if getattr(self.args, 'adaptive_rank', False) and rank_list:
                        rank_counts = {r: rank_list.count(r) for r in sorted(set(rank_list))}
                        logger.info(f"Round {round_num + 1}: Rank distribution: {rank_counts} (min={min(rank_list)}, max={max(rank_list)}, avg={sum(rank_list)/len(rank_list):.1f})")
                    
                    # Índice de Capacidade γ^c por cliente
                    gamma_list = [self.clients_info[cid].get('capacity_index', 1.0) for cid in sorted(self.clients_info.keys())]
                    self.rs_client_gamma.append(gamma_list)
                    if gamma_list:
                        logger.info(f"Round {round_num + 1}: γ distribution: min={min(gamma_list):.3f}, max={max(gamma_list):.3f}, avg={sum(gamma_list)/len(gamma_list):.3f}")
                    
                    training_time_list = [self.clients_info[cid].get('training_time', 0.0) for cid in sorted(self.clients_info.keys())]
                    self.rs_client_training_time.append(training_time_list)
                    
                    comm_time_list = [self.clients_info[cid].get('comm_time', 0.0) for cid in sorted(self.clients_info.keys())]
                    self.rs_client_comm_time.append(comm_time_list)
                    
                    eval_time_list = [self.clients_info[cid].get('client_eval_time', 0.0) for cid in sorted(self.clients_info.keys())]
                    self.rs_client_eval_time.append(eval_time_list)
                    
                    server_proc_list = [self.clients_info[cid].get('server_processing_time', 0.0) for cid in sorted(self.clients_info.keys())]
                    self.rs_server_processing_time.append(server_proc_list)
                    
                    self.server_pruning_time.append(round_pruning_time)
                    
                    # Cancela e limpa qualquer thread de cliente bloqueada no socket
                    self._cleanup_pending_round_threads(
                        round_num + 1, active_round_threads, round_cancel_event
                    )

                    successful_notifications = 0
                    for cid, conn, _ in active_round_threads:
                        if (
                            cid not in completed_clients
                            or not self.clients_info[cid].get('active', True)
                        ):
                            continue
                        try:
                            if self.args.model in ['slm', 'lfm']:
                                send_data(conn, "end")
                            else:
                                conn.send('end'.encode('utf-8'))
                            successful_notifications += 1
                        except (OSError, BrokenPipeError, ConnectionResetError) as e:
                            logger.error(f"Error notifying client {cid}: {e}")
                            self._deactivate_client(cid, conn, shutdown=False)

                    logger.info(
                        f"Round {round_num + 1}: Notified {successful_notifications} clients."
                    )
                else:
                    logger.warning(
                        f"Round {round_num + 1}: No client updates received "
                        "(timeout or client failure)."
                    )
                    self._cleanup_pending_round_threads(
                        round_num + 1, active_round_threads, round_cancel_event, log_timeout=False
                    )
                    if not any(info.get('active', True) for info in self.clients_info.values()):
                        logger.error("Nenhum cliente ativo restante. Encerrando treinamento.")
                        break
            
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
                weights_to_save = self._slm_trainable_state()
            elif self.args.model == 'lfm':
                weights_to_save = self._lfm_trainable_state()
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
    parser.add_argument('--model', type=str, default="cnn", choices=["cnn", "clip", "slm", "lfm"])
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
    parser.add_argument('--adaptive-paca-min', type=int, default=2, help='Número mínimo de camadas que o PaCA adaptativo pode alcançar (default: 2)')
    parser.add_argument('--annealing-ratio', type=float, default=0.82, help='Fração de rodadas a partir da qual congela busca e poda estrutural (default: 0.82)')
    parser.add_argument('--adaptive-rank', action='store_true', help='Ativa rank adaptativo: servidor ajusta automaticamente o rank por cliente baseado em γ^c')
    parser.add_argument('--adaptive-rank-min', type=int, default=4, help='Rank mínimo para o rank adaptativo (default: 4)')
    parser.add_argument('--adaptive-rank-max', type=int, default=8, help='Rank máximo para o rank adaptativo')
    parser.add_argument('--min-rank', type=int, default=2, help='Rank mínimo permitido por módulo SoRA durante poda iterativa')
    parser.add_argument('--sora-prune', action='store_true', help='Ativa poda iterativa de rank do SoRA (independente do --prune CNN)')
    parser.add_argument('--delta-coding', action='store_true', help='Ativa LHDQ (Low Huffman-coded Delta Quantization) em vez de int8')
    parser.add_argument('--adaptive-paca-cooldown', type=int, default=2, help='Número de rodadas de cooldown entre reduções consecutivas de PaCA (default: 2)')
    parser.add_argument('--adaptive-rank-cooldown', type=int, default=3, help='Número de rodadas de cooldown para reduções de Rank (default: 3)')
    parser.add_argument('--allow-rank-upscale', action='store_true', default=True, help='Permite subida de rank quando condições melhoram ou para reversão de pânico (default: True)')
    parser.add_argument('--gamma-delta', type=float, default=0.5, help='Expoente de sensibilidade da acurácia no índice de capacidade γ (default: 0.5)')
    parser.add_argument('--plateau-phi-min', type=float, default=0.45, help='Piso mínimo de capacidade no platô (default: 0.45)')
    parser.add_argument('--plateau-decay-rate', type=float, default=0.025, help='Taxa de decaimento por rodada no platô (default: 0.025)')
    parser.add_argument('--qlora', action='store_true', help='Ativa QLoRA (suportado para --model slm e --model lfm): backbone 4-bit NF4, agrega só adaptadores LoRA 16 bits')
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
