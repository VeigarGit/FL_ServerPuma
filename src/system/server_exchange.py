"""Per-client socket exchange methods for FederatedLearningServer."""
import copy
import logging
import pickle
import struct
import threading
import time

import torch
import torch.nn as nn

from lora_clip.sora import get_trainable_state_dict
from lora_slm.sora import get_trainable_state_dict as get_trainable_state_dict_slm
from prunning import prune_and_restructure
from prunning_nisp import prune_fc1
from prunning_snip import snip_pruning, apply_mask
from utils.fl_math import quantization, dequantization
from utils.lhdq import lhdq_encode, lhdq_decode, is_lhdq_encoded
from utils.network_utils import send_data, recv_data

logger = logging.getLogger(__name__)

def get_state_dict_size_mb(state):
    """Calcula matematicamente o tamanho em MB sem serializar em pickle, evitando travar o GIL e a CPU."""
    total_bytes = 0
    if hasattr(state, 'state_dict'):
        state = state.state_dict()

    if isinstance(state, dict) and state.get('__lhdq__'):
        for k, v in state['keys'].items():
            if 'raw' in v:
                total_bytes += v['raw'].numel() * v['raw'].element_size()
            else:
                total_bytes += len(v['packed']) + 4  # 4 bytes for mu/sigma (float16)
    elif isinstance(state, dict):
        for v in state.values():
            if isinstance(v, torch.Tensor):
                total_bytes += v.numel() * v.element_size()
            elif isinstance(v, dict) and v.get('dtype') == 'quantized_int8':
                if 'weights' in v and isinstance(v['weights'], torch.Tensor):
                    total_bytes += v['weights'].numel() * v['weights'].element_size()
                if 'scale' in v and isinstance(v['scale'], torch.Tensor):
                    total_bytes += v['scale'].numel() * v['scale'].element_size()
            elif isinstance(v, (int, float)):
                total_bytes += 8
    return total_bytes / (1024 * 1024)

class ServerExchangeMixin:
    def recvall(self, conn, n):
        data = b''
        while len(data) < n:
            packet = conn.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data
    def handle_client(self, conn, client_updates, client_weights, round_num, client_id, client_accuracies, client_losses):
        bit_rate = []
        masks = []
        if 'socket_lock' not in self.clients_info[client_id]:
            self.clients_info[client_id]['socket_lock'] = threading.Lock()

        self.clients_info[client_id]['socket_lock'].acquire()
        try:
            start_time = time.time()
            logger.info(f"Round {round_num}: Handling client {client_id}")

            with self.lock:
                if self.args.model == 'clip':
                    current_global_state = get_trainable_state_dict(self.global_model)
                elif self.args.model == 'slm':
                    current_global_state = get_trainable_state_dict_slm(self.global_model)
                else:
                    current_global_state = self.global_state.copy()

            # --- PaCA DINÂMICO (para CLIP e SLM, a partir do round 2, se ativado) ---
            if self.args.model in ['clip', 'slm'] and round_num >= 1 and getattr(self.args, 'adaptive_paca', False):
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

            # --- RANK ADAPTATIVO (para CLIP e SLM, a partir do round 2, se ativado) ---
            if self.args.model in ['clip', 'slm'] and round_num >= 1 and getattr(self.args, 'adaptive_rank', False):
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


            # --- Problema #4: Envia PaCA dinâmico para que o cliente treine apenas as camadas necessárias ---
            if self.args.model in ['clip', 'slm']:
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
                size_before = (len(pickle.dumps(current_global_state)) + 4) / (1024 * 1024)
                if use_lhdq:
                    prev_state = self.previous_sent_state.get(client_id, {})
                    current_global_state = lhdq_encode(current_global_state, prev_state)
                    # Re-decode to ensure identical reference state with the client.
                    # This guarantees both sides use the same base reference for future deltas.
                    self.previous_sent_state[client_id] = lhdq_decode(current_global_state, prev_state)
                else:
                    if getattr(self.args, 'delta_coding', False):
                        self.previous_sent_state[client_id] = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in current_global_state.items()}
                    if not getattr(self.args, 'delta_coding', False):
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

            updated_state, rate, recv_bytes = recv_data(conn, return_bytes=True)
            bit_rate.append(rate)
            if updated_state is not None:
                recv_size_mb = recv_bytes / (1024 * 1024)
                with self.lock:
                    self.received_ammount.append(self.received_ammount[-1] + recv_size_mb)

            if updated_state is None:
                logger.warning(f"Round {round_num}: Cliente {client_id} falhou/desconectou. Abortando thread.")
                self.clients_info[client_id]['training_time'] = time.time() - start_time
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

            data, rate = recv_data(conn)
            self.client_data[client_id] = data

            dataset_size = data if isinstance(data, (int, float)) else 1.0
            self.clients_info[client_id]['dataset_size'] = dataset_size

            logger.debug(f"Client {client_id} dataset size: {dataset_size}")

            self.argalgo, rate = recv_data(conn)

            # Recebe tempo de treino puro medido pelo cliente
            client_training_time, rate = recv_data(conn)

            client_test_acc, _ = recv_data(conn)
            client_test_loss, _ = recv_data(conn)

            # Recebe overhead de processamento do cliente (eval + quant + dequant)
            client_processing_overhead, _ = recv_data(conn)
            if not isinstance(client_processing_overhead, (int, float)):
                client_processing_overhead = 0.0

            if client_test_acc is not None:
                with self.lock:
                    client_accuracies.append(client_test_acc)
                    client_losses.append(client_test_loss)
                self.clients_info[client_id]['last_loss'] = client_test_loss

            media_rate = (sum(bit_rate) / len(bit_rate)) / 8 / (1024 * 1024)
            self.clients_info[client_id]['bandwidth'] = media_rate
            self.clients_info[client_id]['data'] = self.client_data[client_id]

            # --- PaCA DINÂMICO: Remove chaves devolvidas que o cliente NÃO treinou ---
            if self.args.model in ['clip', 'slm'] and round_num >= 1 and getattr(self.args, 'adaptive_paca', False):
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
            self.clients_info[client_id]['comm_time'] = max(0.0, comm_time)
            self.clients_info[client_id]['client_eval_time'] = client_processing_overhead
            self.clients_info[client_id]['server_processing_time'] = server_processing_time

            logger.info(f"Round {round_num}: Client {client_id} tempos: "
                        f"treino={training_time:.2f}s, "
                        f"comm_rede={max(0.0, comm_time):.2f}s, "
                        f"client_overhead={client_processing_overhead:.2f}s, "
                        f"server_proc={server_processing_time:.2f}s")

            if updated_state is not None:
                with self.lock:
                    client_updates.append(updated_state)
                    client_weights.append(dataset_size)

                self.clients_info[client_id]['training_time'] = training_time
            else:
                logger.warning(f"Round {round_num}: No update received from client {client_id}")
                self.clients_info[client_id]['training_time'] = training_time

        # Tratamento Exato de Exceções de Rede e Pickle
        except (OSError, pickle.PickleError, struct.error) as e:
            logger.warning(f"Round {round_num}: Cliente {client_id} excedeu o timeout ou desconectou.")
        except Exception as e:
            logger.exception(f"Round {round_num}: Critical error handling client {client_id}")
        finally:
            self.clients_info[client_id]['socket_lock'].release()
