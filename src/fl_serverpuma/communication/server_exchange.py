# Responsabilidade: Executa a troca de uma rodada com um cliente: prepara o estado a enviar,
# aplica as políticas e recebe atualização e métricas sob os locks existentes.
#
# Principais funções: handle_client.

"""Server server exchange."""

import pickle
import struct
import threading
import copy
import time
import logging
import torch
import torch.nn as nn
from fl_serverpuma.models.clip.sora_pruning import get_trainable_state_dict
from fl_serverpuma.models.slm.sora_pruning import get_trainable_state_dict as get_trainable_state_dict_slm
from fl_serverpuma.algorithms.pruning.structural import prune_and_restructure
from fl_serverpuma.algorithms.pruning.nisp import prune_fc1
from fl_serverpuma.algorithms.pruning.snip import snip_pruning
from fl_serverpuma.algorithms.pruning.snip import apply_mask
from fl_serverpuma.communication.tcp import send_data
from fl_serverpuma.communication.tcp import recv_data
from fl_serverpuma.algorithms.fl_math import quantization
from fl_serverpuma.algorithms.fl_math import dequantization
from fl_serverpuma.communication.lhdq import lhdq_encode
from fl_serverpuma.communication.lhdq import lhdq_decode
from fl_serverpuma.communication.lhdq import is_lhdq_encoded
from fl_serverpuma.algorithms.adaptation.pruning import new_set_amount_prune
from fl_serverpuma.algorithms.adaptation.paca import calculate_adaptive_paca
from fl_serverpuma.algorithms.adaptation.paca import _is_excluded_by_paca
from fl_serverpuma.algorithms.adaptation.rank import calculate_adaptive_rank
from fl_serverpuma.data.server_data import load_test_data
from fl_serverpuma.models.size import get_state_dict_size_mb

logger = logging.getLogger(__name__)

def handle_client(state, conn, client_updates, client_weights, round_num, client_id, client_accuracies, client_losses):
    bit_rate = []
    masks = []
    if 'socket_lock' not in state.clients_info[client_id]:
        state.clients_info[client_id]['socket_lock'] = threading.Lock()

    state.clients_info[client_id]['socket_lock'].acquire()
    try:
        start_time = time.time()
        logger.info(f"Round {round_num}: Handling client {client_id}")

        with state.lock:
            if state.args.model == 'clip':
                current_global_state = get_trainable_state_dict(state.global_model)
            elif state.args.model == 'slm':
                current_global_state = get_trainable_state_dict_slm(state.global_model)
            else:
                current_global_state = state.global_state.copy()

        # --- PaCA DINÂMICO (para CLIP e SLM, a partir do round 2, se ativado) ---
        if state.args.model in ['clip', 'slm'] and round_num >= 1 and getattr(state.args, 'adaptive_paca', False):
            ideal_paca = calculate_adaptive_paca(state, client_id)
            state.clients_info[client_id]['current_paca'] = ideal_paca

            # Filtra: remove chaves de camadas fora do PaCA ideal
            min_layer = state.total_encoder_layers - ideal_paca
            if min_layer > 0:
                keys_before = len(current_global_state)
                current_global_state = {
                    k: v for k, v in current_global_state.items()
                    if not _is_excluded_by_paca(state, k, min_layer)
                }
                keys_after = len(current_global_state)
                if keys_before != keys_after:
                    logger.info(f"PaCA filtro {client_id}: {keys_before} -> {keys_after} chaves "
                                f"(PaCA={ideal_paca}/{state.total_encoder_layers})")

        # --- RANK ADAPTATIVO (para CLIP e SLM, a partir do round 2, se ativado) ---
        if state.args.model in ['clip', 'slm'] and round_num >= 1 and getattr(state.args, 'adaptive_rank', False):
            ideal_rank = calculate_adaptive_rank(state, client_id)
            state.clients_info[client_id]['current_rank'] = ideal_rank

            # Slicing de Rank
            for k, v in current_global_state.items():
                if '.lora_A' in k:
                    current_global_state[k] = v[:ideal_rank, :].clone()
                elif '.lora_B' in k:
                    current_global_state[k] = v[:, :ideal_rank].clone()
                elif '.gate' in k:
                    current_global_state[k] = v[:, :ideal_rank].clone()


        # --- Problema #4: Envia PaCA dinâmico para que o cliente treine apenas as camadas necessárias ---
        if state.args.model in ['clip', 'slm']:
            client_paca = state.clients_info[client_id].get('current_paca', state.args.paca) if getattr(state.args, 'adaptive_paca', False) else state.args.paca
            send_data(conn, client_paca)

            client_rank = state.clients_info[client_id].get('current_rank', state.args.rank) if getattr(state.args, 'adaptive_rank', False) else state.args.rank
            send_data(conn, client_rank)

        if round_num >= 2 and state.prune == 1 and state.args.model == 'cnn':
            logger.info("--- SERVER: PRUNING START (Round 2) ---")
            cnn_prune_start = time.time()
            if state.clients_info[client_id]['original_model_size'] is None:
                max_amount = new_set_amount_prune(state, client_id)
                state.clients_info[client_id]['pruning_rate'] = max_amount
            else:
                max_amount = state.clients_info[client_id]['pruning_rate']

            state.clients_info[client_id]['original_model_size'] = get_state_dict_size_mb(state.global_model)
            logger.info(f"--- SERVER: Calculated pruning rate: {max_amount:.4f}")

            g_model_pruned = copy.deepcopy(state.global_model)
            mask = None

            if state.args.pm == 'OPALA':
                g_model_pruned, mask = prune_and_restructure(model=g_model_pruned, pruning_rate=max_amount, size_fc=state.size_fc, data=state.args.dataset)
            elif state.args.pm == 'NISP':
                trainloader = load_test_data(state, "Cifar10", client_id, 32)
                g_model_pruned, _ = prune_fc1(model=g_model_pruned, dataloader=trainloader, pruning_ratio=max_amount, device=state.args.device)
                g_model_pruned, mask = prune_and_restructure(model=g_model_pruned, pruning_rate=0.0, size_fc=state.size_fc, data=state.args.dataset)
            elif state.args.pm == 'SNIP':
                trainloader = load_test_data(state, "Cifar10", client_id, 32)
                state.mask = snip_pruning(model=g_model_pruned, dataloader=trainloader, criterion=nn.CrossEntropyLoss(), pruning_ratio=max_amount, device=state.args.device)
                g_model_pruned = apply_mask(g_model_pruned, state.mask)
                g_model_pruned, mask = prune_and_restructure(model=g_model_pruned, pruning_rate=0.0, size_fc=state.size_fc, data=state.args.dataset)

            masks.append(mask)
            state.client_masks[client_id] = mask
            g_model_pruned = g_model_pruned.state_dict()
            cnn_prune_time = time.time() - cnn_prune_start
            state.clients_info[client_id]['server_pruning_time'] = cnn_prune_time
            logger.info(f"--- SERVER: PRUNING COMPLETE ({cnn_prune_time:.2f}s) ---")

        # --- Medição granular: quantização do servidor ---
        server_quant_start = time.time()
        comm_start = time.time()

        # Decide se usa LHDQ ou int8
        use_lhdq = (getattr(state.args, 'delta_coding', False) 
                    and round_num >= 2 
                    and client_id in state.previous_sent_state)

        if round_num >= 2 and state.prune == 1 and state.args.model == 'cnn':
            size_before = (len(pickle.dumps(g_model_pruned)) + 4) / (1024 * 1024)
            if use_lhdq:
                prev_state = state.previous_sent_state.get(client_id, {})
                g_model_pruned = lhdq_encode(g_model_pruned, prev_state)
                # Re-decode to ensure identical reference state with the client.
                # This guarantees both sides use the same base reference for future deltas.
                state.previous_sent_state[client_id] = lhdq_decode(g_model_pruned, prev_state)
            else:
                if getattr(state.args, 'delta_coding', False):
                    state.previous_sent_state[client_id] = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in g_model_pruned.items()}
                if not getattr(state.args, 'delta_coding', False):
                    g_model_pruned = quantization(g_model_pruned)
            server_quant_time = time.time() - server_quant_start

            bytes_sent = send_data(conn, g_model_pruned)
            send_data(conn, state.prune)
            send_data(conn, max_amount)
            size_after = bytes_sent / (1024 * 1024)
            with state.lock:
                state.sended_withouquant.append(state.sended_withouquant[-1] + size_before)
                state.sended_ammount.append(state.sended_ammount[-1] + size_after)
        else:
            size_before = (len(pickle.dumps(current_global_state)) + 4) / (1024 * 1024)
            if use_lhdq:
                prev_state = state.previous_sent_state.get(client_id, {})
                current_global_state = lhdq_encode(current_global_state, prev_state)
                # Re-decode to ensure identical reference state with the client.
                # This guarantees both sides use the same base reference for future deltas.
                state.previous_sent_state[client_id] = lhdq_decode(current_global_state, prev_state)
            else:
                if getattr(state.args, 'delta_coding', False):
                    state.previous_sent_state[client_id] = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in current_global_state.items()}
                if not getattr(state.args, 'delta_coding', False):
                    current_global_state = quantization(current_global_state)
            server_quant_time = time.time() - server_quant_start

            bytes_sent = send_data(conn, current_global_state)
            send_data(conn, state.prune)
            size_after = bytes_sent / (1024 * 1024)
            with state.lock:
                state.sended_withouquant.append(state.sended_withouquant[-1] + size_before)
                state.sended_ammount.append(state.sended_ammount[-1] + size_after)

        size_saved = size_before - size_after
        logger.debug(f"Tamanho antes: {size_before:.2f} MB | Tamanho depois: {size_after:.2f} MB | Economia: {size_saved:.2f} MB")
        logger.info(f"Round {round_num}: Sent global model to client {client_id}")

        updated_state, rate, recv_bytes = recv_data(conn, return_bytes=True)
        bit_rate.append(rate)
        if updated_state is not None:
            recv_size_mb = recv_bytes / (1024 * 1024)
            with state.lock:
                state.received_ammount.append(state.received_ammount[-1] + recv_size_mb)

        if updated_state is None:
            logger.warning(f"Round {round_num}: Cliente {client_id} falhou/desconectou. Abortando thread.")
            state.clients_info[client_id]['training_time'] = time.time() - start_time
            return

        # --- Medição granular: dequantização do servidor ---
        server_dequant_start = time.time()
        if is_lhdq_encoded(updated_state):
            prev_recv = state.previous_recv_state.get(client_id, {})
            updated_state = lhdq_decode(updated_state, prev_recv)
        else:
            if not getattr(state.args, 'delta_coding', False):
                updated_state = dequantization(updated_state)
        server_dequant_time = time.time() - server_dequant_start

        # LHDQ: cachear o update recebido para a próxima rodada
        if getattr(state.args, 'delta_coding', False) and updated_state is not None:
            state.previous_recv_state[client_id] = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in updated_state.items()}

        data, rate = recv_data(conn)
        state.client_data[client_id] = data

        dataset_size = data if isinstance(data, (int, float)) else 1.0
        state.clients_info[client_id]['dataset_size'] = dataset_size

        logger.debug(f"Client {client_id} dataset size: {dataset_size}")

        state.argalgo, rate = recv_data(conn)

        # Recebe tempo de treino puro medido pelo cliente
        client_training_time, rate = recv_data(conn)

        client_test_acc, _ = recv_data(conn)
        client_test_loss, _ = recv_data(conn)

        # Recebe overhead de processamento do cliente (eval + quant + dequant)
        client_processing_overhead, _ = recv_data(conn)
        if not isinstance(client_processing_overhead, (int, float)):
            client_processing_overhead = 0.0

        if client_test_acc is not None:
            with state.lock:
                client_accuracies.append(client_test_acc)
                client_losses.append(client_test_loss)
            state.clients_info[client_id]['last_loss'] = client_test_loss

        media_rate = (sum(bit_rate) / len(bit_rate)) / 8 / (1024 * 1024)
        state.clients_info[client_id]['bandwidth'] = media_rate
        state.clients_info[client_id]['data'] = state.client_data[client_id]

        # --- PaCA DINÂMICO: Remove chaves devolvidas que o cliente NÃO treinou ---
        if state.args.model in ['clip', 'slm'] and round_num >= 1 and getattr(state.args, 'adaptive_paca', False):
            ideal_paca = state.clients_info[client_id].get('current_paca', state.args.paca)
            min_layer = state.total_encoder_layers - ideal_paca
            if min_layer > 0 and updated_state is not None:
                updated_state = {
                    k: v for k, v in updated_state.items()
                    if not _is_excluded_by_paca(state, k, min_layer)
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
        state.clients_info[client_id]['comm_time'] = max(0.0, comm_time)
        state.clients_info[client_id]['client_eval_time'] = client_processing_overhead
        state.clients_info[client_id]['server_processing_time'] = server_processing_time

        logger.info(f"Round {round_num}: Client {client_id} tempos: "
                    f"treino={training_time:.2f}s, "
                    f"comm_rede={max(0.0, comm_time):.2f}s, "
                    f"client_overhead={client_processing_overhead:.2f}s, "
                    f"server_proc={server_processing_time:.2f}s")

        if updated_state is not None:
            with state.lock:
                client_updates.append(updated_state)
                client_weights.append(dataset_size)

            state.clients_info[client_id]['training_time'] = training_time
        else:
            logger.warning(f"Round {round_num}: No update received from client {client_id}")
            state.clients_info[client_id]['training_time'] = training_time

    # Tratamento Exato de Exceções de Rede e Pickle 
    except (OSError, pickle.PickleError, struct.error) as e:
        logger.warning(f"Round {round_num}: Cliente {client_id} excedeu o timeout ou desconectou.")
    except Exception as e:
        logger.exception(f"Round {round_num}: Critical error handling client {client_id}")
    finally:
        state.clients_info[client_id]['socket_lock'].release()
