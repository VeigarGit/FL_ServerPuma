# Responsabilidade: Coordena conexões e rodadas, dispara trocas por cliente, chama agregação,
# avaliação e persistência e mantém a sequência do experimento.
#
# Principais funções: run_server.

"""Server round coordinator."""

import socket
import threading
import time
import logging
from pathlib import Path
import torch
from fl_serverpuma.models.clip.sora_pruning import get_trainable_state_dict
from fl_serverpuma.algorithms.fl_math import resize_model_to_pruned
from fl_serverpuma.models.clip.clip_sora_utils import reduce_sora_state_dict_rank
from fl_serverpuma.models.slm.sora_pruning import get_trainable_state_dict as get_trainable_state_dict_slm
from fl_serverpuma.models.size import get_trainable_size_and_params
from fl_serverpuma.communication.tcp import send_data
from fl_serverpuma.communication.tcp import recv_data
from fl_serverpuma.algorithms.fl_math import evaluate_model
from fl_serverpuma.algorithms.aggregation import aggregate_models
from fl_serverpuma.algorithms.adaptation.latency import set_threshold
from fl_serverpuma.communication.server_exchange import handle_client
from fl_serverpuma.data.server_data import load_test_data
from fl_serverpuma.persistence.server_results import save_results
from functools import partial

logger = logging.getLogger(__name__)

def run_server(state, times):
    logger.info("=== Federated Learning Server ===")
    logger.info(f"Host: {state.args.host}:{state.args.port}")
    logger.info(f"Dataset: {state.args.dataset}")
    logger.info(f"Clients per round: {state.args.clients_per_round}")
    logger.info(f"Total rounds: {state.args.rounds}")
    logger.info("=======================================")

    if state.args.model == 'slm':
        state.time_threshold = 1800.0
    elif state.args.model == 'clip':
        state.time_threshold = 600.0
    else:
        state.time_threshold = 120.0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((state.args.host, state.args.port))
        s.listen(state.args.max_clients)

        logger.info(f"Server listening on {state.args.host}:{state.args.port}")
        logger.info(f"Waiting for {state.args.clients_per_round} clients to connect...")

        clientsid = []
        state.client_data = {index: None for index in range(1, state.args.clients_per_round + 1)}

        while len(state.client_connections) < state.args.clients_per_round:
            conn, addr = s.accept()
            idx, rate = recv_data(conn)
            logger.info(f"Client idx: {idx}") 
            clientsid.append(idx)

            state.clients_info[idx] = {
                'training_time': None, 
                'bandwidth': None, 
                'original_model_size': None, 
                'pruning_rate': None,
                'data_size': None,
                'last_flops': None,
                'last_params': None,
                'dataset_size': 1.0,
                'current_paca': state.args.paca,  # PaCA atual do cliente (dinâmico)
                'current_rank': state.args.rank,  # Rank atual do cliente (dinâmico)
            }

            logger.info(f"Client {len(state.client_connections) + 1} connected: {addr}")
            state.client_idx.append(idx)
            state.client_connections.append(conn)
            state.client_addresses.append(addr)

        logger.info(f"All {state.args.clients_per_round} clients connected. Starting training...")

        for round_num in range(state.args.rounds):
            time.sleep(5)
            state.bit = []
            logger.info(f"\n--- Round {round_num + 1}/{state.args.rounds} ---")

            client_updates = []
            client_weights = []
            client_accuracies = []
            client_losses = []
            threads = []
            state.stop_event = threading.Event()
            round_pruning_time = 0.0

            for i, conn in enumerate(state.client_connections):
                t = threading.Thread(
                    target=partial(handle_client, state), 
                    daemon=True, 
                    args=(conn, client_updates, client_weights, round_num + 1, clientsid[i], client_accuracies, client_losses)
                )
                t.start()
                threads.append(t)

            init_time = time.time()
            for t in threads:
                remaining_time = max(0.0, state.time_threshold - (time.time() - init_time))
                t.join(timeout=remaining_time)

            if round_num == 0 or (state.args.model == 'slm' and client_updates):
                set_threshold(state)

            round_duration = time.time() - init_time
            state.round_time.append(round_duration)

            logger.info(f"Round {round_num + 1} duration: {round_duration:.2f} seconds")
            logger.debug(f"Client updates length: {len(client_updates)}")

            if client_updates:
                logger.info(f"Round {round_num + 1}: Aggregating {len(client_updates)} updates")
                state.aggregated_clients.append(len(client_updates))

                # Garantir acesso seguro no momento de ler os updates!
                with state.lock:
                    safe_client_updates = list(client_updates)
                    safe_client_weights = list(client_weights)

                aggregated_state = aggregate_models(safe_client_updates, state.global_state, safe_client_weights, device=state.args.device, client_masks=state.client_masks)

                if state.args.model in ['clip', 'slm'] and 'sora' in state.args.strategy and state.args.prune_freq > 0:
                    # Verifica se a rodada atual é múltipla da frequência desejada
                    if (round_num + 1) % state.args.prune_freq == 0:
                        logger.info(f"--- SERVER: Iniciando Poda Iterativa do SoRA (Round {round_num + 1}) ---")

                        sora_prune_start = time.time()
                        # Threshold adaptativo: sobe progressivamente ao longo das rodadas
                        # Limitado a 5x para não ser destrutivo
                        progress = (round_num + 1) / state.args.rounds  # 0.0 → 1.0
                        decay_factor = 1.0 + progress * 9.0  # 1x → 10x
                        adaptive_threshold = 1e-3 * decay_factor
                        logger.info(f"   Adaptive threshold: {adaptive_threshold:.6f} (progress={progress:.2f})")

                        if state.args.model == 'slm':
                            from fl_serverpuma.models.slm.slm_sora_utils import reduce_sora_state_dict_rank as reduce_sora_state_dict_rank_slm
                            prune_fn = reduce_sora_state_dict_rank_slm
                        else:
                            prune_fn = reduce_sora_state_dict_rank

                        pruned_state, before, after, ranks_info = prune_fn(
                            aggregated_state,
                            threshold=adaptive_threshold,
                            min_rank=state.args.min_rank
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
                            state.global_model = resize_model_to_pruned(state.global_model, aggregated_state)

                with state.lock:
                    state.global_state = aggregated_state
                    state.global_model.load_state_dict(state.global_state, strict=False)

                if state.test_loader is not None or state.client_idx:
                    try:
                        total_correct_weight = 0.0
                        total_loss_weight = 0.0
                        total_samples = 0
                        eval_clients = state.client_idx if state.client_idx else ([state.args.test_client_idx] if state.args.test_client_idx is not None else [0])
                        for i in eval_clients:
                            if i not in state.test_loaders_cache:
                                state.test_loaders_cache[i] = load_test_data(state, state.args.dataset, i, state.args.batch_size)
                            client_loader = state.test_loaders_cache[i]
                            if client_loader is not None:
                                accuracy, avg_loss = evaluate_model(state.global_model, client_loader)
                                n_samples = len(client_loader.dataset)
                                total_correct_weight += accuracy * n_samples
                                total_loss_weight += avg_loss * n_samples
                                total_samples += n_samples

                        final_accuracy = total_correct_weight / total_samples if total_samples > 0 else 0.0
                        final_loss = total_loss_weight / total_samples if total_samples > 0 else 0.0
                        state.rs_test_acc.append(final_accuracy)
                        state.rs_test_loss.append(final_loss)
                        logger.info(f"Round {round_num + 1}: Test Acc: {final_accuracy:.2f}% | Loss: {final_loss:.4f}")
                    except Exception as e:
                        logger.exception(f"Round {round_num + 1}: Erro durante avaliação global no servidor: {e}")
                        if client_accuracies:
                            fallback_acc = sum(client_accuracies) / len(client_accuracies)
                            fallback_loss = sum(client_losses) / len(client_losses) if client_losses else 0.0
                            state.rs_test_acc.append(fallback_acc)
                            state.rs_test_loss.append(fallback_loss)
                            logger.warning(f"Round {round_num + 1}: Usando fallback de acurácia reportada pelos clientes: {fallback_acc:.2f}%")
                        else:
                            state.rs_test_acc.append(0.0)
                            state.rs_test_loss.append(0.0)
                else:
                    logger.info(f"Round {round_num + 1}: Model aggregated (no test data)")

                size_trainable_mb, num_trainable_params = get_trainable_size_and_params(state.global_model)
                logger.info(f'Size Trainable Adapters: {size_trainable_mb:.2f} MB | Trainable Params: {num_trainable_params:,}')
                state.model_size_per_round.append(size_trainable_mb)
                state.trainable_params_per_round.append(num_trainable_params)

                # Total transmitido neste round: (server→clients) + (clients→server)
                sent_this_round = state.sended_ammount[-1] - (state.sended_ammount[-len(client_updates)-1] if len(state.sended_ammount) > len(client_updates) else 0)
                recv_this_round = state.received_ammount[-1] - (state.received_ammount[-len(client_updates)-1] if len(state.received_ammount) > len(client_updates) else 0)
                state.total_transmitted_per_round.append(sent_this_round + recv_this_round)

                paca_list = [state.clients_info[cid].get('current_paca', state.args.paca) for cid in sorted(state.clients_info.keys())]
                state.rs_client_paca.append(paca_list)

                training_time_list = [state.clients_info[cid].get('training_time', 0.0) for cid in sorted(state.clients_info.keys())]
                state.rs_client_training_time.append(training_time_list)

                comm_time_list = [state.clients_info[cid].get('comm_time', 0.0) for cid in sorted(state.clients_info.keys())]
                state.rs_client_comm_time.append(comm_time_list)

                eval_time_list = [state.clients_info[cid].get('client_eval_time', 0.0) for cid in sorted(state.clients_info.keys())]
                state.rs_client_eval_time.append(eval_time_list)

                server_proc_list = [state.clients_info[cid].get('server_processing_time', 0.0) for cid in sorted(state.clients_info.keys())]
                state.rs_server_processing_time.append(server_proc_list)

                state.server_pruning_time.append(round_pruning_time)

                successful_notifications = 0
                for conn in state.client_connections:
                    try:
                        if state.args.model == 'slm':
                            send_data(conn, "end")
                        else:
                            conn.send('end'.encode('utf-8'))
                        successful_notifications += 1
                    except (OSError, BrokenPipeError, ConnectionResetError) as e:
                        logger.error(f"Error notifying client: {e}")

                logger.info(f"Round {round_num + 1}: Notified {successful_notifications} clients.")
            else:
                logger.warning(f"Round {round_num + 1}: No client updates received (timeout or client failure).")

        logger.info(f"\nTraining completed after {state.args.rounds} rounds!")

        # Tratamento tolerante para fechamento residual dos soquetes TCP 
        for conn in state.client_connections:
            try:
                conn.close()
            except OSError as e:
                logger.debug(f"Socket close exception (safe to ignore): {e}")
        logger.info("All client connections closed.")

    if state.args.save_model:
        logger.info(f"Salvando adaptadores e gates treinados...")
        save_path = Path(state.args.save_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if state.args.model == 'clip':
            # Salva APENAS o LoRA/SoRA e a Head de classificação para economizar espaço
            weights_to_save = get_trainable_state_dict(state.global_model)
        elif state.args.model == 'slm':
            weights_to_save = get_trainable_state_dict_slm(state.global_model)
        else:
            weights_to_save = state.global_model.state_dict()

        torch.save(weights_to_save, save_path)
        logger.info(f"Modelo salvo com sucesso em: {save_path}")

    save_results(state, times)
