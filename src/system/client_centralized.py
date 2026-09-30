"""Centralized client/server training loop."""
import logging
import socket
import sys
import time

import torch

from client_results import save_results
from client_workload import local_training
from utils.fl_math import dequantization, evaluate_model, quantization, resize_model_to_pruned
from utils.lhdq import lhdq_decode, lhdq_encode, is_lhdq_encoded
from utils.network_utils import recv_data, send_data

logger = logging.getLogger(__name__)

def run_centralized(args, model, train_loader, test_loader, ala, run_config, metrics):
    rs_global_acc = metrics.rs_global_acc
    rs_global_loss = metrics.rs_global_loss
    rs_ala_acc = metrics.rs_ala_acc
    rs_local_acc = metrics.rs_local_acc
    rs_train_acc = metrics.rs_train_acc
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        connected = False
        for attempt in range(10):
            try:
                s.connect((args.host, args.port))
                connected = True
                break
            except OSError as e:
                logger.warning(f"Connection failed, retrying in 5 seconds... (Attempt {attempt+1}/10)")
                time.sleep(5)
        if not connected:
            logger.exception("Connection failed after multiple attempts")
            sys.exit(1)
        if args.model == 'slm':
            s.settimeout(1800)
        else:
            s.settimeout(120)
        previous_recv_state = {}
        previous_sent_state = {}
        send_data(s, args.client_idx)
        logger.info(f"Connected to server {args.host}:{args.port}")
        for round_num in range(args.rounds):
            logger.info(f"\n--- Round {round_num + 1}/{args.rounds} ---")
            if args.model in ['clip', 'slm']:
                server_paca, _ = recv_data(s)
                if server_paca is None or server_paca == "end":
                    break
                if isinstance(server_paca, int) and not isinstance(server_paca, bool):
                    if server_paca != args.paca:
                        args.paca = server_paca
                        if "paca" in run_config.get("model", {}):
                            run_config["model"]["paca"]["upper_layers"] = server_paca
                server_r, _ = recv_data(s)
                if server_r is None or server_r == "end":
                    break
                if isinstance(server_r, int) and not isinstance(server_r, bool):
                    if server_r != args.rank:
                        args.rank = server_r
                        if "lora" in run_config.get("model", {}):
                            run_config["model"]["lora"]["r"] = server_r
            global_state, _ = recv_data(s)
            if global_state is None or global_state == "end" or not isinstance(global_state, dict):
                break
            prune, _ = recv_data(s)
            if prune is None or prune == "end":
                break
            logger.info("Received global model.")
            client_dequant_start = time.time()
            if is_lhdq_encoded(global_state):
                global_state = lhdq_decode(global_state, previous_recv_state)
                previous_recv_state = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in global_state.items()}
            else:
                if args.delta_coding:
                    previous_recv_state = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in global_state.items()}
                else:
                    global_state = dequantization(global_state)
            client_dequant_time = time.time() - client_dequant_start
            if round_num + 1 >= 2 and prune == 1 and args.model == 'cnn':
                ammount, _ = recv_data(s)
            eval_start = time.time()
            if round_num + 1 >= 2:
                if args.model in ['clip', 'slm']:
                    old_local_weights = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
                local = resize_model_to_pruned(model, global_state)
                test_accuracy, test_loss = evaluate_model(local, test_loader)
                if args.model in ['clip', 'slm']:
                    for n, p in model.named_parameters():
                        if p.requires_grad and n in old_local_weights:
                            if p.data.shape == old_local_weights[n].shape:
                                p.data.copy_(old_local_weights[n])
            else:
                test_accuracy, test_loss = evaluate_model(model, test_loader)
            client_pre_eval_time = time.time() - eval_start
            logger.info(f"Client {args.client_idx}: Global Model Test Accuracy: {test_accuracy:.2f}% | Test Loss: {test_loss:.4f}")
            rs_global_acc.append(test_accuracy)
            rs_global_loss.append(test_loss)
            train_start = time.time()
            updated_state, personalized_acc = local_training(
                model=model,
                state_dict=global_state,
                prune=prune,
                train_loader=train_loader,
                test_loader=test_loader,
                learning_rate=args.learning_rate,
                round_num=round_num + 1,
                alaarg=args.ala,
                ala=ala,
                model_type=args.model,
                run_config=run_config if args.model in ['clip', 'slm'] else None
            )
            client_training_time = time.time() - train_start
            if personalized_acc is not None:
                rs_ala_acc.append(personalized_acc)
            else:
                rs_ala_acc.append(test_accuracy)
            logger.info(f"Local training completed in {client_training_time:.2f}s.")
            post_eval_start = time.time()
            if not args.skip_post_eval:
                local_test_acc, local_test_loss = evaluate_model(model, test_loader)
                rs_local_acc.append(local_test_acc)
                if not args.skip_train_eval:
                    train_accuracy, train_loss = evaluate_model(model, train_loader)
                    rs_train_acc.append(train_accuracy)
                else:
                    train_accuracy = 0.0
                    rs_train_acc.append(0.0)
                logger.info(f"Client {args.client_idx}: Post-Training Test Accuracy: {local_test_acc:.2f}% | Training Accuracy: {train_accuracy:.2f}%")
            else:
                local_test_acc = test_accuracy
                train_accuracy = 0.0
                rs_local_acc.append(test_accuracy)
                rs_train_acc.append(0.0)
            client_post_eval_time = time.time() - post_eval_start
            quant_start = time.time()
            use_lhdq = (args.delta_coding and round_num >= 1 and previous_sent_state)
            if use_lhdq:
                updated_state = lhdq_encode(updated_state, previous_sent_state)
                previous_sent_state = lhdq_decode(updated_state, previous_sent_state)
            else:
                if args.delta_coding:
                    previous_sent_state = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in updated_state.items()}
                else:
                    updated_state = quantization(updated_state)
            client_quant_time = time.time() - quant_start
            client_eval_time = client_pre_eval_time + client_post_eval_time
            client_processing_overhead = client_eval_time + client_quant_time + client_dequant_time
            logger.info(f"Client {args.client_idx}: Tempos - treino={client_training_time:.2f}s, eval={client_eval_time:.2f}s, quant={client_quant_time:.2f}s, dequant={client_dequant_time:.2f}s")
            try:
                send_data(s, updated_state)
                send_data(s, len(train_loader.dataset))
                send_data(s, args.ala)
                send_data(s, client_training_time)
                send_data(s, test_accuracy)
                send_data(s, test_loss)
                send_data(s, client_processing_overhead)
                logger.info("Client update sent.")
                if args.model == 'slm':
                    round_ack, _ = recv_data(s)
                    if round_ack is None:
                        break
                else:
                    s.recv(3)
                logger.info("Ready for next round...")
            except (OSError, BrokenPipeError, ConnectionResetError, socket.timeout) as e:
                break
    save_results(args, rs_global_acc, rs_global_loss, rs_ala_acc, rs_local_acc, rs_train_acc, idx=args.client_idx, argalgo=args.ala)
    logger.info("\nTraining completed!")
