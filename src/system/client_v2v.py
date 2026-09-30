"""Decentralized client loop driven by SUMO encounter signals."""
import logging
import socket
import time
from pathlib import Path

import torch

from client_results import save_results
from client_workload import local_training
from utils.fl_math import evaluate_model
from utils.network_utils import recv_data, send_data

logger = logging.getLogger(__name__)

def run_decentralized(args, model, train_loader, test_loader, ala, metrics):
    rs_global_acc = metrics.rs_global_acc
    rs_global_loss = metrics.rs_global_loss
    rs_ala_acc = metrics.rs_ala_acc
    rs_local_acc = metrics.rs_local_acc
    rs_train_acc = metrics.rs_train_acc
    rs_post_p2p_acc = metrics.rs_post_p2p_acc
    rs_post_p2p_encounter_ids = metrics.rs_post_p2p_encounter_ids
    rs_p2p_data_sent_mb = metrics.rs_p2p_data_sent_mb
    rs_p2p_data_sent_enc_ids = metrics.rs_p2p_data_sent_enc_ids
    import json
    import threading
    logger.info(f"Client {args.client_idx}: Modo Descentralizado (D-PSGD) Ativado.")
    encounters_dir = Path(__file__).resolve().parents[1] / "results" / "encounters"
    in_memory_weights = {}
    in_memory_lock = threading.Lock()
    def wait_for_round_signal(encounters_dir: Path, round_num: int) -> bool:
        while True:
            if (encounters_dir / ".simulation_done").exists():
                return False
            if (encounters_dir / f".start_round_{round_num}").exists():
                return True
            time.sleep(1)
    def wait_for_encounter(encounters_dir: Path, client_idx: int, last_processed: int, current_round: int):
        while True:
            if (encounters_dir / ".simulation_done").exists():
                return None
            if (encounters_dir / f".start_round_{current_round}").exists():
                return "skip"
            encounter_files = list(encounters_dir.glob("encounter_*.json"))
            if encounter_files:
                encounter_files.sort(key=lambda x: int(x.stem.split("_")[-1]))
                for ef in encounter_files:
                    try:
                        with open(ef, "r") as f:
                            data = json.load(f)
                        enc_id = data.get("encounter_id", 0)
                        if enc_id <= last_processed:
                            continue
                        clients_in_enc = data.get("clients", [])
                        if client_idx in clients_in_enc:
                            return data
                    except Exception:
                        pass
            time.sleep(1)
    def p2p_server_thread():
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', 9000))
        server_socket.listen(50)
        logger.info(f"Client {args.client_idx}: P2P Server listening on port 9000")
        while True:
            try:
                conn, addr = server_socket.accept()
                data, _ = recv_data(conn)
                if data is not None:
                    sender_id, enc_id, state = data
                    with in_memory_lock:
                        if enc_id not in in_memory_weights:
                            in_memory_weights[enc_id] = {}
                        in_memory_weights[enc_id][sender_id] = state
                    logger.info(f"Client {args.client_idx}: Recebeu pesos do Cliente {sender_id} para encontro {enc_id}")
                conn.close()
            except Exception as e:
                logger.error(f"Erro no P2P Server: {e}")
    threading.Thread(target=p2p_server_thread, daemon=True).start()
    epoch = 1
    last_processed_encounter = 0
    max_epochs = args.max_epochs if args.max_epochs > 0 else float('inf')
    while epoch <= max_epochs:
        logger.info(f"Client {args.client_idx}: Aguardando inicio da Rodada {epoch-1}...")
        if not wait_for_round_signal(encounters_dir, epoch - 1):
            logger.info(f"Client {args.client_idx}: Sinal de fim de simulacao (.simulation_done) recebido!")
            break
        logger.info(f"\n--- Iniciando Epoca {epoch}{f'/{args.max_epochs}' if args.max_epochs > 0 else ''} (Rodada {epoch-1}) ---")
        if args.model in ['clip', 'slm']:
            if args.model == 'slm':
                from lora_slm.sora import get_trainable_state_dict as get_trainable_state_dict_slm
                dummy_state = get_trainable_state_dict_slm(model)
            else:
                from lora_clip.sora import get_trainable_state_dict as get_trainable_state_dict_clip
                dummy_state = get_trainable_state_dict_clip(model)
        else:
            dummy_state = model.state_dict()
        updated_state, personalized_acc = local_training(
            model=model,
            state_dict=dummy_state,
            prune=args.prune,
            train_loader=train_loader,
            test_loader=test_loader,
            learning_rate=args.learning_rate,
            round_num=epoch,
            alaarg=args.ala,
            ala=ala,
            model_type=args.model,
        )
        # ── FASE 3: Avaliação de Treino (Pré-Encontro) ─────────────────────
        if not args.skip_post_eval:
            local_test_acc, _ = evaluate_model(model, test_loader)
            rs_local_acc.append(local_test_acc)
            if not args.skip_train_eval:
                train_accuracy, _ = evaluate_model(model, train_loader)
                rs_train_acc.append(train_accuracy)
            else:
                rs_train_acc.append(0.0)
            logger.info(f"Client {args.client_idx}: Epoca {epoch} Local Test Acc: {local_test_acc:.2f}%")

        save_results(args, rs_global_acc, rs_global_loss, rs_ala_acc, rs_local_acc, rs_train_acc,
                     idx=args.client_idx, argalgo=args.ala,
                     rs_post_p2p_acc=rs_post_p2p_acc,
                     rs_post_p2p_encounter_ids=rs_post_p2p_encounter_ids,
                     rs_p2p_data_sent_mb=rs_p2p_data_sent_mb,
                     rs_p2p_data_sent_enc_ids=rs_p2p_data_sent_enc_ids)

        # ── FASE 4: Sinalizar conclusão da época ────────────────────────
        (encounters_dir / f".epoch_done_client_{args.client_idx}").touch()
        logger.info(f"Client {args.client_idx}: Epoca concluida. Aguardando encontro P2P...")

        # ── FASE 5: Aguardar e Processar Encontro ───────────────────────
        encounter_data = wait_for_encounter(encounters_dir, args.client_idx, last_processed_encounter, current_round=epoch)

        if encounter_data is None:
            logger.info(f"Client {args.client_idx}: Sinal de fim de simulacao recebido. Encerrando.")
            break

        if encounter_data == "skip":
            logger.info(f"Client {args.client_idx}: Encontro nao ocorreu para mim nesta rodada. Avancando para prox epoca.")
            epoch += 1
            continue

        enc_id = encounter_data.get("encounter_id", 0)
        clients_in_encounter = encounter_data.get("clients", [])
        etc_seconds = encounter_data.get("etc_seconds", 60)

        if isinstance(etc_seconds, str) and etc_seconds == "Carros Parados":
            poll_timeout = 115
        else:
            poll_timeout = max(5, int(etc_seconds - 5))

        logger.info(f"Client {args.client_idx}: [D-PSGD] Encontro {enc_id} detectado! ETC={etc_seconds}s. Iniciando troca P2P.")

        # ── FASE 6: Troca P2P e Agregação (D-PSGD) ──────────────────────
        if args.model == 'clip':
            from lora_clip.sora import get_trainable_state_dict
            snapshot = get_trainable_state_dict(model)
        else:
            import copy
            snapshot = copy.deepcopy(model.state_dict())

        other_clients = [c for c in clients_in_encounter if c != args.client_idx]

        for c in other_clients:
            target_host = f"fl-client-v2v-{c}"
            sent = False
            for attempt in range(poll_timeout):
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(2.0)
                        s.connect((target_host, 9000))
                        import pickle
                        snapshot_bytes = len(pickle.dumps((args.client_idx, enc_id, snapshot)))
                        rs_p2p_data_sent_mb.append(snapshot_bytes / (1024 * 1024))
                        rs_p2p_data_sent_enc_ids.append(enc_id)
                        send_data(s, (args.client_idx, enc_id, snapshot))
                    logger.info(f"Client {args.client_idx}: Enviou pesos para {target_host} ({snapshot_bytes / (1024*1024):.2f} MB)")
                    sent = True
                    break
                except (ConnectionRefusedError, socket.timeout):
                    time.sleep(1)
            if not sent:
                logger.warning(f"Client {args.client_idx}: Falhou ao enviar pesos para {target_host} apos {poll_timeout}s")

        # Aguardar pesos dos vizinhos
        logger.info(f"Client {args.client_idx}: Aguardando receber pesos dos vizinhos {other_clients}...")
        received_states = []
        for attempt in range(poll_timeout):
            with in_memory_lock:
                if enc_id in in_memory_weights:
                    missing = [c for c in other_clients if c not in in_memory_weights[enc_id]]
                    if not missing:
                        for c in other_clients:
                            received_states.append(in_memory_weights[enc_id][c])
                        break
            time.sleep(1)

        if len(received_states) == len(other_clients):
            logger.info(f"Client {args.client_idx}: Todos os pesos recebidos! Agregando D-PSGD...")
            if args.model in ['clip', 'slm']:
                for k in snapshot.keys():
                    tensors = [snapshot[k].float()] + [s[k].float() for s in received_states]
                    avg_tensor = torch.mean(torch.stack(tensors), dim=0).to(snapshot[k].dtype)
                    snapshot[k].copy_(avg_tensor)

                if args.model == 'clip':
                    from lora_clip.sora import set_trainable_state_dict
                    set_trainable_state_dict(model, snapshot)
                else:
                    from lora_slm.sora import set_trainable_state_dict as set_trainable_state_dict_slm
                    set_trainable_state_dict_slm(model, snapshot)
            else:
                for k in snapshot.keys():
                    tensors = [snapshot[k].float()] + [s[k].float() for s in received_states]
                    snapshot[k] = torch.mean(torch.stack(tensors), dim=0).to(snapshot[k].dtype)
                model.load_state_dict(snapshot)

            # Limpar RAM
            with in_memory_lock:
                del in_memory_weights[enc_id]

            # ── FASE 7: Avaliação Pós-Agregação ──────────────────────
            if not args.skip_post_eval:
                post_p2p_acc, _ = evaluate_model(model, test_loader)
                logger.info(f"Client {args.client_idx}: Post-D-PSGD Test Accuracy: {post_p2p_acc:.2f}%")
                rs_post_p2p_acc.append(post_p2p_acc)
                rs_post_p2p_encounter_ids.append(enc_id)
                save_results(args, rs_global_acc, rs_global_loss, rs_ala_acc, rs_local_acc, rs_train_acc,
                             idx=args.client_idx, argalgo=args.ala,
                             rs_post_p2p_acc=rs_post_p2p_acc,
                             rs_post_p2p_encounter_ids=rs_post_p2p_encounter_ids,
                             rs_p2p_data_sent_mb=rs_p2p_data_sent_mb,
                             rs_p2p_data_sent_enc_ids=rs_p2p_data_sent_enc_ids)

            # Sinalizar ao SUMO que a troca terminou
            (encounters_dir / f".done_enc_{enc_id}_client_{args.client_idx}").touch()
            logger.info(f"Client {args.client_idx}: Agregacao concluida. .done_enc enviado.")
        else:
            logger.warning(f"Client {args.client_idx}: Timeout ({poll_timeout}s) esperando pesos. Recebeu {len(received_states)}/{len(other_clients)}.")
            (encounters_dir / f".done_enc_{enc_id}_client_{args.client_idx}").touch()

        last_processed_encounter = enc_id
        epoch += 1

    # Ao final do max_epochs ou simulation_done
    logger.info(f"Client {args.client_idx}: Loop descentralizado finalizado.")
    save_results(args, rs_global_acc, rs_global_loss, rs_ala_acc, rs_local_acc, rs_train_acc,
                 idx=args.client_idx, argalgo=args.ala,
                 rs_post_p2p_acc=rs_post_p2p_acc,
                 rs_post_p2p_encounter_ids=rs_post_p2p_encounter_ids,
                 rs_p2p_data_sent_mb=rs_p2p_data_sent_mb,
                 rs_p2p_data_sent_enc_ids=rs_p2p_data_sent_enc_ids)
    logger.info("\nTraining completed!")
