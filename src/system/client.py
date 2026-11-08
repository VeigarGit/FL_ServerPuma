"""Ponto de entrada unificado e modo centralizado do cliente Federated Learning."""

import argparse
import logging
import os
import random
import socket
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from ALA import ALA  # noqa: E402
from ALA_LoRA import ALA_LoRA  # noqa: E402
from client_data import (  # noqa: E402
    ClientHistory,
    build_client_dataloaders,
    load_data,
    save_results,
)
from client_training import (  # noqa: E402
    build_client_model,
    local_initialization,
    local_training,
    map_sequential_to_simplemodel,
    resolve_device,
    resolve_heterogeneous_paca,
)
from client_v2x import (  # noqa: E402
    P2PWeightStore,
    exchange_v2i_model,
    run_decentralized_client,
    send_p2p_model,
    wait_for_encounter,
    wait_for_round_signal,
    write_v2i_failure,
)
from lora_slm.qlora import set_qlora_state_dict  # noqa: E402

from utils.fl_math import (  # noqa: E402
    dequantization,
    evaluate_model,
    quantization,
    resize_model_to_pruned,
)
from utils.lhdq import is_lhdq_encoded, lhdq_decode, lhdq_encode  # noqa: E402
from utils.network_utils import recv_data, send_data  # noqa: E402

logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

__all__ = [
    "ClientHistory",
    "P2PWeightStore",
    "build_client_dataloaders",
    "build_client_model",
    "exchange_v2i_model",
    "load_data",
    "local_initialization",
    "local_training",
    "main",
    "map_sequential_to_simplemodel",
    "parse_args",
    "resolve_device",
    "resolve_heterogeneous_paca",
    "run_centralized_client",
    "run_decentralized_client",
    "save_results",
    "send_p2p_model",
    "wait_for_encounter",
    "wait_for_round_signal",
    "write_v2i_failure",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="Cifar100")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="Diretório com config.json, train/ e test/; mantém o nome lógico do dataset",
    )
    parser.add_argument("--client-idx", type=int, default=0)
    parser.add_argument("--in-features", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--dim", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--random-client", action="store_true")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("-did", "--device_id", type=str, default="0")
    parser.add_argument("--ala", type=int, default=0)
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument("--config", type=str, default="lora_clip/train_config.yml")
    parser.add_argument("--experiments", type=int, default=1)
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--execution-id", default="", help="ID da simulação V2V/V2I")
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--encounters-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp-name", type=str, default="default_exp", help="Nome da sessão")
    parser.add_argument(
        "--timestamp", type=str, default="", help="Timestamp para diferenciar rodadas de simulação"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="lora",
        choices=["lora", "adalora", "sora_with_schedule", "sora_no_schedule"],
    )
    parser.add_argument("--rank", type=int, default=8, help="Rank para o SoRA/LoRA")
    parser.add_argument(
        "--paca", type=int, default=12, help="Número de camadas para a estratégia PaCA"
    )

    # --- Parâmetros de Sparsidade (SoRA) ---
    parser.add_argument(
        "--sparse-lr",
        type=float,
        default=None,
        help="Taxa de aprendizado para a poda (Softshrink)",
    )
    parser.add_argument(
        "--max-lambda", type=float, default=None, help="Força máxima da penalidade L1 (max_lambda)"
    )

    # --- PaCA Heterogêneo ---
    parser.add_argument(
        "--random-paca",
        action="store_true",
        help="Sorteia um valor de PaCA aleatório para este cliente",
    )
    parser.add_argument(
        "--paca-min", type=int, default=1, help="Valor mínimo de PaCA no sorteio aleatório"
    )
    parser.add_argument(
        "--paca-max", type=int, default=12, help="Valor máximo de PaCA no sorteio aleatório"
    )
    parser.add_argument(
        "--paca-list",
        type=str,
        default=None,
        help='Lista de PaCAs pré-definidos por cliente (ex: "4,6,8,12"). '
        "O cliente usa o valor na posição client_idx %% len(lista)",
    )
    parser.add_argument(
        "--prune",
        type=int,
        default=1,
        choices=[0, 1],
        help="Habilitar pruning estrutural (1=sim, 0=não, default: 1)",
    )
    parser.add_argument(
        "--mode", type=str, default="centralized", choices=["centralized", "decentralized"]
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=0,
        help="Limite de epocas no modo descentralizado (0 = infinito)",
    )
    parser.add_argument(
        "--delta-coding",
        action="store_true",
        help="Ativa LHDQ (Low Huffman-coded Delta Quantization) em vez de int8",
    )

    # --- Otimização de Avaliação ---
    parser.add_argument(
        "--skip-post-eval",
        action="store_true",
        help="Pula todas as avaliações pós-treino no cliente (local_test_acc e train_acc) "
        "para máxima velocidade, mantendo apenas Global Model Test Acc",
    )
    parser.add_argument(
        "--skip-train-eval",
        action="store_true",
        help="Pula apenas a avaliação demorada no conjunto de treino pós-treino",
    )

    # --- QLoRA (backbone 4-bit NF4 + adaptadores LoRA 16 bits) ---
    parser.add_argument(
        "--qlora",
        action="store_true",
        help="Ativa QLoRA (apenas --model slm --strategy lora, modo centralizado)",
    )

    return parser.parse_args()


# =====================================================================
# MODO CENTRALIZADO (EXECUÇÃO POR RODADAS TCP)
# =====================================================================


def _decode_server_payload(
    global_state: dict, previous_recv_state: dict, delta_coding: bool
) -> tuple[dict, dict, float]:
    """Descomprime o payload recebido do servidor (LHDQ ou INT8) e atualiza a referência."""
    start = time.time()
    if is_lhdq_encoded(global_state):
        global_state = lhdq_decode(global_state, previous_recv_state)
        previous_recv_state = {
            k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in global_state.items()
        }
    else:
        if delta_coding:
            previous_recv_state = {
                k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in global_state.items()
            }
        else:
            global_state = dequantization(global_state)
    return global_state, previous_recv_state, time.time() - start


def _encode_client_payload(
    updated_state: dict,
    previous_sent_state: dict,
    round_num: int,
    delta_coding: bool,
    qlora: bool,
) -> tuple[dict, dict, float]:
    """Comprime os pesos locais (LHDQ, INT8 ou passthrough 16-bit QLoRA) para envio."""
    start = time.time()
    use_lhdq = delta_coding and round_num >= 1 and bool(previous_sent_state)
    if use_lhdq:
        updated_state = lhdq_encode(updated_state, previous_sent_state)
        previous_sent_state = lhdq_decode(updated_state, previous_sent_state)
    else:
        if delta_coding:
            previous_sent_state = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in updated_state.items()
            }
        elif not qlora:  # QLoRA: adaptadores trafegam em 16 bits (sem int8)
            updated_state = quantization(updated_state)
    return updated_state, previous_sent_state, time.time() - start


def _evaluate_global_before_training(
    model: nn.Module,
    global_state: dict,
    test_loader: DataLoader,
    round_num: int,
    model_type: str,
    qlora: bool,
) -> tuple[float, float, float]:
    """Avalia o modelo global recebido antes do treino local, preservando pesos locais."""
    eval_start = time.time()
    if round_num + 1 >= 2:
        old_local_weights = None
        if model_type in ["clip", "slm"]:
            old_local_weights = {
                n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad
            }
        if qlora:
            set_qlora_state_dict(model, global_state)
            local = model
        else:
            local = resize_model_to_pruned(model, global_state)
        test_accuracy, test_loss = evaluate_model(local, test_loader)
        if old_local_weights is not None:
            for n, p in model.named_parameters():
                if (
                    p.requires_grad
                    and n in old_local_weights
                    and p.data.shape == old_local_weights[n].shape
                ):
                    p.data.copy_(old_local_weights[n])
    else:
        test_accuracy, test_loss = evaluate_model(model, test_loader)
    return test_accuracy, test_loss, time.time() - eval_start


def _evaluate_post_training(
    args: argparse.Namespace,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    fallback_test_acc: float,
    history: ClientHistory,
) -> float:
    """Executa avaliação pós-treino local respeitando --skip-post-eval e --skip-train-eval."""
    post_eval_start = time.time()
    if not args.skip_post_eval:
        local_test_acc, _ = evaluate_model(model, test_loader)
        history.local_acc.append(local_test_acc)
        if not args.skip_train_eval:
            train_accuracy, _ = evaluate_model(model, train_loader)
            history.train_acc.append(train_accuracy)
        else:
            train_accuracy = 0.0
            history.train_acc.append(0.0)
        logger.info(
            f"Client {args.client_idx}: Post-Training Test Accuracy: {local_test_acc:.2f}% | "
            f"Training Accuracy: {train_accuracy:.2f}%"
        )
    else:
        history.local_acc.append(fallback_test_acc)
        history.train_acc.append(0.0)
    return time.time() - post_eval_start


def _connect_to_central_server(args: argparse.Namespace, sock: socket.socket) -> None:
    connected = False
    for attempt in range(10):
        try:
            sock.connect((args.host, args.port))
            connected = True
            break
        except OSError:
            logger.warning(
                f"Connection failed, retrying in 5 seconds... (Attempt {attempt + 1}/10)"
            )
            time.sleep(5)
    if not connected:
        logger.exception("Connection failed after multiple attempts")
        sys.exit(1)
    sock.settimeout(1800 if args.model == "slm" else 120)
    send_data(sock, args.client_idx)
    logger.info(f"Connected to server {args.host}:{args.port}")


def run_centralized_client(
    args: argparse.Namespace,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    ala,
    run_config: dict | None,
    history: ClientHistory,
) -> None:
    """Executa o ciclo completo de treinamento federado centralizado via TCP."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        _connect_to_central_server(args, s)
        previous_recv_state: dict = {}
        previous_sent_state: dict = {}

        for round_num in range(args.rounds):
            logger.info(f"\n--- Round {round_num + 1}/{args.rounds} ---")
            if args.model in ["clip", "slm"]:
                server_paca, _ = recv_data(s)
                if server_paca is None or server_paca == "end":
                    break
                if (
                    isinstance(server_paca, int)
                    and not isinstance(server_paca, bool)
                    and server_paca != args.paca
                ):
                    args.paca = server_paca
                    if run_config and "paca" in run_config.get("model", {}):
                        run_config["model"]["paca"]["upper_layers"] = server_paca

                server_r, _ = recv_data(s)
                if server_r is None or server_r == "end":
                    break
                if (
                    isinstance(server_r, int)
                    and not isinstance(server_r, bool)
                    and server_r != args.rank
                ):
                    args.rank = server_r
                    if run_config and "lora" in run_config.get("model", {}):
                        run_config["model"]["lora"]["r"] = server_r

            global_state, _ = recv_data(s)
            if global_state is None or global_state == "end" or not isinstance(global_state, dict):
                break
            prune, _ = recv_data(s)
            if prune is None or prune == "end":
                break
            logger.info("Received global model.")

            global_state, previous_recv_state, client_dequant_time = _decode_server_payload(
                global_state, previous_recv_state, args.delta_coding
            )

            if round_num + 1 >= 2 and prune == 1 and args.model == "cnn":
                _amount, _ = recv_data(s)

            test_accuracy, test_loss, client_pre_eval_time = _evaluate_global_before_training(
                model, global_state, test_loader, round_num, args.model, args.qlora
            )
            logger.info(
                f"Client {args.client_idx}: Global Model Test Accuracy: {test_accuracy:.2f}% | "
                f"Test Loss: {test_loss:.4f}"
            )
            history.global_acc.append(test_accuracy)
            history.global_loss.append(test_loss)

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
                run_config=run_config if args.model in ["clip", "slm"] else None,
            )
            client_training_time = time.time() - train_start
            history.ala_acc.append(
                personalized_acc if personalized_acc is not None else test_accuracy
            )
            logger.info(f"Local training completed in {client_training_time:.2f}s.")

            client_post_eval_time = _evaluate_post_training(
                args, model, train_loader, test_loader, test_accuracy, history
            )
            updated_state, previous_sent_state, client_quant_time = _encode_client_payload(
                updated_state, previous_sent_state, round_num, args.delta_coding, args.qlora
            )

            client_eval_time = client_pre_eval_time + client_post_eval_time
            client_processing_overhead = client_eval_time + client_quant_time + client_dequant_time
            logger.info(
                f"Client {args.client_idx}: Tempos - treino={client_training_time:.2f}s, "
                f"eval={client_eval_time:.2f}s, quant={client_quant_time:.2f}s, "
                f"dequant={client_dequant_time:.2f}s"
            )
            try:
                send_data(s, updated_state)
                send_data(s, len(train_loader))
                send_data(s, args.ala)
                send_data(s, client_training_time)
                send_data(s, test_accuracy)
                send_data(s, test_loss)
                send_data(s, client_processing_overhead)
                logger.info("Client update sent.")
                if args.model == "slm":
                    round_ack, _ = recv_data(s)
                    if round_ack is None:
                        break
                else:
                    s.recv(3)
                logger.info("Ready for next round...")
            except (OSError, BrokenPipeError, ConnectionResetError, TimeoutError):
                break

    history.save(args)
    logger.info("\nTraining completed!")


# =====================================================================
# PONTO DE ENTRADA PRINCIPAL
# =====================================================================


def main():
    args = parse_args()
    args.training_seed = args.seed + args.client_idx
    random.seed(args.training_seed)
    np.random.seed(args.training_seed)
    torch.manual_seed(args.training_seed)
    args.encounter_metrics = []

    device = resolve_device(args)
    resolve_heterogeneous_paca(args)

    logger.info("=== Federated Learning Client ===")
    logger.info(f"Host: {args.host}:{args.port}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Client: {args.client_idx} | PaCA: {args.paca}")
    logger.info(f"Rounds: {args.rounds}")
    logger.info(f"Batch size: {args.batch_size} | LR: {args.learning_rate}")
    logger.info("=================================")

    model, run_config = build_client_model(args, device)
    train_loader, test_loader, test_majority_baseline = build_client_dataloaders(
        args, device, run_config
    )

    loss = nn.CrossEntropyLoss()
    if args.model == "clip":
        ala = ALA_LoRA(args.client_idx, loss, train_loader, 32, 80, 1.0, device)
    else:
        ala = ALA(args.client_idx, loss, train_loader, 32, 80, 2, 1.0, device)

    history = ClientHistory()
    time.sleep(10)

    if args.mode == "centralized":
        run_centralized_client(args, model, train_loader, test_loader, ala, run_config, history)
    elif args.mode == "decentralized":
        run_decentralized_client(
            args, model, train_loader, test_loader, ala, test_majority_baseline, history
        )


if __name__ == "__main__":
    main()
