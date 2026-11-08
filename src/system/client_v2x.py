"""Modo descentralizado V2V / V2I D-PSGD Over-the-Air para o cliente FL."""

import argparse
import json
import logging
import math
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from client_data import ClientHistory  # noqa: E402
from client_training import (  # noqa: E402
    extract_trainable_model_state,
    load_trainable_model_state,
    local_training,
)

from utils.fl_math import evaluate_model  # noqa: E402
from utils.model_utils import write_json_atomic  # noqa: E402
from utils.network_utils import (  # noqa: E402
    V2IProtocolError,
    make_v2i_message,
    make_v2v_message,
    recv_data,
    recv_v2i_message,
    send_data,
    send_v2i_message,
    validate_v2v_message,
)

logger = logging.getLogger(__name__)


class P2PWeightStore:
    """Armazena e sincroniza de forma thread-safe os pesos recebidos via TCP P2P (V2V)."""

    def __init__(self, client_idx: int, encounters_dir: Path, default_timeout: float = 600.0):
        self.client_idx = client_idx
        self.encounters_dir = encounters_dir
        self.timeout = default_timeout
        self.weights: dict[int, dict[int, dict]] = {}
        self.bytes_received: dict[int, dict[int, int]] = {}
        self.lock = threading.Lock()
        self._started = False

    def start_if_needed(self) -> None:
        if not self._started:
            threading.Thread(target=self._server_loop, daemon=True).start()
            self._started = True

    def _server_loop(self) -> None:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(("0.0.0.0", 9000))
        server_socket.listen(50)
        logger.info(f"Client {self.client_idx}: P2P Server listening on port 9000")
        while True:
            try:
                conn, _ = server_socket.accept()
                with conn:
                    data, _, received_bytes = recv_data(
                        conn,
                        return_bytes=True,
                        deadline=time.monotonic() + self.timeout,
                        cancelled=lambda: (self.encounters_dir / ".simulation_done").exists(),
                    )
                    if data is not None:
                        validate_v2v_message(data)
                        sender_id, enc_id = data["client_id"], data["encounter_id"]
                        with self.lock:
                            self.weights.setdefault(enc_id, {})[sender_id] = data
                            enc_bytes = self.bytes_received.setdefault(enc_id, {})
                            enc_bytes[sender_id] = enc_bytes.get(sender_id, 0) + received_bytes
                        logger.info(
                            f"Client {self.client_idx}: Recebeu pesos do Cliente {sender_id} "
                            f"para encontro {enc_id}"
                        )
            except Exception as exc:
                logger.error(f"Erro no P2P Server: {exc}")

    def total_bytes_for_encounter(self, enc_id: int) -> int:
        with self.lock:
            return sum(self.bytes_received.get(enc_id, {}).values())

    def clear_encounter(self, enc_id: int) -> None:
        with self.lock:
            self.weights.pop(enc_id, None)
            self.bytes_received.pop(enc_id, None)


# =====================================================================
# COMUNICAÇÃO P2P (V2V / V2I) E UTILITÁRIOS DE ESTADO DE MODELO
# =====================================================================


def send_p2p_model(host: str, payload: dict, *, deadline: float, cancelled=None) -> int:
    """Conecta com tentativas curtas e envia o modelo até o prazo do encontro.

    O limite de conexão não limita o envio de todos os pesos. O transporte
    compartilhado mantém o progresso no socket durante esperas/retransmissões.
    """
    while True:
        if cancelled is not None and cancelled():
            raise ConnectionAbortedError("Simulação encerrada durante o envio V2V.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Prazo de envio V2V para {host} esgotado.")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
                connection.settimeout(min(2.0, remaining))
                connection.connect((host, 9000))
                return send_data(connection, payload, deadline=deadline, cancelled=cancelled)
        except ConnectionAbortedError:
            raise
        except OSError as exc:
            logger.warning("Envio V2V para %s: %s", host, exc)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Prazo de envio V2V para {host} esgotado: {exc}") from exc
            time.sleep(min(0.2, remaining))


def exchange_v2i_model(
    model, encounter: dict, client_id: int, *, train_samples: int, cancelled=None, metrics=None
) -> dict:
    """Troca uma CNN por TCP e confirma APPLIED somente após carregar o resultado.

    Em caso de falha, preserva os pesos locais anteriores ao encontro. O prazo
    abrange conexão, upload, espera, download, aplicação e confirmação.
    """
    host = encounter.get("rsu_host")
    port = encounter.get("rsu_port")
    timeout = encounter.get("exchange_timeout")
    if (
        not isinstance(host, str)
        or not host.strip()
        or type(port) is not int
        or not 1 <= port <= 65535
    ):
        raise V2IProtocolError("Endereço TCP da RSU inválido no sinal de encontro.")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise V2IProtocolError("exchange_timeout deve ser positivo e finito.")
    if client_id not in encounter.get("clients", []):
        raise V2IProtocolError("Cliente ausente dos participantes do encontro.")
    context = {key: encounter.get(key) for key in ("run_id", "rsu_id", "encounter_id")}
    context["client_id"] = client_id
    started = time.monotonic()
    metrics = metrics if metrics is not None else {}
    deadline = started + timeout
    snapshot = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
    upload = make_v2i_message("UPLOAD", **context, weights=snapshot, train_samples=train_samples)
    load_started = False

    def remaining() -> float:
        if cancelled is not None and cancelled():
            raise ConnectionAbortedError("Simulação encerrada durante a troca V2I.")
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("Prazo da troca V2I esgotado.")
        return value

    try:
        with socket.create_connection((host, port), timeout=remaining()) as connection:
            sent = send_v2i_message(connection, upload, deadline=deadline, cancelled=cancelled)
            metrics["upload_bytes"] = sent
            del upload
            response, _, received = recv_v2i_message(
                connection,
                **context,
                deadline=deadline,
                cancelled=cancelled,
            )
            if response["type"] == "ERROR":
                metrics["control_download_bytes"] = received
                raise V2IProtocolError(f"RSU recusou o encontro: {response['error']}")
            if response["type"] != "RESULT":
                raise V2IProtocolError("Esperado RESULT após UPLOAD.")
            metrics["download_bytes"] = received
            metrics["aggregation_time"] = response.get("aggregation_time", 0.0)
            state = response["weights"]
            if state.keys() != snapshot.keys() or any(
                state[key].shape != snapshot[key].shape or state[key].dtype != snapshot[key].dtype
                for key in snapshot
            ):
                raise V2IProtocolError("O resultado da RSU é incompatível com a CNN local.")
            remaining()
            load_started = True
            model.load_state_dict(state, strict=True)
            remaining()
            control = send_v2i_message(
                connection,
                make_v2i_message("APPLIED", **context),
                deadline=deadline,
                cancelled=cancelled,
            )
        metrics.update(applied=True, control_upload_bytes=control)
        return {
            "upload_bytes": sent,
            "download_bytes": received,
            "control_bytes": control,
            "protocol_time": time.monotonic() - started,
        }
    except Exception:
        if load_started:
            # Copiar os tensores também desfaz uma aplicação interrompida por um hook.
            with torch.no_grad():
                for key, tensor in model.state_dict().items():
                    tensor.copy_(snapshot[key])
        raise
    finally:
        metrics["protocol_time"] = time.monotonic() - started


def write_v2i_failure(
    encounters_dir: Path, encounter: dict, client_id: int, error: str, metrics: dict | None = None
) -> None:
    """Publica a falha com os contadores já concluídos, sem emitir .done."""
    data = dict(metrics or {})
    data.update({key: encounter.get(key) for key in ("run_id", "rsu_id", "encounter_id")})
    data.update(client_id=client_id, status="failed", error=error)
    write_json_atomic(
        data, encounters_dir / f".failed_enc_{data['encounter_id']}_client_{client_id}.json"
    )


def wait_for_round_signal(encounters_dir: Path, round_num: int) -> bool:
    """Aguarda o sinal .start_round_<r> ou retorna False se .simulation_done surgir."""
    while True:
        if (encounters_dir / ".simulation_done").exists():
            return False
        if (encounters_dir / f".start_round_{round_num}").exists():
            return True
        time.sleep(1)


def wait_for_encounter(
    encounters_dir: Path,
    client_idx: int,
    last_processed: int,
    current_round: int,
    execution_id: str = "",
) -> dict | str | None:
    """Aguarda um arquivo encounter_<id>.json para este cliente, 'skip' ou fim da simulação."""
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
                    with open(ef) as f:
                        data = json.load(f)
                    if execution_id and data.get("run_id") != execution_id:
                        continue
                    enc_id = data.get("encounter_id", 0)
                    if enc_id <= last_processed:
                        continue
                    if client_idx in data.get("clients", []):
                        return data
                except Exception:
                    pass
        time.sleep(1)


def _build_encounter_metrics(
    args: argparse.Namespace,
    encounter_data: dict,
    encounter_type: str,
    enc_id: int,
    clients_in_encounter: list,
    epoch: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    test_majority_baseline: float | None,
    history: ClientHistory,
) -> dict:
    """Cria o dicionário padronizado de telemetria para um encontro V2V ou V2I."""
    return {
        "schema_version": 1,
        "run_id": encounter_data.get("run_id"),
        "rsu_id": encounter_data.get("rsu_id"),
        "mode": encounter_type,
        "encounter_id": enc_id,
        "client_id": args.client_idx,
        "participant_count": len(clients_in_encounter),
        "epoch": epoch,
        "status": "pending",
        "applied": False,
        "is_aggregator": False,
        "upload_bytes": 0,
        "download_bytes": 0,
        "control_upload_bytes": 0,
        "control_download_bytes": 0,
        "received_upload_bytes": 0,
        "sent_download_bytes": 0,
        "protocol_time": 0.0,
        "aggregation_time": 0.0,
        "evaluation_time": 0.0,
        "accuracy_before": (
            history.local_acc[-1] if history.local_acc and not args.skip_post_eval else None
        ),
        "accuracy_after": None,
        "majority_class_baseline": test_majority_baseline,
        "test_samples": len(test_loader.dataset),
        "train_samples": len(train_loader.dataset),
        "evaluation_skipped": args.skip_post_eval,
        "error": None,
    }


def _handle_v2i_encounter(
    args: argparse.Namespace,
    model: nn.Module,
    encounter_data: dict,
    positions: dict[int, tuple[float, float]],
    enc_id: int,
    epoch: int,
    encounters_dir: Path,
    train_loader: DataLoader,
    test_loader: DataLoader,
    metrics: dict,
    history: ClientHistory,
) -> str | None:
    """Executa a troca de modelo V2I com a RSU, avalia e persiste métricas.

    Retorna None em caso de sucesso ou a mensagem de erro em caso de falha.
    """
    try:
        if args.model != "cnn":
            raise NotImplementedError("A troca V2I por TCP está disponível para CNN.")
        rsu_id = encounter_data["rsu_id"]
        rsu_position = encounter_data["rsu_position"]
        my_position = positions[args.client_idx]
        enc_mean_distance = math.hypot(
            my_position[0] - rsu_position[0],
            my_position[1] - rsu_position[1],
        )
        logger.info(
            "Client %d: [V2I] Encontro %d com %s via %s:%s.",
            args.client_idx,
            enc_id,
            rsu_id,
            encounter_data.get("rsu_host"),
            encounter_data.get("rsu_port"),
        )
        exchange = exchange_v2i_model(
            model,
            encounter_data,
            args.client_idx,
            train_samples=len(train_loader.dataset),
            cancelled=lambda: (encounters_dir / ".simulation_done").exists(),
            metrics=metrics,
        )
        logger.info("Client %d: Resultado da RSU aplicado; APPLIED enviado.", args.client_idx)
        history.p2p_data_sent_mb.append(exchange["upload_bytes"] / (1024 * 1024))
        history.p2p_data_sent_enc_ids.append(enc_id)

        metrics["mean_distance"] = enc_mean_distance
        metrics["status"] = "applied"
        write_json_atomic(
            metrics,
            encounters_dir / f".applied_enc_{enc_id}_client_{args.client_idx}.json",
        )
        # APPLIED é enviado antes da avaliação; .done exige métricas salvas.
        if not args.skip_post_eval:
            evaluation_started = time.monotonic()
            try:
                post_p2p_acc, _ = evaluate_model(model, test_loader)
            finally:
                metrics["evaluation_time"] = time.monotonic() - evaluation_started
            if not math.isfinite(post_p2p_acc):
                raise ValueError("A avaliação pós-V2I retornou uma acurácia inválida.")
            metrics["accuracy_after"] = float(post_p2p_acc)
            history.post_p2p_acc.append(post_p2p_acc)
            history.post_p2p_encounter_ids.append(enc_id)
            logger.info(
                "Client %d: Post-V2I Test Accuracy: %.2f%%",
                args.client_idx,
                post_p2p_acc,
            )
        metrics["status"] = "completed"
        history.ota_encounter_epochs.append(epoch)
        history.ota_mean_distances.append(enc_mean_distance)
        history.ota_is_aggregator.append(0)
        history.save(args)
        write_json_atomic(metrics, encounters_dir / f".done_enc_{enc_id}_client_{args.client_idx}")
        logger.info("Client %d: Encontro V2I concluído; métricas salvas.", args.client_idx)
        return None
    except Exception as exc:
        failure = str(exc) or type(exc).__name__
        logger.error(
            "Client %d: Falha no encontro V2I %d: %s",
            args.client_idx,
            enc_id,
            failure,
        )
        metrics.update(status="failed", error=failure)
        write_v2i_failure(encounters_dir, encounter_data, args.client_idx, failure, metrics)
        return failure


def _handle_v2v_encounter(
    args: argparse.Namespace,
    model: nn.Module,
    encounter_data: dict,
    positions: dict[int, tuple[float, float]],
    clients_in_encounter: list[int],
    etc_seconds,
    poll_timeout: int,
    encounter_started: float,
    enc_id: int,
    epoch: int,
    encounters_dir: Path,
    train_loader: DataLoader,
    test_loader: DataLoader,
    p2p_store: P2PWeightStore,
    metrics: dict,
    history: ClientHistory,
) -> str | None:
    """Executa a troca P2P e agregação OTA de um encontro V2V (Medoid Aggregator vs TX).

    Retorna None em caso de sucesso ou a mensagem de erro em caso de falha.
    """
    from v2v_fl.over_the_air.ota_aggregation import ota_aggregate
    from v2v_fl.over_the_air.ota_channel import distances_from_positions, find_medoid

    try:
        p2p_timeout = encounter_data.get("exchange_timeout", poll_timeout)
        if (
            isinstance(p2p_timeout, bool)
            or not isinstance(p2p_timeout, (int, float))
            or not math.isfinite(p2p_timeout)
            or p2p_timeout <= 0
        ):
            raise ValueError("exchange_timeout V2V deve ser positivo e finito.")
        p2p_store.timeout = p2p_timeout
        exchange_deadline = encounter_started + p2p_timeout
        cancelled = (encounters_dir / ".simulation_done").exists
        p2p_store.start_if_needed()

        aggregator_id = find_medoid(positions)
        is_aggregator = args.client_idx == aggregator_id
        metrics["is_aggregator"] = is_aggregator

        logger.info(
            f"Client {args.client_idx}: [D-PSGD] Encontro {enc_id} detectado! ETC={etc_seconds}s."
        )
        logger.info(
            f"Client {args.client_idx}: Aggregator eleito: {aggregator_id}. "
            f"Sou aggregator? {is_aggregator}"
        )

        _, enc_distances = distances_from_positions(positions, aggregator_id)
        enc_mean_distance = float(np.mean(enc_distances)) if len(enc_distances) > 0 else 0.0

        if args.model == "clip":
            from lora_clip.sora import get_trainable_state_dict

            snapshot = get_trainable_state_dict(model)
        else:
            snapshot = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

        metrics["mean_distance"] = enc_mean_distance
        other_clients = [c for c in clients_in_encounter if c != args.client_idx]
        received_states: list[dict] = []

        # 1. Transmissores enviam seus pesos para o Aggregator e aguardam RESULT
        if not is_aggregator:
            target_host = f"fl-client-v2v-{aggregator_id}"
            snapshot_bytes = send_p2p_model(
                target_host,
                make_v2v_message(
                    "UPLOAD",
                    run_id=encounter_data["run_id"],
                    client_id=args.client_idx,
                    encounter_id=enc_id,
                    weights=snapshot,
                    train_samples=len(train_loader.dataset),
                ),
                deadline=exchange_deadline,
                cancelled=cancelled,
            )
            metrics["upload_bytes"] += snapshot_bytes
            history.p2p_data_sent_mb.append(snapshot_bytes / (1024 * 1024))
            history.p2p_data_sent_enc_ids.append(enc_id)
            logger.info(
                "Client %d: Enviou pesos para %s (%d bytes).",
                args.client_idx,
                target_host,
                snapshot_bytes,
            )
            logger.info(
                f"Client {args.client_idx}: Aguardando resposta do Aggregator {aggregator_id}..."
            )
            while time.monotonic() < exchange_deadline:
                if cancelled():
                    raise ConnectionAbortedError("Simulação encerrada aguardando resultado V2V.")
                with p2p_store.lock:
                    if enc_id in p2p_store.weights and aggregator_id in p2p_store.weights[enc_id]:
                        response = validate_v2v_message(
                            p2p_store.weights[enc_id][aggregator_id],
                            type="RESULT",
                            run_id=encounter_data["run_id"],
                            encounter_id=enc_id,
                            client_id=aggregator_id,
                        )
                        received_states.append(response)
                        break
                time.sleep(1)

        # 2. Aggregator aguarda UPLOAD de todos os transmissores
        if is_aggregator:
            logger.info(
                f"Client {args.client_idx}: Aguardando receber pesos dos transmissores "
                f"{other_clients}..."
            )
            while time.monotonic() < exchange_deadline:
                if cancelled():
                    raise ConnectionAbortedError("Simulação encerrada aguardando uploads V2V.")
                with p2p_store.lock:
                    if enc_id in p2p_store.weights:
                        missing = [c for c in other_clients if c not in p2p_store.weights[enc_id]]
                        if not missing:
                            for c in other_clients:
                                upload = validate_v2v_message(
                                    p2p_store.weights[enc_id][c],
                                    type="UPLOAD",
                                    run_id=encounter_data["run_id"],
                                    encounter_id=enc_id,
                                    client_id=c,
                                )
                                received_states.append(upload)
                            break
                time.sleep(1)

        received_bytes = p2p_store.total_bytes_for_encounter(enc_id)
        metrics["received_upload_bytes" if is_aggregator else "download_bytes"] = received_bytes

        # 3. Consolidação OTA (No Aggregator) ou Atualização (Nos Transmissores)
        expected_received = (is_aggregator and len(received_states) == len(other_clients)) or (
            not is_aggregator and len(received_states) == 1
        )
        if not expected_received:
            raise TimeoutError(f"Pesos V2V incompletos: {len(received_states)} recebidos.")

        if is_aggregator:
            logger.info(
                f"Client {args.client_idx}: Todos os pesos recebidos! Iniciando agregação OTA..."
            )
            aggregation_started = time.monotonic()
            try:
                participant_states = {
                    upload["client_id"]: upload["weights"] for upload in received_states
                }
                participant_states[args.client_idx] = snapshot
                train_samples = {
                    upload["client_id"]: upload["train_samples"] for upload in received_states
                }
                train_samples[args.client_idx] = len(train_loader.dataset)
                diagnostics: dict = {}
                snapshot = ota_aggregate(
                    participant_states,
                    positions,
                    {
                        "x": positions[aggregator_id][0],
                        "y": positions[aggregator_id][1],
                        "n_antennas": 8,
                    },
                    train_samples=train_samples,
                    local_client_id=aggregator_id,
                    diagnostics=diagnostics,
                )
                metrics["ota_diagnostics"] = diagnostics
            finally:
                metrics["aggregation_time"] = time.monotonic() - aggregation_started

            load_trainable_model_state(model, snapshot, args.model)
        else:
            logger.info(f"Client {args.client_idx}: Pesos consolidados recebidos do Aggregator!")
            aggregated_state = received_states[0]["weights"]
            if args.model in ["clip", "slm"]:
                for k in snapshot:
                    snapshot[k].copy_(aggregated_state[k].to(snapshot[k].dtype))
            else:
                for k in snapshot:
                    snapshot[k] = aggregated_state[k].to(snapshot[k].dtype)
            load_trainable_model_state(model, snapshot, args.model)

        metrics["applied"] = True

        # 4. Aggregator envia o resultado de volta para os transmissores
        if is_aggregator:
            for c in other_clients:
                target_host = f"fl-client-v2v-{c}"
                snapshot_bytes = send_p2p_model(
                    target_host,
                    make_v2v_message(
                        "RESULT",
                        run_id=encounter_data["run_id"],
                        client_id=args.client_idx,
                        encounter_id=enc_id,
                        weights=snapshot,
                    ),
                    deadline=exchange_deadline,
                    cancelled=cancelled,
                )
                metrics["sent_download_bytes"] += snapshot_bytes
                history.p2p_data_sent_mb.append(snapshot_bytes / (1024 * 1024))
                history.p2p_data_sent_enc_ids.append(enc_id)
                logger.info(
                    "Client %d: Retornou pesos consolidados para %s (%d bytes).",
                    args.client_idx,
                    target_host,
                    snapshot_bytes,
                )

        metrics.update(
            applied=True,
            status="applied",
            protocol_time=time.monotonic() - encounter_started,
        )
        write_json_atomic(
            metrics,
            encounters_dir / f".applied_enc_{enc_id}_client_{args.client_idx}.json",
        )
        p2p_store.clear_encounter(enc_id)

        # 5. Avaliação Pós-Agregação
        if not args.skip_post_eval:
            evaluation_started = time.monotonic()
            try:
                post_p2p_acc, _ = evaluate_model(model, test_loader)
            finally:
                metrics["evaluation_time"] = time.monotonic() - evaluation_started
            if not math.isfinite(post_p2p_acc):
                raise ValueError("A avaliação pós-V2V retornou uma acurácia inválida.")
            metrics["accuracy_after"] = float(post_p2p_acc)
            logger.info(
                f"Client {args.client_idx}: Post-D-PSGD Test Accuracy: {post_p2p_acc:.2f}%"
            )
            history.post_p2p_acc.append(post_p2p_acc)
            history.post_p2p_encounter_ids.append(enc_id)

        metrics["status"] = "completed"
        history.ota_encounter_epochs.append(epoch)
        history.ota_mean_distances.append(enc_mean_distance)
        history.ota_is_aggregator.append(1 if is_aggregator else 0)
        history.save(args)

        write_json_atomic(metrics, encounters_dir / f".done_enc_{enc_id}_client_{args.client_idx}")
        logger.info(f"Client {args.client_idx}: Agregacao concluida. .done_enc enviado.")
        return None
    except Exception as exc:
        failure = str(exc) or type(exc).__name__
        received_bytes = p2p_store.total_bytes_for_encounter(enc_id)
        byte_key = "received_upload_bytes" if metrics["is_aggregator"] else "download_bytes"
        metrics[byte_key] = max(metrics[byte_key], received_bytes)
        if metrics["status"] == "pending":
            metrics["protocol_time"] = time.monotonic() - encounter_started
        metrics.update(status="failed", error=failure)
        write_v2i_failure(encounters_dir, encounter_data, args.client_idx, failure, metrics)
        logger.error("Falha no encontro V2V %d: %s", enc_id, failure)
        return failure


def run_decentralized_client(
    args: argparse.Namespace,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    ala,
    test_majority_baseline: float | None,
    history: ClientHistory,
) -> None:
    """Executa o loop D-PSGD descentralizado (V2V e V2I) sincronizado com o SUMO."""
    logger.info(f"Client {args.client_idx}: Modo Descentralizado (D-PSGD) Ativado.")
    encounters_dir = (
        args.encounters_dir or Path(__file__).resolve().parents[1] / "results" / "encounters"
    )
    p2p_store = P2PWeightStore(args.client_idx, encounters_dir)
    encounter_failure = None
    epoch = 1
    last_processed_encounter = 0
    max_epochs = args.max_epochs if args.max_epochs > 0 else float("inf")

    while epoch <= max_epochs:
        logger.info(f"Client {args.client_idx}: Aguardando inicio da Rodada {epoch - 1}...")
        if not wait_for_round_signal(encounters_dir, epoch - 1):
            logger.info(
                f"Client {args.client_idx}: Sinal de fim de simulacao (.simulation_done) recebido!"
            )
            break

        max_ep_str = f"/{args.max_epochs}" if args.max_epochs > 0 else ""
        logger.info(f"\n--- Iniciando Epoca {epoch}{max_ep_str} (Rodada {epoch - 1}) ---")

        dummy_state = extract_trainable_model_state(model, args.model)
        local_training(
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
            history.local_acc.append(local_test_acc)
            if not args.skip_train_eval:
                train_accuracy, _ = evaluate_model(model, train_loader)
                history.train_acc.append(train_accuracy)
            else:
                history.train_acc.append(0.0)
            logger.info(
                f"Client {args.client_idx}: Epoca {epoch} Local Test Acc: {local_test_acc:.2f}%"
            )

        history.save(args)

        # ── FASE 4: Sinalizar conclusão da época ────────────────────────
        (encounters_dir / f".epoch_done_client_{args.client_idx}").touch()
        logger.info(f"Client {args.client_idx}: Epoca concluida. Aguardando encontro P2P...")

        # ── FASE 5: Aguardar e Processar Encontro ───────────────────────
        encounter_data = wait_for_encounter(
            encounters_dir,
            args.client_idx,
            last_processed_encounter,
            current_round=epoch,
            execution_id=args.execution_id,
        )

        if encounter_data is None:
            logger.info(
                f"Client {args.client_idx}: Sinal de fim de simulacao recebido. Encerrando."
            )
            break

        if encounter_data == "skip":
            logger.info(
                f"Client {args.client_idx}: Encontro nao ocorreu para mim nesta rodada. "
                "Avancando para prox epoca."
            )
            epoch += 1
            continue

        enc_id = encounter_data.get("encounter_id", 0)
        clients_in_encounter = encounter_data.get("clients", [])
        etc_seconds = encounter_data.get("etc_seconds", 60)
        encounter_type = encounter_data.get("type", "v2v")
        positions = {int(k): tuple(v) for k, v in encounter_data.get("positions", {}).items()}

        if isinstance(etc_seconds, str) and etc_seconds == "Carros Parados":
            poll_timeout = 115
        else:
            poll_timeout = max(5, int(etc_seconds - 5))

        encounter_started = time.monotonic()
        metrics = _build_encounter_metrics(
            args,
            encounter_data,
            encounter_type,
            enc_id,
            clients_in_encounter,
            epoch,
            train_loader,
            test_loader,
            test_majority_baseline,
            history,
        )
        args.encounter_metrics.append(metrics)

        if encounter_type == "v2i":
            encounter_failure = _handle_v2i_encounter(
                args,
                model,
                encounter_data,
                positions,
                enc_id,
                epoch,
                encounters_dir,
                train_loader,
                test_loader,
                metrics,
                history,
            )
        else:
            encounter_failure = _handle_v2v_encounter(
                args,
                model,
                encounter_data,
                positions,
                clients_in_encounter,
                etc_seconds,
                poll_timeout,
                encounter_started,
                enc_id,
                epoch,
                encounters_dir,
                train_loader,
                test_loader,
                p2p_store,
                metrics,
                history,
            )

        if encounter_failure is not None:
            break

        last_processed_encounter = enc_id
        epoch += 1

    logger.info(f"Client {args.client_idx}: Loop descentralizado finalizado.")
    history.save(args)
    if encounter_failure is not None:
        raise RuntimeError(
            f"Falha na execução descentralizada do cliente {args.client_idx}: {encounter_failure}"
        )
    logger.info("\nTraining completed!")
