import argparse
import copy
import hashlib
import json
import logging
import math
import os
import random
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# Desativa checagem online redundante do Hugging Face Hub (usa cache local instantaneamente)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
import torchvision.transforms as T  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

# --- HACK PERMITIDO: Inserindo a pasta 'src' no path do Python ---
# Nota: Esta é uma mitigação aceita enquanto o projeto não for
# completamente empacotado como biblioteca via pyproject.toml
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))
# -----------------------------------------------------------------

from ALA import ALA  # noqa: E402
from ALA_LoRA import ALA_LoRA  # noqa: E402
from lora_clip.clip_setup import (  # noqa: E402
    build_model,
    build_run_config,
    load_config,
    resolve_run_modes,
)
from lora_slm.qlora import (  # noqa: E402
    enable_qlora_in_config,
    get_qlora_state_dict,
    is_qlora_enabled,
    set_qlora_state_dict,
)
from model import SimpleModel  # noqa: E402

from utils.data_utils import read_client_data  # noqa: E402
from utils.fl_math import (  # noqa: E402
    dequantization,
    evaluate_model,
    quantization,
    resize_model_to_pruned,
    set_parameters,
)
from utils.lhdq import is_lhdq_encoded, lhdq_decode, lhdq_encode  # noqa: E402
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

# --- CONFIGURAÇÃO DE OBSERVABILIDADE (LOGGING ENXUTO) ---
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
# --------------------------------------------------------


@dataclass
class ClientHistory:
    """Acumula o histórico de métricas por rodada e por encontro do cliente FL."""

    global_acc: list[float] = field(default_factory=list)
    global_loss: list[float] = field(default_factory=list)
    ala_acc: list[float] = field(default_factory=list)
    local_acc: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    post_p2p_acc: list[float] = field(default_factory=list)
    post_p2p_encounter_ids: list[int] = field(default_factory=list)
    p2p_data_sent_mb: list[float] = field(default_factory=list)
    p2p_data_sent_enc_ids: list[int] = field(default_factory=list)
    ota_encounter_epochs: list[int] = field(default_factory=list)
    ota_mean_distances: list[float] = field(default_factory=list)
    ota_is_aggregator: list[int] = field(default_factory=list)

    def save(self, args: argparse.Namespace) -> None:
        """Persiste todas as séries históricas do cliente no arquivo HDF5."""
        save_results(
            args,
            self.global_acc,
            self.global_loss,
            self.ala_acc,
            self.local_acc,
            self.train_acc,
            idx=args.client_idx,
            argalgo=args.ala,
            rs_post_p2p_acc=self.post_p2p_acc,
            rs_post_p2p_encounter_ids=self.post_p2p_encounter_ids,
            rs_p2p_data_sent_mb=self.p2p_data_sent_mb,
            rs_p2p_data_sent_enc_ids=self.p2p_data_sent_enc_ids,
            rs_ota_encounter_epochs=self.ota_encounter_epochs,
            rs_ota_mean_distances=self.ota_mean_distances,
            rs_ota_is_aggregator=self.ota_is_aggregator,
        )


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


def map_sequential_to_simplemodel(state_dict: dict) -> dict:
    """Mapeia chaves de nn.Sequential (pós-poda OPALA) de volta para SimpleModel."""
    mapped_dict = {}
    mapping = {
        "0.weight": "conv1.0.weight",
        "0.bias": "conv1.0.bias",
        "3.weight": "conv2.0.weight",
        "3.bias": "conv2.0.bias",
        "7.weight": "fc1.0.weight",
        "7.bias": "fc1.0.bias",
        "9.weight": "fc.weight",
        "9.bias": "fc.bias",
    }
    for sequential_key, simple_key in mapping.items():
        if sequential_key in state_dict:
            mapped_dict[simple_key] = state_dict[sequential_key]
    return mapped_dict


def _extract_trainable_model_state(
    model: nn.Module, model_type: str, run_config: dict | None = None
) -> dict:
    """Extrai o dicionário de pesos treináveis de acordo com a arquitetura e modo QLoRA."""
    if model_type in ("clip", "slm"):
        if run_config is not None and is_qlora_enabled(run_config):
            return get_qlora_state_dict(model)
        if model_type == "clip":
            from lora_clip.sora import get_trainable_state_dict

            return get_trainable_state_dict(model)
        from lora_slm.sora import get_trainable_state_dict as get_trainable_state_dict_slm

        return get_trainable_state_dict_slm(model)
    return model.state_dict()


def _load_trainable_model_state(model: nn.Module, state_dict: dict, model_type: str) -> None:
    """Carrega pesos agregados no modelo local (suporta CNN, CLIP e SLM)."""
    if model_type == "clip":
        from lora_clip.sora import set_trainable_state_dict

        set_trainable_state_dict(model, state_dict)
    elif model_type == "slm":
        from lora_slm.sora import set_trainable_state_dict as set_trainable_state_dict_slm

        set_trainable_state_dict_slm(model, state_dict)
    else:
        model.load_state_dict(state_dict)


# =====================================================================
# TREINAMENTO LOCAL, FEDALA E CARREGAMENTO DE DADOS
# =====================================================================


def local_initialization(ala, received_global_model, model, mask=None):
    ala.adaptive_local_aggregation(received_global_model, model, mask=mask)


def _log_trainable_param_count(model: nn.Module) -> None:
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if getattr(model, "_last_trainable_count", None) != trainable_count:
        logger.info(f"[DIAG] Parâmetros treináveis neste round: {trainable_count:,}")
        model._last_trainable_count = trainable_count


def local_training(
    model,
    state_dict,
    prune,
    train_loader,
    test_loader,
    learning_rate=0.01,
    round_num=2,
    alaarg=1,
    ala=None,
    model_type="cnn",
    run_config=None,
):
    # Verifica se os pesos precisam ser desquantizados (se não forem tensores)
    if state_dict and not isinstance(next(iter(state_dict.values())), torch.Tensor):
        state_dict = dequantization(state_dict)

    personalized_acc = None

    if round_num >= 2 and prune == 1 and model_type == "cnn":
        state_dict = map_sequential_to_simplemodel(state_dict)

    # === SINCRONIZAÇÃO DE PESOS GLOBAIS E FEDALA ===
    if model_type in ["clip", "slm"]:
        # FedALA_LoRA só se aplica a CLIP (model_type != 'slm').
        # Evita deepcopy desnecessário do SLM inteiro (~4-6 GB).
        needs_ala = alaarg == 0 and round_num >= 2 and model_type != "slm"

        if needs_ala:
            global_model = copy.deepcopy(model)
            global_model = resize_model_to_pruned(global_model, state_dict)
            logger.info(f"Client {ala.cid}: Applying FedALA_LoRA (Adaptive Local Aggregation)...")
            local_initialization(ala, global_model, model)
            personalized_acc, _ = evaluate_model(model, test_loader)
            logger.info(f"Client {ala.cid}: Post-ALA Test Accuracy: {personalized_acc:.2f}%")
            del global_model
        elif is_qlora_enabled(run_config):
            # QLoRA: injeta apenas os adaptadores LoRA + head (set_peft_model_state_dict)
            set_qlora_state_dict(model, state_dict)
        else:
            resize_model_to_pruned(model, state_dict)
    else:
        # --- LÓGICA PARA CNN ---
        local_model = resize_model_to_pruned(model, state_dict)
        state = copy.deepcopy(local_model)

        if alaarg == 0 and round_num == 2:
            logger.info(f"Client {ala.cid}: Applying FedALA (Adaptive Local Aggregation)...")
            local_initialization(ala, state, model)

        set_parameters(model, state)

        if alaarg == 0 and round_num >= 2:
            personalized_acc, _ = evaluate_model(model, test_loader)
            logger.info(f"Client {ala.cid}: Post-ALA Test Accuracy: {personalized_acc:.2f}%")

    model.train()
    loss_fn = nn.CrossEntropyLoss()
    device = next(model.parameters()).device

    # === 1. SELEÇÃO DE OTIMIZADOR E LOOP DE ÉPOCA ===
    if model_type in ["clip", "slm"]:
        if model_type == "clip":
            from lora_clip.clip_setup import build_optimizer, train_epoch, update_paca_runtime
        else:
            from lora_slm.slm_setup import build_optimizer, train_epoch, update_paca_runtime

        paca_config = run_config["model"].get("paca", {})
        if paca_config.get("enabled"):
            update_paca_runtime(model, paca_config.get("upper_layers"))

        _log_trainable_param_count(model)

        # VRAM OPTIMIZATION: Habilita Gradient Checkpointing para SLM
        if (
            model_type == "slm"
            and hasattr(model, "slm_model")
            and hasattr(model.slm_model, "gradient_checkpointing_enable")
        ):
            model.slm_model.gradient_checkpointing_enable()

        optimizer, sparse_optimizer = build_optimizer(model, run_config)
        is_sora = run_config["model"]["lora"]["mode"] in [
            "with_sora_no_schedule",
            "with_sora_schedule",
        ]
        sparse_lambda = (
            run_config["model"].get("sora", {}).get("sparse_lambda", 0.0) if is_sora else 0.0
        )

        if model_type == "slm":
            accumulation_steps = run_config.get("training", {}).get(
                "gradient_accumulation_steps", 1
            )
            metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                sparse_optimizer,
                sparse_lambda,
                accumulation_steps,
            )
        else:
            metrics = train_epoch(model, train_loader, optimizer, sparse_optimizer, sparse_lambda)

        logger.info(
            f"Métricas do Treino Local - CE Loss: {metrics['ce_loss']:.4f} | "
            f"Sparse Loss: {metrics['sparse_loss']:.4f}"
        )
    else:
        _log_trainable_param_count(model)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        model_dtype = next(model.parameters()).dtype

        for x, y in train_loader:
            x = x.to(device, dtype=model_dtype) if x.is_floating_point() else x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(x)
            if isinstance(output, dict):
                output = output["logits"]

            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

    # === 2. SELEÇÃO DOS PESOS QUE SERÃO ENVIADOS PARA O SERVIDOR ===
    return _extract_trainable_model_state(model, model_type, run_config), personalized_acc


def _tensor_to_pil_rgb(x_tensor: torch.Tensor) -> Image.Image:
    """Reverte normalização (CLIP ou [-1, 1]) e converte tensor CHW para imagem PIL RGB."""
    x_tensor = x_tensor.cpu().float()

    # Se o tensor vier com normalização do CLIP (valores negativos < -0.1)
    if x_tensor.min() < -0.1:
        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1)
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1)
        if x_tensor.shape[0] == 3:
            x_tensor = x_tensor * clip_std + clip_mean
        else:
            x_tensor = x_tensor * 0.5 + 0.5
    elif x_tensor.min() < 0:
        x_tensor = x_tensor * 0.5 + 0.5

    if x_tensor.max() <= 1.0:
        x_tensor = x_tensor * 255.0

    x_numpy = torch.clamp(x_tensor, 0, 255).byte().numpy()
    if x_numpy.shape[0] in [1, 3]:  # (C, H, W) -> (H, W, C)
        x_numpy = np.transpose(x_numpy, (1, 2, 0))
    if x_numpy.shape[-1] == 1:
        x_numpy = np.squeeze(x_numpy, axis=-1)

    pil_img = Image.fromarray(x_numpy)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return pil_img


def load_data(
    dataset,
    client_idx,
    device,
    is_train=True,
    batch_size=32,
    is_clip=False,
    is_slm=False,
    slm_config=None,
    dataset_dir=None,
):
    train_data = read_client_data(dataset, client_idx, is_train, dataset_dir=dataset_dir)
    X, y = zip(*train_data, strict=False)
    X = torch.stack(X)

    if is_slm:
        from lora_slm.slm_setup import CustomCollator
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(slm_config["model"]["name"])
        label_to_idx = {i: i for i in range(1000)}
        label_to_idx.update({str(i): i for i in range(1000)})
        collate_fn = CustomCollator(processor, label_to_idx)

        slm_data = [
            {"image": _tensor_to_pil_rgb(x_tensor), "label": y_tensor.item()}
            for x_tensor, y_tensor in zip(X, y, strict=False)
        ]
        pin_mem = hasattr(device, "type") and device.type == "cuda"
        return DataLoader(
            slm_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=pin_mem,
        )

    # Redimensionamento e normalização vitais para o CLIP
    if is_clip:
        if X.shape[-2:] != (224, 224):
            resize = T.Resize((224, 224), antialias=True)
            X = resize(X)
        if not X.is_floating_point():
            X = X.float() / 255.0
        elif X.min() >= 0.0 and X.max() > 1.0:
            X = X / 255.0
        if X.min() >= 0.0:
            normalize = T.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            )
            X = normalize(X)

    X = X.to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)

    tensor_dataset = torch.utils.data.TensorDataset(X, y)
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)


# =====================================================================
# PERSISTÊNCIA DE RESULTADOS (HDF5)
# =====================================================================


def save_results(
    args,
    rs_global_acc,
    rs_global_loss,
    rs_ala_acc,
    rs_local_acc,
    rs_train_acc,
    idx=0,
    argalgo=0,
    rs_post_p2p_acc=None,
    rs_post_p2p_encounter_ids=None,
    rs_p2p_data_sent_mb=None,
    rs_p2p_data_sent_enc_ids=None,
    rs_ota_encounter_epochs=None,
    rs_ota_mean_distances=None,
    rs_ota_is_aggregator=None,
):
    b = "FedALA" if argalgo == 0 else "FedAVG"

    paca_val = (
        args.paca
        if (
            isinstance(args.paca, (int, float))
            and not isinstance(args.paca, bool)
            and args.paca > 0
        )
        else 0
    )
    rank_val = (
        args.rank
        if (
            isinstance(args.rank, (int, float))
            and not isinstance(args.rank, bool)
            and args.rank > 0
        )
        else 8
    )
    ts_suffix = f"_{args.timestamp}" if hasattr(args, "timestamp") and args.timestamp else ""
    algo = (
        f"client_{idx}_{args.dataset}_{args.strategy}_rank{rank_val}_"
        f"paca{paca_val}_{b}_run{args.run_id}{ts_suffix}"
    )

    result_path = (
        getattr(args, "results_dir", None) or current_dir.parent / "results" / args.exp_name
    )
    result_path.mkdir(parents=True, exist_ok=True)

    file_path = result_path / f"{algo}.h5"
    logger.info(f"Salvando resultados em: {file_path}")

    with h5py.File(file_path, "w") as hf:
        hf.create_dataset("rs_global_acc", data=rs_global_acc)
        hf.create_dataset("rs_global_loss", data=rs_global_loss)
        if rs_ala_acc is not None and len(rs_ala_acc) > 0:
            hf.create_dataset("rs_ala_acc", data=rs_ala_acc)
        hf.create_dataset("rs_local_acc", data=rs_local_acc)
        hf.create_dataset("rs_train_acc", data=rs_train_acc)
        if rs_post_p2p_acc is not None and len(rs_post_p2p_acc) > 0:
            hf.create_dataset("rs_post_p2p_acc", data=rs_post_p2p_acc)
        if rs_post_p2p_encounter_ids is not None and len(rs_post_p2p_encounter_ids) > 0:
            hf.create_dataset("rs_post_p2p_encounter_ids", data=rs_post_p2p_encounter_ids)
        if rs_p2p_data_sent_mb is not None and len(rs_p2p_data_sent_mb) > 0:
            hf.create_dataset("rs_p2p_data_sent_mb", data=rs_p2p_data_sent_mb)
            hf["rs_p2p_data_sent_mb"].attrs["unit"] = "MiB (2**20 bytes)"
        if rs_p2p_data_sent_enc_ids is not None and len(rs_p2p_data_sent_enc_ids) > 0:
            hf.create_dataset("rs_p2p_data_sent_enc_ids", data=rs_p2p_data_sent_enc_ids)
        if rs_ota_encounter_epochs is not None and len(rs_ota_encounter_epochs) > 0:
            hf.create_dataset("rs_ota_encounter_epochs", data=rs_ota_encounter_epochs)
        if rs_ota_mean_distances is not None and len(rs_ota_mean_distances) > 0:
            hf.create_dataset("rs_ota_mean_distances", data=rs_ota_mean_distances)
        if rs_ota_is_aggregator is not None and len(rs_ota_is_aggregator) > 0:
            hf.create_dataset("rs_ota_is_aggregator", data=rs_ota_is_aggregator)
        hf.attrs["paca_used"] = paca_val
        hf.attrs["execution_id"] = getattr(args, "execution_id", "")
        hf.attrs["seed"] = getattr(args, "seed", 42)
        if getattr(args, "dataset_dir", None) is not None:
            hf.attrs["dataset_dir"] = str(args.dataset_dir)
        if getattr(args, "initial_model_sha256", None) is not None:
            hf.attrs["initial_model_sha256"] = args.initial_model_sha256
            hf.attrs["initialization_seed"] = args.seed
            hf.attrs["training_seed"] = args.training_seed
        records = getattr(args, "encounter_metrics", [])
        if records:
            group = hf.create_group("encounters")
            group.attrs["bytes_unit"] = "bytes of completed application frames"
            group.attrs["time_unit"] = "seconds"
            group.attrs["accuracy_unit"] = "percent"
            group.create_dataset(
                "records",
                data=[json.dumps(r, allow_nan=False) for r in records],
                dtype=h5py.string_dtype("utf-8"),
            )
            for key in (
                "encounter_id",
                "participant_count",
                "upload_bytes",
                "download_bytes",
                "control_upload_bytes",
                "control_download_bytes",
                "aggregation_time",
                "protocol_time",
                "evaluation_time",
                "accuracy_before",
                "accuracy_after",
                "majority_class_baseline",
                "test_samples",
                "epoch",
                "mean_distance",
                "is_aggregator",
            ):
                group.create_dataset(
                    key, data=[r.get(key) if r.get(key) is not None else np.nan for r in records]
                )


# =====================================================================
# CONFIGURAÇÃO, CONSTRUÇÃO DE MODELO E RESOLUÇÃO DE AMBIENTE
# =====================================================================


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


def resolve_device(args: argparse.Namespace) -> torch.device:
    """Resolve o acelerador disponível (MPS > CUDA > CPU) e configura CUDA_VISIBLE_DEVICES."""
    if args.device == "mps":
        if torch.backends.mps.is_available():
            logger.info("Using Metal Performance Shaders (MPS) for MacBook GPU acceleration.")
        else:
            logger.warning(
                "MPS is not available. Falling back to CUDA if available, otherwise CPU."
            )
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
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
    return torch.device(args.device)


def resolve_heterogeneous_paca(args: argparse.Namespace) -> None:
    """Ajusta args.paca in-place quando configurado via lista determinística ou sorteio."""
    if args.paca_list is not None:
        paca_values = [int(x.strip()) for x in args.paca_list.split(",")]
        args.paca = paca_values[args.client_idx % len(paca_values)]
        logger.info(
            f"PaCA Heterogêneo (lista): Cliente {args.client_idx} usando PaCA = {args.paca}"
        )
    elif args.random_paca:
        rng = random.Random(args.client_idx * 1000 + args.run_id)
        args.paca = rng.randint(args.paca_min, args.paca_max)
        logger.info(
            f"PaCA Heterogêneo (aleatório): Cliente {args.client_idx} sorteou PaCA = {args.paca} "
            f"(intervalo [{args.paca_min}, {args.paca_max}])"
        )


def _compute_model_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode("utf-8"))
        flat_tensor = tensor.detach().cpu().contiguous().reshape(-1)
        digest.update(flat_tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_client_model(
    args: argparse.Namespace, device: torch.device
) -> tuple[nn.Module, dict | None]:
    """Instancia o modelo (CNN, CLIP ou SLM) e retorna (model, run_config)."""
    run_config = None
    match args.model:
        case "cnn":
            with torch.random.fork_rng(devices=[], enabled=args.mode == "decentralized"):
                if args.mode == "decentralized":
                    torch.random.default_generator.manual_seed(args.seed)
                match args.dataset:
                    case "MNIST":
                        model = SimpleModel(in_features=1, num_classes=10, dim=1024)
                    case "Cifar10":
                        model = SimpleModel(
                            in_features=args.in_features, num_classes=10, dim=args.dim
                        )
                    case "Cifar100":
                        model = SimpleModel(
                            in_features=args.in_features,
                            num_classes=args.num_classes,
                            dim=args.dim,
                        )
                    case _:
                        logger.error(f"Dataset inválido: {args.dataset}")
                        sys.exit(1)
            if args.mode == "decentralized":
                args.initial_model_sha256 = _compute_model_sha256(model)
                logger.info(
                    "Client %d: CNN inicial comum | seed=%d | treino=%d | SHA256=%s",
                    args.client_idx,
                    args.seed,
                    args.training_seed,
                    args.initial_model_sha256,
                )
        case "clip" | "slm":
            if args.model == "clip":
                config = load_config(args.config)
            else:
                from lora_slm.slm_setup import load_config as load_config_slm

                config = load_config_slm(args.config)

            strategy_to_mode = {
                "lora": "with_lora",
                "adalora": "with_adalora",
                "sora_with_schedule": "with_sora_schedule",
                "sora_no_schedule": "with_sora_no_schedule",
            }
            if args.strategy in strategy_to_mode:
                config["model"]["lora"]["mode"] = strategy_to_mode[args.strategy]

            if args.qlora:
                if args.model != "slm" or args.mode != "centralized":
                    logger.error(
                        "--qlora é suportado apenas com --model slm no modo centralizado."
                    )
                    sys.exit(1)
                enable_qlora_in_config(config, args.strategy)
                logger.info(
                    "QLoRA ATIVADO: backbone 4-bit NF4 + DoubleQuant, adaptadores LoRA 16 bits."
                )

            config["model"].setdefault("paca", {})
            if "lora" in config["model"]:
                config["model"]["lora"]["r"] = args.rank

            if args.paca is not None and args.paca > 0:
                config["model"]["paca"]["enabled"] = True
                config["model"]["paca"]["upper_layers"] = args.paca
            else:
                config["model"]["paca"]["enabled"] = False
                config["model"]["paca"]["upper_layers"] = None

            if args.model == "clip":
                run_mode = resolve_run_modes(config)[0]
                run_config = build_run_config(config, run_mode=run_mode)
                model = build_model(config=run_config, num_classes=args.num_classes, device=device)
            else:
                from lora_slm.slm_setup import (
                    build_model as build_model_slm,
                )
                from lora_slm.slm_setup import (
                    build_run_config as build_run_config_slm,
                )
                from lora_slm.slm_setup import (
                    resolve_run_modes as resolve_run_modes_slm,
                )

                run_mode = resolve_run_modes_slm(config)[0]
                run_config = build_run_config_slm(config, run_mode=run_mode)
                model = build_model_slm(
                    config=run_config, num_classes=args.num_classes, device=device
                )
        case _:
            logger.error(f"Modelo inválido: {args.model}")
            sys.exit(1)

    if not args.qlora:  # QLoRA: backbone 4-bit já alocado via device_map
        model = model.to(device)
    return model, run_config


def build_client_dataloaders(
    args: argparse.Namespace, device: torch.device, run_config: dict | None
) -> tuple[DataLoader, DataLoader, float | None]:
    """Carrega train_loader, test_loader e calcula o baseline da classe majoritária."""
    try:
        is_clip_flag = args.model == "clip"
        is_slm_flag = args.model == "slm"
        slm_cfg = run_config if is_slm_flag else None
        train_loader = load_data(
            args.dataset,
            args.client_idx,
            device,
            is_train=True,
            batch_size=args.batch_size,
            is_clip=is_clip_flag,
            is_slm=is_slm_flag,
            slm_config=slm_cfg,
            dataset_dir=args.dataset_dir,
        )
        test_loader = load_data(
            args.dataset,
            args.client_idx,
            device,
            is_train=False,
            batch_size=args.batch_size,
            is_clip=is_clip_flag,
            is_slm=is_slm_flag,
            slm_config=slm_cfg,
            dataset_dir=args.dataset_dir,
        )
        test_majority_baseline = None
        if hasattr(test_loader.dataset, "tensors"):
            labels = test_loader.dataset.tensors[1]
            counts = torch.unique(labels, return_counts=True)[1]
            test_majority_baseline = 100.0 * counts.max().item() / len(labels)
        logger.info(
            f"Data loaded successfully - Train batches: {len(train_loader)}, "
            f"Test batches: {len(test_loader)}"
        )
        return train_loader, test_loader, test_majority_baseline
    except Exception:
        logger.exception("Error loading data")
        sys.exit(1)


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
# MODO DESCENTRALIZADO (V2V / V2I D-PSGD OVER-THE-AIR)
# =====================================================================


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

            _load_trainable_model_state(model, snapshot, args.model)
        else:
            logger.info(f"Client {args.client_idx}: Pesos consolidados recebidos do Aggregator!")
            aggregated_state = received_states[0]["weights"]
            if args.model in ["clip", "slm"]:
                for k in snapshot:
                    snapshot[k].copy_(aggregated_state[k].to(snapshot[k].dtype))
            else:
                for k in snapshot:
                    snapshot[k] = aggregated_state[k].to(snapshot[k].dtype)
            _load_trainable_model_state(model, snapshot, args.model)

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

        dummy_state = _extract_trainable_model_state(model, args.model)
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
