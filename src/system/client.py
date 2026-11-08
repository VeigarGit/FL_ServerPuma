
import socket
import copy
import time
import os
import sys
import argparse
import logging
import random
import math
import json
import hashlib
from pathlib import Path

# Desativa checagem online redundante do Hugging Face Hub (usa cache local instantaneamente)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import h5py

from lora_clip.clip_setup import (
    load_config, resolve_run_modes, build_run_config, build_model
)

from lora_clip.clip_setup import build_optimizer
from lora_clip.sora import get_trainable_state_dict
from lora_slm.qlora import (
    is_qlora_enabled, enable_qlora_in_config, get_qlora_state_dict, set_qlora_state_dict
)
# --- HACK PERMITIDO: Inserindo a pasta 'src' no path do Python ---
# Nota: Esta é uma mitigação aceita enquanto o projeto não for
# completamente empacotado como biblioteca via pyproject.toml
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))
# -----------------------------------------------------------------

from utils.model_utils import write_json_atomic  # noqa: E402
from utils.data_utils import read_client_data
from prunning import prune_and_restructure
from ALA import ALA
from ALA_LoRA import ALA_LoRA
from model import SimpleModel
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
from utils.fl_math import quantization, dequantization, evaluate_model, set_parameters
from utils.fl_math import resize_model_to_pruned
from utils.lhdq import lhdq_encode, lhdq_decode, is_lhdq_encoded

# --- CONFIGURAÇÃO DE OBSERVABILIDADE (LOGGING ENXUTO) ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)
# --------------------------------------------------------


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
    upload = make_v2i_message(
        "UPLOAD", **context, weights=snapshot, train_samples=train_samples
    )
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


def map_sequential_to_simplemodel(state_dict):
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

# def dequantization(global_state):
#     dequantized_state_dict = {}
#     for k, v in global_state.items():
#         if isinstance(v, dict) and v.get('dtype') == 'quantized_int8':
#             scale = v['scale']
#             dequantized_state_dict[k] = v['weights'].float() * scale
#         else:
#             dequantized_state_dict[k] = v
#     return dequantized_state_dict

# def quantization(state_dict):
#     quantized_state_dict = {}
#     for k, v in state_dict.items():
#         if isinstance(v, torch.Tensor):
#             scale = torch.max(torch.abs(v)) / 127.0
#             quantized_weights = torch.clamp((v / scale).round(), -128, 127).to(torch.int8)
#             quantized_state_dict[k] = {
#                 'dtype': 'quantized_int8',
#                 'scale': scale,
#                 'weights': quantized_weights
#             }
#         else:
#             quantized_state_dict[k] = v
#     return quantized_state_dict

def local_training(model, state_dict, prune, train_loader, test_loader, learning_rate=0.01, round_num=2, alaarg=1, ala=None, model_type='cnn', run_config=None):
    # Verifica se os pesos precisam ser desquantizados (se não forem tensores)
    if state_dict and not isinstance(next(iter(state_dict.values())), torch.Tensor):
        state_dict = dequantization(state_dict)
    
    personalized_acc = None  # Inicializa a variável que guardará a acurácia pós-ALA
    
    if round_num >= 2 and prune == 1:
        if model_type == 'cnn':
            state_dict = map_sequential_to_simplemodel(state_dict)
    # === CORREÇÃO ESTRUTURAL PARA CLIP E SLM ===
    if model_type in ['clip', 'slm']:
        # FedALA_LoRA só se aplica a CLIP (model_type != 'slm').
        # Evita deepcopy desnecessário do SLM inteiro (~4-6 GB).
        needs_ala = (alaarg == 0 and round_num >= 2 and model_type != 'slm')
        
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
        # --- LÓGICA ANTIGA PARA CNN MANTIDA INTACTA ---
        local_model = resize_model_to_pruned(model, state_dict)
        state = copy.deepcopy(local_model)
        
        if alaarg == 0:
            # Aviso: Este código da CNN continua bugado (round_num == 2)
            if round_num == 2:
                logger.info(f"Client {ala.cid}: Applying FedALA (Adaptive Local Aggregation)...")
                local_initialization(ala, state, model)
            
        set_parameters(model, state)
        
        if alaarg == 0 and round_num >= 2:
            personalized_acc, _ = evaluate_model(model, test_loader)
            logger.info(f"Client {ala.cid}: Post-ALA Test Accuracy: {personalized_acc:.2f}%")
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    device = next(model.parameters()).device
    
    # --- 1. SELEÇÃO DE OTIMIZADOR ---
    if model_type in ['clip', 'slm']:
        if model_type == 'clip':
            from lora_clip.clip_setup import build_optimizer, train_epoch, update_paca_runtime
        else:
            from lora_slm.slm_setup import build_optimizer, train_epoch, update_paca_runtime
        
        # --- PaCA RUNTIME UPDATE ---
        paca_config = run_config["model"].get("paca", {})
        if paca_config.get("enabled"):
            update_paca_runtime(model, paca_config.get("upper_layers"))
        
        # --- DIAGNÓSTICO: Contagem de parâmetros treináveis (após PaCA update) ---
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if not hasattr(model, '_last_trainable_count') or model._last_trainable_count != trainable_count:
            logger.info(f"[DIAG] Parâmetros treináveis neste round: {trainable_count:,}")
            model._last_trainable_count = trainable_count
            
        # VRAM OPTIMIZATION: Habilita Gradient Checkpointing para SLM (Ação 1)
        if model_type == 'slm':
            if hasattr(model, "slm_model") and hasattr(model.slm_model, "gradient_checkpointing_enable"):
                model.slm_model.gradient_checkpointing_enable()
                # logger.debug("Gradient Checkpointing ativado no SLM (Redução de VRAM).")
        
        optimizer, sparse_optimizer = build_optimizer(model, run_config)
        is_sora = run_config["model"]["lora"]["mode"] in ["with_sora_no_schedule", "with_sora_schedule"]
        sparse_lambda = run_config["model"].get("sora", {}).get("sparse_lambda", 0.0) if is_sora else 0.0
        
        if model_type == 'slm':
            accumulation_steps = run_config.get("training", {}).get("gradient_accumulation_steps", 1)
            metrics = train_epoch(model, train_loader, optimizer, sparse_optimizer, sparse_lambda, accumulation_steps)
        else:
            metrics = train_epoch(model, train_loader, optimizer, sparse_optimizer, sparse_lambda)
        
        logger.info(f"Métricas do Treino Local - CE Loss: {metrics['ce_loss']:.4f} | Sparse Loss: {metrics['sparse_loss']:.4f}")
    else:
        # --- DIAGNÓSTICO: Contagem de parâmetros treináveis ---
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if not hasattr(model, '_last_trainable_count') or model._last_trainable_count != trainable_count:
            logger.info(f"[DIAG] Parâmetros treináveis neste round: {trainable_count:,}")
            model._last_trainable_count = trainable_count
        
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        sparse_optimizer = None

        model_dtype = next(model.parameters()).dtype
        
        for x, y in train_loader:
            if x.is_floating_point():
                x = x.to(device, dtype=model_dtype)
            else:
                x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            if sparse_optimizer is not None:
                sparse_optimizer.zero_grad()
                
            output = model(x)
            if isinstance(output, dict):
                    output = output["logits"]
                    
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            
            if sparse_optimizer is not None:
                sparse_optimizer.step()
    
    # --- 2. SELEÇÃO DOS PESOS QUE SERÃO ENVIADOS PARA O SERVIDOR ---
    if model_type in ['clip', 'slm']:
        if is_qlora_enabled(run_config):
            # QLoRA: envia SOMENTE os adaptadores LoRA 16 bits + head (nunca o backbone 4-bit)
            return get_qlora_state_dict(model), personalized_acc
        if model_type == 'clip':
            from lora_clip.sora import get_trainable_state_dict
        else:
            from lora_slm.sora import get_trainable_state_dict
        return get_trainable_state_dict(model), personalized_acc 
    else:
        return model.state_dict(), personalized_acc 



import torchvision.transforms as T
from PIL import Image

def load_data(dataset, client_idx, device, is_train=True, batch_size=32, is_clip=False, is_slm=False, slm_config=None, dataset_dir=None):
    train_data = read_client_data(dataset, client_idx, is_train, dataset_dir=dataset_dir)
    X, y = zip(*train_data)
    X = torch.stack(X)
    
    if is_slm:
        from lora_slm.slm_setup import CustomCollator
        from transformers import AutoProcessor
        
        processor = AutoProcessor.from_pretrained(slm_config["model"]["name"])
        label_to_idx = {i: i for i in range(1000)}
        label_to_idx.update({str(i): i for i in range(1000)})
        collate_fn = CustomCollator(processor, label_to_idx)
        
        slm_data = []
        for x_tensor, y_tensor in zip(X, y):
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
                # Normalização padrão [-1, 1]
                x_tensor = x_tensor * 0.5 + 0.5
                
            if x_tensor.max() <= 1.0:
                x_tensor = x_tensor * 255.0
                
            # Clamp to [0, 255] range
            x_tensor = torch.clamp(x_tensor, 0, 255)
            x_numpy = x_tensor.byte().numpy()
            
            if x_numpy.shape[0] in [1, 3]: # (C, H, W) -> (H, W, C)
                x_numpy = np.transpose(x_numpy, (1, 2, 0))
            if x_numpy.shape[-1] == 1:
                x_numpy = np.squeeze(x_numpy, axis=-1)
                
            pil_img = Image.fromarray(x_numpy)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            
            slm_data.append({"image": pil_img, "label": y_tensor.item()})
            
        pin_mem = True if hasattr(device, 'type') and device.type == 'cuda' else False
        return DataLoader(slm_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=pin_mem)

    # Redimensionamento e normalização vitais para o CLIP
    if is_clip:
        if X.shape[-2:] != (224, 224):
            resize = T.Resize((224, 224), antialias=True)
            X = resize(X)
        if not X.is_floating_point():
            X = X.float() / 255.0
        elif X.min() >= 0.0 and X.max() > 1.0:
            X = X / 255.0
        # Se os dados não tiverem normalização do CLIP, aplica a normalização oficial do CLIP
        if X.min() >= 0.0:
            normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
            X = normalize(X)
        
    # Garante que o rótulo é inteiro longo e move os dados INTEIROS para a GPU
    X = X.to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)
    
    dataset = torch.utils.data.TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


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
    algo = f"client_{idx}_{args.dataset}_{args.strategy}_rank{rank_val}_paca{paca_val}_{b}_run{args.run_id}{ts_suffix}"

    current_dir = Path(__file__).resolve().parent
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
        # Métricas OTA (Fase 3): distância, época do encontro, papel do cliente
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


def local_initialization(ala, received_global_model, model, mask=None):
    ala.adaptive_local_aggregation(received_global_model, model, mask=mask)


# def resize_model_to_pruned(model, pruned_dict):
#     """ Redimensiona o modelo existente para as dimensões podadas """
#     with torch.no_grad():
#         for name, param in model.named_parameters():
#             if name in pruned_dict:
#                 pruned_weight = pruned_dict[name]
                
#                 if param.shape != pruned_weight.shape:
#                     logger.debug(f"Redimensionando {name}: {param.shape} -> {pruned_weight.shape}")
#                     new_param = nn.Parameter(pruned_weight.to(param.device))
                    
#                     if '.' in name:
#                         parts = name.split('.')
#                         module = model
#                         for part in parts[:-1]:
#                             module = getattr(module, part)
#                         setattr(module, parts[-1], new_param)
#                         if hasattr(module, 'r') and 'lora_A' in name:
#                             module.r = pruned_weight.shape[0]
#                     else:
#                         setattr(model, name, new_param)
#                 else:
#                     param.data.copy_(pruned_weight.to(param.device))
#     return model

def parse_args():
    # Colocar um argumento -t para servir como um for 
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='Cifar100')
    parser.add_argument('--dataset-dir', type=Path, help='Diretório com config.json, train/ e test/; mantém o nome lógico do dataset')
    parser.add_argument('--client-idx', type=int, default=0)
    parser.add_argument('--in-features', type=int, default=3)
    parser.add_argument('--num-classes', type=int, default=100)
    parser.add_argument('--dim', type=int, default=1600)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--random-client', action='store_true')
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument("--ala", type=int, default=0)
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument('--config', type=str, default="lora_clip/train_config.yml")
    parser.add_argument('--experiments', type=int, default=1)
    parser.add_argument('--run-id', type=int, default=1)
    parser.add_argument('--execution-id', default='', help='ID da simulação V2V/V2I')
    parser.add_argument('--results-dir', type=Path)
    parser.add_argument('--encounters-dir', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exp-name', type=str, default='default_exp', help='Nome da sessão')
    parser.add_argument('--timestamp', type=str, default='', help='Timestamp para diferenciar rodadas de simulação')
    parser.add_argument('--strategy', type=str, default='lora', choices=['lora', 'adalora', 'sora_with_schedule', 'sora_no_schedule'])
    parser.add_argument('--rank', type=int, default=8, help='Rank para o SoRA/LoRA') 
    parser.add_argument('--paca', type=int, default=12, help='Número de camadas para a estratégia PaCA')
    
    # --- Parâmetros de Sparsidade (SoRA) ---
    parser.add_argument('--sparse-lr', type=float, default=None, help='Taxa de aprendizado para a poda (Softshrink)')
    parser.add_argument('--max-lambda', type=float, default=None, help='Força máxima da penalidade L1 (max_lambda)')
    
    # --- PaCA Heterogêneo ---
    parser.add_argument('--random-paca', action='store_true', help='Sorteia um valor de PaCA aleatório para este cliente')
    parser.add_argument('--paca-min', type=int, default=1, help='Valor mínimo de PaCA no sorteio aleatório')
    parser.add_argument('--paca-max', type=int, default=12, help='Valor máximo de PaCA no sorteio aleatório')
    parser.add_argument('--paca-list', type=str, default=None, help='Lista de PaCAs pré-definidos por cliente (ex: "4,6,8,12"). O cliente usa o valor na posição client_idx %% len(lista)')
    parser.add_argument('--prune', type=int, default=1, choices=[0, 1], help='Habilitar pruning adaptativo (0=sim, 1=não, default: 1)')
    parser.add_argument('--mode', type=str, default='centralized', choices=['centralized', 'decentralized'])
    parser.add_argument('--max-epochs', type=int, default=0, help='Limite de epocas no modo descentralizado (0 = infinito)')
    parser.add_argument('--delta-coding', action='store_true', help='Ativa LHDQ (Low Huffman-coded Delta Quantization) em vez de int8')
    
    # --- Otimização de Avaliação ---
    parser.add_argument('--skip-post-eval', action='store_true', help='Pula todas as avaliações pós-treino no cliente (local_test_acc e train_acc) para máxima velocidade, mantendo apenas Global Model Test Acc')
    parser.add_argument('--skip-train-eval', action='store_true', help='Pula apenas a avaliação demorada no conjunto de treino pós-treino')
    
    # --- QLoRA (backbone 4-bit NF4 + adaptadores LoRA 16 bits) ---
    parser.add_argument('--qlora', action='store_true', help='Ativa QLoRA (apenas --model slm --strategy lora, modo centralizado)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    args.training_seed = args.seed + args.client_idx
    random.seed(args.training_seed)
    np.random.seed(args.training_seed)
    torch.manual_seed(args.training_seed)
    args.encounter_metrics = []
    
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
    
    device = torch.device(args.device)
    
    # --- Resolução do PaCA Heterogêneo ---
    if args.paca_list is not None:
        # Modo lista: valores pré-definidos por cliente (determinístico)
        paca_values = [int(x.strip()) for x in args.paca_list.split(',')]
        args.paca = paca_values[args.client_idx % len(paca_values)]
        logger.info(f"PaCA Heterogêneo (lista): Cliente {args.client_idx} usando PaCA = {args.paca}")
    elif args.random_paca:
        # Modo aleatório: sorteia uma vez no início (seed baseada no client_idx + run_id para reprodutibilidade)
        rng = random.Random(args.client_idx * 1000 + args.run_id)
        args.paca = rng.randint(args.paca_min, args.paca_max)
        logger.info(f"PaCA Heterogêneo (aleatório): Cliente {args.client_idx} sorteou PaCA = {args.paca} (intervalo [{args.paca_min}, {args.paca_max}])")
    
    logger.info("=== Federated Learning Client ===")
    logger.info(f"Host: {args.host}:{args.port}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Client: {args.client_idx} | PaCA: {args.paca}")
    logger.info(f"Rounds: {args.rounds}")
    logger.info(f"Batch size: {args.batch_size} | LR: {args.learning_rate}")
    logger.info("=================================")
    
    # Adoção de Structural Pattern Matching
    match args.model:
        case "cnn":
            # Isolar a inicialização comum preserva o RNG de treino de cada cliente.
            # SimpleModel é construído em CPU; os geradores dos aceleradores não mudam.
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
                initial_digest = hashlib.sha256()
                for name, tensor in sorted(model.state_dict().items()):
                    initial_digest.update(
                        json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode("utf-8")
                    )
                    initial_tensor = tensor.detach().cpu().contiguous().reshape(-1)
                    initial_digest.update(initial_tensor.view(torch.uint8).numpy().tobytes())
                args.initial_model_sha256 = initial_digest.hexdigest()
                logger.info(
                    "Client %d: CNN inicial comum | seed=%d | treino=%d | SHA256=%s",
                    args.client_idx,
                    args.seed,
                    args.training_seed,
                    args.initial_model_sha256,
                )
        case 'clip' | 'slm':
            if args.model == 'clip':
                config = load_config(args.config)
            else:
                from lora_slm.slm_setup import load_config as load_config_slm
                config = load_config_slm(args.config)
            
            if args.strategy == 'lora':
                config["model"]["lora"]["mode"] = "with_lora"
            elif args.strategy == 'adalora':
                config["model"]["lora"]["mode"] = "with_adalora"
            elif args.strategy == 'sora_with_schedule':
                config["model"]["lora"]["mode"] = "with_sora_schedule"
            elif args.strategy == 'sora_no_schedule':
                config["model"]["lora"]["mode"] = "with_sora_no_schedule"

            if args.qlora:
                if args.model != 'slm' or args.mode != 'centralized':
                    logger.error("--qlora é suportado apenas com --model slm no modo centralizado.")
                    sys.exit(1)
                enable_qlora_in_config(config, args.strategy)
                logger.info("QLoRA ATIVADO: backbone 4-bit NF4 + DoubleQuant, adaptadores LoRA 16 bits.")

                
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
                model = build_model(config=run_config, num_classes=args.num_classes, device=device)
            else:
                from lora_slm.slm_setup import resolve_run_modes as resolve_run_modes_slm
                from lora_slm.slm_setup import build_run_config as build_run_config_slm
                from lora_slm.slm_setup import build_model as build_model_slm
                run_mode = resolve_run_modes_slm(config)[0]
                run_config = build_run_config_slm(config, run_mode=run_mode)
                model = build_model_slm(config=run_config, num_classes=args.num_classes, device=device)
            
    if not args.qlora:  # QLoRA: backbone 4-bit já alocado via device_map
        model = model.to(device)
    loss = nn.CrossEntropyLoss()
    rs_global_acc = []
    rs_global_loss = []
    rs_ala_acc = []
    rs_local_acc = []
    rs_train_acc = []
    rs_post_p2p_acc = []
    rs_post_p2p_encounter_ids = []
    rs_p2p_data_sent_mb = []
    rs_p2p_data_sent_enc_ids = []
    # Métricas OTA (Fase 3): registram informações de cada encontro
    rs_ota_encounter_epochs = []   # Em qual época cada encontro ocorreu
    rs_ota_mean_distances = []     # Distância média (metros) dos TX ao Aggregator
    rs_ota_is_aggregator = []      # 1 se este cliente foi o Aggregator, 0 se TX
    
    try:
        is_clip_flag = (args.model == 'clip')
        is_slm_flag = (args.model == 'slm')
        train_loader = load_data(args.dataset, args.client_idx, device, is_train=True, batch_size=args.batch_size, is_clip=is_clip_flag, is_slm=is_slm_flag, slm_config=run_config if is_slm_flag else None, dataset_dir=args.dataset_dir)
        test_loader = load_data(args.dataset, args.client_idx, device, is_train=False, batch_size=args.batch_size, is_clip=is_clip_flag, is_slm=is_slm_flag, slm_config=run_config if is_slm_flag else None, dataset_dir=args.dataset_dir)
        test_majority_baseline = None
        if hasattr(test_loader.dataset, "tensors"):
            labels = test_loader.dataset.tensors[1]
            counts = torch.unique(labels, return_counts=True)[1]
            test_majority_baseline = 100.0 * counts.max().item() / len(labels)
        logger.info(f"Data loaded successfully - Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    except Exception as e:
        logger.exception("Error loading data")
        sys.exit(1)
    
    if args.model == 'clip':
        ala = ALA_LoRA(args.client_idx, loss, train_loader, 32, 80, 1.0, device)
    else:
        ala = ALA(args.client_idx, loss, train_loader, 32, 80, 2, 1.0, device)
        
    time.sleep(10)
    
    if args.mode == 'centralized':
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
                    if args.qlora:
                        set_qlora_state_dict(model, global_state)
                        local = model
                    else:
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
                    elif not args.qlora:  # QLoRA: adaptadores trafegam em 16 bits (sem int8)
                        updated_state = quantization(updated_state)
                client_quant_time = time.time() - quant_start
                client_eval_time = client_pre_eval_time + client_post_eval_time
                client_processing_overhead = client_eval_time + client_quant_time + client_dequant_time
                logger.info(f"Client {args.client_idx}: Tempos - treino={client_training_time:.2f}s, eval={client_eval_time:.2f}s, quant={client_quant_time:.2f}s, dequant={client_dequant_time:.2f}s")
                try:
                    send_data(s, updated_state)
                    send_data(s, len(train_loader))
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

    elif args.mode == 'decentralized':
        import threading
        logger.info(f"Client {args.client_idx}: Modo Descentralizado (D-PSGD) Ativado.")
        encounters_dir = args.encounters_dir or Path(__file__).resolve().parents[1] / "results" / "encounters"
        in_memory_weights = {}
        in_memory_bytes = {}
        p2p_timeout = 600.0
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
                            if args.execution_id and data.get("run_id") != args.execution_id:
                                continue
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
                    with conn:
                        data, _, received_bytes = recv_data(
                            conn, return_bytes=True,
                            deadline=time.monotonic() + p2p_timeout,
                            cancelled=lambda: (encounters_dir / ".simulation_done").exists(),
                        )
                        if data is not None:
                            validate_v2v_message(data)
                            sender_id, enc_id = data["client_id"], data["encounter_id"]
                            with in_memory_lock:
                                if enc_id not in in_memory_weights:
                                    in_memory_weights[enc_id] = {}
                                in_memory_weights[enc_id][sender_id] = data
                                in_memory_bytes.setdefault(enc_id, {})[sender_id] = (
                                    in_memory_bytes.get(enc_id, {}).get(sender_id, 0) + received_bytes
                                )
                            logger.info(f"Client {args.client_idx}: Recebeu pesos do Cliente {sender_id} para encontro {enc_id}")
                except Exception as e:
                    logger.error(f"Erro no P2P Server: {e}")
        p2p_started = False
        encounter_failure = None
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
                logger.info(
                    f"Client {args.client_idx}: Epoca {epoch} Local Test Acc: {local_test_acc:.2f}%"
                )

            save_results(
                args,
                rs_global_acc,
                rs_global_loss,
                rs_ala_acc,
                rs_local_acc,
                rs_train_acc,
                idx=args.client_idx,
                argalgo=args.ala,
                rs_post_p2p_acc=rs_post_p2p_acc,
                rs_post_p2p_encounter_ids=rs_post_p2p_encounter_ids,
                rs_p2p_data_sent_mb=rs_p2p_data_sent_mb,
                rs_p2p_data_sent_enc_ids=rs_p2p_data_sent_enc_ids,
            )

            # ── FASE 4: Sinalizar conclusão da época ────────────────────────
            (encounters_dir / f".epoch_done_client_{args.client_idx}").touch()
            logger.info(f"Client {args.client_idx}: Epoca concluida. Aguardando encontro P2P...")

            # ── FASE 5: Aguardar e Processar Encontro ───────────────────────
            encounter_data = wait_for_encounter(
                encounters_dir, args.client_idx, last_processed_encounter, current_round=epoch
            )

            if encounter_data is None:
                logger.info(
                    f"Client {args.client_idx}: Sinal de fim de simulacao recebido. Encerrando."
                )
                break

            if encounter_data == "skip":
                logger.info(
                    f"Client {args.client_idx}: Encontro nao ocorreu para mim nesta rodada. Avancando para prox epoca."
                )
                epoch += 1
                continue

            enc_id = encounter_data.get("encounter_id", 0)
            clients_in_encounter = encounter_data.get("clients", [])
            etc_seconds = encounter_data.get("etc_seconds", 60)
            encounter_type = encounter_data.get("type", "v2v")  # "v2v" ou "v2i"

            # Converter chaves do JSON de volta para int
            positions = {int(k): tuple(v) for k, v in encounter_data.get("positions", {}).items()}

            if isinstance(etc_seconds, str) and etc_seconds == "Carros Parados":
                poll_timeout = 115
            else:
                poll_timeout = max(5, int(etc_seconds - 5))

            encounter_started = time.monotonic()
            metrics = {
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
                "accuracy_before": rs_local_acc[-1]
                if rs_local_acc and not args.skip_post_eval
                else None,
                "accuracy_after": None,
                "majority_class_baseline": test_majority_baseline,
                "test_samples": len(test_loader.dataset),
                "train_samples": len(train_loader.dataset),
                "evaluation_skipped": args.skip_post_eval,
                "error": None,
            }
            args.encounter_metrics.append(metrics)

            # ══════════════════════════════════════════════════════════════════
            # BRANCH V2I: CNN → rsu-service por TCP → CNN
            # ══════════════════════════════════════════════════════════════════
            if encounter_type == "v2i":
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
                    logger.info(
                        "Client %d: Resultado da RSU aplicado; APPLIED enviado.", args.client_idx
                    )
                    rs_p2p_data_sent_mb.append(exchange["upload_bytes"] / (1024 * 1024))
                    rs_p2p_data_sent_enc_ids.append(enc_id)

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
                        rs_post_p2p_acc.append(post_p2p_acc)
                        rs_post_p2p_encounter_ids.append(enc_id)
                        logger.info(
                            "Client %d: Post-V2I Test Accuracy: %.2f%%",
                            args.client_idx,
                            post_p2p_acc,
                        )
                    metrics["status"] = "completed"
                    rs_ota_encounter_epochs.append(epoch)
                    rs_ota_mean_distances.append(enc_mean_distance)
                    rs_ota_is_aggregator.append(0)
                    save_results(
                        args,
                        rs_global_acc,
                        rs_global_loss,
                        rs_ala_acc,
                        rs_local_acc,
                        rs_train_acc,
                        idx=args.client_idx,
                        argalgo=args.ala,
                        rs_post_p2p_acc=rs_post_p2p_acc,
                        rs_post_p2p_encounter_ids=rs_post_p2p_encounter_ids,
                        rs_p2p_data_sent_mb=rs_p2p_data_sent_mb,
                        rs_p2p_data_sent_enc_ids=rs_p2p_data_sent_enc_ids,
                        rs_ota_encounter_epochs=rs_ota_encounter_epochs,
                        rs_ota_mean_distances=rs_ota_mean_distances,
                        rs_ota_is_aggregator=rs_ota_is_aggregator,
                    )
                    write_json_atomic(
                        metrics, encounters_dir / f".done_enc_{enc_id}_client_{args.client_idx}"
                    )
                    logger.info(
                        "Client %d: Encontro V2I concluído; métricas salvas.", args.client_idx
                    )
                except Exception as exc:
                    encounter_failure = str(exc) or type(exc).__name__
                    logger.error(
                        "Client %d: Falha no encontro V2I %d: %s",
                        args.client_idx,
                        enc_id,
                        encounter_failure,
                    )
                    metrics.update(status="failed", error=encounter_failure)
                    write_v2i_failure(
                        encounters_dir, encounter_data, args.client_idx, encounter_failure, metrics
                    )
                    break
                last_processed_encounter = enc_id
                epoch += 1
                continue

            # ══════════════════════════════════════════════════════════════════
            # BRANCH V2V: Fluxo original com OTA e eleição de Aggregator
            # ══════════════════════════════════════════════════════════════════
            try:
                p2p_timeout = encounter_data.get("exchange_timeout", poll_timeout)
                if (
                    isinstance(p2p_timeout, bool)
                    or not isinstance(p2p_timeout, (int, float))
                    or not math.isfinite(p2p_timeout)
                    or p2p_timeout <= 0
                ):
                    raise ValueError("exchange_timeout V2V deve ser positivo e finito.")
                exchange_deadline = encounter_started + p2p_timeout
                cancelled = (encounters_dir / ".simulation_done").exists
                if not p2p_started:
                    threading.Thread(target=p2p_server_thread, daemon=True).start()
                    p2p_started = True
                # Eleição do Aggregator via Medoid
                from v2v_fl.over_the_air.ota_channel import find_medoid

                aggregator_id = find_medoid(positions)
                is_aggregator = args.client_idx == aggregator_id
                metrics["is_aggregator"] = is_aggregator

                logger.info(
                    f"Client {args.client_idx}: [D-PSGD] Encontro {enc_id} detectado! ETC={etc_seconds}s."
                )
                logger.info(
                    f"Client {args.client_idx}: Aggregator eleito: {aggregator_id}. Sou aggregator? {is_aggregator}"
                )

                # Calcular distância média TX→Aggregator para métricas OTA
                from v2v_fl.over_the_air.ota_channel import distances_from_positions

                _, enc_distances = distances_from_positions(positions, aggregator_id)
                enc_mean_distance = (
                    float(np.mean(enc_distances)) if len(enc_distances) > 0 else 0.0
                )

                # ── FASE 6: Troca P2P e Agregação (D-PSGD Assimétrico) ──────────────
                if args.model == "clip":
                    from lora_clip.sora import get_trainable_state_dict

                    snapshot = get_trainable_state_dict(model)
                else:

                    snapshot = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }

                metrics["mean_distance"] = enc_mean_distance
                other_clients = [c for c in clients_in_encounter if c != args.client_idx]

                # 1. Transmissores enviam seus pesos para o Aggregator
                if not is_aggregator:
                    target_host = f"fl-client-v2v-{aggregator_id}"
                    snapshot_bytes = send_p2p_model(
                        target_host,
                        make_v2v_message(
                            "UPLOAD", run_id=encounter_data["run_id"],
                            client_id=args.client_idx, encounter_id=enc_id,
                            weights=snapshot, train_samples=len(train_loader.dataset),
                        ),
                        deadline=exchange_deadline, cancelled=cancelled,
                    )
                    metrics["upload_bytes"] += snapshot_bytes
                    rs_p2p_data_sent_mb.append(snapshot_bytes / (1024 * 1024))
                    rs_p2p_data_sent_enc_ids.append(enc_id)
                    logger.info("Client %d: Enviou pesos para %s (%d bytes).",
                                args.client_idx, target_host, snapshot_bytes)

                    # Aguardar resposta do Aggregator (pesos consolidados)
                    logger.info(
                        f"Client {args.client_idx}: Aguardando resposta do Aggregator {aggregator_id}..."
                    )
                    received_states = []
                    while time.monotonic() < exchange_deadline:
                        if cancelled():
                            raise ConnectionAbortedError(
                                "Simulação encerrada aguardando resultado V2V."
                            )
                        with in_memory_lock:
                            if (
                                enc_id in in_memory_weights
                                and aggregator_id in in_memory_weights[enc_id]
                            ):
                                response = validate_v2v_message(
                                    in_memory_weights[enc_id][aggregator_id], type="RESULT",
                                    run_id=encounter_data["run_id"], encounter_id=enc_id,
                                    client_id=aggregator_id,
                                )
                                received_states.append(response)
                                break
                        time.sleep(1)

                # 2. Aggregator recebe de todos os transmissores e consolida
                if is_aggregator:
                    logger.info(
                        f"Client {args.client_idx}: Aguardando receber pesos dos transmissores {other_clients}..."
                    )
                    received_states = []
                    while time.monotonic() < exchange_deadline:
                        if cancelled():
                            raise ConnectionAbortedError(
                                "Simulação encerrada aguardando uploads V2V."
                            )
                        with in_memory_lock:
                            if enc_id in in_memory_weights:
                                missing = [
                                    c for c in other_clients if c not in in_memory_weights[enc_id]
                                ]
                                if not missing:
                                    for c in other_clients:
                                        upload = validate_v2v_message(
                                            in_memory_weights[enc_id][c], type="UPLOAD",
                                            run_id=encounter_data["run_id"], encounter_id=enc_id,
                                            client_id=c,
                                        )
                                        received_states.append(upload)
                                    break
                        time.sleep(1)

                with in_memory_lock:
                    received_bytes = sum(in_memory_bytes.get(enc_id, {}).values())
                metrics["received_upload_bytes" if is_aggregator else "download_bytes"] = (
                    received_bytes
                )

                # 3. Consolidação (No Aggregator) ou Atualização (Nos Transmissores)
                if (is_aggregator and len(received_states) == len(other_clients)) or (
                    not is_aggregator and len(received_states) == 1
                ):
                    # ── AGGREGATOR: Pipeline OTA (Over-the-Air) ──────────────────
                    if is_aggregator:
                        logger.info(
                            f"Client {args.client_idx}: Todos os pesos recebidos! Iniciando agregação OTA..."
                        )

                        from v2v_fl.over_the_air.ota_aggregation import (
                            ota_aggregate,
                        )

                        aggregation_started = time.monotonic()
                        try:
                            participant_states = {
                                upload["client_id"]: upload["weights"]
                                for upload in received_states
                            }
                            participant_states[args.client_idx] = snapshot
                            train_samples = {
                                upload["client_id"]: upload["train_samples"]
                                for upload in received_states
                            }
                            train_samples[args.client_idx] = len(train_loader.dataset)
                            diagnostics = {}
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
                        # Carregar o modelo agregado no PyTorch
                        if args.model in ["clip", "slm"]:
                            if args.model == "clip":
                                from lora_clip.sora import set_trainable_state_dict

                                set_trainable_state_dict(model, snapshot)
                            else:
                                from lora_slm.sora import (
                                    set_trainable_state_dict as set_trainable_state_dict_slm,
                                )

                                set_trainable_state_dict_slm(model, snapshot)
                        else:
                            model.load_state_dict(snapshot)

                    # ── TRANSMISSOR: Apenas carrega o modelo recebido do Aggregator ──
                    else:
                        logger.info(
                            f"Client {args.client_idx}: Pesos consolidados recebidos do Aggregator!"
                        )
                        # O Transmissor recebe o modelo já agregado (OTA + local do Aggregator)
                        # e simplesmente o carrega como seu novo modelo local
                        aggregated_state = received_states[0]["weights"]

                        if args.model in ["clip", "slm"]:
                            for k in snapshot:
                                snapshot[k].copy_(aggregated_state[k].to(snapshot[k].dtype))
                            if args.model == "clip":
                                from lora_clip.sora import set_trainable_state_dict

                                set_trainable_state_dict(model, snapshot)
                            else:
                                from lora_slm.sora import (
                                    set_trainable_state_dict as set_trainable_state_dict_slm,
                                )

                                set_trainable_state_dict_slm(model, snapshot)
                        else:
                            for k in snapshot:
                                snapshot[k] = aggregated_state[k].to(snapshot[k].dtype)
                            model.load_state_dict(snapshot)

                    metrics["applied"] = True
                    # 4. Aggregator envia o resultado de volta para os transmissores
                    if is_aggregator:
                        for c in other_clients:
                            target_host = f"fl-client-v2v-{c}"
                            snapshot_bytes = send_p2p_model(
                                target_host,
                                make_v2v_message(
                                    "RESULT", run_id=encounter_data["run_id"],
                                    client_id=args.client_idx, encounter_id=enc_id,
                                    weights=snapshot,
                                ),
                                deadline=exchange_deadline, cancelled=cancelled,
                            )
                            metrics["sent_download_bytes"] += snapshot_bytes
                            rs_p2p_data_sent_mb.append(snapshot_bytes / (1024 * 1024))
                            rs_p2p_data_sent_enc_ids.append(enc_id)
                            logger.info(
                                "Client %d: Retornou pesos consolidados para %s (%d bytes).",
                                args.client_idx, target_host, snapshot_bytes,
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
                    # Limpar RAM
                    with in_memory_lock:
                        if enc_id in in_memory_weights:
                            del in_memory_weights[enc_id]
                        in_memory_bytes.pop(enc_id, None)

                    # ── FASE 7: Avaliação Pós-Agregação ──────────────────────
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
                        rs_post_p2p_acc.append(post_p2p_acc)
                        rs_post_p2p_encounter_ids.append(enc_id)

                    metrics["status"] = "completed"
                    # Registrar métricas OTA deste encontro (dentro ou fora do skip_post_eval)
                    rs_ota_encounter_epochs.append(epoch)
                    rs_ota_mean_distances.append(enc_mean_distance)
                    rs_ota_is_aggregator.append(1 if is_aggregator else 0)

                    save_results(
                        args,
                        rs_global_acc,
                        rs_global_loss,
                        rs_ala_acc,
                        rs_local_acc,
                        rs_train_acc,
                        idx=args.client_idx,
                        argalgo=args.ala,
                        rs_post_p2p_acc=rs_post_p2p_acc,
                        rs_post_p2p_encounter_ids=rs_post_p2p_encounter_ids,
                        rs_p2p_data_sent_mb=rs_p2p_data_sent_mb,
                        rs_p2p_data_sent_enc_ids=rs_p2p_data_sent_enc_ids,
                        rs_ota_encounter_epochs=rs_ota_encounter_epochs,
                        rs_ota_mean_distances=rs_ota_mean_distances,
                        rs_ota_is_aggregator=rs_ota_is_aggregator,
                    )

                    # Sinalizar ao SUMO que a troca terminou
                    write_json_atomic(
                        metrics, encounters_dir / f".done_enc_{enc_id}_client_{args.client_idx}"
                    )
                    logger.info(
                        f"Client {args.client_idx}: Agregacao concluida. .done_enc enviado."
                    )
                else:
                    raise TimeoutError(f"Pesos V2V incompletos: {len(received_states)} recebidos.")
            except Exception as exc:
                encounter_failure = str(exc) or type(exc).__name__
                with in_memory_lock:
                    received_bytes = sum(in_memory_bytes.get(enc_id, {}).values())
                byte_key = (
                    "received_upload_bytes" if metrics["is_aggregator"] else "download_bytes"
                )
                metrics[byte_key] = max(metrics[byte_key], received_bytes)
                if metrics["status"] == "pending":
                    metrics["protocol_time"] = time.monotonic() - encounter_started
                metrics.update(status="failed", error=encounter_failure)
                write_v2i_failure(
                    encounters_dir, encounter_data, args.client_idx, encounter_failure, metrics
                )
                logger.error("Falha no encontro V2V %d: %s", enc_id, encounter_failure)
                break

            last_processed_encounter = enc_id
            epoch += 1

        # Ao final do max_epochs ou simulation_done
        logger.info(f"Client {args.client_idx}: Loop descentralizado finalizado.")
        save_results(
            args,
            rs_global_acc,
            rs_global_loss,
            rs_ala_acc,
            rs_local_acc,
            rs_train_acc,
            idx=args.client_idx,
            argalgo=args.ala,
            rs_post_p2p_acc=rs_post_p2p_acc,
            rs_post_p2p_encounter_ids=rs_post_p2p_encounter_ids,
            rs_p2p_data_sent_mb=rs_p2p_data_sent_mb,
            rs_p2p_data_sent_enc_ids=rs_p2p_data_sent_enc_ids,
            rs_ota_encounter_epochs=rs_ota_encounter_epochs,
            rs_ota_mean_distances=rs_ota_mean_distances,
            rs_ota_is_aggregator=rs_ota_is_aggregator,
        )
        if encounter_failure is not None:
            raise RuntimeError(
                f"Falha na execução descentralizada do cliente {args.client_idx}: "
                f"{encounter_failure}"
            )
        logger.info("\nTraining completed!")


if __name__ == '__main__':
    main()
