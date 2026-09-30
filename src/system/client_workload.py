"""Client data preparation and local model training."""
import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader

from utils.data_utils import read_client_data
from utils.fl_math import dequantization, evaluate_model, resize_model_to_pruned, set_parameters

logger = logging.getLogger(__name__)

def map_sequential_to_simplemodel(state_dict):
    mapped_dict = {}
    mapping = {
        '0.weight': 'conv1.0.weight',
        '0.bias': 'conv1.0.bias',
        '3.weight': 'conv2.0.weight',
        '3.bias': 'conv2.0.bias',
        '7.weight': 'fc1.0.weight',
        '7.bias': 'fc1.0.bias',
        '9.weight': 'fc.weight',
        '9.bias': 'fc.bias'
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
        if model_type == 'clip':
            from lora_clip.sora import get_trainable_state_dict
        else:
            from lora_slm.sora import get_trainable_state_dict
        return get_trainable_state_dict(model), personalized_acc
    else:
        return model.state_dict(), personalized_acc
def load_data(dataset, client_idx, device, is_train=True, batch_size=32, is_clip=False, is_slm=False, slm_config=None):
    train_data = read_client_data(dataset, client_idx, is_train)
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
        elif X.max() > 1.0:
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
