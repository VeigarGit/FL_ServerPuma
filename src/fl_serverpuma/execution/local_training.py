# Responsabilidade: Executa uma etapa local de treino e personalização e seleciona os pesos
# retornados ao servidor.
#
# Principais funções: local_training, local_initialization.

"""Local training."""

import copy
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from fl_serverpuma.models.clip.optimization import build_optimizer
from fl_serverpuma.models.clip.sora_pruning import get_trainable_state_dict
from fl_serverpuma.algorithms.fl_math import dequantization
from fl_serverpuma.algorithms.fl_math import evaluate_model
from fl_serverpuma.algorithms.fl_math import set_parameters
from fl_serverpuma.algorithms.fl_math import resize_model_to_pruned
from fl_serverpuma.models.state_mapping import map_sequential_to_simplemodel

logger = logging.getLogger(__name__)

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
            from fl_serverpuma.models.clip.optimization import build_optimizer
            from fl_serverpuma.models.clip.training import train_epoch
            from fl_serverpuma.models.clip.adaptation import update_paca_runtime
        else:
            from fl_serverpuma.models.slm.optimization import build_optimizer
            from fl_serverpuma.models.slm.training import train_epoch
            from fl_serverpuma.models.slm.adaptation import update_paca_runtime

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
            from fl_serverpuma.models.clip.sora_pruning import get_trainable_state_dict
        else:
            from fl_serverpuma.models.slm.sora_pruning import get_trainable_state_dict
        return get_trainable_state_dict(model), personalized_acc 
    else:
        return model.state_dict(), personalized_acc


def local_initialization(ala, received_global_model, model, mask=None):
    ala.adaptive_local_aggregation(received_global_model, model, mask=mask)
