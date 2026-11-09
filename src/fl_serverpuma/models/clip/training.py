# Responsabilidade: Executa uma época de treino, mede esparsidade e avalia o classificador.
# Família CLIP.
#
# Principais funções: train_epoch, compute_gate_sparsity, evaluate.

"""CLIP model support — training."""

import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def train_epoch(model, loader, optimizer, sparse_optimizer=None, sparse_lambda=0.0):
    """
    Loop de Treino por Época.
    Calcula a perda CE (Cross-Entropy Loss) e a penalidade de esparsidade, executando o backpropagation 
    em ambos os otimizadores (se aplicável).
    """
    model.train()
    total_ce_loss = 0.0
    total_sparse_loss = 0.0
    total_loss = 0.0

    device = next(model.parameters()).device

    # Ativa o schedule linear de lambda (ex: 3e-4 → 7e-4) para aumentar a pressão
    # de esparsidade. No FL, o otimizador é recriado a cada rodada, então avançamos
    # o schedule imediatamente para usar o lambda mais forte.
    if sparse_optimizer is not None and hasattr(sparse_optimizer, 'step_lambda'):
        sparse_optimizer.step_lambda()

    # Cache dos gate_params: evita scan de named_parameters() a cada batch
    # Os objetos Parameter não mudam de identidade durante o treino, apenas seus .data
    if sparse_optimizer is not None and sparse_lambda > 0:
        gate_params = [p for n, p in model.named_parameters() if "sora" in n and "gate" in n]
        gate_params_total = sum(p.numel() for p in gate_params)
    else:
        gate_params = []
        gate_params_total = 0

    # Cache: evita chamada a next(model.parameters()) a cada batch
    model_dtype = next(model.parameters()).dtype

    # Identifica o AdaLoRA do PEFT, que envolve apenas o vision_model.
    is_adalora = (
        hasattr(model, "vision_model")
        and hasattr(model.vision_model, "base_model")
        and hasattr(model.vision_model.base_model, "update_and_allocate")
    )
    if is_adalora and not hasattr(model, "adalora_global_step"):
        model.adalora_global_step = 0

    for pixel_values, labels in tqdm(loader, desc="train", leave=False, disable=True):
        pixel_values = pixel_values.to(device, dtype=model_dtype)
        labels = labels.to(device)

        # set_to_none=True: mais rápido que zerar tensores (evita alocação de zeros na GPU)
        optimizer.zero_grad(set_to_none=True)
        # GateSparsifier.zero_grad() é no-op (gates estão no otimizador unificado).
        # Mantido para compatibilidade com SparseAdamW no trainer standalone (main.py).
        if sparse_optimizer is not None:
            sparse_optimizer.zero_grad(set_to_none=True)

        outputs = model(pixel_values=pixel_values, labels=labels)
        ce_loss = outputs["loss"]
        loss = ce_loss

        loss.backward()
        optimizer.step()

        if is_adalora:
            model.vision_model.base_model.update_and_allocate(model.adalora_global_step)
            model.adalora_global_step += 1
        # GateSparsifier.step(): aplica APENAS softshrink (proximal step)
        # SparseAdamW.step(): AdamW completo + softshrink (trainer standalone)
        if sparse_optimizer is not None:
            sparse_optimizer.step()

        # # Poda dinâmica do AdaLoRA (PEFT)
        # if is_adalora:
        #     model.base_model.update_and_allocate(model.adalora_global_step)
        #     model.adalora_global_step += 1

        total_ce_loss += ce_loss.item()
        total_loss += loss.item()

    # Calcula a esparsidade APENAS 1 vez no final da época (Remove o gargalo de I/O da GPU)
    sparse_loss_val = 0.0
    if gate_params_total > 0:
        with torch.no_grad():
            sparse_loss = sum(torch.sum(torch.abs(p)) for p in gate_params)
            sparse_loss_val = sparse_loss.item() / gate_params_total

    n = max(len(loader), 1)
    return {
        "ce_loss": total_ce_loss / n,
        "sparse_loss": sparse_loss_val,
        "total_loss": total_loss / n,
    }


def compute_gate_sparsity(model):
    """Monitoramento de Esparsidade. Calcula a porcentagem de portões SoRA que foram zerados."""
    total = 0
    zeros = 0
    for n, p in model.named_parameters():
        if "sora" in n and "gate" in n:
            total += p.numel()
            zeros += (p.data == 0).sum().item()
    return zeros, total


def evaluate(model, loader):
    """Validação de Resultados. Calcula a acurácia final usando scikit-learn."""
    model.eval()
    preds = []
    true_labels = []

    with torch.no_grad():
        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype

        for pixel_values, labels in tqdm(loader, desc="eval", leave=False, disable=True):
            pixel_values = pixel_values.to(device, dtype=model_dtype)
            outputs = model(pixel_values=pixel_values)
            logits = outputs["logits"]
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pred)
            true_labels.extend(labels.cpu().numpy())

    return accuracy_score(true_labels, preds)
