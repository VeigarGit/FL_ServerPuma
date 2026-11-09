# Responsabilidade: Executa uma época de treino, mede esparsidade e avalia o classificador.
# Família SLM.
#
# Principais funções: train_epoch, compute_gate_sparsity, evaluate.

"""SLM model support — training."""

import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def train_epoch(model, loader, optimizer, sparse_optimizer=None, sparse_lambda=0.0, accumulation_steps=1):
    model.train()
    total_ce_loss = 0.0
    total_sparse_loss = 0.0
    total_loss = 0.0
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    if sparse_optimizer is not None and hasattr(sparse_optimizer, 'step_lambda'):
        sparse_optimizer.step_lambda()

    if sparse_optimizer is not None and sparse_lambda > 0:
        gate_params = [p for n, p in model.named_parameters() if "sora" in n and "gate" in n]
        gate_params_total = sum(p.numel() for p in gate_params)
    else:
        gate_params = []
        gate_params_total = 0

    optimizer.zero_grad(set_to_none=True)
    if sparse_optimizer is not None: sparse_optimizer.zero_grad(set_to_none=True)

    for step, (inputs_dict, labels) in enumerate(tqdm(loader, desc="train", leave=False, disable=True)):

        labels = labels.to(device=device, non_blocking=True)
        for k in inputs_dict.keys():
            if inputs_dict[k].is_floating_point():
                inputs_dict[k] = inputs_dict[k].to(device=device, dtype=model_dtype, non_blocking=True)
            else:
                inputs_dict[k] = inputs_dict[k].to(device=device, non_blocking=True)

        outputs = model(labels=labels, **inputs_dict)
        ce_loss = outputs["loss"]
        loss = ce_loss / accumulation_steps

        loss.backward()

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
            optimizer.step()
            if sparse_optimizer is not None: sparse_optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            if sparse_optimizer is not None: sparse_optimizer.zero_grad(set_to_none=True)

        total_ce_loss += ce_loss.item()
        total_loss += (loss.item() * accumulation_steps)

    sparse_loss_val = 0.0
    if gate_params_total > 0:
        with torch.no_grad():
            sparse_loss = sum(torch.sum(torch.abs(p)) for p in gate_params)
            sparse_loss_val = sparse_loss.item() / gate_params_total

    n = max(len(loader), 1)
    return {"ce_loss": total_ce_loss / n, "sparse_loss": sparse_loss_val, "total_loss": total_loss / n}


def compute_gate_sparsity(model):
    total = 0
    zeros = 0
    for n, p in model.named_parameters():
        if "sora" in n and "gate" in n:
            total += p.numel()
            zeros += (p.data == 0).sum().item()
    return zeros, total


def evaluate(model, loader):
    model.eval()
    preds = []
    true_labels = []

    with torch.no_grad():
        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype

        for inputs_dict, labels in tqdm(loader, desc="eval", leave=False, disable=True):
            labels = labels.to(device=device, non_blocking=True)
            for k in inputs_dict.keys():
                if inputs_dict[k].is_floating_point():
                    inputs_dict[k] = inputs_dict[k].to(device=device, dtype=model_dtype, non_blocking=True)
                else:
                    inputs_dict[k] = inputs_dict[k].to(device=device, non_blocking=True)

            outputs = model(**inputs_dict)
            logits = outputs["logits"]
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pred)
            true_labels.extend(labels.cpu().numpy())

    return accuracy_score(true_labels, preds)
