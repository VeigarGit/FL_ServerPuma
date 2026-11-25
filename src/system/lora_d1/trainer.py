"""Loop de treino e avaliação para o modelo DecisionD1 (LiquidAI/d1-2B)."""

import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def decision_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    brier_weight: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Perda híbrida Cross-Entropy + Brier Score para calibração de probabilidades."""
    weight = torch.as_tensor(brier_weight, device=logits.device, dtype=torch.float32)
    if not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("brier_weight deve ser finito e >= 0")
    if reduction not in {"none", "sum", "mean"}:
        raise ValueError("reduction deve ser none, sum ou mean")

    logits = logits.float()
    if weight.ndim == 0 and weight == 0:
        return F.cross_entropy(logits, targets, reduction=reduction)

    ce = F.cross_entropy(logits, targets, reduction="none")
    probabilities = logits.softmax(dim=-1)
    target_one_hot = F.one_hot(targets, logits.shape[-1]).to(probabilities.dtype)
    brier = (probabilities - target_one_hot).square().sum(dim=-1)

    loss = ce + weight * brier
    if reduction == "none":
        return loss
    return getattr(loss, reduction)()


def _move_batch_to_device(batch: dict, device: torch.device, model_dtype: torch.dtype) -> dict:
    """Transfere tensores de lote para o dispositivo alvo respeitando dtypes e host_inputs."""
    moved = {}
    for key, val in batch.items():
        if key == "image_grid_thw" or not isinstance(val, torch.Tensor):
            moved[key] = val
        elif val.is_floating_point():
            moved[key] = val.to(device, dtype=model_dtype, non_blocking=True)
        else:
            moved[key] = val.to(device, non_blocking=True)
    return moved


def train_epoch_d1(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    device: torch.device,
    brier_weight: float = 1.0,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    sparse_optimizer: object | None = None,
) -> tuple[float, float]:
    """Executa 1 época de fine-tuning local para o LiquidAI/d1-2B (LoRA/QLoRA e SoRA/QSoRA)."""
    model.train()
    total_loss_tensor = torch.zeros((), device=device, dtype=torch.float32)
    correct_tensor = torch.zeros((), device=device, dtype=torch.long)
    total_samples = 0

    if sparse_optimizer is not None and hasattr(sparse_optimizer, "step_lambda"):
        sparse_optimizer.step_lambda()

    optimizer.zero_grad(set_to_none=True)
    use_autocast = device.type == "cuda"
    model_dtype = getattr(model, "compute_dtype", torch.bfloat16)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    for step, batch in enumerate(train_loader):
        batch = _move_batch_to_device(batch, device, model_dtype)
        targets = batch["targets"]
        batch_size_curr = targets.size(0)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_autocast):
            logits = model(batch)
            loss = decision_loss(logits, targets, brier_weight=brier_weight)
            loss_scaled = loss / grad_accum_steps

        if loss_scaled.requires_grad:
            loss_scaled.backward()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct_tensor += (preds == targets).sum()
            total_loss_tensor += loss.detach().float() * batch_size_curr
        total_samples += batch_size_curr

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
            if trainable_params and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
            optimizer.step()
            if sparse_optimizer is not None:
                sparse_optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    if scheduler is not None:
        scheduler.step()

    denom = max(1, total_samples)
    return total_loss_tensor.item() / denom, 100.0 * correct_tensor.item() / denom


@torch.no_grad()
def evaluate_d1(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    brier_weight: float = 1.0,
) -> tuple[float, float]:
    """Avalia acurácia (%) e perda média do DecisionD1."""
    model.eval()
    total_loss_tensor = torch.zeros((), device=device, dtype=torch.float32)
    correct_tensor = torch.zeros((), device=device, dtype=torch.long)
    total_samples = 0
    use_autocast = device.type == "cuda"
    model_dtype = getattr(model, "compute_dtype", torch.bfloat16)

    for batch in eval_loader:
        batch = _move_batch_to_device(batch, device, model_dtype)
        targets = batch["targets"]
        batch_size_curr = targets.size(0)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_autocast):
            logits = model(batch)
            loss = decision_loss(logits, targets, brier_weight=brier_weight)

        preds = logits.argmax(dim=-1)
        correct_tensor += (preds == targets).sum()
        total_loss_tensor += loss.detach().float() * batch_size_curr
        total_samples += batch_size_curr

    denom = max(1, total_samples)
    return 100.0 * correct_tensor.item() / denom, total_loss_tensor.item() / denom
