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
    """
    Função de perda híbrida do BonsaiToClef:
    Loss = CrossEntropy(logits, targets) + brier_weight * sum((probabilities - one_hot(target))^2)
    Calibra as probabilidades das 77 alternativas e penaliza previsões superconfiantes incorretas.
    """
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


def train_epoch_lfm(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    device: torch.device,
    brier_weight: float = 1.0,
    grad_accum_steps: int = 1,
) -> tuple[float, float]:
    """Loop de treino local de 1 época do DecisionLFM."""
    model.train()
    total_loss_tensor = torch.zeros((), device=device, dtype=torch.float32)
    correct_tensor = torch.zeros((), device=device, dtype=torch.long)
    total_samples = 0

    optimizer.zero_grad(set_to_none=True)
    use_autocast = device.type == "cuda"

    for step, batch in enumerate(train_loader):
        # Transfere tensores do batch para a GPU de forma assíncrona com pin_memory
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val.to(device, non_blocking=True)

        targets = batch["targets"]
        batch_size_curr = targets.size(0)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_autocast):
            logits = model(batch)
            loss = decision_loss(logits, targets, brier_weight=brier_weight)
            loss_scaled = loss / grad_accum_steps

        loss_scaled.backward()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct_tensor += (preds == targets).sum()
            total_loss_tensor += loss.detach().float() * batch_size_curr
        total_samples += batch_size_curr

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    if scheduler is not None:
        scheduler.step()

    denom = max(1, total_samples)
    avg_loss = total_loss_tensor.item() / denom
    accuracy = 100.0 * correct_tensor.item() / denom
    return avg_loss, accuracy


@torch.no_grad()
def evaluate_lfm(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    brier_weight: float = 1.0,
) -> tuple[float, float]:
    """Avaliação local ou no servidor do DecisionLFM em dados de teste."""
    model.eval()
    total_loss_tensor = torch.zeros((), device=device, dtype=torch.float32)
    correct_tensor = torch.zeros((), device=device, dtype=torch.long)
    total_samples = 0
    use_autocast = device.type == "cuda"

    for batch in eval_loader:
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val.to(device, non_blocking=True)

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
    avg_loss = total_loss_tensor.item() / denom
    accuracy = 100.0 * correct_tensor.item() / denom
    return accuracy, avg_loss
