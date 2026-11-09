# Responsabilidade: Constrói otimizadores e scheduler conforme o modo de adaptação. Família
# SLM.
#
# Principais funções: build_optimizer, build_scheduler.

"""SLM model support — optimization."""

from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from fl_serverpuma.models.slm.sora_optimization import GateSparsifier
from fl_serverpuma.models.slm.config import SORA_MODES

def build_optimizer(model, config):
    optimizer_config = config["optimizer"]
    sora_config = config["model"].get("sora", {})
    mode = config["model"]["lora"]["mode"]
    is_sora = mode in SORA_MODES

    if is_sora:
        gate_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            if "sora" in name and "gate" in name: gate_params.append(param)
            else: other_params.append(param)
        sparse_lr = sora_config.get("sparse_lr") or optimizer_config["lr"]
        unified_optimizer = AdamW([
            {'params': other_params, 'lr': optimizer_config["lr"], 'weight_decay': optimizer_config["weight_decay"]},
            {'params': gate_params, 'lr': sparse_lr, 'weight_decay': 0.0},
        ])

        lambda_schedule = sora_config.get("lambda_schedule") if mode == "with_sora_schedule" else None
        max_lambda = sora_config.get("max_lambda") if mode == "with_sora_schedule" else None
        lambda_num = sora_config.get("lambda_num") if mode == "with_sora_schedule" else None

        gate_sparsifier = GateSparsifier(
            gate_params=gate_params,
            sparse_lambda=sora_config.get("sparse_lambda_2", 3e-4),
            gate_lr=sparse_lr,
            lambda_schedule=lambda_schedule,
            max_lambda=max_lambda,
            lambda_num=lambda_num,
        )
        return unified_optimizer, gate_sparsifier

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return AdamW(trainable_params, lr=optimizer_config["lr"], weight_decay=optimizer_config["weight_decay"]), None


def build_scheduler(optimizer, config):
    scheduler_config = config["scheduler"]
    return StepLR(optimizer, step_size=scheduler_config["step_size"], gamma=scheduler_config["gamma"])
