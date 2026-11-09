# Responsabilidade: Constrói otimizadores e scheduler conforme o modo de adaptação. Família
# CLIP.
#
# Principais funções: build_optimizer, build_scheduler.

"""CLIP model support — optimization."""

from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from fl_serverpuma.models.clip.sora_optimization import GateSparsifier
from fl_serverpuma.models.clip.config import SORA_MODES

def build_optimizer(model, config):
    """
    Configuração da Otimização.
    Instancia o AdamW tradicional e o SparseAdamW (para SoRA), lidando com a 
    separação de parâmetros de portão (gates) e adaptadores.
    """
    optimizer_config = config["optimizer"]
    sora_config = config["model"].get("sora", {})
    mode = config["model"]["lora"]["mode"]
    is_sora = mode in SORA_MODES

    if is_sora:
        gate_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "sora" in name and "gate" in name:
                gate_params.append(param)
            else:
                other_params.append(param)

        if not other_params and not gate_params:
            raise ValueError("No trainable parameters found.")

        sparse_lr = sora_config.get("sparse_lr") or optimizer_config["lr"]

        # --- OTIMIZAÇÃO FL: Otimizador Unificado ---
        # Antes: 2 otimizadores separados (AdamW + SparseAdamW)
        #   → 2× zero_grad(), 2× step(), 2× estado interno (momentum/variância)
        # Agora: 1 AdamW com 2 param_groups + GateSparsifier leve
        #   → 1× zero_grad(), 1× step(), 1× estado + softshrink
        #
        # A matemática do SoRA (Proximal Gradient Descent) é preservada:
        #   Passo 1: AdamW atualiza TODOS os params (adapters + gates) → grupo unificado
        #   Passo 2: GateSparsifier aplica softshrink nos gates → proximal step do L1
        unified_optimizer = AdamW([
            {'params': other_params, 'lr': optimizer_config["lr"],
             'weight_decay': optimizer_config["weight_decay"]},
            {'params': gate_params, 'lr': sparse_lr, 'weight_decay': 0.0},
        ])

        if mode == "with_sora_schedule":
            lambda_schedule = sora_config.get("lambda_schedule")
            max_lambda = sora_config.get("max_lambda")
            lambda_num = sora_config.get("lambda_num")
        else:
            lambda_schedule = None
            max_lambda = None
            lambda_num = None

        # GateSparsifier: aplica APENAS o proximal step (softshrink) nos gates.
        # Substitui o SparseAdamW completo, que era um AdamW inteiro + softshrink.
        gate_sparsifier = GateSparsifier(
            gate_params=gate_params,
            sparse_lambda=sora_config.get("sparse_lambda_2", 3e-4),
            gate_lr=sparse_lr,
            lambda_schedule=lambda_schedule,
            max_lambda=max_lambda,
            lambda_num=lambda_num,
        )

        # print(f"Gate params: {sum(p.numel() for p in gate_params):,}")
        # print(f"Other trainable params: {sum(p.numel() for p in other_params):,}")
        return unified_optimizer, gate_sparsifier

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found. Check the LoRA and classifier setup.")

    return AdamW(
        trainable_params,
        lr=optimizer_config["lr"],
        weight_decay=optimizer_config["weight_decay"],
    ), None


def build_scheduler(optimizer, config):
    """Controle de Decay. Define a estratégia de redução da taxa de aprendizado."""
    scheduler_config = config["scheduler"]
    return StepLR(
        optimizer,
        step_size=scheduler_config["step_size"],
        gamma=scheduler_config["gamma"],
    )
