import logging
import re
from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModel, BitsAndBytesConfig

logger = logging.getLogger(__name__)

VISION_SKIP_MODULES = ("vision_tower", "multi_modal_projector")
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|w1|w2|w3)"


def _resolve_lora_targets(targets: str | list[str] | None) -> str | list[str]:
    """Restringe os módulos-alvo do LoRA estritamente ao language_model (padrão BonsaiToClef)."""
    if not targets:
        return LORA_TARGETS
    if isinstance(targets, str):
        return targets
    joined = "|".join(str(t) for t in targets)
    return rf".*language_model.*\.({joined})"


def make_bnb_config(
    quant_type: str = "nf4",
    double_quant: bool = True,
    compute_dtype: torch.dtype = torch.bfloat16,
    skip_modules: tuple[str, ...] = VISION_SKIP_MODULES,
) -> BitsAndBytesConfig:
    """Configuração BitsAndBytes padrão para 4-bit NormalFloat com Dupla Quantização."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
        llm_int8_skip_modules=list(skip_modules) if skip_modules else None,
    )


def load_quantized_lfm_backbone(
    model_name: str,
    revision: str,
    device: torch.device | str = "cuda",
) -> nn.Module:
    """Carrega o backbone do LiquidAI/LFM3-VL-2B em 4-bit NF4 congelado."""
    bnb_config = make_bnb_config()
    target_device = str(device) if isinstance(device, torch.device) else device
    device_map = {"": target_device} if target_device.startswith("cuda") else "auto"

    backbone = AutoModel.from_pretrained(
        model_name,
        revision=revision,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
    )
    backbone.eval()
    return backbone


def _prepare_kbit_backbone(backbone: nn.Module) -> nn.Module:
    """Congela o backbone 4-bit e ativa gradient checkpointing seguro antes dos adaptadores."""
    for param in backbone.parameters():
        param.requires_grad = False
    return prepare_model_for_kbit_training(
        backbone,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )


def apply_qlora(backbone: nn.Module, lora_config: dict[str, Any]) -> nn.Module:
    """Prepara o backbone 4-bit e injeta adaptadores LoRA com gradient checkpointing seguro."""
    backbone = _prepare_kbit_backbone(backbone)

    rank = lora_config.get("r", 8)
    alpha = lora_config.get("alpha", 16)
    dropout = lora_config.get("dropout", 0.05)
    targets = _resolve_lora_targets(lora_config.get("target_modules", LORA_TARGETS))

    peft_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=targets,
    )
    return get_peft_model(backbone, peft_config)


def apply_qsora(
    backbone: nn.Module,
    lora_config: dict[str, Any],
    compute_dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Prepara o backbone 4-bit e injeta adaptadores SoRA em 16 bits (QSoRA) para LFM."""
    from lora_slm.sora import apply_sora_modules

    backbone = _prepare_kbit_backbone(backbone)
    targets = _resolve_lora_targets(lora_config.get("target_modules", LORA_TARGETS))
    n_wrapped = apply_sora_modules(
        backbone,
        target_pattern=targets,
        r=int(lora_config.get("r", 8)),
        lora_alpha=int(lora_config.get("alpha", 16)),
        lora_dropout=float(lora_config.get("dropout", 0.05)),
        adapter_dtype=compute_dtype,
    )
    logger.info(f"[LFM QSoRA] SoRA aplicado a {n_wrapped} módulos Linear4bit.")
    return backbone


def verify_qlora_freezing(model: nn.Module) -> None:
    """
    Verificação estrita de segurança: garante que 100% dos tensores 4-bit estejam congelados
    e que apenas adaptadores LoRA/SoRA e a DecisionHead tenham requires_grad=True.
    """
    trainable_count = 0
    params4bit_trainable = 0
    params4bit_total = 0
    offenders = []

    for name, param in model.named_parameters():
        param_type = type(param).__name__
        is_4bit = "Params4bit" in param_type or "int4" in param_type.lower()
        if is_4bit:
            params4bit_total += 1
            if param.requires_grad:
                params4bit_trainable += 1
                logger.error(f"[QLoRA ERROR] Parâmetro 4-bit com gradiente ativo: {name}")
        elif param.requires_grad:
            if not ("lora_" in name or ".sora." in name or name.startswith("head.")):
                offenders.append(name)
            trainable_count += param.numel()

    if params4bit_trainable > 0 or offenders:
        raise RuntimeError(
            f"Falha de isolamento QLoRA/QSoRA: {params4bit_trainable} tensores 4-bit e "
            f"{len(offenders)} tensores de backbone treináveis ({offenders[:5]})!"
        )

    logger.info(
        f"[QLoRA/QSoRA Freezing Check] Aprovado! {params4bit_total} tensores 4-bit congelados. "
        f"Total de parâmetros treináveis (Adaptadores + Head): {trainable_count:,}"
    )


def build_lfm_optimizer(
    model: nn.Module,
    run_config: dict[str, Any],
    is_qlora: bool = False,
) -> torch.optim.Optimizer:
    """
    Constrói otimizador para DecisionHead, LoRA/SoRA e portões SoRA (weight_decay=0.0),
    utilizando PagedAdamW8bit no modo QLoRA/QSoRA ou AdamW nativo em BF16.
    """
    opt_cfg = run_config.get("optimizer", {})
    sora_cfg = run_config.get("model", {}).get("sora", {})
    head_lr = float(opt_cfg.get("head_lr", 0.0003))
    lora_lr = float(opt_cfg.get("lora_lr", 0.00003))
    sparse_lr = float(sora_cfg.get("sparse_lr") or lora_lr)
    weight_decay = float(opt_cfg.get("weight_decay", 0.01))

    head_params = []
    gate_params = []
    lora_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "head" in name:
            head_params.append(param)
        elif "sora" in name and "gate" in name:
            gate_params.append(param)
        else:
            lora_params.append(param)

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lora_lr, "weight_decay": weight_decay})
    if gate_params:
        param_groups.append({"params": gate_params, "lr": sparse_lr, "weight_decay": 0.0})

    if is_qlora:
        try:
            import bitsandbytes as bnb

            logger.info("Instanciando bitsandbytes.optim.PagedAdamW8bit para LFM...")
            return bnb.optim.PagedAdamW8bit(param_groups)
        except ImportError:
            logger.warning("bitsandbytes não disponível para otimizador; usando AdamW padrão.")
            return torch.optim.AdamW(param_groups)

    return torch.optim.AdamW(param_groups)


def assert_adapter_only_payload(payload: dict[str, Any]) -> None:
    """
    Invariante crítico de rede federada: assegura que nenhum peso base do backbone
    seja serializado no payload de comunicação cliente-servidor.
    """
    allowed_pattern = re.compile(r"^(head\..*|.*(?:lora_|sora\.(?:lora_A|lora_B|gate)).*)$")
    violations = [k for k in payload if not allowed_pattern.search(k)]

    if violations:
        raise ValueError(
            f"Violação de isolamento de rede! {len(violations)} tensores base "
            f"encontrados no payload: {violations[:5]}"
        )
    for key, val in payload.items():
        if isinstance(val, torch.Tensor) and val.dtype == torch.uint8:
            raise ValueError(f"Tensor quantizado 4-bit detectado no payload LFM: {key}")
