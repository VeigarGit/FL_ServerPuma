"""Suporte a QLoRA 4-bit NF4 e isolamento de payload para o LiquidAI/d1-2B."""

import logging
import re
from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModel, BitsAndBytesConfig

logger = logging.getLogger(__name__)

LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|w1|w2|w3)"
SKIP_QUANT_MODULES = (
    "vision_tower",
    "model.vision_tower",
    "patchifier",
    "multi_modal_projector",
    "model.multi_modal_projector",
    "lm_head",
)

DEFAULT_QLORA_CONFIG: dict[str, Any] = {
    "quant_type": "nf4",
    "double_quant": True,
    "compute_dtype": "bfloat16",
    "optimizer": "adamw",
    "prepare_kbit": False,
}


def _resolve_lora_targets(targets: str | list[str] | None) -> str | list[str]:
    """Restringe os módulos-alvo do LoRA ao language_model do d1-2B."""
    if not targets:
        return LORA_TARGETS
    if isinstance(targets, str):
        return targets
    joined = "|".join(str(t) for t in targets)
    return rf".*language_model.*\.({joined})"


def make_bnb_config(qcfg: dict[str, Any] | None = None) -> BitsAndBytesConfig:
    """Configuração 4-bit NF4 preservando patchifier/vision_tower/lm_head em 16 bits."""
    cfg = {**DEFAULT_QLORA_CONFIG, **(qcfg or {})}
    compute_dtype = getattr(torch, cfg.get("compute_dtype", "bfloat16"))
    if compute_dtype == torch.bfloat16 and not (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ):
        logger.warning("[d1 QLoRA] bf16 indisponível nesta GPU; usando float16.")
        compute_dtype = torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.get("quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(cfg.get("double_quant", True)),
        bnb_4bit_compute_dtype=compute_dtype,
        llm_int8_skip_modules=list(SKIP_QUANT_MODULES),
    )


def load_quantized_d1_backbone(
    model_name: str,
    revision: str,
    device: torch.device | str = "cuda",
    qcfg: dict[str, Any] | None = None,
) -> nn.Module:
    """Carrega o backbone LiquidAI/d1-2B em 4-bit NF4 congelado."""
    bnb_config = make_bnb_config(qcfg)
    target_device = str(device) if isinstance(device, torch.device) else device
    device_map = {"": target_device} if target_device.startswith("cuda") else "auto"

    backbone = AutoModel.from_pretrained(
        model_name,
        revision=revision,
        quantization_config=bnb_config,
        device_map=device_map,
        dtype=bnb_config.bnb_4bit_compute_dtype,
        trust_remote_code=True,
    )
    backbone.eval()
    return backbone


def apply_qlora(
    backbone: nn.Module,
    lora_config: dict[str, Any],
    *,
    gradient_checkpointing: bool = True,
    prepare_kbit: bool = False,
) -> nn.Module:
    """Congela o backbone 4-bit e injeta adaptadores LoRA evitando cast FP32 desnecessário."""
    for param in backbone.parameters():
        param.requires_grad = False

    if prepare_kbit:
        backbone = prepare_model_for_kbit_training(
            backbone,
            use_gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    else:
        if hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()
        if gradient_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if hasattr(backbone, "config"):
            backbone.config.use_cache = False

    targets = _resolve_lora_targets(lora_config.get("target_modules", LORA_TARGETS))
    peft_config = LoraConfig(
        r=int(lora_config.get("r", 8)),
        lora_alpha=int(lora_config.get("alpha", 16)),
        lora_dropout=float(lora_config.get("dropout", 0.05)),
        bias=str(lora_config.get("bias", "none")),
        target_modules=targets,
    )
    return get_peft_model(backbone, peft_config)


def verify_qlora_freezing(model: nn.Module) -> None:
    """Garante que nenhum tensor 4-bit possua gradiente ativo no d1-2B."""
    trainable_count = 0
    params4bit_trainable = 0
    params4bit_total = 0

    for name, param in model.named_parameters():
        param_type = type(param).__name__
        is_4bit = "Params4bit" in param_type or "int4" in param_type.lower()
        if is_4bit:
            params4bit_total += 1
            if param.requires_grad:
                params4bit_trainable += 1
                logger.error(f"[d1 QLoRA] Parâmetro 4-bit com gradiente ativo: {name}")
        elif param.requires_grad:
            trainable_count += param.numel()

    if params4bit_trainable > 0:
        raise RuntimeError(
            f"Falha de isolamento QLoRA d1: {params4bit_trainable} tensores 4-bit treináveis!"
        )

    logger.info(
        f"[d1 QLoRA] Freezing OK: {params4bit_total} tensores 4-bit congelados | "
        f"Treináveis: {trainable_count:,}"
    )


def build_d1_optimizer(
    model: nn.Module,
    run_config: dict[str, Any],
    is_qlora: bool = False,
) -> torch.optim.Optimizer:
    """Constrói AdamW FP32 (estável para Non-IID) ou PagedAdamW8bit se configurado."""
    opt_cfg = run_config.get("optimizer", {})
    lr = float(opt_cfg.get("lr", opt_cfg.get("lora_lr", 5e-5)))
    weight_decay = float(opt_cfg.get("weight_decay", 0.01))
    params = [p for p in model.parameters() if p.requires_grad]

    if is_qlora:
        qopt = (
            run_config.get("model", {})
            .get("qlora", {})
            .get("optimizer", DEFAULT_QLORA_CONFIG["optimizer"])
        )
        if qopt == "paged_adamw_8bit":
            import bitsandbytes as bnb

            return bnb.optim.PagedAdamW8bit(params, lr=lr, weight_decay=weight_decay)

    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def assert_adapter_only_payload(payload: dict[str, Any]) -> None:
    """Assegura que somente adaptadores LoRA/SoRA trafeguem pela rede federada."""
    allowed_pattern = re.compile(r".*(lora|sora)_.*")
    violations = [k for k in payload if not allowed_pattern.search(k)]
    if violations:
        raise ValueError(
            f"Violação de isolamento de rede d1! {len(violations)} tensores não-PEFT "
            f"encontrados no payload: {violations[:5]}"
        )
