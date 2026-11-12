"""
QLoRA (Quantized Low-Rank Adaptation) para o pipeline federado de SLM.

Resumo da teoria aplicada ao FL:
  * Backbone: carregado em 4 bits NF4 + Dupla Quantização (BitsAndBytes), com
    computação em 16 bits (bf16, ou fp16 se a GPU não suportar bf16).
  * O backbone fica 100% congelado. Apenas os adaptadores LoRA (16 bits) e a head
    de classificação recebem gradiente.
  * Comunicação: SOMENTE o state_dict dos adaptadores (get/set_peft_model_state_dict)
    + a head trafegam pela rede. O backbone de 4 bits nunca é transmitido nem agregado.

Formato das chaves trafegadas (wire format):
  * Adaptadores: "slm_model." + chave canônica do PEFT (sem o nome do adapter),
    ex.: "slm_model.base_model.model.language_model.layers.3.self_attn.q_proj.lora_A.weight"
  * Head:       "classifier.weight", "classifier.bias"
  Os padrões ".layers.N." e ".lora_A"/".lora_B" são mantidos, então o PaCA e o
  rank adaptativo do servidor funcionam sem alteração.

Ativação: flag `--qlora` (run.sh / client.py / server.py), apenas com `--strategy lora`.
"""
import logging

import torch

logger = logging.getLogger(__name__)

HEAD_PREFIX = "classifier."
PEFT_PREFIX = "slm_model."
ADAPTER_NAME = "default"
DEFAULT_QLORA_CONFIG = {
    "quant_type": "nf4",
    "double_quant": True,
    "compute_dtype": "bfloat16",
    # AdamW FP32 nos ~8.8M parâmetros treináveis (~67 MB/cliente) evita
    # instabilidade dos estados 8-bit em rodadas FL Non-IID
    "optimizer": "adamw",
    "prepare_kbit": False,
}


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
def is_qlora_enabled(config) -> bool:
    return bool(config and config.get("model", {}).get("qlora", {}).get("enabled", False))


def enable_qlora_in_config(config, strategy: str) -> None:
    """Liga o QLoRA no dicionário de config (chamado por client/server a partir da flag --qlora)."""
    if strategy != "lora":
        raise ValueError(
            f"--qlora só é suportado com --strategy lora (recebido: '{strategy}'). "
            "SoRA/AdaLoRA sobre backbone 4-bit não estão implementados."
        )
    qcfg = {**DEFAULT_QLORA_CONFIG, **config["model"].get("qlora", {})}
    qcfg["enabled"] = True
    config["model"]["qlora"] = qcfg


def resolve_compute_dtype(qcfg) -> torch.dtype:
    requested = getattr(torch, qcfg.get("compute_dtype", "bfloat16"))
    if requested == torch.bfloat16 and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        logger.warning("[QLoRA] bf16 indisponível nesta GPU, usando float16 como compute dtype.")
        return torch.float16
    return requested


def build_bnb_config(qcfg):
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=qcfg.get("quant_type", "nf4"),
        bnb_4bit_use_double_quant=qcfg.get("double_quant", True),
        bnb_4bit_compute_dtype=resolve_compute_dtype(qcfg),
    )


# ---------------------------------------------------------------------------
# Construção do modelo
# ---------------------------------------------------------------------------
def _device_map_for(device) -> dict:
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("[QLoRA] BitsAndBytes 4-bit requer GPU CUDA.")
    return {"": device.index if device.index is not None else torch.cuda.current_device()}


def load_quantized_backbone(model_name: str, qcfg, device):
    """Carrega o SLM com backbone em 4 bits (NF4 + Double Quant)."""
    from transformers import AutoModel

    compute_dtype = resolve_compute_dtype(qcfg)
    backbone = AutoModel.from_pretrained(
        model_name,
        quantization_config=build_bnb_config(qcfg),
        dtype=compute_dtype,
        device_map=_device_map_for(device),
        attn_implementation="sdpa",
    )
    logger.info(
        f"[QLoRA] Backbone carregado em 4-bit ({qcfg.get('quant_type')}, "
        f"double_quant={qcfg.get('double_quant')}, compute={compute_dtype})"
    )
    return backbone, compute_dtype


def apply_qlora(model, lora_config, qcfg, compute_dtype):
    """Congela o backbone 4-bit e injeta adaptadores LoRA em 16 bits (in-place em model.slm_model)."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # Nota: prepare_model_for_kbit_training do PEFT converte embeddings/LayerNorms (311M params no Qwen3-VL)
    # para fp32, o que foi pensado para evitar underflow em fp16 mas desperdiça ~600 MB/cliente em bf16.
    # Em bf16 (mesmo range dinâmico do fp32), habilitamos gradient checkpointing + input grads mantendo bf16.
    if qcfg.get("prepare_kbit", False) or compute_dtype == torch.float16:
        model.slm_model = prepare_model_for_kbit_training(
            model.slm_model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    else:
        if hasattr(model.slm_model, "gradient_checkpointing_enable"):
            model.slm_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if hasattr(model.slm_model, "enable_input_require_grads"):
            model.slm_model.enable_input_require_grads()

    # Congelamento explícito de 100% do backbone (antes da injeção PEFT)
    for param in model.slm_model.parameters():
        param.requires_grad = False

    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["alpha"],
        target_modules=lora_config["target_modules"],
        lora_dropout=lora_config["dropout"],
        bias=lora_config["bias"],
    )
    # autocast_adapter_dtype=False: impede o upcast automático para fp32 -> adaptadores em 16 bits
    model.slm_model = get_peft_model(model.slm_model, peft_config, autocast_adapter_dtype=False)
    for name, param in model.slm_model.named_parameters():
        if "lora_" in name:
            param.data = param.data.to(compute_dtype)

    # Head de classificação em 16 bits, treinável, no mesmo device do backbone
    device = next(p.device for p in model.slm_model.parameters())
    model.classifier.to(device=device, dtype=compute_dtype)
    for param in model.classifier.parameters():
        param.requires_grad = True

    model.compute_dtype = compute_dtype
    verify_qlora_freezing(model)
    return model


def verify_qlora_freezing(model) -> None:
    """Garante que apenas LoRA/head são treináveis e que nenhum peso 4-bit recebe gradiente."""
    offenders = [
        n for n, p in model.named_parameters()
        if p.requires_grad and "lora_" not in n and not n.startswith(HEAD_PREFIX)
    ]
    quantized_trainable = [
        n for n, p in model.named_parameters()
        if p.requires_grad and p.__class__.__name__ == "Params4bit"
    ]
    if offenders or quantized_trainable:
        raise RuntimeError(f"[QLoRA] Parâmetros do backbone treináveis: {(offenders + quantized_trainable)[:5]}")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_4bit = sum(1 for p in model.parameters() if p.__class__.__name__ == "Params4bit")
    lora_dtypes = {p.dtype for n, p in model.named_parameters() if "lora_" in n}
    logger.info(f"[QLoRA] Treináveis: {n_train:,} | Tensores 4-bit congelados: {n_4bit} | dtype LoRA: {lora_dtypes}")


# ---------------------------------------------------------------------------
# Otimizador
# ---------------------------------------------------------------------------
def build_qlora_optimizer(model, config):
    optimizer_config = config["optimizer"]
    params = [p for p in model.parameters() if p.requires_grad]
    name = config["model"].get("qlora", {}).get("optimizer", "paged_adamw_8bit")
    if name == "paged_adamw_8bit":
        import bitsandbytes as bnb

        return bnb.optim.PagedAdamW8bit(params, lr=optimizer_config["lr"], weight_decay=optimizer_config["weight_decay"])
    return torch.optim.AdamW(params, lr=optimizer_config["lr"], weight_decay=optimizer_config["weight_decay"])


# ---------------------------------------------------------------------------
# Comunicação federada: extração / injeção do state_dict dos adaptadores
# ---------------------------------------------------------------------------
def _canonical_to_param_name(canonical_key: str) -> str:
    """'...q_proj.lora_A.weight' -> '...q_proj.lora_A.default.weight' (nome em named_parameters)."""
    for tag in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", "lora_magnitude_vector"):
        marker = f".{tag}."
        if marker in canonical_key:
            return canonical_key.replace(marker, f".{tag}.{ADAPTER_NAME}.", 1)
    return canonical_key


def get_qlora_state_dict(model, trainable_only: bool = True) -> dict:
    """
    Extrai SOMENTE os adaptadores LoRA (via get_peft_model_state_dict) + head, em 16 bits na CPU.
    trainable_only=True respeita o PaCA (não envia adaptadores de camadas congeladas no round).
    """
    from peft import get_peft_model_state_dict

    compute_dtype = getattr(model, "compute_dtype", torch.bfloat16)
    params = dict(model.slm_model.named_parameters())
    state = {}
    for key, tensor in get_peft_model_state_dict(model.slm_model, adapter_name=ADAPTER_NAME).items():
        param = params.get(_canonical_to_param_name(key))
        if trainable_only and param is not None and not param.requires_grad:
            continue
        state[PEFT_PREFIX + key] = tensor.detach().to("cpu", dtype=compute_dtype).clone()
    for key, tensor in model.classifier.state_dict().items():
        state[HEAD_PREFIX + key] = tensor.detach().to("cpu", dtype=compute_dtype).clone()

    assert_adapter_only_payload(state)
    return state


def set_qlora_state_dict(model, state: dict) -> None:
    """Injeta adaptadores (set_peft_model_state_dict) + head. Suporta mudança de rank (rank adaptativo)."""
    from peft import set_peft_model_state_dict

    try:
        from utils.fl_math import resize_model_to_pruned
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).resolve().parents[2]))
        from utils.fl_math import resize_model_to_pruned

    assert_adapter_only_payload(state)
    peft_state = {k[len(PEFT_PREFIX):]: v for k, v in state.items() if k.startswith(PEFT_PREFIX)}
    head_state = {k[len(HEAD_PREFIX):]: v for k, v in state.items() if k.startswith(HEAD_PREFIX)}

    # 1) Rank adaptativo: redimensiona as matrizes LoRA cujo shape mudou
    params = dict(model.slm_model.named_parameters())
    to_resize = {}
    for key, tensor in peft_state.items():
        pname = _canonical_to_param_name(key)
        if pname in params and params[pname].shape != tensor.shape:
            to_resize[pname] = tensor
    if to_resize:
        resize_model_to_pruned(model.slm_model, to_resize)

    # 2) Injeção dos pesos dos adaptadores (API oficial do PEFT)
    if peft_state:
        result = set_peft_model_state_dict(model.slm_model, peft_state, adapter_name=ADAPTER_NAME)
        unexpected = getattr(result, "unexpected_keys", [])
        if unexpected:
            logger.warning(f"[QLoRA] {len(unexpected)} chaves inesperadas ignoradas (ex.: {unexpected[:2]})")

    # 3) Head de classificação
    if head_state:
        with torch.no_grad():
            for key, tensor in head_state.items():
                target = getattr(model.classifier, key)
                target.data.copy_(tensor.to(device=target.device, dtype=target.dtype))


def assert_adapter_only_payload(state: dict) -> None:
    """Proteção: o backbone 4-bit NUNCA pode ir para a rede."""
    for key, value in state.items():
        is_adapter = key.startswith(PEFT_PREFIX) and "lora_" in key
        if not (is_adapter or key.startswith(HEAD_PREFIX)):
            raise RuntimeError(f"[QLoRA] Chave fora dos adaptadores no payload: {key}")
        if isinstance(value, torch.Tensor) and value.dtype == torch.uint8:
            raise RuntimeError(f"[QLoRA] Tensor quantizado 4-bit detectado no payload: {key}")

