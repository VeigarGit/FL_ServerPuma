"""Núcleo compartilhado de adaptadores esparsos SoRA/PEFT e poda de rank."""

from .pld import ProgressiveLayerDropper, apply_pld_runtime
from .sora import (
    SORA_MODES,
    GateSparsifier,
    SoRALinear,
    SoRAWrappedLinear,
    SparseAdamW,
    apply_peft_lora_modules,
    apply_sora_modules,
    build_sora_gate_sparsifier,
    get_trainable_state_dict,
    pre_prune_whole_model,
    prune_sora_iterative,
    prune_sora_to_lora_and_report,
    re_freeze_vision_model,
    set_trainable_state_dict,
)
from .sora_utils import reduce_sora_state_dict_rank

__all__ = [
    "GateSparsifier",
    "ProgressiveLayerDropper",
    "SORA_MODES",
    "SoRALinear",
    "SoRAWrappedLinear",
    "SparseAdamW",
    "apply_peft_lora_modules",
    "apply_pld_runtime",
    "apply_sora_modules",
    "build_sora_gate_sparsifier",
    "get_trainable_state_dict",
    "pre_prune_whole_model",
    "prune_sora_iterative",
    "prune_sora_to_lora_and_report",
    "re_freeze_vision_model",
    "reduce_sora_state_dict_rank",
    "set_trainable_state_dict",
]
