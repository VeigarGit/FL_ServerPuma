"""Shim de retrocompatibilidade: reexporta o núcleo SoRA de peft_common.sora."""

import sys
from pathlib import Path

_SYSTEM_DIR = str(Path(__file__).resolve().parents[1])
if _SYSTEM_DIR not in sys.path:
    sys.path.insert(0, _SYSTEM_DIR)

from peft_common.sora import (  # noqa: E402
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

__all__ = [
    "GateSparsifier",
    "SORA_MODES",
    "SoRALinear",
    "SoRAWrappedLinear",
    "SparseAdamW",
    "apply_peft_lora_modules",
    "apply_sora_modules",
    "build_sora_gate_sparsifier",
    "get_trainable_state_dict",
    "pre_prune_whole_model",
    "prune_sora_iterative",
    "prune_sora_to_lora_and_report",
    "re_freeze_vision_model",
    "set_trainable_state_dict",
]
