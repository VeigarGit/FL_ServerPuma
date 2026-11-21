"""Shim de retrocompatibilidade: reexporta reduce_sora_state_dict_rank de peft_common."""

import sys
from pathlib import Path

_SYSTEM_DIR = str(Path(__file__).resolve().parents[1])
if _SYSTEM_DIR not in sys.path:
    sys.path.insert(0, _SYSTEM_DIR)

from peft_common.sora_utils import reduce_sora_state_dict_rank  # noqa: E402

__all__ = ["reduce_sora_state_dict_rank"]
