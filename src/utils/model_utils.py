import os
import tempfile
from pathlib import Path

import torch


def save_state_dict_atomic(state_dict: dict, path: Path) -> None:
    """Publish complete weights readable by both container and host users."""
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as tmp_file:
            torch.save(state_dict, tmp_file)
            tmp_file.flush()
            # mkstemp defaults to 0600; Docker and host may have different UIDs.
            os.fchmod(tmp_file.fileno(), 0o644)
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def dequantization(global_state):
    """Dequantizes a state dictionary."""
    dequantized_state_dict = {}
    for k, v in global_state.items():
        if isinstance(v, dict) and v.get('dtype') == 'quantized_int8':
            # Recupera tensores quantizados
            scale = v['scale']
            dequantized_state_dict[k] = v['weights'].float() * scale
        else:
            # Mantém tensores normais
            dequantized_state_dict[k] = v
    return dequantized_state_dict

def quantization(state_dict):
    """Quantizes a state dictionary to int8."""
    quantized_state_dict = {}
    keys = list(state_dict.keys())
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            scale = torch.max(torch.abs(v)) / 127.0
            quantized_weights = torch.clamp((v / scale).round(), -128, 127).to(torch.int8)
            quantized_state_dict[k] = {
                'dtype': 'quantized_int8',
                'scale': scale,
                'weights': quantized_weights
            }
        else:
            quantized_state_dict[k] = v
    return quantized_state_dict
