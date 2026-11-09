# Responsabilidade: Alinha tensores locais/podados às dimensões globais e constrói máscaras de
# cobertura para agregação.
#
# Principais funções: _align_to_global_structure, _align_with_mask, _get_mask.

"""Alignment of partial and pruned parameter structures."""

import logging
import torch

logger = logging.getLogger(__name__)

def _are_compatible_with_pruning(local_val, global_val, key):
    if not hasattr(local_val, 'shape') or not hasattr(global_val, 'shape'):
        try:
            len_local = len(local_val) if hasattr(local_val, '__len__') else 1
            len_global = len(global_val) if hasattr(global_val, '__len__') else 1
            return len_local == len_global
        except TypeError:
            return True

    if local_val.ndim != global_val.ndim:
        return False

    if 'conv' in key and 'weight' in key:
        if local_val.ndim == 4:
            return local_val.shape[2:] == global_val.shape[2:]
    elif 'weight' in key and local_val.ndim == 2:
        return True

    return local_val.ndim == global_val.ndim


def _align_to_global_structure(local_val, global_val, key, model_state, client_idx, client_masks):
    if local_val.shape == global_val.shape:
        return local_val

    if client_idx in client_masks and key in client_masks[client_idx]:
        return _align_with_mask(local_val, global_val, client_masks[client_idx][key])

    if 'conv' in key and 'weight' in key and local_val.ndim == 4:
        return _align_conv_weights(local_val, global_val)
    elif ('weight' in key or 'lora_' in key or 'gate' in key) and local_val.ndim == 2:
        return _align_linear_weights(local_val, global_val)
    elif 'bias' in key and local_val.ndim == 1:
        return _align_bias(local_val, global_val)

    logger.warning(f"Não foi possível alinhar {key}. Usando valor local.")
    return local_val


def _align_with_mask(local_val, global_val, mask_info):
    aligned = torch.zeros_like(global_val)
    if 'indices' in mask_info:
        indices = mask_info['indices']
        if len(indices) == len(local_val):
            for i, idx in enumerate(indices):
                if idx is not None:
                    aligned[idx] = local_val[i]
    else:
        slices = tuple(slice(0, min(l, g)) for l, g in zip(local_val.shape, global_val.shape))
        aligned[slices] = local_val[slices]
    return aligned


def _get_mask(local_val, global_val):
    mask = torch.zeros_like(global_val)
    if local_val.ndim == 2:
        min_out = min(local_val.shape[0], global_val.shape[0])
        min_in = min(local_val.shape[1], global_val.shape[1])
        mask[:min_out, :min_in] = 1.0
    elif local_val.ndim == 1:
        min_len = min(local_val.shape[0], global_val.shape[0])
        mask[:min_len] = 1.0
    elif local_val.ndim == 4:
        min_out = min(local_val.shape[0], global_val.shape[0])
        min_in = min(local_val.shape[1], global_val.shape[1])
        mask[:min_out, :min_in, :, :] = 1.0
    return mask


def _align_conv_weights(local_val, global_val):
    if local_val.ndim != 4 or global_val.ndim != 4:
        return local_val

    local_out, local_in, kh, kw = local_val.shape
    global_out, global_in, kh_global, kw_global = global_val.shape

    if kh != kh_global or kw != kw_global:
        logger.warning("Kernel size incompatível para convolução")
        return local_val

    aligned = torch.zeros_like(global_val)
    min_out = min(local_out, global_out)
    min_in = min(local_in, global_in)
    aligned[:min_out, :min_in, :, :] = local_val[:min_out, :min_in, :, :]
    return aligned


def _align_linear_weights(local_val, global_val):
    if local_val.ndim != 2 or global_val.ndim != 2:
        return local_val

    local_out, local_in = local_val.shape
    global_out, global_in = global_val.shape

    aligned = torch.zeros_like(global_val)
    min_out = min(local_out, global_out)
    min_in = min(local_in, global_in)
    aligned[:min_out, :min_in] = local_val[:min_out, :min_in]
    return aligned


def _align_bias(local_val, global_val):
    if local_val.ndim != 1 or global_val.ndim != 1:
        return local_val

    aligned = torch.zeros_like(global_val)
    min_len = min(local_val.shape[0], global_val.shape[0])
    aligned[:min_len] = local_val[:min_len]
    return aligned
