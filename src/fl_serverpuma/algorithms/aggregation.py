# Responsabilidade: Agrega estados parciais com pesos por cliente; recebe dispositivo e
# máscaras explicitamente, sem depender do estado completo do servidor.
#
# Principais funções: aggregate_models.

"""Weighted aggregation."""

import logging
import torch
from fl_serverpuma.algorithms.alignment import _are_compatible_with_pruning
from fl_serverpuma.algorithms.alignment import _align_to_global_structure
from fl_serverpuma.algorithms.alignment import _get_mask

logger = logging.getLogger(__name__)

def aggregate_models(model_list, global_model, client_weights=None, *, device, client_masks):
    agg_state = global_model.copy()

    if client_weights is None:
        client_weights = [1.0] * len(model_list)

    total_weight = sum(client_weights)
    normalized_weights = [w / total_weight for w in client_weights]

    for key in global_model.keys():
        weighted_sum = None
        weight_mask = None
        global_val = global_model[key]

        for idx, model_state in enumerate(model_list):
            if key in model_state:
                local_val = model_state[key]

                if _are_compatible_with_pruning(local_val, global_val, key):
                    if not isinstance(global_val, torch.Tensor):
                        global_val = torch.tensor(global_val, device=device)

                    if not isinstance(local_val, torch.Tensor):
                        local_val = torch.tensor(local_val, device=global_val.device)
                    else:
                        local_val = local_val.to(global_val.device)

                    aligned_val = _align_to_global_structure(
                        local_val, global_val, key, model_state, idx, client_masks
                    )

                    mask = _get_mask(local_val, global_val)
                    weight = normalized_weights[idx]

                    if weighted_sum is None:
                        weighted_sum = torch.zeros_like(global_val)
                        weight_mask = torch.zeros_like(global_val)

                    weighted_sum += aligned_val * weight
                    weight_mask += mask * weight

        if weighted_sum is not None:
            valid_mask = weight_mask > 0
            agg_state[key] = global_val.clone()
            agg_state[key][valid_mask] = weighted_sum[valid_mask] / weight_mask[valid_mask]

    return agg_state
