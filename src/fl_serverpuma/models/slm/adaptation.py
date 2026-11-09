# Responsabilidade: Aplica SoRA e controla as camadas/gradientes habilitados por PaCA em tempo
# de execução. Família SLM.
#
# Principais funções: apply_sora, update_paca_runtime.

"""SLM model support — adaptation."""

import torch.nn as nn
from fl_serverpuma.models.slm.sora_layers import SoRAWrappedLinear

def apply_sora(model, lora_config, upper_k=None):
    target_modules = lora_config["target_modules"]
    r = lora_config["r"]
    alpha = lora_config["alpha"]
    dropout = lora_config["dropout"]

    base_lm = model.slm_model
    layers_container = _get_encoder_layers(model)

    total_layers = len(layers_container)
    modules_map = dict(base_lm.named_modules())
    replacements = []

    for name in list(modules_map.keys()):
        for target in target_modules:
            if not name.endswith(target):
                continue

            if upper_k is not None:
                if ".layers." not in name:
                    continue
                try:
                    layer_idx = int(name.split(".layers.")[1].split(".")[0])
                    if layer_idx < total_layers - upper_k:
                        continue 
                except:
                    continue

            parts = name.rsplit(".", 1)
            parent = modules_map[parts[0]] if len(parts) == 2 else base_lm
            child_name = parts[1] if len(parts) == 2 else parts[0]

            original_linear = getattr(parent, child_name)
            if isinstance(original_linear, nn.Linear):
                replacements.append((parent, child_name, original_linear))

    for parent, child_name, original_linear in replacements:
        wrapped = SoRAWrappedLinear(original_linear, r=r, lora_alpha=alpha, lora_dropout=dropout)
        setattr(parent, child_name, wrapped)

    print(f"SoRA aplicado a {len(replacements)} módulos (Últimas {upper_k or 'todas'} camadas)")


def _get_encoder_layers(model):
    base_lm = getattr(model, "slm_model", model)
    if hasattr(base_lm, 'language_model') and hasattr(base_lm.language_model, 'layers'):
        return base_lm.language_model.layers
    if hasattr(base_lm, 'model') and hasattr(base_lm.model, 'layers'):
        return base_lm.model.layers
    return getattr(base_lm, 'layers', [])


def _apply_paca_gradient_boundary(model, upper_k):
    layers = _get_encoder_layers(model)
    total_layers = len(layers)
    if upper_k is None or upper_k >= total_layers: return
    boundary_idx = total_layers - upper_k - 1

    def detach_hook(module, input, output):
        if not module.training: return output
        if isinstance(output, tuple):
            return (output[0].detach(),) + output[1:]
        return output.detach()

    handle = layers[boundary_idx].register_forward_hook(detach_hook)
    if not hasattr(model, '_paca_hooks'):
        model._paca_hooks = []
    model._paca_hooks.append(handle)


def update_paca_runtime(model, new_upper_k):
    layers = _get_encoder_layers(model)
    total_layers = len(layers)
    if hasattr(model, '_paca_hooks'):
        for handle in model._paca_hooks:
            handle.remove()
        model._paca_hooks = []
    if new_upper_k is not None:
        _apply_paca_gradient_boundary(model, new_upper_k)

    boundary_layer = 0 if (new_upper_k is None or new_upper_k >= total_layers) else total_layers - new_upper_k
    for name, param in model.named_parameters():
        if "sora" not in name and "lora" not in name: continue
        if ".layers." not in name: continue
        try:
            layer_idx = int(name.split(".layers.")[1].split(".")[0])
            if layer_idx < boundary_layer:
                if param.requires_grad: param.requires_grad = False
            else:
                if not param.requires_grad: param.requires_grad = True
        except:
            continue
