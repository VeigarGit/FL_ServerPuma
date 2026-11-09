# Responsabilidade: Aplica SoRA e controla as camadas/gradientes habilitados por PaCA em tempo
# de execução. Família CLIP.
#
# Principais funções: apply_sora, update_paca_runtime.

"""CLIP model support — adaptation."""

import torch.nn as nn
from fl_serverpuma.models.clip.sora_layers import SoRAWrappedLinear

def apply_sora(model, lora_config, upper_k=None):
    """
    Injeção de SoRA (Sparse LoRA).
    Substitui camadas Lineares por versões 'Wrapped' que contêm adaptadores esparsos.
    Suporta PaCA (Partial Calibration) se upper_k for definido.
    """
    target_modules = lora_config["target_modules"]
    r = lora_config["r"]
    alpha = lora_config["alpha"]
    dropout = lora_config["dropout"]

    vision = model.vision_model
    total_layers = len(vision.encoder.layers)
    modules_map = dict(vision.named_modules())
    replacements = []

    for name in list(modules_map.keys()):
        for target in target_modules:
            if not name.endswith(target):
                continue

            # Filtro PaCA: Verifica se a camada está entre as últimas 'upper_k'
            if upper_k is not None:
                if "encoder.layers." not in name:
                    continue
                try:
                    layer_idx = int(name.split("encoder.layers.")[1].split(".")[0])
                    if layer_idx < total_layers - upper_k:
                        continue 
                except:
                    continue

            parts = name.rsplit(".", 1)
            parent = modules_map[parts[0]] if len(parts) == 2 else vision
            child_name = parts[1] if len(parts) == 2 else parts[0]

            original_linear = getattr(parent, child_name)
            if isinstance(original_linear, nn.Linear):
                replacements.append((parent, child_name, original_linear))

    for parent, child_name, original_linear in replacements:
        wrapped = SoRAWrappedLinear(original_linear, r=r, lora_alpha=alpha, lora_dropout=dropout)
        setattr(parent, child_name, wrapped)

    print(f"SoRA aplicado a {len(replacements)} módulos (Últimas {upper_k or 'todas'} camadas)")


def _get_encoder_layers(model):
    """Encontra as camadas do encoder em modelos PEFT-wrapped ou raw."""
    vision = model.vision_model
    # PEFT envolve o modelo: base_model.model contém o original
    if hasattr(vision, 'base_model') and hasattr(vision.base_model, 'model'):
        return vision.base_model.model.encoder.layers
    return vision.encoder.layers


def _apply_paca_gradient_boundary(model, upper_k):
    """
    Registra um forward hook que interrompe a propagação de gradientes
    nas camadas abaixo do boundary do PaCA.

    Referência teórica: Em partial fine-tuning, camadas inferiores sem
    parâmetros treináveis não precisam de gradientes de ativação.
    (Head2Toe — Evci et al. 2022, Ladder Side-Tuning — Sung et al. 2022)
    """
    encoder_layers = _get_encoder_layers(model)
    total_layers = len(encoder_layers)

    if upper_k >= total_layers:
        return  # PaCA cobre todas as camadas, nada a cortar

    boundary_idx = total_layers - upper_k - 1

    def detach_hook(module, input, output):
        # Só interrompe gradientes durante treino; em eval é no-op
        if not module.training:
            return output
        # CLIPEncoderLayer retorna (hidden_states,) ou (hidden_states, attn_weights)
        if isinstance(output, tuple):
            return (output[0].detach(),) + output[1:]
        return output.detach()

    handle = encoder_layers[boundary_idx].register_forward_hook(detach_hook)
    # Armazena o handle no modelo para poder remover se necessário
    if not hasattr(model, '_paca_hooks'):
        model._paca_hooks = []
    model._paca_hooks.append(handle)

    print(f"PaCA gradient boundary: camadas 0-{boundary_idx} não propagarão gradientes "
          f"(economia de ~{(boundary_idx + 1) / total_layers * 100:.0f}% no backward pass)")


def update_paca_runtime(model, new_upper_k):
    """
    Atualiza o PaCA em runtime sem reconstruir o modelo.

    Chamado pelo cliente quando o servidor ajusta o PaCA adaptativamente.
    Garante que a economia computacional reflita o novo valor de PaCA,
    atualizando tanto o corte de gradientes quanto o estado dos adaptadores.

    Operações:
      1. Remove os forward hooks de gradiente existentes (_paca_hooks)
      2. Registra novo hook de boundary na posição correta
      3. Congela adaptadores (SoRA/LoRA) em camadas fora do novo PaCA
      4. Descongela adaptadores dentro do novo PaCA (necessário quando PaCA sobe)

    Referência teórica: PaCA (Partial Calibration) restringe o fine-tuning
    às últimas upper_k camadas do encoder. Camadas abaixo do boundary não
    precisam de gradientes, e seus adaptadores não devem ser atualizados.

    Args:
        model: O modelo CLIPForClassification com adaptadores injetados.
        new_upper_k: Número de camadas superiores a treinar (novo valor PaCA).
                     Se None, todas as camadas ficam ativas.
    """
    encoder_layers = _get_encoder_layers(model)
    total_layers = len(encoder_layers)

    # --- 1. REMOVE HOOKS ANTIGOS ---
    # Os hooks são armazenados em model._paca_hooks pelo _apply_paca_gradient_boundary.
    # Remover antes de registrar novos evita hooks duplicados ou conflitantes.
    if hasattr(model, '_paca_hooks'):
        for handle in model._paca_hooks:
            handle.remove()
        model._paca_hooks = []

    # --- 2. REGISTRA NOVO HOOK NA POSIÇÃO CORRETA ---
    # _apply_paca_gradient_boundary já lida com o caso upper_k >= total_layers
    # (retorna sem registrar hook, permitindo backward completo).
    if new_upper_k is not None:
        _apply_paca_gradient_boundary(model, new_upper_k)

    # --- 3/4. FREEZE/UNFREEZE DE ADAPTADORES ---
    # Camadas abaixo do boundary não precisam de gradientes nos adaptadores.
    # Congelar seus parâmetros traz 3 benefícios:
    #   a) PyTorch não constrói o grafo de autograd para esses ops (menos memória)
    #   b) O optimizer não itera sobre eles (menos overhead por step)
    #   c) get_trainable_state_dict não os inclui (menos dados enviados ao servidor)
    if new_upper_k is None or new_upper_k >= total_layers:
        boundary_layer = 0  # Todas as camadas estão ativas
    else:
        boundary_layer = total_layers - new_upper_k

    frozen_count = 0
    unfrozen_count = 0

    for name, param in model.named_parameters():
        # Filtra apenas parâmetros de adaptadores (SoRA ou LoRA via PEFT)
        if "sora" not in name and "lora" not in name:
            continue
        # Filtra apenas parâmetros dentro de encoder layers
        if "encoder.layers." not in name:
            continue

        try:
            layer_idx = int(name.split("encoder.layers.")[1].split(".")[0])
        except (ValueError, IndexError):
            continue

        if layer_idx < boundary_layer:
            # Camada fora do PaCA: congelar adaptador
            if param.requires_grad:
                param.requires_grad = False
                frozen_count += 1
        else:
            # Camada dentro do PaCA: garantir que o adaptador está ativo
            if not param.requires_grad:
                param.requires_grad = True
                unfrozen_count += 1

    if frozen_count > 0 or unfrozen_count > 0:
        print(f"PaCA runtime update: upper_k={new_upper_k}, "
              f"boundary=layer {boundary_layer}/{total_layers}, "
              f"frozen={frozen_count} params, unfrozen={unfrozen_count} params")
