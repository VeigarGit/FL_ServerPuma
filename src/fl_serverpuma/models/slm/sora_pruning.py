# Responsabilidade: Compacta o rank SoRA, converte para LoRA, ajusta congelamento e extrai
# parâmetros treináveis. Família SLM.
#
# Principais funções: prune_sora_to_lora_and_report, prune_sora_iterative,
# pre_prune_whole_model, re_freeze_vision_model, get_trainable_state_dict.

"""SLM SoRA — sora_pruning."""

import torch
import torch.nn as nn
from fl_serverpuma.models.slm.sora_layers import SoRAWrappedLinear

@torch.no_grad()
def prune_sora_to_lora_and_report(model, min_rank=2):
    """
    Poda Estrutural Pós-Treino.
    Analisa os 'gates' do SoRA, remove as dimensões do rank que foram zeradas pelo 
    SparseAdamW e absorve os gates restantes na matriz B. Converte o SoRA em um 
    LoRA estático extremamente compacto e eficiente para inferência.
    Respeita um rank mínimo (min_rank) por módulo para evitar over-pruning.
    """
    import sys
    import pickle
    total_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sora_modules = 0

    print("\n" + "="*75)
    print("PRUNING SoRA → LoRA COMPACTO")
    print("="*75)

    for name, module in model.named_modules():
        if not isinstance(module, SoRAWrappedLinear):
            continue

        sora = module.sora
        gate = sora.gate.data.squeeze(0)

        # Identifica quais dimensões manter (onde gate > 1e-4)
        mask = torch.abs(gate) > 1e-4
        keep_idx = torch.where(mask)[0]
        r_new = len(keep_idx)
        r_original = len(gate)

        # Garante que o rank não caia abaixo de min_rank
        effective_min = min(min_rank, r_original)
        if r_new < effective_min:
            keep_idx = torch.topk(gate.abs(), k=effective_min).indices
            keep_idx, _ = torch.sort(keep_idx)  # Preserva ordem ordinal
            r_new = len(keep_idx)
            print(f"    {name:<50} → Rank forçado para {r_new} (min_rank={min_rank})")

        # Compacta as matrizes A e B e absorve o gate na matriz B
        A_pruned = sora.lora_A.data[keep_idx, :]
        B_pruned = sora.lora_B.data[:, keep_idx]
        g_pruned = gate[keep_idx]
        new_B = B_pruned * g_pruned.unsqueeze(0)

        # Define uma classe leve para o LoRA já podado
        class PrunedLoRA(nn.Module):
            def __init__(self, A, B, scaling):
                super().__init__()
                self.lora_A = nn.Parameter(A.clone())
                self.lora_B = nn.Parameter(B.clone())
                self.scaling = scaling

            def forward(self, x):
                return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

        module.sora = PrunedLoRA(A_pruned, new_B, sora.scaling)
        sora_modules += 1

        print(f"   {name:<50} Rank: {r_original:3d} → {r_new:3d}")

    total_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("="*70)
    print(f"   RESUMO DA PODA: {sora_modules} módulos processados")
    print(f"   Redução de parâmetros treináveis: {((1 - total_after / total_before) * 100):.2f}%")
    print("="*70)
    return model


@torch.no_grad()
def prune_sora_iterative(model, min_rank=2):
    """
    Poda Iterativa (Intermediária) para Aprendizado Federado.
    Reduz o tamanho das matrizes A, B e do gate, MAS MANTÉM o gate vivo
    para que o modelo continue a aprender esparsidade nas próximas rodadas.
    Respeita um rank mínimo (min_rank) por módulo para evitar over-pruning.
    """
    total_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sora_modules = 0

    print("\n" + "="*75)
    print("PRUNING ITERATIVO (SoRA → SoRA Compacto)")
    print("="*75)

    for name, module in model.named_modules():
        if not isinstance(module, SoRAWrappedLinear):
            continue

        sora = module.sora
        gate = sora.gate.data.squeeze(0)

        # Identifica quais dimensões manter (onde gate > 1e-4)
        mask = torch.abs(gate) > 1e-4
        keep_idx = torch.where(mask)[0]
        r_new = len(keep_idx)
        r_original = len(gate)

        # Garante que o rank não caia abaixo de min_rank
        effective_min = min(min_rank, r_original)
        if r_new < effective_min:
            keep_idx = torch.topk(gate.abs(), k=effective_min).indices
            keep_idx, _ = torch.sort(keep_idx)  # Preserva ordem ordinal
            r_new = len(keep_idx)
            print(f"    {name:<50} → Rank forçado para {r_new} (min_rank={min_rank})")

        # Se o rank não diminuiu nesta camada, não fazemos nada
        if r_new == r_original:
            continue

        # --- A GRANDE MÁGICA ACONTECE AQUI ---
        # 1. Fatiamos as matrizes usando apenas os índices úteis (keep_idx)
        A_pruned = sora.lora_A.data[keep_idx, :]
        B_pruned = sora.lora_B.data[:, keep_idx]
        g_pruned = sora.gate.data[:, keep_idx] # O gate continua a existir!

        # 2. Re-instanciamos os parâmetros de rede para avisar o PyTorch da mudança
        sora.lora_A = nn.Parameter(A_pruned)
        sora.lora_B = nn.Parameter(B_pruned)
        sora.gate = nn.Parameter(g_pruned)

        # 3. Atualizamos a variável de controle do rank
        sora.r = r_new 

        # Nota de Engenharia: NÃO alteramos o sora.scaling! 
        # Manter o scaling original evita que os gradientes explodam após a poda.

        sora_modules += 1
        print(f"   {name:<50} Rank: {r_original:3d} → {r_new:3d}")

    total_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("="*70)
    print(f"   RESUMO DA PODA ITERATIVA: {sora_modules} módulos afetados")
    print(f"   Redução de parâmetros treináveis: {((1 - total_after / total_before) * 100):.2f}%")
    print("="*70)

    return model


@torch.no_grad()
def pre_prune_whole_model(model, prune_ratio=0.3, device="cpu"):
    """
    Otimização Preventiva do Backbone.
    Realiza uma poda conservadora das camadas MLP do CLIP original baseada na norma L1. 
    Reduz o número de parâmetros do modelo "base" antes mesmo do treino começar, 
    economizando memória e acelerando o processamento.
    """
    print(f"\n  PRE-PRUNING CONSERVADOR (ratio = {prune_ratio*100:.0f}%)")
    vision_model = model.vision_model

    for name, module in vision_model.named_modules():
        if not isinstance(module, nn.Linear) or "mlp" not in name:
            continue

        weight = module.weight.data
        if "fc1" in name: # Camada de expansão (Fan-out)
            norms = torch.norm(weight, dim=1, p=1)
            keep_idx = torch.topk(norms, int(module.out_features * (1 - prune_ratio)), largest=True).indices
            new_linear = nn.Linear(module.in_features, len(keep_idx), bias=module.bias is not None).to(device)
            new_linear.weight.data = weight[keep_idx, :]
        else: # Camada de projeção (Fan-in)
            norms = torch.norm(weight, dim=0, p=1)
            keep_idx = torch.topk(norms, int(module.in_features * (1 - prune_ratio)), largest=True).indices
            new_linear = nn.Linear(len(keep_idx), module.out_features, bias=module.bias is not None).to(device)
            new_linear.weight.data = weight[:, keep_idx]

        if module.bias is not None:
            new_linear.bias.data = module.bias.data[keep_idx] if "fc1" in name else module.bias.data

        # Substitui a camada no backbone pelo novo tensor podado
        parent = vision_model.get_submodule(".".join(name.split(".")[:-1]))
        setattr(parent, name.split(".")[-1], new_linear)

    return model


def re_freeze_vision_model(model):
    """Integridade do Backbone. Garante que o vision_model não seja treinado após as modificações estruturais, mantendo apenas os adaptadores (SoRA/LoRA) ativos."""
    for name, param in model.vision_model.named_parameters():
        if "sora" in name or "lora" in name:
            continue
        param.requires_grad = False
    return model


def get_trainable_state_dict(model):
    """Extração de Pesos. Retorna apenas os adaptadores treinados, ignorando o backbone congelado."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
