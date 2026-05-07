import math
import numpy as np
import torch
import torch.nn as nn


class SoRALinear(nn.Module):
    """
    Camada fundamental do Sparse LoRA (SoRA).
    Implementa uma projeção de rank baixo (matrizes A e B) multiplicada por um vetor 
    de portões (gate) treinável. O gate permite ao modelo "aprender" a esparsidade, 
    desativando dimensões inúteis do rank durante o treinamento.
    """

    def __init__(self, in_features, out_features, r=8, lora_alpha=16, lora_dropout=0.0):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.gate = nn.Parameter(torch.randn(1, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        """Calcula a saída do adaptador: (x * A * gate) * B."""
        return ((self.lora_dropout(x) @ self.lora_A.T).mul(self.gate) @ self.lora_B.T) * self.scaling


class SoRAWrappedLinear(nn.Module):
    """
    Integração não-invasiva de adaptadores.
    Envolve uma camada linear original (congelada) e adiciona o SoRALinear em paralelo. 
    A saída final é a soma da projeção original com a correção de rank baixo.
    """

    def __init__(self, original_linear, r=8, lora_alpha=16, lora_dropout=0.0):
        super().__init__()
        self.original = original_linear
        self.sora = SoRALinear(
            original_linear.in_features,
            original_linear.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        # Congela os pesos originais para garantir que apenas o adaptador aprenda
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        # Proteção: se o SoRA for removido (em versões futuras ou poda radical), retorna apenas o original
        if self.sora is None:
            return self.original(x)
        return self.original(x) + self.sora(x)

@torch.no_grad()
def prune_sora_to_lora_and_report(model):
    """
    Poda Estrutural Pós-Treino.
    Analisa os 'gates' do SoRA, remove as dimensões do rank que foram zeradas pelo 
    SparseAdamW e absorve os gates restantes na matriz B. Converte o SoRA em um 
    LoRA estático extremamente compacto e eficiente para inferência.
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
        
        # Identifica quais dimensões manter (onde gate > 1e-8)
        mask = torch.abs(gate) > 1e-8
        keep_idx = torch.where(mask)[0]
        r_new = len(keep_idx)
        r_original = len(gate)

        # Garante que o rank não seja zero (mantém ao menos 1 dimensão)
        if r_new == 0:
            keep_idx = torch.topk(gate.abs(), k=1).indices
            r_new = 1
            print(f"    {name:<50} → Rank zerado forçado para 1")

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

class SparseAdamW(torch.optim.AdamW):
    """
    Otimizador para Aprendizado de Estrutura.
    Implementa uma penalidade de esparsidade via 'Soft-Thresholding' (Proximal Gradient). 
    A cada passo, ele "limpa" os portões SoRA, forçando valores insignificantes a zero, 
    o que permite a posterior poda estrutural.
    """

    def __init__(self, sparse_lambda=0.1, lambda_schedule=None, max_lambda=None, lambda_num=None, **kwargs):
        super().__init__(**kwargs)
        self.sparse_lambda = sparse_lambda
        self.lambda_idx = 0
        self.lambda_schedule = lambda_schedule
        self._build_lambda_list(max_lambda, lambda_num)

    def _build_lambda_list(self, max_lambda, lambda_num):
        """Constrói agendas dinâmicas para o hiperparâmetro lambda de esparsidade."""
        if self.lambda_schedule is None:
            self._lambdas = None
            return
        if isinstance(self.lambda_schedule, list):
            self._lambdas = self.lambda_schedule
            return
        if max_lambda is None or lambda_num is None:
            raise ValueError(
                f"max_lambda and lambda_num are required for schedule '{self.lambda_schedule}', "
                f"got max_lambda={max_lambda}, lambda_num={lambda_num}"
            )
        if self.lambda_schedule == "linear":
            self._lambdas = np.linspace(self.sparse_lambda, max_lambda, lambda_num)
        elif self.lambda_schedule == "log_linear":
            self._lambdas = np.log(np.linspace(np.exp(self.sparse_lambda), np.exp(max_lambda), lambda_num))
        elif self.lambda_schedule == "exp_linear":
            self._lambdas = np.exp(np.linspace(np.log(self.sparse_lambda), np.log(max_lambda), lambda_num))
        else:
            raise ValueError(f"Unknown lambda_schedule: {self.lambda_schedule}")

    def step_lambda(self):
        """Avança o índice na agenda de lambdas, aumentando a pressão por esparsidade."""
        if self._lambdas is None:
            return
        if self.lambda_idx < len(self._lambdas) - 1:
            self.lambda_idx += 1
            self.sparse_lambda = self._lambdas[self.lambda_idx]
            print(f"[SparseAdamW] lambda={self.sparse_lambda}")

    @torch.no_grad()
    def step(self, closure=None):
        """Executa o passo AdamW tradicional seguido do limiar de esparsidade (Proximal Step)."""
        loss = super().step(closure)

        for group in self.param_groups:
            for p in group["params"]:
                if self.sparse_lambda > 0:
                    # Aplica Soft-Thresholding: p = sign(p) * max(0, |p| - lambda)
                    p.data[p.data > self.sparse_lambda] -= self.sparse_lambda
                    p.data[p.data < -self.sparse_lambda] += self.sparse_lambda
                    p.data[abs(p.data) < self.sparse_lambda] = 0.0

        return loss
