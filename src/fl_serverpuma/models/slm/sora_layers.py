# Responsabilidade: Define as camadas SoRALinear e SoRAWrappedLinear e sua propagação direta.
# Família SLM.
#
# Principais funções/classes: SoRALinear, SoRAWrappedLinear.

"""SLM SoRA — sora_layers."""

import math
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
        self.gate = nn.Parameter(torch.full((1, r), 0.1))

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

        self.sora.to(original_linear.weight.dtype)

        # Congela os pesos originais para garantir que apenas o adaptador aprenda
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        # Proteção: se o SoRA for removido (em versões futuras ou poda radical), retorna apenas o original
        if self.sora is None:
            return self.original(x)
        return self.original(x) + self.sora(x)
