# Responsabilidade: Implementa SparseAdamW e GateSparsifier para atualizar e esparsificar
# gates. Família CLIP.
#
# Principais funções/classes: SparseAdamW, GateSparsifier.

"""CLIP SoRA — sora_optimization."""

import numpy as np
import torch

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
            # O threshold verdadeiro de Proximal Gradient é scaled pelo learning rate
            lr = group.get("lr", 1.0)
            threshold = self.sparse_lambda * lr
            for p in group["params"]:
                if threshold > 0:
                    # Aplica Soft-Thresholding vetorizado nativo (extremamente mais rápido)
                    p.data = torch.nn.functional.softshrink(p.data, threshold)

        return loss


class GateSparsifier:
    """
    Passo Proximal Leve para Portões SoRA (Federated Learning).

    No algoritmo SoRA (Ding et al.), a esparsidade estrutural é aprendida via
    Proximal Gradient Descent, que consiste em dois passos por iteração:
      1. Gradient step (AdamW padrão — feito pelo otimizador unificado)
      2. Proximal step (soft-thresholding nos gates — feito aqui)

    Esta classe substitui o SparseAdamW completo no contexto de FL, eliminando
    o overhead de um segundo otimizador (zero_grad + AdamW state + step) sem
    alterar a matemática do SoRA. Os gates são incluídos como param_group no
    AdamW unificado, e esta classe aplica APENAS o operador proximal do L1.

    Nota: O SparseAdamW original é mantido para o trainer standalone (main.py),
    que precisa de compatibilidade com schedulers (StepLR).
    """

    def __init__(self, gate_params, sparse_lambda, gate_lr,
                 lambda_schedule=None, max_lambda=None, lambda_num=None):
        self.gate_params = list(gate_params)
        self.sparse_lambda = sparse_lambda
        self.gate_lr = gate_lr
        self.lambda_idx = 0
        self.lambda_schedule = lambda_schedule
        self._lambdas = None
        self._build_lambda_list(max_lambda, lambda_num)

    def _build_lambda_list(self, max_lambda, lambda_num):
        """Constrói agendas dinâmicas para o hiperparâmetro lambda de esparsidade."""
        if self.lambda_schedule is None:
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
            # print(f"[GateSparsifier] lambda={self.sparse_lambda}")

    def zero_grad(self, set_to_none=False):
        """No-op: os gradientes dos gates são geridos pelo otimizador unificado."""
        pass

    @torch.no_grad()
    def step(self):
        """
        Proximal step: aplica soft-thresholding (operador proximal do L1) nos gates.

        Matematicamente: gate ← S_λ(gate), onde S_λ é o operador de soft-shrinkage
        com threshold = sparse_lambda × learning_rate.

        Este é o segundo passo do Proximal Gradient Descent usado no SoRA.
        """
        threshold = self.sparse_lambda * self.gate_lr
        if threshold <= 0:
            return
        for p in self.gate_params:
            p.data = torch.nn.functional.softshrink(p.data, threshold)
