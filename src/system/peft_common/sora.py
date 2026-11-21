"""Núcleo compartilhado do Sparse Low-Rank Adaptation (SoRA) para SLM, LFM e DecisionD1."""

import math
import re

import numpy as np
import torch
import torch.nn as nn

SORA_MODES = ("with_sora_no_schedule", "with_sora_schedule")


class SoRALinear(nn.Module):
    """Camada fundamental do Sparse LoRA (SoRA).

    Implementa uma projeção de rank baixo (matrizes A e B) multiplicada por um vetor
    de portões (gate) treinável. O gate permite ao modelo "aprender" a esparsidade,
    desativando dimensões inúteis do rank durante o treinamento.
    """

    def __init__(self, in_features, out_features, r=8, lora_alpha=16, lora_dropout=0.0):
        super().__init__()
        self.r = r
        self.base_r = r
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
        """Calcula a saída do adaptador: ((x @ A^T) * (gate * scaling)) @ B^T."""
        x_in = x.to(self.lora_A.dtype) if x.dtype != self.lora_A.dtype else x
        h = nn.functional.linear(self.lora_dropout(x_in), self.lora_A)
        h = h.mul(self.gate * self.scaling)
        return nn.functional.linear(h, self.lora_B)


class SoRAWrappedLinear(nn.Module):
    """Integração não-invasiva de adaptadores.

    Envolve uma camada linear original (congelada, inclusive Linear4bit no QSoRA) e
    adiciona o SoRALinear em paralelo. A saída final é a soma da projeção original
    com a correção de rank baixo.
    """

    def __init__(self, original_linear, r=8, lora_alpha=16, lora_dropout=0.0, adapter_dtype=None):
        super().__init__()
        self.original = original_linear
        self.sora = SoRALinear(
            original_linear.in_features,
            original_linear.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        # Em backbones 4-bit (bnb.nn.Linear4bit), original_linear.weight é uint8 empacotado.
        # Evita converter os adaptadores para uint8 usando adapter_dtype e o mesmo device.
        if adapter_dtype is not None:
            target_dtype = adapter_dtype
        elif original_linear.weight.is_floating_point():
            target_dtype = original_linear.weight.dtype
        else:
            target_dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
            )
        self.sora.to(device=original_linear.weight.device, dtype=target_dtype)

        # Congela os pesos originais para garantir que apenas o adaptador aprenda
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        orig_out = self.original(x)
        # Proteção: se o SoRA for removido (poda radical), retorna apenas o original
        if self.sora is None:
            return orig_out
        sora_out = self.sora(x)
        if sora_out.dtype != orig_out.dtype:
            sora_out = sora_out.to(orig_out.dtype)
        return orig_out + sora_out


@torch.no_grad()
def prune_sora_to_lora_and_report(model, min_rank=2):
    """Poda Estrutural Pós-Treino.

    Analisa os 'gates' do SoRA, remove as dimensões do rank que foram zeradas pelo
    SparseAdamW e absorve os gates restantes na matriz B. Converte o SoRA em um
    LoRA estático extremamente compacto e eficiente para inferência.
    Respeita um rank mínimo (min_rank) por módulo para evitar over-pruning.
    """
    total_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sora_modules = 0

    print("\n" + "=" * 75)
    print("PRUNING SoRA → LoRA COMPACTO")
    print("=" * 75)

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
                x_in = x.to(self.lora_A.dtype) if x.dtype != self.lora_A.dtype else x
                return (x_in @ self.lora_A.T @ self.lora_B.T) * self.scaling

        module.sora = PrunedLoRA(A_pruned, new_B, sora.scaling)
        sora_modules += 1

        print(f"   {name:<50} Rank: {r_original:3d} → {r_new:3d}")

    total_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 70)
    print(f"   RESUMO DA PODA: {sora_modules} módulos processados")
    print(f"   Redução de parâmetros treináveis: {((1 - total_after / total_before) * 100):.2f}%")
    print("=" * 70)
    return model


@torch.no_grad()
def prune_sora_iterative(model, min_rank=2):
    """Poda Iterativa (Intermediária) para Aprendizado Federado.

    Reduz o tamanho das matrizes A, B e do gate, MAS MANTÉM o gate vivo
    para que o modelo continue a aprender esparsidade nas próximas rodadas.
    Respeita um rank mínimo (min_rank) por módulo para evitar over-pruning.
    """
    total_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sora_modules = 0

    print("\n" + "=" * 75)
    print("PRUNING ITERATIVO (SoRA → SoRA Compacto)")
    print("=" * 75)

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

        # 1. Fatia as matrizes usando apenas os índices úteis (keep_idx)
        A_pruned = sora.lora_A.data[keep_idx, :]
        B_pruned = sora.lora_B.data[:, keep_idx]
        g_pruned = sora.gate.data[:, keep_idx]

        # 2. Re-instancia os parâmetros de rede para avisar o PyTorch da mudança
        sora.lora_A = nn.Parameter(A_pruned)
        sora.lora_B = nn.Parameter(B_pruned)
        sora.gate = nn.Parameter(g_pruned)

        # 3. Atualiza a variável de controle do rank (mantendo sora.scaling inalterado)
        sora.r = r_new

        sora_modules += 1
        print(f"   {name:<50} Rank: {r_original:3d} → {r_new:3d}")

    total_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 70)
    print(f"   RESUMO DA PODA ITERATIVA: {sora_modules} módulos afetados")
    print(f"   Redução de parâmetros treináveis: {((1 - total_after / total_before) * 100):.2f}%")
    print("=" * 70)

    return model


@torch.no_grad()
def pre_prune_whole_model(model, prune_ratio=0.3, device="cpu"):
    """Otimização Preventiva do Backbone.

    Realiza uma poda conservadora das camadas MLP do CLIP original baseada na norma L1.
    """
    print(f"\n  PRE-PRUNING CONSERVADOR (ratio = {prune_ratio * 100:.0f}%)")
    vision_model = model.vision_model

    for name, module in vision_model.named_modules():
        if not isinstance(module, nn.Linear) or "mlp" not in name:
            continue

        weight = module.weight.data
        if "fc1" in name:  # Camada de expansão (Fan-out)
            norms = torch.norm(weight, dim=1, p=1)
            k = int(module.out_features * (1 - prune_ratio))
            keep_idx = torch.topk(norms, k, largest=True).indices
            new_linear = nn.Linear(
                module.in_features, len(keep_idx), bias=module.bias is not None
            ).to(device)
            new_linear.weight.data = weight[keep_idx, :]
        else:  # Camada de projeção (Fan-in)
            norms = torch.norm(weight, dim=0, p=1)
            k = int(module.in_features * (1 - prune_ratio))
            keep_idx = torch.topk(norms, k, largest=True).indices
            new_linear = nn.Linear(
                len(keep_idx), module.out_features, bias=module.bias is not None
            ).to(device)
            new_linear.weight.data = weight[:, keep_idx]

        if module.bias is not None:
            new_linear.bias.data = (
                module.bias.data[keep_idx] if "fc1" in name else module.bias.data
            )

        # Substitui a camada no backbone pelo novo tensor podado
        parent = vision_model.get_submodule(".".join(name.split(".")[:-1]))
        setattr(parent, name.split(".")[-1], new_linear)

    return model


def re_freeze_vision_model(model):
    """Garante que o vision_model não seja treinado após modificações estruturais."""
    for name, param in model.vision_model.named_parameters():
        if "sora" in name or "lora" in name:
            continue
        param.requires_grad = False
    return model


def get_trainable_state_dict(model):
    """Retorna apenas os adaptadores treinados, ignorando o backbone congelado."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def set_trainable_state_dict(model, state_dict):
    """Injeta pesos treináveis (compatível com rank heterogêneo/poda SoRA)."""
    from utils.fl_math import resize_model_to_pruned

    resize_model_to_pruned(model, state_dict)


class SparseAdamW(torch.optim.AdamW):
    """Otimizador para Aprendizado de Estrutura (Proximal Gradient).

    Implementa uma penalidade de esparsidade via 'Soft-Thresholding'.
    """

    def __init__(
        self,
        sparse_lambda=0.1,
        lambda_schedule=None,
        max_lambda=None,
        lambda_num=None,
        **kwargs,
    ):
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
            self._lambdas = np.log(
                np.linspace(np.exp(self.sparse_lambda), np.exp(max_lambda), lambda_num)
            )
        elif self.lambda_schedule == "exp_linear":
            self._lambdas = np.exp(
                np.linspace(np.log(self.sparse_lambda), np.log(max_lambda), lambda_num)
            )
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
            lr = group.get("lr", 1.0)
            threshold = self.sparse_lambda * lr
            for p in group["params"]:
                if threshold > 0:
                    p.data = torch.nn.functional.softshrink(p.data, threshold)

        return loss


class GateSparsifier:
    """Passo Proximal Leve para Portões SoRA (Federated Learning).

    No algoritmo SoRA (Ding et al.), a esparsidade estrutural é aprendida via
    Proximal Gradient Descent, que consiste em dois passos por iteração:
      1. Gradient step (AdamW padrão — feito pelo otimizador unificado)
      2. Proximal step (soft-thresholding nos gates — feito aqui)
    """

    def __init__(
        self,
        gate_params,
        sparse_lambda,
        gate_lr,
        lambda_schedule=None,
        max_lambda=None,
        lambda_num=None,
    ):
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
            self._lambdas = np.log(
                np.linspace(np.exp(self.sparse_lambda), np.exp(max_lambda), lambda_num)
            )
        elif self.lambda_schedule == "exp_linear":
            self._lambdas = np.exp(
                np.linspace(np.log(self.sparse_lambda), np.log(max_lambda), lambda_num)
            )
        else:
            raise ValueError(f"Unknown lambda_schedule: {self.lambda_schedule}")

    def sync_to_round(self, round_num: int) -> None:
        """Alinha a agenda de esparsidade à rodada federada atual (1-based).

        Posiciona `lambda_idx` em `target_idx - 1` e `sparse_lambda` em `_lambdas[target_idx]`
        para que a chamada inicial de `step_lambda()` no treino avance exatamente para
        `_lambdas[target_idx]`, progredindo monotonicamente ao longo das rodadas mesmo
        quando o sparsifier é reconstruído a cada rodada.
        """
        if self._lambdas is None or len(self._lambdas) == 0 or round_num < 1:
            return
        target_idx = min(max(0, int(round_num) - 1), len(self._lambdas) - 1)
        self.lambda_idx = target_idx - 1
        self.sparse_lambda = self._lambdas[target_idx]

    def step_lambda(self):
        """Avança o índice na agenda de lambdas, aumentando a pressão por esparsidade."""
        if self._lambdas is None:
            return
        if self.lambda_idx < len(self._lambdas) - 1:
            self.lambda_idx += 1
            self.sparse_lambda = self._lambdas[self.lambda_idx]

    def zero_grad(self, set_to_none=False):
        """No-op: os gradientes dos gates são geridos pelo otimizador unificado."""
        pass

    @torch.no_grad()
    def step(self):
        """Proximal step: aplica soft-thresholding (operador proximal do L1) nos gates.

        Matematicamente: gate ← S_λ(gate), onde S_λ é o operador de soft-shrinkage
        com threshold = sparse_lambda × learning_rate.
        Usa operações multi-tensor (_foreach_*) agrupadas por (device, dtype) para
        reduzir o overhead de lançamentos CUDA sem alterar a matemática do SoRA.
        """
        threshold = float(self.sparse_lambda * self.gate_lr)
        if threshold <= 0 or not self.gate_params:
            return
        groups = {}
        for p in self.gate_params:
            groups.setdefault((p.device, p.dtype), []).append(p.data)
        for tensors in groups.values():
            signs = torch._foreach_sign(tensors)
            torch._foreach_abs_(tensors)
            torch._foreach_sub_(tensors, threshold)
            torch._foreach_clamp_min_(tensors, 0.0)
            torch._foreach_mul_(tensors, signs)


def apply_sora_modules(
    root_module: nn.Module,
    target_pattern: str | list[str],
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    adapter_dtype: torch.dtype | None = None,
) -> int:
    """Substitui camadas nn.Linear/Linear4bit correspondentes por SoRAWrappedLinear."""
    regex = re.compile(target_pattern) if isinstance(target_pattern, str) else None
    modules_map = dict(root_module.named_modules())
    replacements = []

    for name in list(modules_map.keys()):
        if regex is not None:
            if not regex.match(name):
                continue
        elif not any(name.endswith(t) for t in target_pattern):
            continue

        parts = name.rsplit(".", 1)
        parent = modules_map[parts[0]] if len(parts) == 2 else root_module
        child_name = parts[1] if len(parts) == 2 else parts[0]
        original_linear = getattr(parent, child_name)
        if isinstance(original_linear, nn.Linear):
            replacements.append((parent, child_name, original_linear))

    for parent, child_name, original_linear in replacements:
        wrapped = SoRAWrappedLinear(
            original_linear,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            adapter_dtype=adapter_dtype,
        )
        setattr(parent, child_name, wrapped)
    return len(replacements)


def apply_peft_lora_modules(
    root_module: nn.Module,
    peft_config,
    adapter_dtype: torch.dtype | None = None,
):
    """Aplica PEFT LoRA mantendo os pesos no dtype alvo (evita upcast para float32)."""
    from peft import get_peft_model

    if adapter_dtype is None:
        return get_peft_model(root_module, peft_config)

    # autocast_adapter_dtype=False impede o upcast automático para fp32 feito pelo peft.
    wrapped = get_peft_model(root_module, peft_config, autocast_adapter_dtype=False)
    for name, param in wrapped.named_parameters():
        if "lora_" in name and param.is_floating_point() and param.dtype != adapter_dtype:
            param.data = param.data.to(adapter_dtype)
    return wrapped


def build_sora_gate_sparsifier(
    model: nn.Module, run_config: dict, *, round_num: int | None = None
) -> GateSparsifier | None:
    """Constrói GateSparsifier para modelos com estratégia SoRA/QSoRA ativa."""
    model_cfg = run_config.get("model", {})
    mode = model_cfg.get("lora", {}).get("mode", "with_lora")
    if mode not in SORA_MODES:
        return None

    gate_params = [
        p for n, p in model.named_parameters() if p.requires_grad and "sora" in n and "gate" in n
    ]
    if not gate_params:
        return None

    sora_cfg = model_cfg.get("sora", {})
    opt_cfg = run_config.get("optimizer", {})
    default_lr = float(opt_cfg.get("lr", opt_cfg.get("lora_lr", 0.01)))
    sparse_lr = float(sora_cfg.get("sparse_lr") or default_lr)
    is_schedule = mode == "with_sora_schedule"
    sparsifier = GateSparsifier(
        gate_params=gate_params,
        sparse_lambda=float(sora_cfg.get("sparse_lambda_2", 0.10)),
        gate_lr=sparse_lr,
        lambda_schedule=sora_cfg.get("lambda_schedule", "linear") if is_schedule else None,
        max_lambda=sora_cfg.get("max_lambda", 0.20) if is_schedule else None,
        lambda_num=sora_cfg.get("lambda_num", 10) if is_schedule else None,
    )
    if round_num is not None:
        sparsifier.sync_to_round(round_num)
    return sparsifier
