"""Progressive Layer Dropping (PLD) para Fine-Tuning Federado com PEFT.

Implementa o algoritmo Progressive Layer Dropping (Zhang & He, NeurIPS 2020)
adaptado a adaptadores LoRA/SoRA/AdaLoRA e acoplado ao Índice de Capacidade γ^c
do motor PUMA-GT Plus.

Referências:
  - Equação (1): x_{l+1} = x_l + G_l * (1 / p_l) * f_block(f_LN(x_l))
  - Equação (2): θ(t) = (1 - θ_bar) * exp(-γ_pld * t) + θ_bar
  - Equação (3): p_l(t) = 1 - (l / L) * (1 - θ(t))
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

VALID_PLD_MODES = ("round", "batch")
VALID_PLD_DIRECTIONS = ("top_drop", "bottom_drop", "uniform")


def extract_layer_idx(param_name: str) -> int | None:
    """Extrai o índice inteiro da camada encoder/decoder a partir do nome do parâmetro."""
    marker = "encoder.layers." if "encoder.layers." in param_name else ".layers."
    if marker not in param_name:
        return None
    try:
        return int(param_name.split(marker, 1)[1].split(".", 1)[0])
    except (ValueError, IndexError):
        return None


def resolve_model_layers_and_paca_updater(model: nn.Module, model_type: str):
    """Retorna `(layers, update_paca_runtime_fn)` para o tipo de modelo informado."""
    if model_type == "clip":
        from lora_clip.clip_setup import _get_encoder_layers, update_paca_runtime
    elif model_type == "slm":
        from lora_slm.slm_setup import _get_encoder_layers, update_paca_runtime
    elif model_type == "lfm":
        from lora_lfm.lfm_setup import _get_encoder_layers, update_paca_runtime
    elif model_type == "d1":
        from lora_d1.d1_setup import _get_encoder_layers, update_paca_runtime
    else:
        raise ValueError(f"PLD não suportado para o modelo '{model_type}'.")

    return _get_encoder_layers(model), update_paca_runtime


def _is_adapter_param(param_name: str, model_type: str) -> bool:
    """Verifica se o parâmetro pertence a um adaptador PEFT (LoRA/AdaLoRA/SoRA)."""
    if model_type == "d1":
        from lora_d1.qlora import is_d1_adapter_key

        return is_d1_adapter_key(param_name)
    return "lora" in param_name or "sora" in param_name


def clear_pld_batch_hooks(model: nn.Module) -> None:
    """Remove hooks de Switchable-Transformer registrados em rodadas anteriores."""
    hooks = getattr(model, "_pld_batch_hooks", None)
    if hooks:
        for handle in hooks:
            handle.remove()
    model._pld_batch_hooks = []


_clear_pld_batch_hooks = clear_pld_batch_hooks


class ProgressiveLayerDropper:
    """Agenda progressiva 2D (tempo x profundidade) e amostrador Bernoulli do PLD."""

    def __init__(
        self,
        total_layers: int,
        total_steps: int,
        *,
        theta_bar: float = 0.5,
        direction: str = "top_drop",
        mode: str = "round",
        decay_constant: float = 5.0,
    ) -> None:
        if total_layers <= 0:
            raise ValueError(f"total_layers deve ser positivo, recebido {total_layers}.")
        if direction not in VALID_PLD_DIRECTIONS:
            raise ValueError(
                f"direction inválido '{direction}'. Esperado um de {VALID_PLD_DIRECTIONS}."
            )
        if mode not in VALID_PLD_MODES:
            raise ValueError(f"mode inválido '{mode}'. Esperado um de {VALID_PLD_MODES}.")

        self.total_layers = int(total_layers)
        self.total_steps = max(1, int(total_steps))
        self.theta_bar = max(0.15, min(1.0, float(theta_bar)))
        self.direction = direction
        self.mode = mode
        self.decay_constant = max(0.0, float(decay_constant))
        self.gamma_pld = self.decay_constant / self.total_steps

    def compute_theta(self, step: int, theta_bar: float | None = None) -> float:
        """Calcula θ(t) = (1 - θ_bar) * exp(-γ_pld * t) + θ_bar (Eq. 2)."""
        target_theta = (
            self.theta_bar
            if theta_bar is None
            else max(0.15, min(1.0, float(theta_bar)))
        )
        t = max(0, int(step))
        if t == 0:
            return 1.0
        theta_t = (1.0 - target_theta) * math.exp(-self.gamma_pld * t) + target_theta
        return max(target_theta, min(1.0, theta_t))

    def compute_keep_probs(
        self, step: int, theta_bar: float | None = None
    ) -> list[float]:
        """Distribui θ(t) ao longo das L camadas (Eq. 3 e variantes)."""
        theta_t = self.compute_theta(step, theta_bar=theta_bar)
        probs: list[float] = []
        for idx in range(self.total_layers):
            if self.direction == "top_drop":
                depth_ratio = (idx + 1) / self.total_layers
            elif self.direction == "bottom_drop":
                depth_ratio = (self.total_layers - idx) / self.total_layers
            else:
                depth_ratio = 1.0
            p_l = 1.0 - depth_ratio * (1.0 - theta_t)
            probs.append(max(0.10, min(1.0, p_l)))
        return probs

    def sample_active_layers(
        self,
        step: int,
        *,
        theta_bar: float | None = None,
        min_active: int = 2,
        rng: random.Random | None = None,
    ) -> tuple[set[int], list[float]]:
        """Sorteia G_l ~ Bernoulli(p_l(t)) e garante piso mínimo de camadas ativas."""
        sampler = rng if rng is not None else random
        probs = self.compute_keep_probs(step, theta_bar=theta_bar)
        active = {idx for idx, p_l in enumerate(probs) if sampler.random() <= p_l}

        target_min = min(max(1, int(min_active)), self.total_layers)
        if len(active) < target_min:
            ranked = sorted(
                range(self.total_layers),
                key=lambda i: probs[i],
                reverse=True,
            )
            for idx in ranked:
                active.add(idx)
                if len(active) >= target_min:
                    break

        return active, probs


def _register_batch_pld_hooks(
    model: nn.Module,
    layers: Any,
    probs: list[float],
    window_start: int = 0,
) -> None:
    """Registra hooks Switchable-Transformer (Eq. 1) por mini-batch na janela ativa.

    Usa o RNG nativo do PyTorch (`torch.rand`) no dispositivo do tensor de entrada para
    garantir determinismo idêntico entre forward e recomputação sob
    `torch.utils.checkpoint(..., preserve_rng_state=True)` (gradient checkpointing).
    """
    clear_pld_batch_hooks(model)
    hooks: list[Any] = []

    for rel_idx, p_l in enumerate(probs):
        layer = layers[window_start + rel_idx]

        def _make_hook(keep_prob: float):
            def _switchable_hook(module, args, kwargs, output):
                if not (module.training and torch.is_grad_enabled()):
                    return output

                if keep_prob >= 1.0 - 1e-9:
                    return output

                x_in = None
                if args and isinstance(args[0], torch.Tensor):
                    x_in = args[0]
                elif isinstance(kwargs, dict) and isinstance(
                    kwargs.get("hidden_states"), torch.Tensor
                ):
                    x_in = kwargs["hidden_states"]

                if x_in is None:
                    return output

                x_out = output[0] if isinstance(output, tuple) else output
                if not isinstance(x_out, torch.Tensor) or x_out.shape != x_in.shape:
                    return output

                # RNG do PyTorch é salvo/restaurado automaticamente pelo gradient checkpointing
                keep = bool(torch.rand((), device=x_in.device).item() <= keep_prob)
                new_hidden = x_in if not keep else x_in + (x_out - x_in) * (1.0 / keep_prob)

                if isinstance(output, tuple):
                    return (new_hidden,) + output[1:]
                return new_hidden

            return _switchable_hook

        hooks.append(layer.register_forward_hook(_make_hook(p_l), with_kwargs=True))

    model._pld_batch_hooks = hooks


def apply_pld_runtime(
    model: nn.Module,
    model_type: str,
    round_num: int,
    run_config: dict[str, Any],
) -> dict[str, Any]:
    """Aplica o Progressive Layer Dropping em runtime antes do treino local da rodada.

    Quando combinado com PaCA (fixo ou adaptativo), opera de forma ortogonal:
      - PaCA define a janela elegível das `k_c` camadas superiores `[L - k_c, ..., L - 1]`,
        garantindo o corte real de backward nas camadas `0 .. L - k_c - 1`.
      - PLD aplica o schedule progressivo 2D (Eq. 2 e Eq. 3) com `theta_bar = pld_theta`
        sobre as `k_c` camadas elegíveis da janela ativa.
    """
    clear_pld_batch_hooks(model)

    model_cfg = run_config.get("model", {})
    pld_cfg = model_cfg.get("pld", {})
    paca_cfg = model_cfg.get("paca", {})

    layers, update_paca_runtime = resolve_model_layers_and_paca_updater(model, model_type)
    total_layers = len(layers)
    if total_layers == 0:
        raise RuntimeError(
            f"Nenhuma camada encoder encontrada para aplicar PLD em '{model_type}'."
        )

    if paca_cfg.get("enabled", False) and paca_cfg.get("upper_layers"):
        paca_k = max(1, min(total_layers, int(paca_cfg["upper_layers"])))
    else:
        paca_k = total_layers
    window_start = total_layers - paca_k
    window_size = paca_k

    theta_bar = max(0.15, min(1.0, float(pld_cfg.get("theta_bar", 0.5))))
    mode = pld_cfg.get("mode", "round")
    direction = pld_cfg.get("direction", "top_drop")
    decay_constant = float(pld_cfg.get("decay_constant", 5.0))
    total_rounds = max(1, int(pld_cfg.get("total_rounds", 1)))
    min_active = min(max(1, int(pld_cfg.get("min_active", 2))), window_size)

    base_seed = int(pld_cfg.get("seed", 42))
    run_id = int(pld_cfg.get("run_id", 1))
    client_idx = int(pld_cfg.get("client_idx", 0))
    round_seed = (base_seed * 100003) + (run_id * 1009) + (client_idx * 97) + int(round_num)
    rng = random.Random(round_seed)

    dropper = ProgressiveLayerDropper(
        total_layers=window_size,
        total_steps=total_rounds,
        theta_bar=theta_bar,
        direction=direction,
        mode=mode,
        decay_constant=decay_constant,
    )

    step = max(0, int(round_num) - 1)
    theta_t = dropper.compute_theta(step, theta_bar=theta_bar)

    if mode == "round":
        active_rel, probs_rel = dropper.sample_active_layers(
            step,
            theta_bar=theta_bar,
            min_active=min_active,
            rng=rng,
        )
        active_layers = {window_start + idx for idx in active_rel}
        l_min = min(active_layers)
        effective_upper_k = total_layers - l_min
        update_paca_runtime(model, effective_upper_k)

        for name, param in model.named_parameters():
            if not _is_adapter_param(name, model_type):
                continue
            layer_idx = extract_layer_idx(name)
            if layer_idx is None:
                continue
            param.requires_grad = layer_idx in active_layers

        probs_full = [0.0] * window_start + probs_rel
        sorted_active = sorted(active_layers)
        logger.info(
            "PLD [round] r=%d | janela=[%d..%d] (PaCA=%d/%d) | θ(t)=%.3f | θ_bar=%.3f | "
            "dir=%s | ativas=%d/%d %s | l_min=%d",
            round_num,
            window_start,
            total_layers - 1,
            paca_k,
            total_layers,
            theta_t,
            theta_bar,
            direction,
            len(sorted_active),
            total_layers,
            sorted_active,
            l_min,
        )
        return {
            "mode": mode,
            "direction": direction,
            "theta_t": theta_t,
            "theta_bar": theta_bar,
            "probs": probs_full,
            "active_layers": sorted_active,
            "window_start": window_start,
            "paca_k": paca_k,
            "l_min": l_min,
        }

    probs_rel = dropper.compute_keep_probs(step, theta_bar=theta_bar)
    if window_size <= min_active:
        probs_rel = [1.0] * window_size
    update_paca_runtime(model, paca_k)
    _register_batch_pld_hooks(model, layers, probs_rel, window_start=window_start)
    probs_full = [0.0] * window_start + probs_rel
    active_window = list(range(window_start, total_layers))
    logger.info(
        "PLD [batch] r=%d | janela=[%d..%d] (PaCA=%d/%d) | θ(t)=%.3f | θ_bar=%.3f | "
        "dir=%s | p_min=%.3f | p_max=%.3f",
        round_num,
        window_start,
        total_layers - 1,
        paca_k,
        total_layers,
        theta_t,
        theta_bar,
        direction,
        min(probs_rel),
        max(probs_rel),
    )
    return {
        "mode": mode,
        "direction": direction,
        "theta_t": theta_t,
        "theta_bar": theta_bar,
        "probs": probs_full,
        "active_layers": active_window,
        "window_start": window_start,
        "paca_k": paca_k,
        "l_min": window_start,
    }
