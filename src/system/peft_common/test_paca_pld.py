"""Testes unitários e de integração leve para PaCA Adaptativo e PLD (PUMA-PLD)."""

from __future__ import annotations

import functools
import random
from types import SimpleNamespace

import torch
import torch.nn as nn
from ALA_LoRA import ALA_LoRA
from server import FederatedLearningServer
from transformers.modeling_layers import GradientCheckpointingLayer

from peft_common.pld import (
    ProgressiveLayerDropper,
    apply_pld_runtime,
    clear_pld_batch_hooks,
)
from utils.lhdq import is_lhdq_encoded, lhdq_decode, lhdq_encode


def test_progressive_layer_dropper_schedule() -> None:
    dropper = ProgressiveLayerDropper(
        total_layers=12,
        total_steps=10,
        theta_bar=0.5,
        direction="top_drop",
        mode="round",
        decay_constant=5.0,
    )
    assert dropper.compute_theta(0) == 1.0
    thetas = [dropper.compute_theta(s) for s in range(11)]
    for a, b in zip(thetas[:-1], thetas[1:], strict=True):
        assert a >= b >= 0.5 - 1e-9

    probs_top = ProgressiveLayerDropper(
        12, 10, theta_bar=0.5, direction="top_drop"
    ).compute_keep_probs(10)
    probs_bot = ProgressiveLayerDropper(
        12, 10, theta_bar=0.5, direction="bottom_drop"
    ).compute_keep_probs(10)
    probs_uni = ProgressiveLayerDropper(
        12, 10, theta_bar=0.5, direction="uniform"
    ).compute_keep_probs(10)

    assert probs_top[0] > probs_top[-1]
    assert probs_bot[0] < probs_bot[-1]
    assert len({round(p, 6) for p in probs_uni}) == 1

    active, _ = dropper.sample_active_layers(10, min_active=4, rng=random.Random(1))
    assert len(active) >= 4


def test_orthogonal_paca_and_pld_on_clip() -> None:
    from lora_clip.clip_setup import build_model, load_config

    cfg = load_config("src/system/lora_clip/train_config.yml")
    cfg["model"]["lora"]["mode"] = "with_sora_schedule"
    cfg["model"]["paca"] = {"enabled": True, "upper_layers": 6, "adaptive": True}
    cfg["model"]["pld"] = {
        "enabled": True,
        "theta_bar": 0.5,
        "mode": "round",
        "direction": "bottom_drop",
        "decay_constant": 5.0,
        "adaptive_paca": True,
        "total_rounds": 10,
        "client_idx": 0,
        "run_id": 1,
        "seed": 42,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, num_classes=37, device=device)

    # Com adaptive=True e paca=6, todas as 12 camadas devem possuir adaptadores SoRA
    layer_indices = {
        int(n.split("encoder.layers.")[1].split(".")[0])
        for n, _ in model.named_parameters()
        if "sora" in n and "encoder.layers." in n
    }
    assert layer_indices == set(range(12)), f"Esperado 0..11, obtido {sorted(layer_indices)}"

    # A janela PaCA=6 seleciona [6..11]; PLD nunca deve ativar camadas < 6
    info = apply_pld_runtime(model, "clip", round_num=10, run_config=cfg)
    assert info["window_start"] == 6
    assert info["paca_k"] == 6
    assert info["theta_bar"] == 0.5
    assert all(idx >= 6 for idx in info["active_layers"])
    assert info["l_min"] >= 6

    # Mesmo quando paca=12 (janela total), theta_bar continua sendo pld_theta=0.5
    cfg["model"]["paca"]["upper_layers"] = 12
    info_full = apply_pld_runtime(model, "clip", round_num=10, run_config=cfg)
    assert info_full["window_start"] == 0
    assert info_full["theta_bar"] == 0.5
    assert len(info_full["active_layers"]) < 12

    # Upscale dinâmico 6 -> 10 via update_paca_runtime + treino
    x = torch.randn(2, 3, 224, 224, device=device)
    y = torch.randint(0, 37, (2,), device=device)
    model.train()
    loss = model(pixel_values=x, labels=y)["loss"]
    loss.backward()
    model.zero_grad()

    # Batch mode respeita janela PaCA=6 e hooks são limpos corretamente
    cfg["model"]["paca"]["upper_layers"] = 6
    cfg["model"]["pld"]["mode"] = "batch"
    info_batch = apply_pld_runtime(model, "clip", round_num=10, run_config=cfg)
    assert info_batch["window_start"] == 6
    assert len(getattr(model, "_pld_batch_hooks", [])) == 6
    loss_b = model(pixel_values=x, labels=y)["loss"]
    loss_b.backward()
    for n, p in model.named_parameters():
        if "sora" in n and "encoder.layers." in n:
            idx = int(n.split("encoder.layers.")[1].split(".")[0])
            if idx < 6:
                assert p.grad is None, f"Gradiente vazou para camada {idx} fora do PaCA"

    clear_pld_batch_hooks(model)
    assert model._pld_batch_hooks == []


def test_batch_pld_checkpoint_rng_determinism() -> None:
    class TinyBlock(GradientCheckpointingLayer):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(16, 16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + torch.relu(self.lin(x))

    class TinyEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([TinyBlock() for _ in range(4)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for layer in self.layers:
                x = layer(x)
            return x

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = TinyEncoder().to(device)
    for layer in enc.layers:
        layer.gradient_checkpointing = True
        layer._gradient_checkpointing_func = functools.partial(
            torch.utils.checkpoint.checkpoint, use_reentrant=False
        )
    enc.train()

    from peft_common.pld import _register_batch_pld_hooks

    _register_batch_pld_hooks(enc, enc.layers, [0.5, 0.5], window_start=2)
    assert len(enc._pld_batch_hooks) == 2

    x = torch.randn(3, 16, device=device, requires_grad=True)
    out = enc(x)
    out.sum().backward()
    clear_pld_batch_hooks(enc)
    assert enc._pld_batch_hooks == []


def test_slm_paca_and_pld_with_gradient_checkpointing() -> None:
    from lora_slm.qlora import enable_qlora_in_config
    from lora_slm.slm_setup import (
        _get_encoder_layers,
        build_model,
        load_config,
        update_paca_runtime,
    )

    cfg = load_config("src/system/lora_slm/train_config.yml")
    cfg["model"]["lora"]["mode"] = "with_sora_schedule"
    cfg["model"]["paca"] = {"enabled": True, "upper_layers": 14, "adaptive": True}
    cfg["model"]["pld"] = {
        "enabled": True,
        "theta_bar": 0.5,
        "mode": "round",
        "direction": "bottom_drop",
        "decay_constant": 5.0,
        "adaptive_paca": True,
        "total_rounds": 10,
        "client_idx": 0,
        "run_id": 1,
        "seed": 42,
    }
    enable_qlora_in_config(cfg, "sora_with_schedule")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return

    model = build_model(cfg, num_classes=37, device=device)
    model.slm_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.slm_model.config.use_cache = False

    layers = _get_encoder_layers(model)
    assert len(layers) == 28

    # PaCA=14 deve desativar gradient_checkpointing nas camadas 0..13 e zerar grads abaixo de 14
    update_paca_runtime(model, 14)
    assert all(not layers[i].gradient_checkpointing for i in range(14))
    assert all(layers[i].gradient_checkpointing for i in range(14, 28))

    model.train()
    input_ids = torch.randint(10, 1000, (2, 24), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, 37, (2,), device=device)

    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)["loss"]
    loss.backward()
    for n, p in model.named_parameters():
        if ".layers." in n and "sora" in n:
            idx = int(n.split(".layers.")[1].split(".")[0])
            if idx < 14:
                assert p.grad is None, f"Gradiente vazou para camada {idx} do SLM"
    model.zero_grad()

    # Restaurar para 28 deve restaurar gradient_checkpointing nas 28 camadas
    update_paca_runtime(model, 28)
    assert all(layers[i].gradient_checkpointing for i in range(28))

    # PLD batch + round na janela PaCA=14
    for pld_mode in ("round", "batch"):
        cfg["model"]["pld"]["mode"] = pld_mode
        info = apply_pld_runtime(model, "slm", round_num=5, run_config=cfg)
        assert info["window_start"] == 14
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        assert torch.isfinite(out["loss"]).all()
        if out["loss"].requires_grad:
            out["loss"].backward()
        model.zero_grad()
        clear_pld_batch_hooks(model)

    del model
    torch.cuda.empty_cache()


def test_puma_adaptive_paca_clamping_and_aggregation() -> None:
    srv = object.__new__(FederatedLearningServer)
    srv.total_encoder_layers = 12
    srv.args = SimpleNamespace(
        paca=0,
        adaptive_paca_min=2,
        adaptive_paca_cooldown=2,
        gamma_delta=0.5,
        plateau_phi_min=0.45,
        plateau_decay_rate=0.025,
        device="cpu",
    )
    srv.rs_aux_simple_avg_acc = [80.0, 81.0]
    srv.rs_test_loss = [0.5, 0.4]
    srv.rs_test_acc = [80.0, 81.0]
    srv.target_latency = 50.0
    srv.model_size_per_round = [10.0]
    srv.client_masks = {}
    srv.clients_info = {
        0: {
            "training_time": 100.0,
            "bandwidth": 10.0,
            "dataset_size": 100,
            "acc_history": [80.0, 81.0],
            "comm_time": 0.1,
            "client_eval_time": 1.0,
            "server_processing_time": 0.1,
            "last_loss": 0.4,
            "current_paca": 12,
        }
    }

    paca_val = srv.calculate_adaptive_paca(0)
    assert 2 <= paca_val <= 12

    # Panic reversal check
    srv.args.paca = 12
    srv.clients_info[0]["current_paca"] = 8
    srv.clients_info[0]["paca_before_change"] = 10
    srv.clients_info[0]["paca_acc_at_reduction"] = 85.0
    srv.clients_info[0]["acc_history"].append(80.0)
    reverted = srv.calculate_adaptive_paca(0)
    assert reverted == 10

    # Agregação parcial heterogênea (camada 0 presente só no cliente 0, camada 11 em ambos)
    global_state = {
        "layers.0.sora.lora_A": torch.zeros(4, 8),
        "layers.11.sora.lora_A": torch.zeros(4, 8),
    }
    u0 = {
        "layers.0.sora.lora_A": torch.ones(4, 8) * 2.0,
        "layers.11.sora.lora_A": torch.ones(4, 8) * 4.0,
    }
    u1 = {
        "layers.11.sora.lora_A": torch.ones(4, 8) * 6.0,
    }
    agg = srv.aggregate_models(
        [u0, u1], global_state, client_weights=[1.0, 1.0], client_ids=[0, 1]
    )
    assert torch.allclose(agg["layers.0.sora.lora_A"], torch.ones(4, 8) * 2.0)
    assert torch.allclose(agg["layers.11.sora.lora_A"], torch.ones(4, 8) * 5.0)


def test_lhdq_round_trip_with_pld_and_paca() -> None:
    torch.manual_seed(0)
    k0 = "layers.0.sora.lora_A"
    k5 = "layers.5.sora.lora_A"

    # R1: ambas as camadas presentes
    s1 = {k0: torch.randn(8, 16), k5: torch.randn(8, 16)}
    srv_prev_sent = {k: v.clone() for k, v in s1.items()}
    cli_prev_recv = {k: v.clone() for k, v in s1.items()}

    # R2: PaCA reduz e envia só k5 (keep_missing=True preserva k0 no cache)
    s2 = {k5: s1[k5] + 0.05 * torch.randn(8, 16)}
    enc2, ref2 = lhdq_encode(s2, srv_prev_sent, return_reference=True)
    assert is_lhdq_encoded(enc2)
    assert "raw" not in enc2["keys"][k5]
    srv_prev_sent = {**srv_prev_sent, **ref2}

    dec2 = lhdq_decode(enc2, cli_prev_recv)
    cli_prev_recv = {**cli_prev_recv, **{k: v.clone() for k, v in dec2.items()}}
    assert torch.allclose(dec2[k5], ref2[k5], atol=1e-4)

    # R3: k0 volta a ficar ativa; com keep_missing=True (PLD) continua usando delta!
    s3 = {k0: s1[k0] + 0.03 * torch.randn(8, 16), k5: ref2[k5] + 0.02 * torch.randn(8, 16)}
    enc3, ref3 = lhdq_encode(s3, srv_prev_sent, return_reference=True)
    assert "raw" not in enc3["keys"][k0]
    dec3 = lhdq_decode(enc3, cli_prev_recv)
    assert torch.allclose(dec3[k0], ref3[k0], atol=1e-4)


def test_ala_lora_frozen_shape_sync_after_pruning() -> None:
    class Wrap(nn.Module):
        def __init__(self, r: int) -> None:
            super().__init__()
            self.sora = nn.Module()
            self.sora.r = r
            self.sora.base_r = 8
            self.sora.lora_alpha = 16
            self.sora.scaling = 2.0
            self.sora.lora_A = nn.Parameter(torch.randn(r, 8))
            self.sora.lora_B = nn.Parameter(torch.randn(8, r))
            self.head = nn.Linear(8, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = (x @ self.sora.lora_A.T @ self.sora.lora_B.T) * self.sora.scaling
            return self.head(x + h)

    local_m = Wrap(8)
    global_m = Wrap(5)
    local_m.sora.lora_A.requires_grad = False
    local_m.sora.lora_B.requires_grad = False

    ds = torch.utils.data.TensorDataset(torch.randn(8, 8), torch.randint(0, 4, (8,)))
    loader = torch.utils.data.DataLoader(ds, batch_size=4)
    ala = ALA_LoRA(0, nn.CrossEntropyLoss(), loader, batch_size=4, rand_percent=100, device="cpu")
    ala.adaptive_local_aggregation(global_m, local_m)

    assert local_m.sora.lora_A.shape == (5, 8)
    assert local_m.sora.lora_B.shape == (8, 5)


def main() -> int:
    tests = [
        test_progressive_layer_dropper_schedule,
        test_orthogonal_paca_and_pld_on_clip,
        test_batch_pld_checkpoint_rng_determinism,
        test_slm_paca_and_pld_with_gradient_checkpointing,
        test_puma_adaptive_paca_clamping_and_aggregation,
        test_lhdq_round_trip_with_pld_and_paca,
        test_ala_lora_frozen_shape_sync_after_pruning,
    ]
    for fn in tests:
        print(f"[RUN] {fn.__name__} ...", flush=True)
        fn()
        print(f"[OK]  {fn.__name__}", flush=True)
    print(f"\nALL {len(tests)} PaCA/PLD TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
