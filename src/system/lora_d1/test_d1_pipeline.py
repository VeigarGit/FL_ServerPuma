"""Testes unitários do pipeline DecisionD1 (LiquidAI/d1-2B)."""

import torch
from PIL import Image
from transformers import AutoTokenizer

from lora_d1.d1_setup import (
    MODEL_NAME,
    MODEL_REVISION,
    D1Collator,
    D1DecisionDataset,
    DecisionD1,
    configure_d1_tokenizer,
    resolve_d1_options,
)
from lora_d1.qlora import assert_adapter_only_payload
from lora_d1.trainer import decision_loss


def test_text_dataset_and_collate():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=True
    )
    configure_d1_tokenizer(tokenizer, MODEL_NAME, MODEL_REVISION)
    options = {"0": "activate card", "1": "lost card", "2": "check balance"}
    ds = D1DecisionDataset(
        [("Where is my card?", 1), ("What is my balance?", 2)],
        tokenizer,
        options,
        question="Identify intent.",
    )
    batch = D1Collator(tokenizer.pad_token_id or 0)([ds[0], ds[1]])

    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["slot_pos"].tolist() == [ds[0]["slot_pos"], ds[1]["slot_pos"]]
    assert batch["targets"].tolist() == [1, 2]
    print("✓ test_text_dataset_and_collate aprovado!")


def test_multimodal_dataset_and_collate():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=True
    )
    configure_d1_tokenizer(tokenizer, MODEL_NAME, MODEL_REVISION)
    options = {"0": "red", "1": "blue"}
    imgs = [
        (Image.new("RGB", (64, 64), (200, 10, 10)), 0),
        (Image.new("RGB", (64, 64), (10, 10, 200)), 1),
    ]
    ds = D1DecisionDataset(imgs, tokenizer, options, question="Dominant color?")
    batch = D1Collator(tokenizer.pad_token_id or 0, media_reader=ds.media_reader)([ds[0], ds[1]])

    assert "pixel_values" in batch
    assert "image_grid_thw" in batch
    assert batch["targets"].tolist() == [0, 1]
    print("✓ test_multimodal_dataset_and_collate aprovado!")


def test_loss_and_payload_isolation():
    logits = torch.randn(4, 5, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 4], dtype=torch.long)
    loss = decision_loss(logits, targets, brier_weight=1.0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

    lora_key = (
        "backbone.base_model.model.model.language_model.layers.23"
        ".self_attn.q_proj.lora_A.default.weight"
    )
    sora_gate_key = "backbone.model.language_model.layers.23.feed_forward.w1.sora.gate"
    valid_payload = {lora_key: torch.randn(8, 2048), sora_gate_key: torch.ones(1, 8)}
    assert_adapter_only_payload(valid_payload)

    invalid_payload = {
        **valid_payload,
        "backbone.base_model.model.lm_head.weight": torch.randn(128000, 2048),
    }
    try:
        assert_adapter_only_payload(invalid_payload)
        raise AssertionError("Deveria ter rejeitado lm_head no payload!")
    except ValueError:
        pass

    opts = resolve_d1_options("Banking77", 77)
    assert len(opts) == 77
    assert DecisionD1 is not None
    print("✓ test_loss_and_payload_isolation aprovado!")


def test_lora_sora_adapter_dtype_parity():
    """Garante que LoRA (PEFT) e SoRA mantenham adaptadores em 16 bits (sem upcast FP32)."""
    from lora_slm.sora import apply_peft_lora_modules, apply_sora_modules
    from peft import LoraConfig

    class _DummyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = torch.nn.Module()
            self.language_model.q_proj = torch.nn.Linear(32, 32, bias=False, dtype=torch.bfloat16)
            self.language_model.w1 = torch.nn.Linear(32, 64, bias=False, dtype=torch.bfloat16)

        def forward(self, x):
            return self.language_model.q_proj(x) + self.language_model.w1(x).sum(
                dim=-1, keepdim=True
            )

    lora_model = _DummyLM()
    lora_model.requires_grad_(False)
    peft_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=r".*language_model\.(q_proj|w1)",
    )
    lora_wrapped = apply_peft_lora_modules(lora_model, peft_cfg, adapter_dtype=torch.bfloat16)

    sora_model = _DummyLM()
    sora_model.requires_grad_(False)
    apply_sora_modules(
        sora_model,
        target_pattern=r".*language_model\.(q_proj|w1)",
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        adapter_dtype=torch.bfloat16,
    )

    lora_dtypes = {
        p.dtype for n, p in lora_wrapped.named_parameters() if p.requires_grad and "lora_" in n
    }
    sora_dtypes = {
        p.dtype for n, p in sora_model.named_parameters() if p.requires_grad and ".sora." in n
    }
    assert lora_dtypes == {torch.bfloat16}, f"Esperado bfloat16 no LoRA, obtido {lora_dtypes}"
    assert sora_dtypes == {torch.bfloat16}, f"Esperado bfloat16 no SoRA, obtido {sora_dtypes}"

    x = torch.randn(2, 32, dtype=torch.bfloat16)
    lora_wrapped(x).sum().backward()
    sora_model(x).sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in lora_wrapped.parameters()
        if p.requires_grad
    )
    print("✓ test_lora_sora_adapter_dtype_parity aprovado!")


def test_sora_forward_and_gate_sparsifier_and_optimizer():
    """Valida SoRALinear, GateSparsifier vetorizado e build_d1_optimizer."""
    from lora_slm.sora import GateSparsifier, SoRALinear

    from lora_d1.qlora import build_d1_optimizer

    torch.manual_seed(0)
    for dtype, atol, grad_atol in ((torch.float32, 1e-5, 1e-5), (torch.bfloat16, 1e-2, 5e-2)):
        mod = SoRALinear(32, 64, r=8, lora_alpha=16, lora_dropout=0.0).to(dtype=dtype)
        with torch.no_grad():
            mod.lora_B.normal_(std=0.1)
            mod.gate.copy_(torch.linspace(-0.3, 0.3, 8, dtype=dtype).unsqueeze(0))

        x = torch.randn(4, 10, 32, dtype=dtype, requires_grad=True)
        out = mod(x)
        ref = ((x @ mod.lora_A.T).mul(mod.gate) @ mod.lora_B.T) * mod.scaling
        assert torch.allclose(out, ref, atol=atol, rtol=atol)

        out.sum().backward()
        g_A, g_B, g_gate, g_x = (
            mod.lora_A.grad.clone(),
            mod.lora_B.grad.clone(),
            mod.gate.grad.clone(),
            x.grad.clone(),
        )
        mod.zero_grad(set_to_none=True)
        x.grad = None
        ref2 = ((x @ mod.lora_A.T).mul(mod.gate) @ mod.lora_B.T) * mod.scaling
        ref2.sum().backward()
        assert torch.allclose(g_A, mod.lora_A.grad, atol=grad_atol, rtol=grad_atol)
        assert torch.allclose(g_B, mod.lora_B.grad, atol=grad_atol, rtol=grad_atol)
        assert torch.allclose(g_gate, mod.gate.grad, atol=grad_atol, rtol=grad_atol)
        assert torch.allclose(g_x, x.grad, atol=grad_atol, rtol=grad_atol)

        # Valida GateSparsifier multi-tensor (_foreach_*) vs F.softshrink
        gates_vec = [torch.nn.Parameter(torch.randn(1, 8, dtype=dtype)) for _ in range(6)]
        gates_ref = [p.data.clone() for p in gates_vec]
        sparsifier = GateSparsifier(gates_vec, sparse_lambda=0.15, gate_lr=0.5)
        sparsifier.step()
        th = 0.15 * 0.5
        for p_vec, p_ref in zip(gates_vec, gates_ref, strict=True):
            expected = torch.nn.functional.softshrink(p_ref, th)
            tol = 1e-6 if dtype == torch.float32 else 5e-3
            assert torch.allclose(p_vec.data, expected, atol=tol, rtol=tol)

    # Limite threshold <= 0 / vazio não deve alterar nem falhar
    GateSparsifier([], sparse_lambda=0.1, gate_lr=0.1).step()
    p_zero = torch.nn.Parameter(torch.tensor([[0.123]], dtype=torch.float32))
    GateSparsifier([p_zero], sparse_lambda=0.0, gate_lr=0.1).step()
    assert torch.isclose(p_zero.data, torch.tensor([[0.123]]))

    # Valida build_d1_optimizer em CPU (fused=False) e CUDA (fused=True)
    cpu_mod = SoRALinear(16, 16, r=4).to(dtype=torch.bfloat16)
    opt_cpu = build_d1_optimizer(
        cpu_mod,
        {"optimizer": {"lr": 1e-4}, "model": {"sora": {"sparse_lr": 1e-2}}},
    )
    assert opt_cpu.defaults["fused"] is False

    if torch.cuda.is_available():
        cuda_mod = SoRALinear(32, 64, r=8).to(device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            cuda_mod.lora_B.normal_(std=0.05)
        opt_cuda = build_d1_optimizer(
            cuda_mod,
            {"optimizer": {"lr": 1e-4}, "model": {"sora": {"sparse_lr": 1e-2}}},
        )
        assert opt_cuda.defaults["fused"] is True
        xc = torch.randn(2, 8, 32, device="cuda", dtype=torch.bfloat16)
        cuda_mod(xc).sum().backward()
        GateSparsifier([cuda_mod.gate], sparse_lambda=0.1, gate_lr=0.01).step()
        opt_cuda.step()
        opt_cuda.zero_grad(set_to_none=True)
        for p in cuda_mod.parameters():
            assert torch.isfinite(p).all()

    print("✓ test_sora_forward_and_gate_sparsifier_and_optimizer aprovado!")


def test_lhdq_vectorized_unpack_and_return_reference():
    """Valida decodificador Huffman vetorizado e referência direta no lhdq_encode."""
    import numpy as np

    from utils.lhdq import _huffman_pack, _huffman_unpack, lhdq_decode, lhdq_encode

    def _ref_unpack(packed_bytes: bytes, total_bits: int, n_elements: int) -> np.ndarray:
        out = np.empty(n_elements, dtype=np.int8)
        b_idx, bit_pos, s_idx, read = 0, 7, 0, 0
        cur = packed_bytes[0] if packed_bytes else 0
        while s_idx < n_elements and read < total_bits:
            b1 = (cur >> bit_pos) & 1
            read += 1
            bit_pos -= 1
            if bit_pos < 0:
                b_idx += 1
                bit_pos = 7
                cur = packed_bytes[b_idx] if b_idx < len(packed_bytes) else 0
            if b1 == 1:
                out[s_idx] = 1
            else:
                b2 = (cur >> bit_pos) & 1
                read += 1
                bit_pos -= 1
                if bit_pos < 0:
                    b_idx += 1
                    bit_pos = 7
                    cur = packed_bytes[b_idx] if b_idx < len(packed_bytes) else 0
                out[s_idx] = 0 if b2 == 1 else 2
            s_idx += 1
        return out

    assert _huffman_unpack(b"", 0, 0).size == 0
    rng = np.random.default_rng(7)
    cases = [
        np.zeros(1, dtype=np.int8),
        np.ones(1, dtype=np.int8),
        np.full(1, 2, dtype=np.int8),
        np.zeros(257, dtype=np.int8),
        np.ones(257, dtype=np.int8),
        np.full(257, 2, dtype=np.int8),
    ] + [rng.integers(0, 3, size=int(rng.integers(1, 4096)), dtype=np.int8) for _ in range(20)]

    for syms in cases:
        pk, tb = _huffman_pack(syms)
        ref = _ref_unpack(pk, tb, len(syms))
        vec = _huffman_unpack(pk, tb, len(syms))
        assert np.array_equal(vec, syms) and np.array_equal(vec, ref)

    torch.manual_seed(11)
    prev = {
        "a.bf16": torch.randn(8, 128, dtype=torch.bfloat16),
        "b.fp32": torch.randn(16, 64, dtype=torch.float32),
        "c.resized": torch.randn(8, 32, dtype=torch.bfloat16),
    }
    curr = {
        "a.bf16": prev["a.bf16"] + 0.02 * torch.randn(8, 128, dtype=torch.bfloat16),
        "b.fp32": prev["b.fp32"] + 0.02 * torch.randn(16, 64, dtype=torch.float32),
        "c.resized": torch.randn(4, 32, dtype=torch.bfloat16),
        "d.new": torch.randn(2, 8, dtype=torch.bfloat16),
        "meta.step": 42,
    }
    enc_only = lhdq_encode(curr, prev)
    enc, ref_state = lhdq_encode(curr, prev, return_reference=True)
    dec_state = lhdq_decode(enc_only, prev)

    assert set(ref_state.keys()) == set(dec_state.keys()) == set(curr.keys())
    assert ref_state["meta.step"] == dec_state["meta.step"] == 42
    for key in ("a.bf16", "b.fp32", "c.resized", "d.new"):
        assert torch.equal(ref_state[key], dec_state[key]), f"Divergência na chave {key}"
    # Clones de fallback raw não devem compartilhar armazenamento mutável com curr
    curr["d.new"].zero_()
    assert not torch.equal(ref_state["d.new"], curr["d.new"])

    print("✓ test_lhdq_vectorized_unpack_and_return_reference aprovado!")


def test_peft_common_reexports_and_round_schedule():
    """Valida reexportações peft_common/lora_slm e agendamento de lambda por rodada."""
    import numpy as np
    from lora_slm.slm_sora_utils import reduce_sora_state_dict_rank as slm_reduce
    from lora_slm.sora import (
        GateSparsifier as SlmGateSparsifier,
    )
    from lora_slm.sora import (
        SoRALinear as SlmSoRALinear,
    )
    from lora_slm.sora import (
        build_sora_gate_sparsifier as slm_build_sparsifier,
    )
    from peft_common.sora import (
        GateSparsifier,
        SoRALinear,
        SoRAWrappedLinear,
        build_sora_gate_sparsifier,
    )
    from peft_common.sora_utils import reduce_sora_state_dict_rank

    assert SoRALinear is SlmSoRALinear
    assert GateSparsifier is SlmGateSparsifier
    assert build_sora_gate_sparsifier is slm_build_sparsifier
    assert reduce_sora_state_dict_rank is slm_reduce

    mod = SoRAWrappedLinear(torch.nn.Linear(8, 8, bias=False), r=4)
    run_cfg = {
        "model": {
            "lora": {"mode": "with_sora_schedule"},
            "sora": {
                "sparse_lambda_2": 0.10,
                "sparse_lr": 0.01,
                "lambda_schedule": "linear",
                "max_lambda": 0.20,
                "lambda_num": 10,
            },
        },
        "optimizer": {"lr": 5e-5},
    }
    expected = np.linspace(0.10, 0.20, 10)
    for round_num in range(1, 13):
        sp = build_sora_gate_sparsifier(mod, run_cfg, round_num=round_num)
        assert sp is not None
        sp.step_lambda()
        exp_val = expected[min(round_num - 1, 9)]
        assert np.isclose(sp.sparse_lambda, exp_val), (
            f"Round {round_num}: esperado {exp_val}, obtido {sp.sparse_lambda}"
        )

    print("✓ test_peft_common_reexports_and_round_schedule aprovado!")


def test_d1_config_and_release_vision_and_runtime_guards():
    """Valida prepare_d1_run_config, detecção textual release_vision e guardas de payload."""
    from types import SimpleNamespace

    from peft_common.sora import SoRAWrappedLinear

    from lora_d1.d1_setup import (
        detect_text_only_dataset,
        get_d1_state_dict,
        prepare_d1_run_config,
        resolve_release_vision,
        set_d1_state_dict,
    )
    from lora_d1.qlora import is_d1_adapter_key, load_quantized_d1_backbone

    base_args = SimpleNamespace(
        config="lora_clip/train_config.yml",
        strategy="sora_with_schedule",
        rank=8,
        paca=12,
        sparse_lr=0.02,
        max_lambda=0.35,
        qlora=True,
        mode="centralized",
        learning_rate=0.01,
    )
    cfg = prepare_d1_run_config(base_args)
    assert cfg["model"]["name"] == MODEL_NAME
    assert cfg["model"]["lora"]["mode"] == "with_sora_schedule"
    assert cfg["model"]["lora"]["alpha"] == 16  # Preserva alpha do YAML quando r=8 não mudou
    assert cfg["model"]["paca"] == {"enabled": True, "upper_layers": 12}
    assert cfg["model"]["sora"]["sparse_lr"] == 0.02
    assert cfg["model"]["sora"]["max_lambda"] == 0.35
    assert cfg["optimizer"]["lr"] == 0.00005  # --learning-rate da CNN não sobrescreve d1
    assert cfg["is_qlora"] is True

    # Novo rank distinto do padrão ajusta alpha=2*rank
    rank_args = SimpleNamespace(**{**base_args.__dict__, "rank": 16})
    cfg_r16 = prepare_d1_run_config(rank_args)
    assert cfg_r16["model"]["lora"]["r"] == 16 and cfg_r16["model"]["lora"]["alpha"] == 32

    # Estratégia inválida e --qlora fora do modo centralizado devem falhar
    for bad_updates in ({"strategy": "adalora"}, {"mode": "decentralized"}):
        bad_args = SimpleNamespace(**{**base_args.__dict__, **bad_updates})
        try:
            prepare_d1_run_config(bad_args)
            raise AssertionError(f"Esperava ValueError para {bad_updates}")
        except ValueError:
            pass

    # Detecção automática de dataset textual / multimodal (quando particionado no disco)
    from pathlib import Path

    if Path("src/dataset/Banking77/train/0.npz").exists():
        assert detect_text_only_dataset("Banking77") is True
        assert resolve_release_vision(cfg, dataset="Banking77") is True
    if Path("src/dataset/OxfordPets/train/0.npz").exists():
        assert detect_text_only_dataset("OxfordPets") is False
        assert resolve_release_vision(cfg, dataset="OxfordPets") is False
    assert (
        resolve_release_vision({"model": {"release_vision": True}}, dataset="OxfordPets") is True
    )

    # Chaves que apenas contêm 'lora' no nome (não adaptadores) devem ser ignoradas/rejeitadas
    assert not is_d1_adapter_key("backbone.floral_weight")
    assert is_d1_adapter_key("backbone.model.language_model.layers.0.q_proj.sora.gate")

    class _DummyD1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.floral_bias = torch.nn.Parameter(torch.ones(2))
            self.q_proj = SoRAWrappedLinear(torch.nn.Linear(8, 8, bias=False), r=4)

    dummy = _DummyD1()
    state = get_d1_state_dict(dummy, trainable_only=False)
    assert "floral_bias" not in state and "q_proj.sora.gate" in state
    try:
        set_d1_state_dict(dummy, {"backbone.lm_head.weight": torch.ones(2, 2)})
        raise AssertionError("set_d1_state_dict deveria bloquear chave fora dos adaptadores!")
    except ValueError:
        pass

    try:
        load_quantized_d1_backbone(MODEL_NAME, MODEL_REVISION, device="cpu")
        raise AssertionError("Esperava RuntimeError para device='cpu'")
    except RuntimeError:
        pass

    print("✓ test_d1_config_and_release_vision_and_runtime_guards aprovado!")


def test_paca_boundary_no_grad_below_and_grad_above():
    """Valida PaCA com GradientCheckpointingLayer: zero grafo/ckpt abaixo, exato acima."""
    from functools import partial

    from peft_common.sora import SoRAWrappedLinear
    from transformers.modeling_layers import GradientCheckpointingLayer

    from lora_d1.d1_setup import update_paca_runtime

    class _CkptLayer(GradientCheckpointingLayer):
        def __init__(self):
            super().__init__()
            self.proj = SoRAWrappedLinear(torch.nn.Linear(16, 16, bias=False), r=4)
            with torch.no_grad():
                self.proj.sora.lora_B.normal_(std=0.1)
            self.gradient_checkpointing = True
            self._gradient_checkpointing_func = partial(
                torch.utils.checkpoint.checkpoint, use_reentrant=False
            )

        def forward(self, hidden_states, **kwargs):
            return self.proj(hidden_states)

    class _FakeBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Linear(16, 16, bias=False)
            self.embed.requires_grad_(False)
            self.language_model = torch.nn.Module()
            self.language_model.layers = torch.nn.ModuleList([_CkptLayer() for _ in range(4)])

        def forward(self, x):
            # Simula enable_input_require_grads() sobre a saída dos embeddings
            h = self.embed(x).detach().requires_grad_(True)
            for layer in self.language_model.layers:
                h = layer(h)
            return h

    torch.manual_seed(17)
    model = torch.nn.Module()
    model.backbone = _FakeBackbone()
    model.train()
    x = torch.randn(3, 16)

    # Referência sem hooks com gradient_checkpointing ativo nas camadas 2..3 e entrada desconectada
    update_paca_runtime(model, 2)
    assert [layer.gradient_checkpointing for layer in model.backbone.language_model.layers] == [
        False,
        False,
        True,
        True,
    ]
    out_paca = model.backbone(x)
    out_paca.square().sum().backward()

    layers = model.backbone.language_model.layers
    for idx in (0, 1):
        assert all(p.grad is None for p in layers[idx].parameters())
        assert not any(p.requires_grad for p in layers[idx].parameters())

    grads_paca = {
        idx: {n: p.grad.clone() for n, p in layers[idx].named_parameters() if p.requires_grad}
        for idx in (2, 3)
    }
    for idx in (2, 3):
        assert len(grads_paca[idx]) == 3
        for g in grads_paca[idx].values():
            assert torch.isfinite(g).all()

    # Compara contra execução manual a partir da camada 2 sem hooks
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        h_ref = model.backbone.embed(x)
        h_ref = layers[0](h_ref)
        h_ref = layers[1](h_ref)
    h_ref = h_ref.detach().requires_grad_(True)
    h_ref = layers[2](h_ref)
    h_ref = layers[3](h_ref)
    h_ref.square().sum().backward()
    for idx in (2, 3):
        for n, p in layers[idx].named_parameters():
            if p.requires_grad:
                assert torch.allclose(p.grad, grads_paca[idx][n], atol=1e-6, rtol=1e-6)

    # Subida adaptativa do PaCA restaura checkpointing e gradientes nas 4 camadas
    model.zero_grad(set_to_none=True)
    update_paca_runtime(model, 4)
    assert all(layer.gradient_checkpointing for layer in layers)
    out_full = model.backbone(x)
    out_full.square().sum().backward()
    for idx in range(4):
        assert all(
            p.requires_grad and p.grad is not None for p in layers[idx].proj.sora.parameters()
        )

    print("✓ test_paca_boundary_no_grad_below_and_grad_above aprovado!")


def test_evaluate_model_async_equivalence():
    """Valida evaluate_model assíncrono em CPU/CUDA e contrato host_input_keys."""
    from torch.utils.data import DataLoader

    from utils.fl_math import evaluate_model

    class _StdDictModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.compute_dtype = torch.float32
            self.proj = torch.nn.Linear(8, 3)
            self.seen_grid_devices = []

        def forward(self, input_ids, image_grid_thw=None, **kwargs):
            if image_grid_thw is not None:
                self.seen_grid_devices.append(image_grid_thw.device.type)
            return self.proj(input_ids.float())

    class _HostDictModel(_StdDictModel):
        host_input_keys = ("image_grid_thw", "video_grid_thw")

    torch.manual_seed(3)
    samples = [
        {
            "input_ids": torch.randn(8),
            "image_grid_thw": torch.tensor([1, 4, 4], dtype=torch.long),
            "labels": torch.tensor(idx % 3, dtype=torch.long),
        }
        for idx in range(7)
    ]
    loader = DataLoader(samples, batch_size=3, shuffle=False)

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for dev in devices:
        std_model = _StdDictModel().to(dev)
        host_model = _HostDictModel().to(dev)
        host_model.load_state_dict(std_model.state_dict())

        acc_std, loss_std = evaluate_model(std_model, loader)
        acc_host, loss_host = evaluate_model(host_model, loader)

        # Modelo padrão (como SLM) move grid_thw para device; modelo host_input_keys mantém em CPU
        assert all(d == dev.type for d in std_model.seen_grid_devices)
        assert all(d == "cpu" for d in host_model.seen_grid_devices)
        # Lote original não deve ser mutado in-place
        assert samples[0]["image_grid_thw"].device.type == "cpu"
        assert samples[0]["input_ids"].device.type == "cpu"

        # Equivalência numérica com referência usando mesmo autocast
        loss_fn = torch.nn.CrossEntropyLoss()
        ref_loss, ref_ok, ref_tot = 0.0, 0, 0
        ac_dtype = torch.float16 if dev.type == "cuda" else torch.bfloat16
        with torch.no_grad():
            for batch in loader:
                with torch.autocast(device_type=dev.type, dtype=ac_dtype):
                    logits = std_model(input_ids=batch["input_ids"].to(dev))
                y = batch["labels"].to(dev)
                ref_loss += loss_fn(logits, y).item()
                ref_ok += (logits.argmax(dim=-1) == y).sum().item()
                ref_tot += y.size(0)
        ref_acc = 100.0 * ref_ok / ref_tot
        ref_avg = ref_loss / len(loader)
        assert abs(acc_std - ref_acc) < 1e-4 and abs(acc_host - ref_acc) < 1e-4
        assert abs(loss_std - ref_avg) < 1e-4 and abs(loss_host - ref_avg) < 1e-4

    print("✓ test_evaluate_model_async_equivalence aprovado!")


if __name__ == "__main__":
    test_text_dataset_and_collate()
    test_multimodal_dataset_and_collate()
    test_loss_and_payload_isolation()
    test_lora_sora_adapter_dtype_parity()
    test_sora_forward_and_gate_sparsifier_and_optimizer()
    test_lhdq_vectorized_unpack_and_return_reference()
    test_peft_common_reexports_and_round_schedule()
    test_d1_config_and_release_vision_and_runtime_guards()
    test_paca_boundary_no_grad_below_and_grad_above()
    test_evaluate_model_async_equivalence()
    print("\nTODOS OS TESTES UNITÁRIOS DO DECISION-D1 FORAM APROVADOS COM SUCESSO!")
