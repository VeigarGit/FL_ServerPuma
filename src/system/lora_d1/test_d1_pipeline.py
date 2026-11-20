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


if __name__ == "__main__":
    test_text_dataset_and_collate()
    test_multimodal_dataset_and_collate()
    test_loss_and_payload_isolation()
    test_lora_sora_adapter_dtype_parity()
    test_sora_forward_and_gate_sparsifier_and_optimizer()
    test_lhdq_vectorized_unpack_and_return_reference()
    print("\nTODOS OS TESTES UNITÁRIOS DO DECISION-D1 FORAM APROVADOS COM SUCESSO!")
