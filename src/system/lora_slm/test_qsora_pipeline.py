"""Testes unitários e de integração ponta a ponta para QSoRA (QLoRA 4-bit NF4 + SoRA) no SLM."""

import copy
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

SYSTEM_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = SYSTEM_DIR.parent
for path in (str(SYSTEM_DIR), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from lora_slm.qlora import (  # noqa: E402
    assert_adapter_only_payload,
    enable_qlora_in_config,
    get_qlora_state_dict,
    is_qsora_model,
    set_qlora_state_dict,
    verify_qlora_freezing,
)
from lora_slm.slm_setup import (  # noqa: E402
    CustomCollator,
    build_model,
    build_optimizer,
    evaluate,
    load_config,
    train_epoch,
    update_paca_runtime,
)
from lora_slm.slm_sora_utils import reduce_sora_state_dict_rank  # noqa: E402
from lora_slm.sora import SoRAWrappedLinear  # noqa: E402

from utils.fl_math import dequantization, quantization  # noqa: E402
from utils.lhdq import lhdq_decode, lhdq_encode  # noqa: E402


def test_qsora_config_validation():
    """Garante que --qlora aceita lora e sora_*, rejeitando estratégias não suportadas."""
    for strategy in ("lora", "sora_with_schedule", "sora_no_schedule"):
        cfg = {"model": {"lora": {"mode": "with_lora"}}}
        enable_qlora_in_config(cfg, strategy)
        assert cfg["model"]["qlora"]["enabled"] is True

    for unsupported in ("adalora", "full_ft", "invalid"):
        cfg = {"model": {"lora": {"mode": "with_lora"}}}
        try:
            enable_qlora_in_config(cfg, unsupported)
            raise AssertionError(f"Deveria ter rejeitado estratégia '{unsupported}' com --qlora!")
        except ValueError:
            pass
    print("✓ test_qsora_config_validation aprovado!")


def test_qsora_linear4bit_and_payload_isolation():
    """Valida SoRAWrappedLinear sobre Linear4bit real e isolamento do payload de rede."""
    if not torch.cuda.is_available():
        print("⊘ test_qsora_linear4bit_and_payload_isolation ignorado (CUDA indisponível).")
        return

    import bitsandbytes as bnb

    device = torch.device("cuda:0")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    fp_linear = torch.nn.Linear(64, 32, bias=False)
    lin4bit = bnb.nn.Linear4bit(
        64,
        32,
        bias=False,
        compute_dtype=compute_dtype,
        quant_type="nf4",
    )
    lin4bit.weight = bnb.nn.Params4bit(
        fp_linear.weight.data,
        requires_grad=False,
        quant_type="nf4",
    )
    lin4bit = lin4bit.to(device)
    assert lin4bit.weight.dtype == torch.uint8

    wrapped = SoRAWrappedLinear(
        lin4bit, r=16, lora_alpha=32, lora_dropout=0.0, adapter_dtype=compute_dtype
    )
    assert wrapped.sora.lora_A.dtype == compute_dtype
    assert wrapped.sora.lora_B.dtype == compute_dtype
    assert wrapped.sora.gate.dtype == compute_dtype
    assert wrapped.sora.lora_A.device == lin4bit.weight.device
    assert wrapped.sora.base_r == 16
    assert wrapped.original.weight.requires_grad is False

    x = torch.randn(2, 4, 64, device=device, dtype=compute_dtype)
    out = wrapped(x)
    assert out.shape == (2, 4, 32)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert wrapped.sora.lora_A.grad is not None
    assert wrapped.sora.lora_B.grad is not None
    assert wrapped.sora.gate.grad is not None

    valid_payload = {
        "slm_model.language_model.layers.27.self_attn.q_proj.sora.lora_A": torch.randn(8, 64),
        "slm_model.language_model.layers.27.self_attn.q_proj.sora.lora_B": torch.randn(32, 8),
        "slm_model.language_model.layers.27.self_attn.q_proj.sora.gate": torch.ones(1, 8),
        "classifier.weight": torch.randn(5, 64),
    }
    assert_adapter_only_payload(valid_payload)

    for bad_key, bad_val in (
        ("slm_model.language_model.layers.0.self_attn.q_proj.weight", torch.randn(64, 64)),
        (
            "slm_model.language_model.layers.0.self_attn.q_proj.sora.lora_A",
            torch.zeros(8, 64, dtype=torch.uint8),
        ),
    ):
        bad_payload = {**valid_payload, bad_key: bad_val}
        try:
            assert_adapter_only_payload(bad_payload)
            raise AssertionError(f"Deveria ter bloqueado chave/tensor inválido: {bad_key}")
        except RuntimeError:
            pass
    print("✓ test_qsora_linear4bit_and_payload_isolation aprovado!")


def test_qsora_slm_end_to_end():
    """Valida build, PaCA, treino SoRA, poda iterativa, resize, INT8 e LHDQ no SLM 4-bit."""
    if not torch.cuda.is_available():
        print("⊘ test_qsora_slm_end_to_end ignorado (CUDA indisponível).")
        return

    config_path = SYSTEM_DIR / "lora_slm" / "train_config.yml"
    config = load_config(str(config_path))
    config["model"]["lora"]["mode"] = "with_sora_schedule"
    config["model"]["lora"]["r"] = 8
    config["model"]["lora"]["alpha"] = 32
    config["model"]["paca"] = {"enabled": True, "upper_layers": 2}
    config["training"]["gradient_accumulation_steps"] = 1
    enable_qlora_in_config(config, "sora_with_schedule")

    device = torch.device("cuda:0")
    num_classes = 5
    model = build_model(config, num_classes=num_classes, device=device)

    assert is_qsora_model(model)
    verify_qlora_freezing(model)

    update_paca_runtime(model, new_upper_k=2)
    paca_state = get_qlora_state_dict(model, trainable_only=True)
    full_state = get_qlora_state_dict(model, trainable_only=False)
    assert len(paca_state) < len(full_state)
    assert any(".layers.27." in k and ".sora.gate" in k for k in paca_state)
    assert not any(".layers.0." in k for k in paca_state)
    assert "classifier.weight" in paca_state

    processor = AutoProcessor.from_pretrained(config["model"]["name"])
    label_to_idx = {i: i for i in range(num_classes)}
    collator = CustomCollator(processor, label_to_idx)
    samples = [
        {"image": Image.new("RGB", (64, 64), (200, 30, 30)), "label": 0},
        {"image": Image.new("RGB", (64, 64), (30, 200, 30)), "label": 1},
        {"image": Image.new("RGB", (64, 64), (30, 30, 200)), "label": 2},
        {"image": Image.new("RGB", (64, 64), (200, 200, 30)), "label": 3},
    ]
    loader = torch.utils.data.DataLoader(samples, batch_size=2, shuffle=False, collate_fn=collator)

    optimizer, gate_sparsifier = build_optimizer(model, config)
    assert gate_sparsifier is not None

    gates_before = {
        k: v.clone()
        for k, v in get_qlora_state_dict(model, trainable_only=True).items()
        if ".sora.gate" in k
    }
    metrics = train_epoch(
        model,
        loader,
        optimizer,
        sparse_optimizer=gate_sparsifier,
        sparse_lambda=config["model"]["sora"].get("sparse_lambda", 0.001),
        accumulation_steps=1,
    )
    assert metrics["ce_loss"] > 0 and torch.isfinite(torch.tensor(metrics["ce_loss"]))
    acc = evaluate(model, loader)
    assert 0.0 <= acc <= 1.0

    after_train_state = get_qlora_state_dict(model, trainable_only=True)
    gates_changed = any(
        not torch.equal(gates_before[k], after_train_state[k]) for k in gates_before
    )
    assert gates_changed, "Os gates SoRA deveriam ser atualizados durante o treino QSoRA!"

    # Simula compressão INT8 e LHDQ na transmissão cliente -> servidor
    int8_payload = quantization(after_train_state)
    recovered_int8 = dequantization(int8_payload)
    assert set(recovered_int8.keys()) == set(after_train_state.keys())

    prev_state = copy.deepcopy(full_state)
    lhdq_payload = lhdq_encode(full_state, prev_state)
    recovered_lhdq = lhdq_decode(lhdq_payload, prev_state)
    assert set(recovered_lhdq.keys()) == set(full_state.keys())

    # Simula poda iterativa de rank no servidor + injeção no cliente (resize dinâmico)
    pruned_state, before_p, after_p, _ = reduce_sora_state_dict_rank(
        full_state, threshold=0.2, min_rank=4
    )
    assert after_p < before_p
    set_qlora_state_dict(model, pruned_state)

    q_proj_sora = model.slm_model.language_model.layers[27].self_attn.q_proj.sora
    assert q_proj_sora.r == 4
    assert q_proj_sora.base_r == 8
    assert q_proj_sora.scaling == q_proj_sora.lora_alpha / 8
    assert q_proj_sora.lora_A.shape[0] == 4
    assert q_proj_sora.lora_A.device.type == "cuda"
    assert q_proj_sora.lora_A.dtype == model.compute_dtype

    # LHDQ com fallback raw quando o rank muda após poda
    lhdq_after_prune = lhdq_encode(pruned_state, full_state)
    decoded_after_prune = lhdq_decode(lhdq_after_prune, full_state)
    assert (
        decoded_after_prune[
            "slm_model.language_model.layers.27.self_attn.q_proj.sora.lora_A"
        ].shape[0]
        == 4
    )

    # Verifica que treino após poda/resize continua funcional
    optimizer2, sparsifier2 = build_optimizer(model, config)
    metrics2 = train_epoch(
        model, loader, optimizer2, sparse_optimizer=sparsifier2, accumulation_steps=1
    )
    assert torch.isfinite(torch.tensor(metrics2["ce_loss"]))
    del model
    torch.cuda.empty_cache()
    print("✓ test_qsora_slm_end_to_end aprovado!")


def test_qsora_lfm_end_to_end():
    """Valida build, PaCA, treino SoRA, poda iterativa e resize no LiquidAI/LFM3-VL-2B 4-bit."""
    if not torch.cuda.is_available():
        print("⊘ test_qsora_lfm_end_to_end ignorado (CUDA indisponível).")
        return

    import yaml
    from lora_lfm.lfm_setup import (
        LFMCollator,
        LFMDecisionDataset,
        get_lfm_state_dict,
        set_lfm_state_dict,
        setup_lfm_model,
        update_paca_runtime,
    )
    from lora_lfm.qlora import (
        assert_adapter_only_payload as assert_lfm_payload,
    )
    from lora_lfm.qlora import (
        build_lfm_optimizer,
    )
    from lora_lfm.qlora import (
        verify_qlora_freezing as verify_lfm_freezing,
    )
    from lora_lfm.trainer import evaluate_lfm, train_epoch_lfm
    from lora_slm.sora import build_sora_gate_sparsifier

    with (SYSTEM_DIR / "lora_lfm" / "train_config.yml").open("r", encoding="utf-8") as f:
        run_config = yaml.safe_load(f)
    run_config["model"]["lora"]["mode"] = "with_sora_schedule"
    run_config["model"]["lora"]["r"] = 8
    run_config["model"]["lora"]["alpha"] = 16
    run_config["model"]["paca"] = {"enabled": True, "upper_layers": 2}

    device = torch.device("cuda:0")
    model, tokenizer = setup_lfm_model(run_config, device, is_qlora=True)
    assert is_qsora_model(model)
    verify_lfm_freezing(model)

    update_paca_runtime(model, 2)
    paca_state = get_lfm_state_dict(model, trainable_only=True)
    full_state = get_lfm_state_dict(model, trainable_only=False)
    assert_lfm_payload(paca_state)
    assert len(paca_state) < len(full_state)
    assert any(".layers.23." in k and ".sora.gate" in k for k in paca_state)
    assert not any(".layers.0." in k for k in paca_state)

    options = {"0": "activate card", "1": "lost card", "2": "check balance"}
    ds = LFMDecisionDataset(
        [
            ("How do I activate my new card?", 0),
            ("My credit card was stolen yesterday.", 1),
            ("What is my current account balance?", 2),
            ("Please check my checking balance.", 2),
        ],
        tokenizer,
        options,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=2, shuffle=False, collate_fn=LFMCollator(tokenizer.pad_token_id)
    )

    optimizer = build_lfm_optimizer(model, run_config, is_qlora=True)
    sparsifier = build_sora_gate_sparsifier(model, run_config)
    assert sparsifier is not None

    gates_before = {k: v.clone() for k, v in paca_state.items() if ".sora.gate" in k}
    loss, acc = train_epoch_lfm(
        model,
        loader,
        optimizer,
        None,
        device,
        brier_weight=1.0,
        grad_accum_steps=1,
        sparse_optimizer=sparsifier,
    )
    assert torch.isfinite(torch.tensor(loss)) and 0.0 <= acc <= 100.0
    eval_acc, eval_loss = evaluate_lfm(model, loader, device)
    assert torch.isfinite(torch.tensor(eval_loss)) and 0.0 <= eval_acc <= 100.0

    after_state = get_lfm_state_dict(model, trainable_only=True)
    assert any(not torch.equal(gates_before[k], after_state[k]) for k in gates_before)

    pruned_state, before_p, after_p, _ = reduce_sora_state_dict_rank(
        full_state, threshold=0.2, min_rank=4
    )
    assert after_p < before_p
    set_lfm_state_dict(model, pruned_state)

    w1_sora = model.backbone.language_model.layers[23].feed_forward.w1.sora
    assert w1_sora.r == 4 and w1_sora.base_r == 8 and w1_sora.lora_A.shape[0] == 4

    optimizer2 = build_lfm_optimizer(model, run_config, is_qlora=True)
    sparsifier2 = build_sora_gate_sparsifier(model, run_config)
    loss2, _ = train_epoch_lfm(
        model, loader, optimizer2, None, device, sparse_optimizer=sparsifier2
    )
    assert torch.isfinite(torch.tensor(loss2))
    del model
    torch.cuda.empty_cache()
    print("✓ test_qsora_lfm_end_to_end aprovado!")


def test_qsora_d1_end_to_end():
    """Valida build, PaCA, treino SoRA, poda iterativa e resize no LiquidAI/d1-2B 4-bit."""
    if not torch.cuda.is_available():
        print("⊘ test_qsora_d1_end_to_end ignorado (CUDA indisponível).")
        return

    import yaml
    from lora_d1.d1_setup import (
        D1Collator,
        D1DecisionDataset,
        get_d1_state_dict,
        set_d1_state_dict,
        setup_d1_model,
        update_paca_runtime,
    )
    from lora_d1.qlora import (
        assert_adapter_only_payload as assert_d1_payload,
    )
    from lora_d1.qlora import (
        build_d1_optimizer,
    )
    from lora_d1.qlora import (
        verify_qlora_freezing as verify_d1_freezing,
    )
    from lora_d1.trainer import evaluate_d1, train_epoch_d1
    from lora_slm.sora import build_sora_gate_sparsifier

    with (SYSTEM_DIR / "lora_d1" / "train_config.yml").open("r", encoding="utf-8") as f:
        run_config = yaml.safe_load(f)
    run_config["model"]["lora"]["mode"] = "with_sora_schedule"
    run_config["model"]["lora"]["r"] = 8
    run_config["model"]["lora"]["alpha"] = 16
    run_config["model"]["paca"] = {"enabled": True, "upper_layers": 2}

    device = torch.device("cuda:0")
    model, tokenizer = setup_d1_model(
        run_config,
        device,
        is_qlora=True,
        dataset="Banking77",
        num_classes=3,
        release_vision=True,
    )
    assert is_qsora_model(model)
    verify_d1_freezing(model)

    update_paca_runtime(model, 2)
    paca_state = get_d1_state_dict(model, trainable_only=True)
    full_state = get_d1_state_dict(model, trainable_only=False)
    assert_d1_payload(paca_state)
    assert len(paca_state) < len(full_state)
    assert any(".layers.23." in k and ".sora.gate" in k for k in paca_state)
    assert not any(".layers.0." in k for k in paca_state)

    options = {"0": "activate card", "1": "lost card", "2": "check balance"}
    ds = D1DecisionDataset(
        [
            ("Where do I activate my card?", 0),
            ("I lost my debit card.", 1),
            ("Show my current balance.", 2),
            ("Check account balance please.", 2),
        ],
        tokenizer,
        options,
        question="Identify the customer's intent.",
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=2, shuffle=False, collate_fn=D1Collator(tokenizer.pad_token_id or 0)
    )

    optimizer = build_d1_optimizer(model, run_config, is_qlora=True)
    sparsifier = build_sora_gate_sparsifier(model, run_config)
    assert sparsifier is not None

    gates_before = {k: v.clone() for k, v in paca_state.items() if ".sora.gate" in k}
    loss, acc = train_epoch_d1(
        model,
        loader,
        optimizer,
        None,
        device,
        brier_weight=1.0,
        grad_accum_steps=1,
        sparse_optimizer=sparsifier,
    )
    assert torch.isfinite(torch.tensor(loss)) and 0.0 <= acc <= 100.0
    eval_acc, eval_loss = evaluate_d1(model, loader, device)
    assert torch.isfinite(torch.tensor(eval_loss)) and 0.0 <= eval_acc <= 100.0

    after_state = get_d1_state_dict(model, trainable_only=True)
    assert any(not torch.equal(gates_before[k], after_state[k]) for k in gates_before)

    pruned_state, before_p, after_p, _ = reduce_sora_state_dict_rank(
        full_state, threshold=0.2, min_rank=4
    )
    assert after_p < before_p
    set_d1_state_dict(model, pruned_state)

    w1_sora = model.backbone.model.language_model.layers[23].feed_forward.w1.sora
    assert w1_sora.r == 4 and w1_sora.base_r == 8 and w1_sora.lora_A.shape[0] == 4

    optimizer2 = build_d1_optimizer(model, run_config, is_qlora=True)
    sparsifier2 = build_sora_gate_sparsifier(model, run_config)
    loss2, _ = train_epoch_d1(
        model, loader, optimizer2, None, device, sparse_optimizer=sparsifier2
    )
    assert torch.isfinite(torch.tensor(loss2))
    del model
    torch.cuda.empty_cache()
    print("✓ test_qsora_d1_end_to_end aprovado!")


if __name__ == "__main__":
    test_qsora_config_validation()
    test_qsora_linear4bit_and_payload_isolation()
    test_qsora_slm_end_to_end()
    test_qsora_lfm_end_to_end()
    test_qsora_d1_end_to_end()
    print("\nTODOS OS TESTES DO QSoRA (SLM, LFM E d1) FORAM APROVADOS COM SUCESSO!")
