"""Testes unitários do pipeline DecisionLFM: encode, collate, DecisionHead, loss e payload."""

import torch
from lora_lfm.lfm_setup import MODEL_NAME, MODEL_REVISION, DecisionHead, collate_records, encode
from lora_lfm.qlora import assert_adapter_only_payload
from lora_lfm.trainer import decision_loss
from transformers import AutoTokenizer


def test_encode_and_collate():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    options = {
        "0": "activate card",
        "1": "lost or stolen card",
        "2": "check account balance",
    }
    r1 = encode(tokenizer, "Where is my card?", "Identify the customer's intent.", options)
    r1["target"] = 1
    r2 = encode(tokenizer, "I want to see my balance", "Identify the customer's intent.", options)
    r2["target"] = 2

    batch = collate_records([r1, r2], pad_id=tokenizer.pad_token_id)

    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "option_weights" in batch
    assert "option_mask" in batch
    assert "targets" in batch
    assert batch["input_ids"].shape[0] == 2
    assert batch["targets"].shape[0] == 2
    assert batch["option_mask"].shape == (2, 3)
    assert batch["option_mask"].all()
    print("✓ test_encode_and_collate aprovado!")


def test_decision_head():
    head = DecisionHead(hidden_size=2048, head_size=256)
    batch_size = 2
    num_options = 3
    query = torch.randn(batch_size, 2048)
    options = torch.randn(batch_size, num_options, 2048)
    valid = torch.ones(batch_size, num_options, dtype=torch.bool)

    scores = head(query, options, valid)
    assert scores.shape == (batch_size, num_options)
    assert torch.isfinite(scores).all()
    print("✓ test_decision_head aprovado!")


def test_loss_backward():
    logits = torch.randn(4, 5, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 4], dtype=torch.long)

    loss = decision_loss(logits, targets, brier_weight=1.0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    print("✓ test_loss_backward aprovado!")


def test_payload_isolation():
    lora_key = "base_model.model.language_model.layers.23.self_attn.q_proj.lora_A.default.weight"
    valid_payload = {
        "head.query.weight": torch.randn(256, 2048),
        "head.option.weight": torch.randn(256, 2048),
        "head.norm.weight": torch.randn(2048),
        lora_key: torch.randn(8, 2048),
    }
    assert_adapter_only_payload(valid_payload)

    invalid_payload = dict(valid_payload)
    invalid_key = "backbone.language_model.layers.0.self_attn.q_proj.weight"
    invalid_payload[invalid_key] = torch.randn(2048, 2048)

    try:
        assert_adapter_only_payload(invalid_payload)
        raise AssertionError("Deveria ter lançado ValueError para tensores base!")
    except ValueError:
        print("✓ test_payload_isolation aprovado!")


if __name__ == "__main__":
    test_encode_and_collate()
    test_decision_head()
    test_loss_backward()
    test_payload_isolation()
    print("\nTODOS OS TESTES UNITÁRIOS DO LFM FORAM APROVADOS COM SUCESSO! 🎉")
