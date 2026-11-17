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


if __name__ == "__main__":
    test_text_dataset_and_collate()
    test_multimodal_dataset_and_collate()
    test_loss_and_payload_isolation()
    print("\nTODOS OS TESTES UNITÁRIOS DO DECISION-D1 FORAM APROVADOS COM SUCESSO!")
