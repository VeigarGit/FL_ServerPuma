"""Setup, dataset e wrapper diferenciável DecisionD1 para o modelo LiquidAI/d1-2B."""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from lora_d1.qlora import assert_adapter_only_payload, is_d1_adapter_key

logger = logging.getLogger(__name__)

MODEL_NAME = "LiquidAI/d1-2B"
MODEL_REVISION = "a6bce774e6dde338dcc3036ceb5d91f78bf4c19e"
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|w1|w2|w3)"
DEFAULT_QUESTION = "Identify the customer's intent or class."
D1_STRATEGY_TO_MODE: dict[str, str] = {
    "lora": "with_lora",
    "sora_with_schedule": "with_sora_schedule",
    "sora_no_schedule": "with_sora_no_schedule",
}


def detect_text_only_dataset(dataset: str, dataset_dir: str | Path | None = None) -> bool:
    """Detecta se a partição train/0.npz armazena amostras textuais (str)."""
    candidates: list[Path] = []
    if dataset_dir is not None:
        candidates.append(Path(dataset_dir) / "train" / "0.npz")
    candidates.extend(
        [
            Path("src/dataset") / dataset / "train" / "0.npz",
            Path(__file__).resolve().parents[2] / "dataset" / dataset / "train" / "0.npz",
        ]
    )
    for path in candidates:
        if not path.exists():
            continue
        try:
            with np.load(path, allow_pickle=True) as loaded:
                raw = loaded["data"].item()
            xs = raw.get("x", []) if isinstance(raw, dict) else []
            if len(xs) > 0:
                return isinstance(xs[0], (str, np.str_))
        except Exception:
            return False
    return False


def resolve_release_vision(
    run_config: dict[str, Any] | None,
    dataset: str | None = None,
    dataset_dir: str | Path | None = None,
) -> bool:
    """Resolve se vision_tower e multi_modal_projector devem ser liberados."""
    raw = (run_config or {}).get("model", {}).get("release_vision", "auto")
    if isinstance(raw, bool):
        return raw
    if str(raw).lower() == "auto" and dataset:
        return detect_text_only_dataset(dataset, dataset_dir=dataset_dir)
    return False


def prepare_d1_run_config(args: Any) -> dict[str, Any]:
    """Carrega e valida a configuração do d1-2B combinando YAML e flags da CLI."""
    import yaml

    cfg_arg = getattr(args, "config", None)
    if not cfg_arg or cfg_arg == "lora_clip/train_config.yml":
        config_path = Path("src/system/lora_d1/train_config.yml")
    else:
        config_path = Path(cfg_arg)
    if not config_path.exists():
        config_path = Path(__file__).resolve().parent / "train_config.yml"
    with config_path.open("r", encoding="utf-8") as f:
        run_config = yaml.safe_load(f) or {}

    strategy = getattr(args, "strategy", "lora")
    if strategy not in D1_STRATEGY_TO_MODE:
        raise ValueError(
            f"--model d1 suporta apenas --strategy em {tuple(D1_STRATEGY_TO_MODE)} "
            f"(recebido: '{strategy}')."
        )

    model_cfg = run_config.setdefault("model", {})
    lora_cfg = model_cfg.setdefault("lora", {})
    paca_cfg = model_cfg.setdefault("paca", {})
    sora_cfg = model_cfg.setdefault("sora", {})

    lora_cfg["mode"] = D1_STRATEGY_TO_MODE[strategy]

    rank = getattr(args, "rank", None)
    if rank:
        if "alpha" not in lora_cfg or int(lora_cfg.get("r", 8)) != int(rank):
            lora_cfg["alpha"] = 2 * int(rank)
        lora_cfg["r"] = int(rank)

    paca = getattr(args, "paca", None)
    if paca is not None and paca > 0:
        paca_cfg["enabled"] = True
        paca_cfg["upper_layers"] = int(paca)
    else:
        paca_cfg["enabled"] = False
        paca_cfg["upper_layers"] = None

    sparse_lr = getattr(args, "sparse_lr", None)
    if sparse_lr is not None:
        sora_cfg["sparse_lr"] = float(sparse_lr)
    max_lambda = getattr(args, "max_lambda", None)
    if max_lambda is not None:
        sora_cfg["max_lambda"] = float(max_lambda)

    is_qlora = bool(getattr(args, "qlora", False))
    if is_qlora and getattr(args, "mode", "centralized") != "centralized":
        raise ValueError("--qlora para d1 é suportado apenas no modo centralizado.")
    run_config["is_qlora"] = is_qlora
    return run_config


def resolve_d1_package(
    model_name: str = MODEL_NAME,
    revision: str = MODEL_REVISION,
) -> dict[str, Any]:
    """Importa dinamicamente os submódulos (prompt, slots, attention, media, runner) do d1-2B."""
    cls = get_class_from_dynamic_module(
        "modeling_d1_lfm3_vl.D1Model",
        model_name,
        revision=revision,
    )
    pkg_name = cls.__module__.rsplit(".", 1)[0]
    return {
        "prompt": importlib.import_module(f"{pkg_name}.prompt"),
        "slots": importlib.import_module(f"{pkg_name}.slots"),
        "attention": importlib.import_module(f"{pkg_name}.attention"),
        "media": importlib.import_module(f"{pkg_name}.media"),
        "runner": importlib.import_module(f"{pkg_name}.runner"),
    }


def resolve_d1_options(
    dataset: str,
    num_classes: int,
    classes_file: str | Path | None = None,
) -> dict[str, str]:
    """Carrega classes.json do dataset ou gera descrições padrão para as N classes."""
    candidates: list[Path] = []
    if classes_file:
        candidates.append(Path(classes_file))
    candidates.extend(
        [
            Path("src/dataset") / dataset / "classes.json",
            Path(__file__).resolve().parents[2] / "dataset" / dataset / "classes.json",
        ]
    )
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            keys = sorted(
                raw.keys(),
                key=lambda k: int(k) if str(k).isdigit() else str(k),
            )
            return {str(k): str(raw[k]) for k in keys}
    return {str(i): f"class {i}" for i in range(max(2, int(num_classes)))}


def configure_d1_tokenizer(
    tokenizer,
    model_name: str = MODEL_NAME,
    revision: str = MODEL_REVISION,
    config=None,
):
    """Inicializa atributos d1_codes e d1_prompt no tokenizer a partir do config do modelo."""
    cfg = config
    if cfg is None:
        cfg = AutoConfig.from_pretrained(model_name, revision=revision, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.d1_codes = int(getattr(cfg, "d1_codes", 2))
    tokenizer.d1_prompt = int(getattr(cfg, "d1_prompt", 1))
    return tokenizer, cfg


def _resolve_lora_targets(targets: str | list[str] | None) -> str | list[str]:
    if not targets:
        return LORA_TARGETS
    if isinstance(targets, str):
        return targets
    joined = "|".join(str(t) for t in targets)
    return rf".*language_model.*\.({joined})"


class D1DecisionDataset(Dataset):
    """Dataset que pré-tokeniza texto ou imagem no formato de answer slots do d1-2B."""

    def __init__(
        self,
        data: list[tuple[Any, int]],
        tokenizer,
        options: dict[str, str],
        question: str = DEFAULT_QUESTION,
        max_length: int = 2048,
        attention_mode: str = "set",
        model_name: str = MODEL_NAME,
        revision: str = MODEL_REVISION,
        max_pixels: int = 1638400,
    ):
        self.mods = resolve_d1_package(model_name, revision)
        d1_prompt = self.mods["prompt"]
        d1_attn = self.mods["attention"]
        d1_media = self.mods["media"]
        d1_runner = self.mods["runner"]

        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.media_reader = None
        self.records: list[dict[str, Any]] = []

        criteria = {str(k): str(v) for k, v in options.items()}
        choice = d1_prompt.Choice(question, criteria)
        lead = d1_prompt.default_lead("lfm3_vl", tokenizer)
        bos = getattr(tokenizer, "bos_token", None)
        bos = bos if isinstance(bos, str) else ""
        suffix, slot_offsets = d1_prompt.suffix_layout(tokenizer, choice, lead, "desc")
        slot_offset = slot_offsets[0]
        suf_ids = tokenizer.encode(suffix, add_special_tokens=False)

        for sample, label in data:
            is_image = isinstance(sample, Image.Image)
            if is_image:
                if self.media_reader is None:
                    processor = d1_runner.auto_processor(model_name, revision)
                    self.media_reader = d1_media.Reader(processor, max_pixels)
                markup, parts = self.media_reader.encode([d1_media.as_media(sample)])
                prefix = d1_prompt.prefix_text(
                    tokenizer,
                    None,
                    bos,
                    d1_prompt.DEFAULT_STATE_STYLE,
                    d1_prompt.DEFAULT_SYSTEM,
                    markup,
                )
            else:
                parts = []
                prefix = d1_prompt.prefix_text(
                    tokenizer,
                    str(sample),
                    bos,
                    d1_prompt.DEFAULT_STATE_STYLE,
                    d1_prompt.DEFAULT_SYSTEM,
                )

            ids = tokenizer.encode(prefix + suffix, add_special_tokens=False)
            if not is_image and len(ids) > max_length:
                # Preserva o sufixo integralmente (pergunta + opções) truncando o texto do estado
                budget = max(1, max_length - len(suf_ids))
                pre_ids = tokenizer.encode(prefix, add_special_tokens=False)[:budget]
                ids = pre_ids + suf_ids

            state_len = d1_attn.both_ways(attention_mode, tokenizer, ids, suffix)
            if state_len is None:
                state_len = len(ids)

            media_inputs = self.media_reader.inputs(parts, ids) if parts else None
            self.records.append(
                {
                    "ids": ids,
                    "state_len": int(state_len),
                    "slot_pos": int(len(ids) - 1 + slot_offset),
                    "target": int(label),
                    "media_inputs": media_inputs,
                }
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.records[idx]


class D1Collator:
    """Collator para empacotar lotes de texto ou imagem no formato aceito pelo DecisionD1."""

    def __init__(self, pad_id: int, media_reader=None):
        self.pad_id = pad_id
        self.media_reader = media_reader

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["ids"]) for item in batch)
        padded_ids = []
        attention_masks = []
        slot_positions = []
        state_lengths = []
        targets = []
        media_list = []

        for item in batch:
            seq = item["ids"]
            pad_len = max_len - len(seq)
            padded_ids.append(seq + [self.pad_id] * pad_len)
            attention_masks.append([1] * len(seq) + [0] * pad_len)
            slot_positions.append(item["slot_pos"])
            state_lengths.append(item["state_len"])
            targets.append(item["target"])
            media_list.append(item.get("media_inputs"))

        collated: dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "slot_pos": torch.tensor(slot_positions, dtype=torch.long),
            "state_len": torch.tensor(state_lengths, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
        }

        if self.media_reader is not None and any(media_list):
            media_batch = self.media_reader.batch(media_list, padded_ids)
            collated.update(media_batch)

        return collated


class DecisionD1(nn.Module):
    """Wrapper diferenciável sobre LiquidAI/d1-2B com leitura via slot_logits nativos."""

    host_input_keys = ("image_grid_thw", "video_grid_thw")

    def __init__(
        self,
        backbone: nn.Module,
        tokenizer,
        options: dict[str, str],
        question: str = DEFAULT_QUESTION,
        model_name: str = MODEL_NAME,
        revision: str = MODEL_REVISION,
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.compute_dtype = compute_dtype
        self.has_lora = False

        mods = resolve_d1_package(model_name, revision)
        self._d1_prompt = mods["prompt"]
        self._d1_slots = mods["slots"]

        choice = self._d1_prompt.Choice(question, {str(k): str(v) for k, v in options.items()})
        _, forms, gather_tokens = self._d1_prompt.readout_table(tokenizer, choice)
        self.register_buffer(
            "gather_tokens",
            torch.from_numpy(gather_tokens).long(),
            persistent=False,
        )
        self.register_buffer(
            "forms",
            torch.from_numpy(forms).long(),
            persistent=False,
        )

    def forward(self, batch: dict[str, Any] | None = None, **kwargs) -> torch.Tensor:
        inputs = dict(batch) if batch is not None else dict(kwargs)
        inputs.pop("targets", None)
        inputs.pop("labels", None)

        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop("attention_mask")
        slot_pos = inputs.pop("slot_pos")
        state_len = inputs.pop("state_len")
        question_mask = inputs.pop("question", None)

        # Mantém metadados de grid na CPU conforme contrato `host_inputs` do LFM3-VL
        vision_kwargs: dict[str, Any] = {}
        for key, val in inputs.items():
            if key in ("image_grid_thw", "video_grid_thw") and isinstance(val, torch.Tensor):
                vision_kwargs[key] = val.cpu()
            else:
                vision_kwargs[key] = val

        res = self._d1_slots.slot_logits(
            self.backbone,
            input_ids,
            attention_mask,
            slot_pos,
            state_len=state_len,
            question=question_mask,
            **vision_kwargs,
        )
        codes_device = res.codes.device
        gather_t = (
            self.gather_tokens
            if self.gather_tokens.device == codes_device
            else self.gather_tokens.to(codes_device)
        )
        forms_t = self.forms if self.forms.device == codes_device else self.forms.to(codes_device)
        return res.codes[:, gather_t][:, forms_t].max(dim=-1).values.float()


def _get_encoder_layers(model: nn.Module):
    """Recupera a lista sequencial de 24 camadas do módulo de linguagem do d1-2B."""
    base = getattr(model, "backbone", model)
    if hasattr(base, "get_base_model"):
        base = base.get_base_model()
    if hasattr(base, "model"):
        base = base.model
    if hasattr(base, "language_model") and hasattr(base.language_model, "layers"):
        return base.language_model.layers
    return getattr(base, "layers", [])


def _apply_paca_gradient_boundary(model: nn.Module, upper_k: int | None) -> None:
    """Desconecta grafo/autograd abaixo das upper_k camadas superiores via hooks."""
    layers = _get_encoder_layers(model)
    total_layers = len(layers)
    if upper_k is None or upper_k <= 0 or upper_k >= total_layers:
        return
    boundary_idx = total_layers - upper_k - 1

    def detach_input_pre_hook(module, args, kwargs):
        if not (module.training and torch.is_grad_enabled()):
            return args, kwargs
        if args and isinstance(args[0], torch.Tensor):
            args = (args[0].detach(),) + args[1:]
        elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = kwargs["hidden_states"].detach()
        return args, kwargs

    def boundary_output_hook(module, inputs, output):
        if not (module.training and torch.is_grad_enabled()):
            return output
        if isinstance(output, tuple):
            return (output[0].detach().requires_grad_(True),) + output[1:]
        return output.detach().requires_grad_(True)

    pre_handle = layers[0].register_forward_pre_hook(detach_input_pre_hook, with_kwargs=True)
    boundary_handle = layers[boundary_idx].register_forward_hook(boundary_output_hook)
    if not hasattr(model, "_paca_hooks"):
        model._paca_hooks = []
    model._paca_hooks.extend([pre_handle, boundary_handle])


def update_paca_runtime(model: nn.Module, new_upper_k: int | None) -> None:
    """Atualiza dinamicamente hooks, checkpointing e flags requires_grad das camadas LoRA."""
    layers = _get_encoder_layers(model)
    total_layers = len(layers)

    if hasattr(model, "_paca_hooks"):
        for handle in model._paca_hooks:
            handle.remove()
        model._paca_hooks = []

    for layer in layers:
        if hasattr(layer, "_paca_orig_gradient_checkpointing"):
            layer.gradient_checkpointing = layer._paca_orig_gradient_checkpointing

    if new_upper_k is not None and 0 < new_upper_k < total_layers:
        boundary_idx = total_layers - new_upper_k - 1
        for idx in range(boundary_idx + 1):
            layer = layers[idx]
            if hasattr(layer, "gradient_checkpointing"):
                if not hasattr(layer, "_paca_orig_gradient_checkpointing"):
                    layer._paca_orig_gradient_checkpointing = layer.gradient_checkpointing
                layer.gradient_checkpointing = False
        _apply_paca_gradient_boundary(model, new_upper_k)

    boundary_layer = (
        0 if (new_upper_k is None or new_upper_k >= total_layers) else total_layers - new_upper_k
    )
    for name, param in model.named_parameters():
        if not is_d1_adapter_key(name) or ".layers." not in name:
            continue
        try:
            layer_idx = int(name.split(".layers.")[1].split(".")[0])
        except (ValueError, IndexError):
            continue
        param.requires_grad = layer_idx >= boundary_layer


def get_d1_state_dict(model: nn.Module, trainable_only: bool = True) -> dict[str, torch.Tensor]:
    """Extrai exclusivamente os adaptadores LoRA/SoRA do DecisionD1 na CPU."""
    state_dict: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if is_d1_adapter_key(name):
            if trainable_only and not param.requires_grad:
                continue
            detached = param.detach()
            state_dict[name] = detached.cpu() if detached.is_cuda else detached.clone()
    assert_adapter_only_payload(state_dict)
    return state_dict


def set_d1_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Carrega adaptadores agregados no DecisionD1 preservando device/dtype e rank dinâmico."""
    from utils.fl_math import resize_model_to_pruned

    assert_adapter_only_payload(state_dict)
    resize_model_to_pruned(model, state_dict)


def _release_unused_vision_modules(backbone: nn.Module) -> None:
    """Libera torre visual e projetor quando o dataset é puramente textual."""
    targets = [backbone]
    if hasattr(backbone, "model"):
        targets.append(backbone.model)
    for holder in targets:
        for attr in ("vision_tower", "multi_modal_projector"):
            if hasattr(holder, attr) and getattr(holder, attr) is not None:
                setattr(holder, attr, None)


def setup_d1_model(
    run_config: dict[str, Any],
    device: torch.device | str,
    *,
    is_qlora: bool = False,
    dataset: str | None = None,
    num_classes: int = 2,
    release_vision: bool | str | None = None,
) -> tuple[DecisionD1, Any]:
    """Instancia o LiquidAI/d1-2B em BF16 ou 4-bit NF4 com LoRA/SoRA e PaCA."""
    from peft_common.sora import SORA_MODES, apply_peft_lora_modules, apply_sora_modules

    torch.set_float32_matmul_precision("high")
    model_cfg = run_config.get("model", {})
    name = model_cfg.get("name", MODEL_NAME)
    revision = model_cfg.get("revision", MODEL_REVISION)
    question = model_cfg.get("question", DEFAULT_QUESTION)
    lora_cfg = model_cfg.get("lora", {})
    qcfg = model_cfg.get("qlora", {})
    is_sora = lora_cfg.get("mode", "with_lora") in SORA_MODES
    target_device = torch.device(device)
    ds_name = dataset or run_config.get("dataset", {}).get("name", "Banking77")
    should_release_vision = (
        release_vision
        if isinstance(release_vision, bool)
        else resolve_release_vision(run_config, dataset=ds_name)
    )
    compute_dtype = (
        torch.bfloat16
        if target_device.type == "cuda"
        and getattr(torch, qcfg.get("compute_dtype", "bfloat16")) == torch.bfloat16
        and torch.cuda.is_bf16_supported()
        else (torch.float16 if target_device.type == "cuda" else torch.float32)
    )

    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=True)
    if is_qlora:
        from lora_d1.qlora import (
            apply_qlora,
            apply_qsora,
            load_quantized_d1_backbone,
            verify_qlora_freezing,
        )

        tag = "QSoRA" if is_sora else "QLoRA"
        logger.info(f"Carregando backbone d1-2B em 4-bit NF4 ({tag})...")
        backbone = load_quantized_d1_backbone(name, revision, target_device, qcfg=qcfg)
        if should_release_vision:
            _release_unused_vision_modules(backbone)
        if is_sora:
            backbone = apply_qsora(
                backbone,
                lora_cfg,
                gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", True)),
                prepare_kbit=bool(qcfg.get("prepare_kbit", False)),
                compute_dtype=compute_dtype,
            )
        else:
            backbone = apply_qlora(
                backbone,
                lora_cfg,
                gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", True)),
                prepare_kbit=bool(qcfg.get("prepare_kbit", False)),
                compute_dtype=compute_dtype,
            )
        configure_d1_tokenizer(tokenizer, name, revision, config=backbone.config)
        verify_qlora_freezing(backbone)
    else:
        logger.info(f"Carregando backbone d1-2B em {compute_dtype}...")
        backbone = AutoModel.from_pretrained(
            name,
            revision=revision,
            dtype=compute_dtype,
            trust_remote_code=True,
        )
        if should_release_vision:
            _release_unused_vision_modules(backbone)
        backbone.requires_grad_(False)
        backbone.eval()
        if model_cfg.get("gradient_checkpointing", True) and hasattr(
            backbone, "gradient_checkpointing_enable"
        ):
            if hasattr(backbone, "enable_input_require_grads"):
                backbone.enable_input_require_grads()
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            backbone.config.use_cache = False

        targets = _resolve_lora_targets(lora_cfg.get("target_modules", LORA_TARGETS))
        if is_sora:
            n_wrapped = apply_sora_modules(
                backbone,
                target_pattern=targets,
                r=int(lora_cfg.get("r", 8)),
                lora_alpha=int(lora_cfg.get("alpha", 16)),
                lora_dropout=float(lora_cfg.get("dropout", 0.05)),
                adapter_dtype=compute_dtype,
            )
            logger.info(f"[d1 SoRA] SoRA aplicado a {n_wrapped} módulos.")
        else:
            peft_config = LoraConfig(
                r=int(lora_cfg.get("r", 8)),
                lora_alpha=int(lora_cfg.get("alpha", 16)),
                lora_dropout=float(lora_cfg.get("dropout", 0.05)),
                bias=str(lora_cfg.get("bias", "none")),
                target_modules=targets,
            )
            backbone = apply_peft_lora_modules(backbone, peft_config, adapter_dtype=compute_dtype)
        configure_d1_tokenizer(tokenizer, name, revision, config=backbone.config)

    classes_file = run_config.get("dataset", {}).get("classes_file")
    if dataset and dataset != run_config.get("dataset", {}).get("name"):
        classes_file = None
    options = resolve_d1_options(ds_name, num_classes, classes_file=classes_file)

    model = DecisionD1(
        backbone=backbone,
        tokenizer=tokenizer,
        options=options,
        question=question,
        model_name=name,
        revision=revision,
        compute_dtype=compute_dtype,
    )
    model.has_lora = True
    if not is_qlora:
        model = model.to(target_device)
    else:
        model.gather_tokens = model.gather_tokens.to(target_device)
        model.forms = model.forms.to(target_device)

    paca_cfg = model_cfg.get("paca", {})
    if paca_cfg.get("enabled", False):
        upper_k = paca_cfg.get("upper_layers", 24)
        update_paca_runtime(model, upper_k)
        logger.info(f"PaCA ativado nas {upper_k} camadas superiores do d1-2B")

    return model, tokenizer
