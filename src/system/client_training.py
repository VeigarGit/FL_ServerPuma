"""Construção de modelos, adaptação FedALA e loop de treinamento local do cliente FL."""

import argparse
import copy
import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from lora_clip.clip_setup import (  # noqa: E402
    build_model,
    build_run_config,
    load_config,
    resolve_run_modes,
)
from lora_slm.qlora import (  # noqa: E402
    enable_qlora_in_config,
    get_qlora_state_dict,
    is_qlora_enabled,
    set_qlora_state_dict,
)
from model import SimpleModel  # noqa: E402

from utils.fl_math import (  # noqa: E402
    dequantization,
    evaluate_model,
    resize_model_to_pruned,
    set_parameters,
)

logger = logging.getLogger(__name__)


def resolve_device(args: argparse.Namespace) -> torch.device:
    """Resolve o acelerador disponível (MPS > CUDA > CPU) e configura CUDA_VISIBLE_DEVICES."""
    if args.device == "mps":
        if torch.backends.mps.is_available():
            logger.info("Using Metal Performance Shaders (MPS) for MacBook GPU acceleration.")
        else:
            logger.warning(
                "MPS is not available. Falling back to CUDA if available, otherwise CPU."
            )
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Checking for MPS...")
            if torch.backends.mps.is_available():
                logger.info("Falling back to Metal Performance Shaders (MPS).")
                args.device = "mps"
            else:
                logger.warning("CUDA and MPS not available. Using CPU.")
                args.device = "cpu"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    logger.info(f"Device set to: {args.device}")
    return torch.device(args.device)


def resolve_heterogeneous_paca(args: argparse.Namespace) -> None:
    """Ajusta args.paca in-place quando configurado via lista determinística ou sorteio."""
    if args.paca_list is not None:
        paca_values = [int(x.strip()) for x in args.paca_list.split(",")]
        args.paca = paca_values[args.client_idx % len(paca_values)]
        logger.info(
            f"PaCA Heterogêneo (lista): Cliente {args.client_idx} usando PaCA = {args.paca}"
        )
    elif args.random_paca:
        rng = random.Random(args.client_idx * 1000 + args.run_id)
        args.paca = rng.randint(args.paca_min, args.paca_max)
        logger.info(
            f"PaCA Heterogêneo (aleatório): Cliente {args.client_idx} sorteou PaCA = {args.paca} "
            f"(intervalo [{args.paca_min}, {args.paca_max}])"
        )


def _compute_model_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode("utf-8"))
        flat_tensor = tensor.detach().cpu().contiguous().reshape(-1)
        digest.update(flat_tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_client_model(
    args: argparse.Namespace, device: torch.device
) -> tuple[nn.Module, dict | None]:
    """Instancia o modelo (CNN, CLIP ou SLM) e retorna (model, run_config)."""
    run_config = None
    match args.model:
        case "cnn":
            with torch.random.fork_rng(devices=[], enabled=args.mode == "decentralized"):
                if args.mode == "decentralized":
                    torch.random.default_generator.manual_seed(args.seed)
                match args.dataset:
                    case "MNIST":
                        model = SimpleModel(in_features=1, num_classes=10, dim=1024)
                    case "Cifar10":
                        model = SimpleModel(
                            in_features=args.in_features, num_classes=10, dim=args.dim
                        )
                    case "Cifar100":
                        model = SimpleModel(
                            in_features=args.in_features,
                            num_classes=args.num_classes,
                            dim=args.dim,
                        )
                    case _:
                        logger.error(f"Dataset inválido: {args.dataset}")
                        sys.exit(1)
            if args.mode == "decentralized":
                args.initial_model_sha256 = _compute_model_sha256(model)
                logger.info(
                    "Client %d: CNN inicial comum | seed=%d | treino=%d | SHA256=%s",
                    args.client_idx,
                    args.seed,
                    args.training_seed,
                    args.initial_model_sha256,
                )
        case "clip" | "slm":
            if args.model == "clip":
                config = load_config(args.config)
            else:
                from lora_slm.slm_setup import load_config as load_config_slm

                config = load_config_slm(args.config)

            strategy_to_mode = {
                "lora": "with_lora",
                "adalora": "with_adalora",
                "sora_with_schedule": "with_sora_schedule",
                "sora_no_schedule": "with_sora_no_schedule",
            }
            if args.strategy in strategy_to_mode:
                config["model"]["lora"]["mode"] = strategy_to_mode[args.strategy]

            if args.qlora:
                if args.model != "slm" or args.mode != "centralized":
                    logger.error(
                        "--qlora é suportado apenas com --model slm no modo centralizado."
                    )
                    sys.exit(1)
                enable_qlora_in_config(config, args.strategy)
                logger.info(
                    "QLoRA ATIVADO: backbone 4-bit NF4 + DoubleQuant, adaptadores LoRA 16 bits."
                )

            config["model"].setdefault("paca", {})
            if "lora" in config["model"]:
                config["model"]["lora"]["r"] = args.rank

            if args.paca is not None and args.paca > 0:
                config["model"]["paca"]["enabled"] = True
                config["model"]["paca"]["upper_layers"] = args.paca
            else:
                config["model"]["paca"]["enabled"] = False
                config["model"]["paca"]["upper_layers"] = None

            if args.model == "clip":
                run_mode = resolve_run_modes(config)[0]
                run_config = build_run_config(config, run_mode=run_mode)
                model = build_model(config=run_config, num_classes=args.num_classes, device=device)
            else:
                from lora_slm.slm_setup import (
                    build_model as build_model_slm,
                )
                from lora_slm.slm_setup import (
                    build_run_config as build_run_config_slm,
                )
                from lora_slm.slm_setup import (
                    resolve_run_modes as resolve_run_modes_slm,
                )

                run_mode = resolve_run_modes_slm(config)[0]
                run_config = build_run_config_slm(config, run_mode=run_mode)
                model = build_model_slm(
                    config=run_config, num_classes=args.num_classes, device=device
                )
        case "lfm":
            import yaml
            from lora_lfm.lfm_setup import setup_lfm_model

            config_path = (
                Path(args.config) if args.config else Path("src/system/lora_lfm/train_config.yml")
            )
            if not config_path.exists():
                config_path = Path("lora_lfm/train_config.yml")
            with config_path.open("r", encoding="utf-8") as f:
                run_config = yaml.safe_load(f)

            if args.rank:
                run_config["model"]["lora"]["r"] = args.rank
                run_config["model"]["lora"]["alpha"] = 2 * args.rank

            if args.paca is not None and args.paca > 0:
                run_config["model"]["paca"]["enabled"] = True
                run_config["model"]["paca"]["upper_layers"] = args.paca
            else:
                run_config["model"]["paca"]["enabled"] = False
                run_config["model"]["paca"]["upper_layers"] = None

            run_config["is_qlora"] = bool(args.qlora)
            if args.qlora:
                logger.info(
                    "QLoRA ATIVADO para LFM: backbone 4-bit NF4 + DoubleQuant, LoRA 16 bits."
                )

            model, _ = setup_lfm_model(run_config, device, is_qlora=bool(args.qlora))
        case _:
            logger.error(f"Modelo inválido: {args.model}")
            sys.exit(1)

    if not args.qlora:  # QLoRA: backbone 4-bit já alocado via device_map
        model = model.to(device)
    return model, run_config


def map_sequential_to_simplemodel(state_dict: dict) -> dict:
    """Mapeia chaves de nn.Sequential (pós-poda OPALA) de volta para SimpleModel."""
    mapped_dict = {}
    mapping = {
        "0.weight": "conv1.0.weight",
        "0.bias": "conv1.0.bias",
        "3.weight": "conv2.0.weight",
        "3.bias": "conv2.0.bias",
        "7.weight": "fc1.0.weight",
        "7.bias": "fc1.0.bias",
        "9.weight": "fc.weight",
        "9.bias": "fc.bias",
    }
    for sequential_key, simple_key in mapping.items():
        if sequential_key in state_dict:
            mapped_dict[simple_key] = state_dict[sequential_key]
    return mapped_dict


def extract_trainable_model_state(
    model: nn.Module, model_type: str, run_config: dict | None = None
) -> dict:
    """Extrai o dicionário de pesos treináveis de acordo com a arquitetura e modo QLoRA."""
    if model_type == "lfm":
        from lora_lfm.lfm_setup import get_lfm_state_dict

        return get_lfm_state_dict(model, trainable_only=True)
    if model_type in ("clip", "slm"):
        if run_config is not None and is_qlora_enabled(run_config):
            return get_qlora_state_dict(model)
        if model_type == "clip":
            from lora_clip.sora import get_trainable_state_dict

            return get_trainable_state_dict(model)
        from lora_slm.sora import get_trainable_state_dict as get_trainable_state_dict_slm

        return get_trainable_state_dict_slm(model)
    return model.state_dict()


def load_trainable_model_state(model: nn.Module, state_dict: dict, model_type: str) -> None:
    """Carrega pesos agregados no modelo local (suporta CNN, CLIP, SLM e LFM)."""
    if model_type == "lfm":
        from lora_lfm.lfm_setup import set_lfm_state_dict

        set_lfm_state_dict(model, state_dict)
    elif model_type == "clip":
        from lora_clip.sora import set_trainable_state_dict

        set_trainable_state_dict(model, state_dict)
    elif model_type == "slm":
        from lora_slm.sora import set_trainable_state_dict as set_trainable_state_dict_slm

        set_trainable_state_dict_slm(model, state_dict)
    else:
        model.load_state_dict(state_dict)


def local_initialization(ala, received_global_model, model, mask=None):
    """Aplica Adaptive Local Aggregation (FedALA / FedALA_LoRA) no modelo local."""
    ala.adaptive_local_aggregation(received_global_model, model, mask=mask)


def _log_trainable_param_count(model: nn.Module) -> None:
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if getattr(model, "_last_trainable_count", None) != trainable_count:
        logger.info(f"[DIAG] Parâmetros treináveis neste round: {trainable_count:,}")
        model._last_trainable_count = trainable_count


def local_training(
    model,
    state_dict,
    prune,
    train_loader,
    test_loader,
    learning_rate=0.01,
    round_num=2,
    alaarg=1,
    ala=None,
    model_type="cnn",
    run_config=None,
):
    """Executa sincronização de pesos globais, FedALA e uma época de treino local."""
    if state_dict and not isinstance(next(iter(state_dict.values())), torch.Tensor):
        state_dict = dequantization(state_dict)

    personalized_acc = None

    if round_num >= 2 and prune == 1 and model_type == "cnn":
        state_dict = map_sequential_to_simplemodel(state_dict)

    device = next(model.parameters()).device

    if model_type == "lfm":
        from lora_lfm.lfm_setup import set_lfm_state_dict, update_paca_runtime
        from lora_lfm.qlora import build_lfm_optimizer
        from lora_lfm.trainer import train_epoch_lfm

        if state_dict:
            set_lfm_state_dict(model, state_dict)

        paca_cfg = run_config["model"].get("paca", {}) if run_config else {}
        if paca_cfg.get("enabled"):
            update_paca_runtime(model, paca_cfg.get("upper_layers"))

        _log_trainable_param_count(model)

        brier_weight = (
            run_config.get("training", {}).get("brier_weight", 1.0) if run_config else 1.0
        )
        grad_accum = (
            run_config.get("training", {}).get("gradient_accumulation_steps", 1)
            if run_config
            else 1
        )
        is_qlora = bool(run_config.get("is_qlora", False)) if run_config else False
        optimizer = build_lfm_optimizer(model, run_config or {}, is_qlora=is_qlora)

        loss, acc = train_epoch_lfm(
            model,
            train_loader,
            optimizer,
            None,
            device,
            brier_weight=brier_weight,
            grad_accum_steps=grad_accum,
        )
        logger.info(f"LFM Treino Local - Loss (CE+Brier): {loss:.4f} | Acc: {acc:.2f}%")
        return extract_trainable_model_state(model, model_type, run_config), personalized_acc

    # === SINCRONIZAÇÃO DE PESOS GLOBAIS E FEDALA ===
    if model_type in ["clip", "slm"]:
        # FedALA_LoRA só se aplica a CLIP (model_type != 'slm').
        # Evita deepcopy desnecessário do SLM inteiro (~4-6 GB).
        needs_ala = alaarg == 0 and round_num >= 2 and model_type != "slm"

        if needs_ala:
            global_model = copy.deepcopy(model)
            global_model = resize_model_to_pruned(global_model, state_dict)
            logger.info(f"Client {ala.cid}: Applying FedALA_LoRA (Adaptive Local Aggregation)...")
            local_initialization(ala, global_model, model)
            personalized_acc, _ = evaluate_model(model, test_loader)
            logger.info(f"Client {ala.cid}: Post-ALA Test Accuracy: {personalized_acc:.2f}%")
            del global_model
        elif is_qlora_enabled(run_config):
            # QLoRA: injeta apenas os adaptadores LoRA + head (set_peft_model_state_dict)
            set_qlora_state_dict(model, state_dict)
        else:
            resize_model_to_pruned(model, state_dict)
    else:
        local_model = resize_model_to_pruned(model, state_dict)
        state = copy.deepcopy(local_model)

        if alaarg == 0 and round_num == 2:
            logger.info(f"Client {ala.cid}: Applying FedALA (Adaptive Local Aggregation)...")
            local_initialization(ala, state, model)

        set_parameters(model, state)

        if alaarg == 0 and round_num >= 2:
            personalized_acc, _ = evaluate_model(model, test_loader)
            logger.info(f"Client {ala.cid}: Post-ALA Test Accuracy: {personalized_acc:.2f}%")

    model.train()
    loss_fn = nn.CrossEntropyLoss()
    device = next(model.parameters()).device

    # === 1. SELEÇÃO DE OTIMIZADOR E LOOP DE ÉPOCA ===
    if model_type in ["clip", "slm"]:
        if model_type == "clip":
            from lora_clip.clip_setup import build_optimizer, train_epoch, update_paca_runtime
        else:
            from lora_slm.slm_setup import build_optimizer, train_epoch, update_paca_runtime

        paca_config = run_config["model"].get("paca", {})
        if paca_config.get("enabled"):
            update_paca_runtime(model, paca_config.get("upper_layers"))

        _log_trainable_param_count(model)

        if (
            model_type == "slm"
            and hasattr(model, "slm_model")
            and hasattr(model.slm_model, "gradient_checkpointing_enable")
        ):
            model.slm_model.gradient_checkpointing_enable()

        optimizer, sparse_optimizer = build_optimizer(model, run_config)
        is_sora = run_config["model"]["lora"]["mode"] in [
            "with_sora_no_schedule",
            "with_sora_schedule",
        ]
        sparse_lambda = (
            run_config["model"].get("sora", {}).get("sparse_lambda", 0.0) if is_sora else 0.0
        )

        if model_type == "slm":
            accumulation_steps = run_config.get("training", {}).get(
                "gradient_accumulation_steps", 1
            )
            metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                sparse_optimizer,
                sparse_lambda,
                accumulation_steps,
            )
        else:
            metrics = train_epoch(model, train_loader, optimizer, sparse_optimizer, sparse_lambda)

        logger.info(
            f"Métricas do Treino Local - CE Loss: {metrics['ce_loss']:.4f} | "
            f"Sparse Loss: {metrics['sparse_loss']:.4f}"
        )
    else:
        _log_trainable_param_count(model)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        model_dtype = next(model.parameters()).dtype

        for x, y in train_loader:
            x = x.to(device, dtype=model_dtype) if x.is_floating_point() else x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(x)
            if isinstance(output, dict):
                output = output["logits"]

            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

    # === 2. SELEÇÃO DOS PESOS QUE SERÃO ENVIADOS PARA O SERVIDOR ===
    return extract_trainable_model_state(model, model_type, run_config), personalized_acc
