import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

# Compatibilidade defensiva com transformers < 5.21.0 para RoPE axial
try:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "axial" not in ROPE_INIT_FUNCTIONS:

        def _compute_axial_rope_parameters(config, device=None, **kwargs):
            rope_params = getattr(config, "rope_parameters", {}) or {}
            base = rope_params.get("rope_theta", 10000.0)
            hidden_size = getattr(config, "hidden_size", 1024)
            num_heads = getattr(config, "num_attention_heads", 16)
            spatial_dim = hidden_size // num_heads // 2
            exponents = (
                torch.arange(0, spatial_dim, 2, dtype=torch.float, device=device) / spatial_dim
            )
            inv_freq = 1.0 / (base**exponents)
            return inv_freq, 1.0

        ROPE_INIT_FUNCTIONS["axial"] = _compute_axial_rope_parameters
except (ImportError, AttributeError):
    pass

MODEL_NAME = "LiquidAI/LFM3-VL-2B"
MODEL_REVISION = "f83d2b0a2296aaa277f394990a9b8b949ddf9691"
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|w1|w2|w3)"


def _resolve_lora_targets(targets: str | list[str] | None) -> str | list[str]:
    """Restringe os módulos-alvo do LoRA estritamente ao language_model (padrão BonsaiToClef)."""
    if not targets:
        return LORA_TARGETS
    if isinstance(targets, str):
        return targets
    joined = "|".join(str(t) for t in targets)
    return rf".*language_model.*\.({joined})"


def encode(
    tokenizer, state: str, question: str, options: dict[str, str], max_length: int = 2048
) -> dict:
    """Codifica o prompt textual contendo as opções e a pergunta em uma única passagem."""
    if not isinstance(state, str) or not isinstance(question, str) or not question.strip():
        raise ValueError("state e question devem ser strings não vazias")
    if len(options) < 2:
        raise ValueError("Informe ao menos duas alternativas em options")

    content = question + "\nOptions:\n"
    spans = []
    for description in options.values():
        content += "- "
        start = len(content)
        content += description
        spans.append((start, len(content)))
        content += "\n"
    content += "\nState:\n" + state + "\nChoose exactly one option."

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    offset = prompt.index(content)
    tokens = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    if len(tokens["input_ids"]) > max_length:
        raise ValueError(
            f"Entrada com {len(tokens['input_ids'])} tokens excede limite de {max_length}"
        )

    positions = []
    opt_rows: list[int] = []
    opt_cols: list[int] = []
    opt_vals: list[float] = []
    for j, (start, end) in enumerate(spans):
        indices = [
            i
            for i, (a, b) in enumerate(tokens["offset_mapping"])
            if b > a and a < offset + end and b > offset + start
        ]
        if not indices:
            raise ValueError("Alternativa sem tokens mapeados no prompt")
        positions.append(indices)
        weight_val = 1.0 / len(indices)
        opt_rows.extend([j] * len(indices))
        opt_cols.extend(indices)
        opt_vals.extend([weight_val] * len(indices))

    return {
        "input_ids": tokens["input_ids"],
        "option_positions": positions,
        "option_ids": list(options.keys()),
        "_input_ids_tensor": torch.tensor(tokens["input_ids"], dtype=torch.long),
        "_opt_row_idx": torch.tensor(opt_rows, dtype=torch.long),
        "_opt_col_idx": torch.tensor(opt_cols, dtype=torch.long),
        "_opt_val": torch.tensor(opt_vals, dtype=torch.float32),
    }


def collate_records(records: list[dict], pad_id: int, device: torch.device | str = "cpu") -> dict:
    """Empacota os registros com padding à direita e cria pesos para pooling das alternativas."""
    batch = len(records)
    length = max(len(r["input_ids"]) for r in records)
    choices = max(len(r["option_ids"]) for r in records)

    ids = torch.full((batch, length), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    weights = torch.zeros(batch, choices, length, dtype=torch.float32)
    valid = torch.zeros(batch, choices, dtype=torch.bool)
    has_targets = "target" in records[0]
    targets = torch.zeros(batch, dtype=torch.long) if has_targets else None

    for i, record in enumerate(records):
        n = len(record["input_ids"])
        if "_input_ids_tensor" in record:
            ids[i, :n] = record["_input_ids_tensor"]
        else:
            ids[i, :n] = torch.tensor(record["input_ids"], dtype=torch.long)
        mask[i, :n] = 1

        if "_opt_row_idx" in record:
            weights[i, record["_opt_row_idx"], record["_opt_col_idx"]] = record["_opt_val"]
            valid[i, : len(record["option_positions"])] = True
        else:
            for j, pos in enumerate(record["option_positions"]):
                weights[i, j, pos] = 1.0 / len(pos)
                valid[i, j] = True

        if has_targets:
            targets[i] = int(record["target"])

    result = {
        "input_ids": ids.to(device),
        "attention_mask": mask.to(device),
        "option_weights": weights.to(device),
        "option_mask": valid.to(device),
    }
    if targets is not None:
        result["targets"] = targets.to(device)
    return result


class LFMDecisionDataset(Dataset):
    """Dataset federado que pré-codifica as perguntas bancárias em memória na inicialização."""

    def __init__(
        self,
        data: list[tuple[str, int]],
        tokenizer,
        options: dict[str, str],
        max_length: int = 2048,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.options = options
        self.max_length = max_length
        self.question = "Identify the customer's banking intent."
        self.records: list[dict] = []
        for text, label in data:
            record = encode(
                self.tokenizer,
                text,
                self.question,
                self.options,
                max_length=self.max_length,
            )
            record["target"] = int(label)
            self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


class LFMCollator:
    """Collator que empacota registros do LFMDecisionDataset."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        return collate_records(batch, self.pad_id)


class DecisionHead(nn.Module):
    """
    Cabeça de Decisão baseada em similaridade de cosseno com projeção query/option
    e escala adaptativa com limitação superior de magnitude.
    """

    def __init__(self, hidden_size: int = 2048, head_size: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.query = nn.Linear(hidden_size, head_size, bias=False)
        self.option = nn.Linear(hidden_size, head_size, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(
        self, query: torch.Tensor, options: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        q = F.normalize(self.query(self.norm(query.float())), dim=-1)
        k = F.normalize(self.option(self.norm(options.float())), dim=-1)
        scores = torch.einsum("bd,bcd->bc", q, k) * self.log_scale.exp().clamp(max=100.0)
        return scores.masked_fill(~valid, -torch.inf)


class DecisionLFM(nn.Module):
    """
    Modelo de Decisão com backbone LiquidAI/LFM3-VL-2B e DecisionHead.
    Executa inferência e treinamento em uma única passagem (sem autorregressão).
    """

    def __init__(self, backbone, tokenizer, head_size: int = 256, max_length: int = 2048):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.head_size = head_size

        # text_config hidden_size padrão do LFM3 é 2048
        hidden_size = getattr(
            getattr(backbone.config, "text_config", backbone.config), "hidden_size", 2048
        )
        self.head = DecisionHead(hidden_size=hidden_size, head_size=head_size)

        # Garante que a cabeça use FP32 para estabilidade numérica de normalização e cosseno
        self.head.to(torch.float32)
        self.has_lora = False

    def forward(self, batch: dict) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state  # [B, L, D]

        # Extrai representação da query no último token válido do prompt
        query_idx = batch["attention_mask"].sum(-1) - 1
        query = hidden[torch.arange(hidden.shape[0], device=hidden.device), query_idx]

        # Pooling das representações das alternativas ponderadas pelos tokens de cada span
        options = torch.bmm(batch["option_weights"].to(hidden.dtype), hidden)

        return self.head(query, options, batch["option_mask"])


def _get_encoder_layers(model: DecisionLFM):
    """Recupera a lista de camadas sequenciais do módulo de linguagem do LFM3."""
    backbone = getattr(model, "backbone", model)
    if hasattr(backbone, "base_model") and hasattr(backbone.base_model, "model"):
        backbone = backbone.base_model.model
    if hasattr(backbone, "language_model") and hasattr(backbone.language_model, "layers"):
        return backbone.language_model.layers
    if hasattr(backbone, "model") and hasattr(backbone.model, "layers"):
        return backbone.model.layers
    return getattr(backbone, "layers", [])


def _apply_paca_gradient_boundary(model: DecisionLFM, upper_k: int | None):
    """Aplica hook de gradiente para desconectar retropropagação abaixo das upper_k camadas."""
    layers = _get_encoder_layers(model)
    total_layers = len(layers)
    if upper_k is None or upper_k >= total_layers:
        return
    boundary_idx = total_layers - upper_k - 1

    def detach_hook(module, inputs, output):
        if not module.training:
            return output
        if isinstance(output, tuple):
            return (output[0].detach(),) + output[1:]
        return output.detach()

    handle = layers[boundary_idx].register_forward_hook(detach_hook)
    if not hasattr(model, "_paca_hooks"):
        model._paca_hooks = []
    model._paca_hooks.append(handle)


def update_paca_runtime(model: DecisionLFM, new_upper_k: int | None):
    """
    Atualiza dinamicamente a fronteira PaCA com base no índice de capacidade do cliente:
      1. Remove forward hooks antigos e registra o novo boundary hook.
      2. Congela (requires_grad=False) adaptadores LoRA abaixo da fronteira
         e ativa (requires_grad=True) adaptadores nas new_upper_k camadas superiores.
    """
    layers = _get_encoder_layers(model)
    total_layers = len(layers)

    if hasattr(model, "_paca_hooks"):
        for handle in model._paca_hooks:
            handle.remove()
        model._paca_hooks = []

    if new_upper_k is not None:
        _apply_paca_gradient_boundary(model, new_upper_k)

    boundary_layer = (
        0 if (new_upper_k is None or new_upper_k >= total_layers) else total_layers - new_upper_k
    )
    for name, param in model.named_parameters():
        if "lora" not in name and "sora" not in name:
            continue
        if ".layers." not in name:
            continue
        try:
            layer_idx = int(name.split(".layers.")[1].split(".")[0])
            if layer_idx < boundary_layer:
                if param.requires_grad:
                    param.requires_grad = False
            else:
                if not param.requires_grad:
                    param.requires_grad = True
        except (ValueError, IndexError):
            continue


def get_lfm_state_dict(model: DecisionLFM, trainable_only: bool = True) -> dict[str, torch.Tensor]:
    """
    Retorna os adaptadores LoRA e parâmetros da DecisionHead.
    Se trainable_only=True, retorna apenas os que têm requires_grad=True (PaCA no cliente).
    Se trainable_only=False, retorna todos os adaptadores e head (servidor global_state).
    Garante que o backbone base de 1.7B parâmetros nunca seja transmitido na rede.
    """
    state_dict = {}
    for name, param in model.named_parameters():
        if "lora" in name or "sora" in name or "head" in name:
            if trainable_only and not param.requires_grad:
                continue
            state_dict[name] = param.detach().cpu()
    return state_dict


def set_lfm_state_dict(model: DecisionLFM, state_dict: dict[str, torch.Tensor]):
    """
    Carrega os pesos treináveis agregados do servidor (LoRA + DecisionHead).
    Suporta dinamicamente mudanças de Rank adaptativo (slicing/redimensionamento).
    """
    from utils.fl_math import resize_model_to_pruned

    resize_model_to_pruned(model, state_dict)


def _release_unused_vision_modules(backbone: nn.Module) -> None:
    """Remove módulos de visão ociosos para economizar VRAM no processamento puramente textual."""
    for attr in ("vision_tower", "multi_modal_projector"):
        if hasattr(backbone, attr) and getattr(backbone, attr) is not None:
            setattr(backbone, attr, None)


def setup_lfm_model(
    run_config: dict, device: torch.device, is_qlora: bool = False
) -> tuple[DecisionLFM, AutoTokenizer]:
    """Instancia o modelo DecisionLFM em BF16 ou 4-bit NF4 com LoRA e DecisionHead."""
    torch.set_float32_matmul_precision("high")
    model_cfg = run_config.get("model", {})
    name = model_cfg.get("name", MODEL_NAME)
    revision = model_cfg.get("revision", MODEL_REVISION)
    head_size = model_cfg.get("head_size", 256)
    max_length = model_cfg.get("max_length", 2048)
    lora_cfg = model_cfg.get("lora", {})

    logger.info(f"Carregando tokenizer oficial de {name}...")
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if is_qlora:
        from lora_lfm.qlora import apply_qlora, load_quantized_lfm_backbone, verify_qlora_freezing

        logger.info("Carregando backbone LFM3 em 4-bit NF4 (QLoRA)...")
        backbone = load_quantized_lfm_backbone(name, revision=revision, device=device)
        _release_unused_vision_modules(backbone)
        backbone = apply_qlora(backbone, lora_cfg)
        model = DecisionLFM(backbone, tokenizer, head_size=head_size, max_length=max_length)
        model.has_lora = True
        model.to(device)
        verify_qlora_freezing(model)
    else:
        logger.info("Carregando backbone LFM3 em bfloat16...")
        backbone = AutoModel.from_pretrained(
            name,
            revision=revision,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        _release_unused_vision_modules(backbone)
        backbone.requires_grad_(False)
        backbone.eval()

        # Ativa gradient checkpointing apenas quando habilitado na configuração
        if model_cfg.get("gradient_checkpointing", False) and hasattr(
            backbone, "gradient_checkpointing_enable"
        ):
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # Injeta adaptadores LoRA estritamente no language_model
        rank = lora_cfg.get("r", 8)
        alpha = lora_cfg.get("alpha", 16)
        dropout = lora_cfg.get("dropout", 0.05)
        targets = _resolve_lora_targets(lora_cfg.get("target_modules", LORA_TARGETS))

        peft_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            target_modules=targets,
        )
        backbone = get_peft_model(backbone, peft_config)

        model = DecisionLFM(backbone, tokenizer, head_size=head_size, max_length=max_length)
        model.has_lora = True
        model.to(device)

    # Configuração de PaCA
    paca_cfg = model_cfg.get("paca", {})
    if paca_cfg.get("enabled", False):
        upper_k = paca_cfg.get("upper_layers", 4)
        update_paca_runtime(model, upper_k)
        logger.info(f"PaCA ativado nas {upper_k} camadas superiores do LFM3")

    return model, tokenizer
