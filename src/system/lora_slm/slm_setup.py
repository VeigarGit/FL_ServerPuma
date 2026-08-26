import argparse
from pathlib import Path
import torch
import torch.nn as nn
import yaml
from datasets import ClassLabel, load_dataset
from peft import LoraConfig, AdaLoraConfig, get_peft_model
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
from PIL import Image

try:
    from sora import SoRAWrappedLinear, SparseAdamW, GateSparsifier
except ImportError:
    from lora_slm.sora import SoRAWrappedLinear, SparseAdamW, GateSparsifier

DEFAULT_CONFIG_PATH = Path("train_config.yml")
VALID_LORA_MODES = {
    "with_lora",
    "with_adalora",
    "without_lora",
    "both",
    "with_sora_no_schedule",
    "with_sora_schedule",
}
SORA_MODES = {"with_sora_no_schedule", "with_sora_schedule"}

class SLMForClassification(nn.Module):
    """
    Adaptador para transformar o SLM (Qwen-VL) em um classificador de imagens.
    Instancia o backbone multimodal e adiciona uma camada linear no topo (head)
    para extrair a classe a partir do último token (Sequence Classification).
    """
    def __init__(self, slm_model, num_classes, freeze_base=True):
        super().__init__()
        self.slm_model = slm_model
        self.num_classes = num_classes
        
        # O Qwen2VLModel retorna hidden_states. O hidden_size está no config
        if hasattr(self.slm_model.config, "hidden_size"):
            hidden_size = self.slm_model.config.hidden_size
        elif hasattr(self.slm_model.config, "text_config"):
            hidden_size = self.slm_model.config.text_config.hidden_size
        else:
            raise ValueError("Could not determine hidden_size from config")
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.classifier.to(next(self.slm_model.parameters()).dtype)

        # Congela o backbone base. O PEFT/LoRA cuidará de injetar adaptadores nas camadas treináveis
        if freeze_base:
            for param in self.slm_model.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None, labels=None, **kwargs):
        # Passagem para frente do Qwen-VL (apenas o modelo base, sem LM Head)
        outputs = self.slm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs
        )
        
        # O Qwen2VLModel retorna 'last_hidden_state' no formato (batch_size, seq_len, hidden_size)
        last_hidden_state = outputs.last_hidden_state
        
        # Seleciona o representation do ÚLTIMO token não-pad da sequência de cada exemplo no batch
        if attention_mask is not None:
            sequence_lengths = attention_mask.cumsum(dim=1).argmax(dim=1)
            batch_size = last_hidden_state.shape[0]
            pooled_output = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            pooled_output = last_hidden_state[:, -1, :]
            
        # Classificação Linear em cima do último token
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        return {"logits": logits, "loss": loss}


class CustomCollator:
    """
    Collator customizado para VLM (Vision-Language Model).
    Prepara o prompt de texto fixo ("Classify this image") + a imagem 
    utilizando o AutoProcessor que cuida da tokenização e 
    criação dos `pixel_values` dinâmicos (image_grid_thw).
    """
    def __init__(self, processor, label_to_idx):
        self.processor = processor
        self.label_to_idx = label_to_idx
        
        # Prompt fixo em formato conversacional
        self.prompt_text = "Classify this image into its category. Respond only with the correct classification."

    def __call__(self, batch):
        texts = []
        images = []
        labels = []
        
        for item in batch:
            # Em imagens PIL, às vezes o formato precisa de conversão para RGB
            image = item["image"]
            if not isinstance(image, Image.Image):
                # Fallback se for tensor ou array
                image = Image.fromarray(image.numpy() if hasattr(image, 'numpy') else image)
            if image.mode != "RGB":
                image = image.convert("RGB")
            images.append(image)
            
            labels.append(encode_label(item["label"], self.label_to_idx))
            
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.prompt_text}
                ]}
            ]
            try:
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except ValueError:
                # Fallback para modelos base (não-instruct) que não possuem chat template
                # Tenta o formato Qwen2-VL primeiro, se não, usa Qwen-VL
                if hasattr(self.processor.tokenizer, 'added_tokens_encoder') and '<|vision_start|>' in self.processor.tokenizer.added_tokens_encoder:
                    text = f"<|vision_start|><|image_pad|><|vision_end|>{self.prompt_text}"
                else:
                    text = f"<image>{self.prompt_text}"
            texts.append(text)
            
        # O processor cria input_ids, attention_mask, pixel_values e image_grid_thw
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        
        return inputs, label_tensor

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SLM for classification.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to YAML config.")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    lora_config = config["model"].get("lora", {})
    lora_mode = lora_config.get("mode", "with_lora")
    lora_config["mode"] = lora_mode
    return config

def resolve_label_metadata(dataset_split):
    label_feature = dataset_split.features.get("label")
    raw_labels = dataset_split["label"]
    unique_labels = sorted(set(raw_labels))
    if isinstance(label_feature, ClassLabel):
        class_names = list(label_feature.names)
        label_to_idx = {idx: idx for idx in range(len(class_names))}
        label_to_idx.update({name: idx for idx, name in enumerate(class_names)})
        return class_names, label_to_idx
    if unique_labels and isinstance(unique_labels[0], str):
        class_names = unique_labels
        label_to_idx = {name: idx for idx, name in enumerate(class_names)}
        return class_names, label_to_idx
    class_names = [str(label) for label in unique_labels]
    label_to_idx = {idx: idx for idx, label in enumerate(unique_labels)}
    label_to_idx.update({str(label): idx for idx, label in enumerate(unique_labels)})
    return class_names, label_to_idx

def encode_label(label, label_to_idx):
    if isinstance(label, torch.Tensor):
        label = label.item()
    if label in label_to_idx:
        return label_to_idx[label]
    string_label = str(label)
    if string_label in label_to_idx:
        return label_to_idx[string_label]
    raise KeyError(f"Unknown label value: {label!r}")

def build_dataloaders(config, processor, device):
    dataset_config = config["dataset"]
    training_config = config["training"]
    dataset = load_dataset(dataset_config["name"])
    split_dataset = dataset["train"].train_test_split(test_size=training_config["test_size"], shuffle=True, seed=training_config["seed"])
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    class_names, label_to_idx = resolve_label_metadata(dataset["train"])
    collate_fn = CustomCollator(processor=processor, label_to_idx=label_to_idx)
    batch_size = training_config["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    return train_loader, eval_loader, class_names

def apply_sora(model, lora_config, upper_k=None):
    target_modules = lora_config["target_modules"]
    r = lora_config["r"]
    alpha = lora_config["alpha"]
    dropout = lora_config["dropout"]

    base_lm = model.slm_model
    layers_container = _get_encoder_layers(model)
    
    total_layers = len(layers_container)
    modules_map = dict(base_lm.named_modules())
    replacements = []

    for name in list(modules_map.keys()):
        for target in target_modules:
            if not name.endswith(target):
                continue

            if upper_k is not None:
                if ".layers." not in name:
                    continue
                try:
                    layer_idx = int(name.split(".layers.")[1].split(".")[0])
                    if layer_idx < total_layers - upper_k:
                        continue 
                except:
                    continue

            parts = name.rsplit(".", 1)
            parent = modules_map[parts[0]] if len(parts) == 2 else base_lm
            child_name = parts[1] if len(parts) == 2 else parts[0]

            original_linear = getattr(parent, child_name)
            if isinstance(original_linear, nn.Linear):
                replacements.append((parent, child_name, original_linear))

    for parent, child_name, original_linear in replacements:
        wrapped = SoRAWrappedLinear(original_linear, r=r, lora_alpha=alpha, lora_dropout=dropout)
        setattr(parent, child_name, wrapped)

    print(f"SoRA aplicado a {len(replacements)} módulos (Últimas {upper_k or 'todas'} camadas)")


def _get_encoder_layers(model):
    base_lm = getattr(model, "slm_model", model)
    if hasattr(base_lm, 'language_model') and hasattr(base_lm.language_model, 'layers'):
        return base_lm.language_model.layers
    if hasattr(base_lm, 'model') and hasattr(base_lm.model, 'layers'):
        return base_lm.model.layers
    return getattr(base_lm, 'layers', [])

def _apply_paca_gradient_boundary(model, upper_k):
    layers = _get_encoder_layers(model)
    total_layers = len(layers)
    if upper_k is None or upper_k >= total_layers: return
    boundary_idx = total_layers - upper_k - 1
    
    def detach_hook(module, input, output):
        if not module.training: return output
        if isinstance(output, tuple):
            return (output[0].detach(),) + output[1:]
        return output.detach()
    
    handle = layers[boundary_idx].register_forward_hook(detach_hook)
    if not hasattr(model, '_paca_hooks'):
        model._paca_hooks = []
    model._paca_hooks.append(handle)

def update_paca_runtime(model, new_upper_k):
    layers = _get_encoder_layers(model)
    total_layers = len(layers)
    if hasattr(model, '_paca_hooks'):
        for handle in model._paca_hooks:
            handle.remove()
        model._paca_hooks = []
    if new_upper_k is not None:
        _apply_paca_gradient_boundary(model, new_upper_k)
        
    boundary_layer = 0 if (new_upper_k is None or new_upper_k >= total_layers) else total_layers - new_upper_k
    for name, param in model.named_parameters():
        if "sora" not in name and "lora" not in name: continue
        if ".layers." not in name: continue
        try:
            layer_idx = int(name.split(".layers.")[1].split(".")[0])
            if layer_idx < boundary_layer:
                if param.requires_grad: param.requires_grad = False
            else:
                if not param.requires_grad: param.requires_grad = True
        except:
            continue

def build_model(config, num_classes, device):
    model_config = config["model"]
    lora_config = model_config["lora"]
    
    model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    
    slm_base = AutoModel.from_pretrained(
        model_config["name"],
        dtype=model_dtype,
        attn_implementation="sdpa" 
    )
    
    model = SLMForClassification(slm_base, num_classes)
    
    mode = lora_config["mode"]
    paca_config = config["model"].get("paca", {})
    upper_k = paca_config.get("upper_layers") if paca_config.get("enabled") else None

    if mode == "with_lora":
        total_layers = len(_get_encoder_layers(model))
        layers_to_transform = None  # Allows dynamic expansion of PaCA in runtime
        peft_config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["alpha"],
            target_modules=lora_config["target_modules"],
            lora_dropout=lora_config["dropout"],
            bias=lora_config["bias"],
            layers_to_transform=layers_to_transform,
            layers_pattern="layers",
        )
        model.slm_model = get_peft_model(model.slm_model, peft_config)
    elif mode in SORA_MODES:
        apply_sora(model, lora_config, upper_k=None)  # Allows dynamic expansion of PaCA in runtime

    if upper_k is not None:
        _apply_paca_gradient_boundary(model, upper_k)

    model.to(device)
    return model


def build_optimizer(model, config):
    optimizer_config = config["optimizer"]
    sora_config = config["model"].get("sora", {})
    mode = config["model"]["lora"]["mode"]
    is_sora = mode in SORA_MODES

    if is_sora:
        gate_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            if "sora" in name and "gate" in name: gate_params.append(param)
            else: other_params.append(param)
        sparse_lr = sora_config.get("sparse_lr") or optimizer_config["lr"]
        unified_optimizer = AdamW([
            {'params': other_params, 'lr': optimizer_config["lr"], 'weight_decay': optimizer_config["weight_decay"]},
            {'params': gate_params, 'lr': sparse_lr, 'weight_decay': 0.0},
        ])
        
        lambda_schedule = sora_config.get("lambda_schedule") if mode == "with_sora_schedule" else None
        max_lambda = sora_config.get("max_lambda") if mode == "with_sora_schedule" else None
        lambda_num = sora_config.get("lambda_num") if mode == "with_sora_schedule" else None
        
        gate_sparsifier = GateSparsifier(
            gate_params=gate_params,
            sparse_lambda=sora_config.get("sparse_lambda_2", 3e-4),
            gate_lr=sparse_lr,
            lambda_schedule=lambda_schedule,
            max_lambda=max_lambda,
            lambda_num=lambda_num,
        )
        return unified_optimizer, gate_sparsifier

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return AdamW(trainable_params, lr=optimizer_config["lr"], weight_decay=optimizer_config["weight_decay"]), None

def build_scheduler(optimizer, config):
    scheduler_config = config["scheduler"]
    return StepLR(optimizer, step_size=scheduler_config["step_size"], gamma=scheduler_config["gamma"])

def train_epoch(model, loader, optimizer, sparse_optimizer=None, sparse_lambda=0.0, accumulation_steps=1):
    model.train()
    total_ce_loss = 0.0
    total_sparse_loss = 0.0
    total_loss = 0.0
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    if sparse_optimizer is not None and hasattr(sparse_optimizer, 'step_lambda'):
        sparse_optimizer.step_lambda()

    if sparse_optimizer is not None and sparse_lambda > 0:
        gate_params = [p for n, p in model.named_parameters() if "sora" in n and "gate" in n]
        gate_params_total = sum(p.numel() for p in gate_params)
    else:
        gate_params = []
        gate_params_total = 0

    optimizer.zero_grad(set_to_none=True)
    if sparse_optimizer is not None: sparse_optimizer.zero_grad(set_to_none=True)

    for step, (inputs_dict, labels) in enumerate(tqdm(loader, desc="train", leave=False, disable=True)):
        
        labels = labels.to(device=device, non_blocking=True)
        for k in inputs_dict.keys():
            if inputs_dict[k].is_floating_point():
                inputs_dict[k] = inputs_dict[k].to(device=device, dtype=model_dtype, non_blocking=True)
            else:
                inputs_dict[k] = inputs_dict[k].to(device=device, non_blocking=True)

        outputs = model(labels=labels, **inputs_dict)
        ce_loss = outputs["loss"]
        loss = ce_loss / accumulation_steps

        loss.backward()
        
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
            optimizer.step()
            if sparse_optimizer is not None: sparse_optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
            if sparse_optimizer is not None: sparse_optimizer.zero_grad(set_to_none=True)

        total_ce_loss += ce_loss.item()
        total_loss += (loss.item() * accumulation_steps)

    sparse_loss_val = 0.0
    if gate_params_total > 0:
        with torch.no_grad():
            sparse_loss = sum(torch.sum(torch.abs(p)) for p in gate_params)
            sparse_loss_val = sparse_loss.item() / gate_params_total

    n = max(len(loader), 1)
    return {"ce_loss": total_ce_loss / n, "sparse_loss": sparse_loss_val, "total_loss": total_loss / n}

def evaluate(model, loader):
    model.eval()
    preds = []
    true_labels = []

    with torch.no_grad():
        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        
        for inputs_dict, labels in tqdm(loader, desc="eval", leave=False, disable=True):
            labels = labels.to(device=device, non_blocking=True)
            for k in inputs_dict.keys():
                if inputs_dict[k].is_floating_point():
                    inputs_dict[k] = inputs_dict[k].to(device=device, dtype=model_dtype, non_blocking=True)
                else:
                    inputs_dict[k] = inputs_dict[k].to(device=device, non_blocking=True)
            
            outputs = model(**inputs_dict)
            logits = outputs["logits"]
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pred)
            true_labels.extend(labels.cpu().numpy())

    return accuracy_score(true_labels, preds)

def resolve_run_modes(config):
    lora_mode = config["model"]["lora"]["mode"]
    return ["with_lora", "without_lora"] if lora_mode == "both" else [lora_mode]

def build_run_config(config, run_mode):
    run_config = yaml.safe_load(yaml.safe_dump(config))
    run_config["model"]["lora"]["mode"] = run_mode
    return run_config

def build_output_path(base_output_path, run_mode, multiple_runs):
    output_path = Path(base_output_path)
    if not multiple_runs: return output_path
    return output_path.with_name(f"{output_path.stem}_{run_mode}{output_path.suffix}")

def compute_gate_sparsity(model):
    total = 0
    zeros = 0
    for n, p in model.named_parameters():
        if "sora" in n and "gate" in n:
            total += p.numel()
            zeros += (p.data == 0).sum().item()
    return zeros, total

def quantize_weights(state_dict):
    quantized_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            scale = torch.max(torch.abs(v)) / 127.0
            q_weight = torch.clamp((v / scale).round(), -128, 127).to(torch.int8)
            quantized_dict[k] = {'dtype': 'int8', 'scale': scale, 'weights': q_weight}
        else:
            quantized_dict[k] = v
    return quantized_dict

def benchmark_attention(model, loader, device, num_batches=10):
    pass
