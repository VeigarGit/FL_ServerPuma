import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from datasets import ClassLabel, load_dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from .sora import SoRAWrappedLinear, SparseAdamW

DEFAULT_CONFIG_PATH = Path("train_config.yml")
VALID_LORA_MODES = {"with_lora", "without_lora", "both", "with_sora_no_schedule", "with_sora_schedule"}
SORA_MODES = {"with_sora_no_schedule", "with_sora_schedule"}

class CLIPForClassification(nn.Module):
    """
    Adaptador para transformar o CLIP em um classificador de imagens.
    Esta classe encapsula o encoder de visão do CLIP e adiciona uma camada linear
    no topo (head) para realizar a classificação em N classes, mantendo o backbone original congelado.
    """
    def __init__(self, vision_model, num_classes, freeze_vision=True):
        super().__init__()
        self.vision_model = vision_model
        self.num_classes = num_classes
        hidden_size = self.vision_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Congela os parâmetros do backbone para garantir um Fine-Tuning eficiente (apenas o head ou adaptadores treinam)
        if freeze_vision:
            for param in self.vision_model.parameters():
                param.requires_grad = False

    def forward(self, pixel_values, labels=None):
        """
        Passagem para frente (Forward pass).
        Extrai as características globais da imagem através do vision_model e as projeta 
        no espaço de classes via o classificador linear.
        """
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        pooled_output = vision_outputs.pooler_output
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        return {"logits": logits, "loss": loss}
    
class CustomCollator:
    """
    Padronização do processamento de lotes (batches).
    Converte imagens brutas e rótulos de texto em tensores prontos para o PyTorch/GPU,
    utilizando o processador oficial do CLIP para garantir que o pré-processamento (resize/norm)
    seja idêntico ao do pré-treinamento.
    """
    def __init__(self, processor, label_to_idx, device):
        self.processor = processor
        self.label_to_idx = label_to_idx
        self.device = device

    def __call__(self, batch):
        images = [item["image"] for item in batch]
        labels = [encode_label(item["label"], self.label_to_idx) for item in batch]
        
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].to(self.device)
        label_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
        
        return pixel_values, label_tensor

def get_device():
    """Detecção de Hardware. Identifica se o treino ocorrerá em GPU (CUDA) ou CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    """Captura o caminho do arquivo de configuração YAML via linha de comando."""
    parser = argparse.ArgumentParser(description="Fine-tune CLIP vision encoder for classification.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def load_config(config_path):
    """Gestão de Configurações. Carrega, valida chaves obrigatórias e normaliza modos de LoRA."""
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("The YAML config must define a mapping at the top level.")

    required_keys = {"dataset", "model", "training", "optimizer", "scheduler", "output"}
    missing_keys = required_keys - config.keys()
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise KeyError(f"Missing required config sections: {missing}")

    lora_config = config["model"].get("lora", {})
    lora_mode = lora_config.get("mode")
    if lora_mode is None:
        if "enabled" in lora_config:
            lora_mode = "with_lora" if lora_config["enabled"] else "without_lora"
        else:
            lora_mode = "with_lora"
        lora_config["mode"] = lora_mode

    if lora_mode not in VALID_LORA_MODES:
        expected = ", ".join(sorted(VALID_LORA_MODES))
        raise ValueError(f"Invalid model.lora.mode: {lora_mode!r}. Expected one of: {expected}")

    return config


def resolve_label_metadata(dataset_split):
    """
    Mapeamento de Metadados de Classe.
    Extrai nomes de classes e cria dicionários de conversão (nome <-> ID) para garantir 
    consistência entre o dataset e a camada de classificação.
    """
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
    label_to_idx = {label: idx for label in unique_labels} 
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    label_to_idx.update({str(label): idx for idx, label in enumerate(unique_labels)})
    return class_names, label_to_idx

def encode_label(label, label_to_idx):
    """Normalização de Labels. Converte valores brutos do dataset para índices inteiros."""
    if isinstance(label, torch.Tensor):
        label = label.item()

    if label in label_to_idx:
        return label_to_idx[label]

    string_label = str(label)
    if string_label in label_to_idx:
        return label_to_idx[string_label]

    raise KeyError(f"Unknown label value: {label!r}")

def build_dataloaders(config, processor, device):
    """
    Pipeline de Dados.
    Carrega o dataset, realiza o split treino/teste e instancia os DataLoaders 
    com o collator customizado para processamento em GPU.
    """
    dataset_config = config["dataset"]
    training_config = config["training"]

    dataset = load_dataset(dataset_config["name"])
    split_dataset = dataset["train"].train_test_split(
        test_size=training_config["test_size"],
        shuffle=True,
        seed=training_config["seed"],
    )
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    class_names, label_to_idx = resolve_label_metadata(dataset["train"])

    collate_fn = CustomCollator(processor=processor,label_to_idx=label_to_idx, device=device)
    batch_size = training_config["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, eval_loader, class_names

def apply_sora(model, lora_config, upper_k=None):
    """
    Injeção de SoRA (Sparse LoRA).
    Substitui camadas Lineares por versões 'Wrapped' que contêm adaptadores esparsos.
    Suporta PaCA (Partial Calibration) se upper_k for definido.
    """
    target_modules = lora_config["target_modules"]
    r = lora_config["r"]
    alpha = lora_config["alpha"]
    dropout = lora_config["dropout"]

    vision = model.vision_model
    total_layers = len(vision.encoder.layers)
    modules_map = dict(vision.named_modules())
    replacements = []

    for name in list(modules_map.keys()):
        for target in target_modules:
            if not name.endswith(target):
                continue
            
            # Filtro PaCA: Verifica se a camada está entre as últimas 'upper_k'
            if upper_k is not None:
                if "encoder.layers." not in name:
                    continue
                try:
                    layer_idx = int(name.split("encoder.layers.")[1].split(".")[0])
                    if layer_idx < total_layers - upper_k:
                        continue 
                except:
                    continue

            parts = name.rsplit(".", 1)
            parent = modules_map[parts[0]] if len(parts) == 2 else vision
            child_name = parts[1] if len(parts) == 2 else parts[0]

            original_linear = getattr(parent, child_name)
            if isinstance(original_linear, nn.Linear):
                replacements.append((parent, child_name, original_linear))

    for parent, child_name, original_linear in replacements:
        wrapped = SoRAWrappedLinear(original_linear, r=r, lora_alpha=alpha, lora_dropout=dropout)
        setattr(parent, child_name, wrapped)

    print(f"SoRA aplicado a {len(replacements)} módulos (Últimas {upper_k or 'todas'} camadas)")


def build_model(config, num_classes, device):
    """
    Fábrica de Modelos.
    Instancia o CLIP com otimização SDPA (Scaled Dot Product Attention) e aplica 
    as técnicas de adaptação (LoRA ou SoRA) especificadas.
    """
    model_config = config["model"]
    lora_config = model_config["lora"]

    # Carrega o modelo com Scaled Dot Product Attention (SDPA)
    clip_model = CLIPModel.from_pretrained(model_config["name"], attn_implementation="sdpa")
    vision_model = clip_model.vision_model
    del clip_model

    model = CLIPForClassification(vision_model, num_classes)

    mode = lora_config["mode"]
    
    # Verifica configurações de PaCA (ajuste apenas em camadas superiores)
    paca_config = config["model"].get("paca", {})
    upper_k = paca_config.get("upper_layers") if paca_config.get("enabled") else None

    if mode == "with_lora":
        peft_config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["alpha"],
            target_modules=lora_config["target_modules"],
            lora_dropout=lora_config["dropout"],
            bias=lora_config["bias"],
        )
        model.vision_model = get_peft_model(model.vision_model, peft_config)
    elif mode in SORA_MODES:
        apply_sora(model, lora_config, upper_k=upper_k)

    # Função auxiliar externa para resumo de parâmetros (omitida no snippet por brevidade)
    # print_trainable_summary(model, mode) 
    model.to(device)
    return model

@torch.no_grad()
def benchmark_attention(model, loader, device, num_batches=10):
    """
    Benchmarking de Infraestrutura.
    Mede a latência por batch e o consumo de VRAM, validando o impacto do SDPA na performance.
    """
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for i, (pixel_values, _) in enumerate(loader):
        if i >= num_batches: break
        _ = model(pixel_values=pixel_values.to(device))
    end.record()
    
    torch.cuda.synchronize()
    time_ms = start.elapsed_time(end)
    vram_mb = torch.cuda.max_memory_allocated(device) / 1024**2

    print(f"\nBENCHMARK SDPA: {time_ms/num_batches:.1f}ms/batch | VRAM: {vram_mb:.1f}MB\n")

def quantize_weights(state_dict):
    """
    Otimização de Armazenamento.
    Converte tensores de ponto flutuante para INT8 linear, reduzindo o tamanho do 
    arquivo final em ~4x com mínima perda de precisão.
    """
    quantized_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            scale = torch.max(torch.abs(v)) / 127.0
            q_weight = torch.clamp((v / scale).round(), -128, 127).to(torch.int8)
            quantized_dict[k] = {'dtype': 'int8', 'scale': scale, 'weights': q_weight}
        else:
            quantized_dict[k] = v
    return quantized_dict


def resolve_run_modes(config):
    """Lógica de Experimentos. Determina se haverá treinos comparativos (com vs sem LoRA)."""
    lora_mode = config["model"]["lora"]["mode"]
    if lora_mode == "both":
        return ["with_lora", "without_lora"]
    return [lora_mode]


def build_run_config(config, run_mode):
    """Gera um clone da configuração global ajustado para um modo de execução específico."""
    run_config = yaml.safe_load(yaml.safe_dump(config))
    run_config["model"]["lora"]["mode"] = run_mode
    return run_config


def build_output_path(base_output_path, run_mode, multiple_runs):
    """Define o caminho final do arquivo de pesos baseado no experimento atual."""
    output_path = Path(base_output_path)
    if not multiple_runs:
        return output_path

    return output_path.with_name(f"{output_path.stem}_{run_mode}{output_path.suffix}")


def build_optimizer(model, config):
    """
    Configuração da Otimização.
    Instancia o AdamW tradicional e o SparseAdamW (para SoRA), lidando com a 
    separação de parâmetros de portão (gates) e adaptadores.
    """
    optimizer_config = config["optimizer"]
    sora_config = config["model"].get("sora", {})
    mode = config["model"]["lora"]["mode"]
    is_sora = mode in SORA_MODES

    if is_sora:
        gate_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "sora" in name and "gate" in name:
                gate_params.append(param)
            else:
                other_params.append(param)

        if not other_params and not gate_params:
            raise ValueError("No trainable parameters found.")

        main_optimizer = AdamW(
            other_params,
            lr=optimizer_config["lr"],
            weight_decay=optimizer_config["weight_decay"],
        )

        sparse_lr = sora_config.get("sparse_lr") or optimizer_config["lr"]

        if mode == "with_sora_schedule":
            lambda_schedule = sora_config.get("lambda_schedule")
            max_lambda = sora_config.get("max_lambda")
            lambda_num = sora_config.get("lambda_num")
        else:
            lambda_schedule = None
            max_lambda = None
            lambda_num = None

        sparse_optimizer = SparseAdamW(
            sparse_lambda=sora_config.get("sparse_lambda_2", 3e-4),
            lambda_schedule=lambda_schedule,
            max_lambda=max_lambda,
            lambda_num=lambda_num,
            params=gate_params,
            lr=sparse_lr,
            weight_decay=0.0,
        )

        print(f"Gate params: {sum(p.numel() for p in gate_params):,}")
        print(f"Other trainable params: {sum(p.numel() for p in other_params):,}")
        return main_optimizer, sparse_optimizer

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found. Check the LoRA and classifier setup.")

    return AdamW(
        trainable_params,
        lr=optimizer_config["lr"],
        weight_decay=optimizer_config["weight_decay"],
    ), None


def build_scheduler(optimizer, config):
    """Controle de Decay. Define a estratégia de redução da taxa de aprendizado."""
    scheduler_config = config["scheduler"]
    return StepLR(
        optimizer,
        step_size=scheduler_config["step_size"],
        gamma=scheduler_config["gamma"],
    )


def train_epoch(model, loader, optimizer, sparse_optimizer=None, sparse_lambda=0.0):
    """
    Loop de Treino por Época.
    Calcula a perda CE (Cross-Entropy Loss) e a penalidade de esparsidade, executando o backpropagation 
    em ambos os otimizadores (se aplicável).
    """
    model.train()
    total_ce_loss = 0.0
    total_sparse_loss = 0.0
    total_loss = 0.0

    for pixel_values, labels in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad()
        if sparse_optimizer is not None:
            sparse_optimizer.zero_grad()

        outputs = model(pixel_values=pixel_values, labels=labels)
        ce_loss = outputs["loss"]
        loss = ce_loss

        sparse_loss_val = 0.0
        if sparse_optimizer is not None and sparse_lambda > 0:
            gate_params = [p for n, p in model.named_parameters() if "sora" in n and "gate" in n]
            sparse_loss = sum(torch.sum(torch.abs(p)) for p in gate_params)
            p_total = sum(p.numel() for p in gate_params)
            if p_total > 0:
                sparse_loss_val = sparse_loss.item() / p_total
                loss = ce_loss + sparse_lambda * sparse_loss / p_total

        loss.backward()
        optimizer.step()
        if sparse_optimizer is not None:
            sparse_optimizer.step()

        total_ce_loss += ce_loss.item()
        total_sparse_loss += sparse_loss_val
        total_loss += loss.item()

    n = max(len(loader), 1)
    return {
        "ce_loss": total_ce_loss / n,
        "sparse_loss": total_sparse_loss / n,
        "total_loss": total_loss / n,
    }


def compute_gate_sparsity(model):
    """Monitoramento de Esparsidade. Calcula a porcentagem de portões SoRA que foram zerados."""
    total = 0
    zeros = 0
    for n, p in model.named_parameters():
        if "sora" in n and "gate" in n:
            total += p.numel()
            zeros += (p.data == 0).sum().item()
    return zeros, total


def evaluate(model, loader):
    """Validação de Resultados. Calcula a acurácia final usando scikit-learn."""
    model.eval()
    preds = []
    true_labels = []

    with torch.no_grad():
        for pixel_values, labels in tqdm(loader, desc="eval", leave=False):
            outputs = model(pixel_values=pixel_values)
            logits = outputs["logits"]
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pred)
            true_labels.extend(labels.cpu().numpy())

    return accuracy_score(true_labels, preds)