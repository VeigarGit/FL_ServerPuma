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
from transformers import CLIPModel, CLIPProcessor

from .sora import SoRAWrappedLinear, SparseAdamW, GateSparsifier

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
        self.classifier.to(next(self.vision_model.parameters()).dtype)

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


def _get_encoder_layers(model):
    """Encontra as camadas do encoder em modelos PEFT-wrapped ou raw."""
    vision = model.vision_model
    # PEFT envolve o modelo: base_model.model contém o original
    if hasattr(vision, 'base_model') and hasattr(vision.base_model, 'model'):
        return vision.base_model.model.encoder.layers
    return vision.encoder.layers


def _apply_paca_gradient_boundary(model, upper_k):
    """
    Registra um forward hook que interrompe a propagação de gradientes
    nas camadas abaixo do boundary do PaCA.
    
    Referência teórica: Em partial fine-tuning, camadas inferiores sem
    parâmetros treináveis não precisam de gradientes de ativação.
    (Head2Toe — Evci et al. 2022, Ladder Side-Tuning — Sung et al. 2022)
    """
    encoder_layers = _get_encoder_layers(model)
    total_layers = len(encoder_layers)
    
    if upper_k >= total_layers:
        return  # PaCA cobre todas as camadas, nada a cortar
    
    boundary_idx = total_layers - upper_k - 1
    
    def detach_hook(module, input, output):
        # Só interrompe gradientes durante treino; em eval é no-op
        if not module.training:
            return output
        # CLIPEncoderLayer retorna (hidden_states,) ou (hidden_states, attn_weights)
        if isinstance(output, tuple):
            return (output[0].detach(),) + output[1:]
        return output.detach()
    
    handle = encoder_layers[boundary_idx].register_forward_hook(detach_hook)
    # Armazena o handle no modelo para poder remover se necessário
    if not hasattr(model, '_paca_hooks'):
        model._paca_hooks = []
    model._paca_hooks.append(handle)
    
    print(f"PaCA gradient boundary: camadas 0-{boundary_idx} não propagarão gradientes "
          f"(economia de ~{(boundary_idx + 1) / total_layers * 100:.0f}% no backward pass)")


def update_paca_runtime(model, new_upper_k):
    """
    Atualiza o PaCA em runtime sem reconstruir o modelo.

    Chamado pelo cliente quando o servidor ajusta o PaCA adaptativamente.
    Garante que a economia computacional reflita o novo valor de PaCA,
    atualizando tanto o corte de gradientes quanto o estado dos adaptadores.

    Operações:
      1. Remove os forward hooks de gradiente existentes (_paca_hooks)
      2. Registra novo hook de boundary na posição correta
      3. Congela adaptadores (SoRA/LoRA) em camadas fora do novo PaCA
      4. Descongela adaptadores dentro do novo PaCA (necessário quando PaCA sobe)

    Referência teórica: PaCA (Partial Calibration) restringe o fine-tuning
    às últimas upper_k camadas do encoder. Camadas abaixo do boundary não
    precisam de gradientes, e seus adaptadores não devem ser atualizados.

    Args:
        model: O modelo CLIPForClassification com adaptadores injetados.
        new_upper_k: Número de camadas superiores a treinar (novo valor PaCA).
                     Se None, todas as camadas ficam ativas.
    """
    encoder_layers = _get_encoder_layers(model)
    total_layers = len(encoder_layers)

    # --- 1. REMOVE HOOKS ANTIGOS ---
    # Os hooks são armazenados em model._paca_hooks pelo _apply_paca_gradient_boundary.
    # Remover antes de registrar novos evita hooks duplicados ou conflitantes.
    if hasattr(model, '_paca_hooks'):
        for handle in model._paca_hooks:
            handle.remove()
        model._paca_hooks = []

    # --- 2. REGISTRA NOVO HOOK NA POSIÇÃO CORRETA ---
    # _apply_paca_gradient_boundary já lida com o caso upper_k >= total_layers
    # (retorna sem registrar hook, permitindo backward completo).
    if new_upper_k is not None:
        _apply_paca_gradient_boundary(model, new_upper_k)

    # --- 3/4. FREEZE/UNFREEZE DE ADAPTADORES ---
    # Camadas abaixo do boundary não precisam de gradientes nos adaptadores.
    # Congelar seus parâmetros traz 3 benefícios:
    #   a) PyTorch não constrói o grafo de autograd para esses ops (menos memória)
    #   b) O optimizer não itera sobre eles (menos overhead por step)
    #   c) get_trainable_state_dict não os inclui (menos dados enviados ao servidor)
    if new_upper_k is None or new_upper_k >= total_layers:
        boundary_layer = 0  # Todas as camadas estão ativas
    else:
        boundary_layer = total_layers - new_upper_k

    frozen_count = 0
    unfrozen_count = 0

    for name, param in model.named_parameters():
        # Filtra apenas parâmetros de adaptadores (SoRA ou LoRA via PEFT)
        if "sora" not in name and "lora" not in name:
            continue
        # Filtra apenas parâmetros dentro de encoder layers
        if "encoder.layers." not in name:
            continue

        try:
            layer_idx = int(name.split("encoder.layers.")[1].split(".")[0])
        except (ValueError, IndexError):
            continue

        if layer_idx < boundary_layer:
            # Camada fora do PaCA: congelar adaptador
            if param.requires_grad:
                param.requires_grad = False
                frozen_count += 1
        else:
            # Camada dentro do PaCA: garantir que o adaptador está ativo
            if not param.requires_grad:
                param.requires_grad = True
                unfrozen_count += 1

    if frozen_count > 0 or unfrozen_count > 0:
        print(f"PaCA runtime update: upper_k={new_upper_k}, "
              f"boundary=layer {boundary_layer}/{total_layers}, "
              f"frozen={frozen_count} params, unfrozen={unfrozen_count} params")


def build_model(config, num_classes, device):
    """
    Fábrica de Modelos.
    Instancia o CLIP com otimização SDPA (Scaled Dot Product Attention) e aplica 
    as técnicas de adaptação (LoRA ou SoRA) especificadas.
    """
    model_config = config["model"]
    lora_config = model_config["lora"]

    # Carrega o modelo com Scaled Dot Product Attention (SDPA)
    clip_model = CLIPModel.from_pretrained(
            model_config["name"], 
            attn_implementation="sdpa",
            torch_dtype=torch.float32
        )    
    vision_model = clip_model.vision_model
    del clip_model

    model = CLIPForClassification(vision_model, num_classes)

    mode = lora_config["mode"]
    
    # Verifica configurações de PaCA (ajuste apenas em camadas superiores)
    paca_config = config["model"].get("paca", {})
    upper_k = paca_config.get("upper_layers") if paca_config.get("enabled") else None

    if mode == "with_lora":
        # PaCA: restringe LoRA às últimas upper_k camadas do encoder
        # Referência: PEFT LoraConfig.layers_to_transform (documentação oficial HuggingFace)
        total_layers = len(vision_model.encoder.layers)
        layers_to_transform = list(range(total_layers - upper_k, total_layers)) if upper_k else None
        
        peft_config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["alpha"],
            target_modules=lora_config["target_modules"],
            lora_dropout=lora_config["dropout"],
            bias=lora_config["bias"],
            layers_to_transform=layers_to_transform,
        )
        model.vision_model = get_peft_model(model.vision_model, peft_config)
    elif mode == "with_adalora":
        # PaCA: restringe LoRA às últimas upper_k camadas do encoder
        # Referência: PEFT LoraConfig.layers_to_transform (documentação oficial HuggingFace)
        total_layers = len(vision_model.encoder.layers)
        layers_to_transform = list(range(total_layers - upper_k, total_layers)) if upper_k else None
        adalora_config = config["model"]["adalora"]

        peft_config = AdaLoraConfig(
            r=lora_config["r"],
            init_r=adalora_config["init_r"],
            target_r=adalora_config["target_r"],
            tinit=adalora_config["tinit"],
            tfinal=adalora_config["tfinal"],
            deltaT=adalora_config["deltaT"],
            beta1=adalora_config["beta1"],
            beta2=adalora_config["beta2"],
            total_step=adalora_config["total_step"],
            orth_reg_weight=adalora_config.get("orth_reg_weight", 0.5),
            lora_alpha=lora_config["alpha"],
            target_modules=lora_config["target_modules"],
            lora_dropout=lora_config["dropout"],
            bias=lora_config["bias"],
            layers_to_transform=layers_to_transform,
        )
        model.vision_model = get_peft_model(model.vision_model, peft_config)

    # elif mode == "with_adalora":
    #     total_layers = len(vision_model.encoder.layers)
    #     layers_to_transform = list(range(total_layers - upper_k, total_layers)) if upper_k else None
    #     
    #     adalora_config = config["model"].get("adalora", {})
    #     
    #     peft_config = AdaLoraConfig(
    #         init_r=lora_config["r"],
    #         target_r=adalora_config.get("target_r", lora_config["r"]),
    #         tinit=adalora_config.get("tinit", 200),
    #         tfinal=adalora_config.get("tfinal", 1000),
    #         deltaT=adalora_config.get("deltaT", 10),
    #         beta1=adalora_config.get("beta1", 0.85),
    #         beta2=adalora_config.get("beta2", 0.85),
    #         total_step=adalora_config.get("total_step", 1500),
    #         lora_alpha=lora_config["alpha"],
    #         target_modules=lora_config["target_modules"],
    #         lora_dropout=lora_config["dropout"],
    #         layers_to_transform=layers_to_transform,
    #     )
    #     model.vision_model = get_peft_model(model.vision_model, peft_config)
    elif mode in SORA_MODES:
        apply_sora(model, lora_config, upper_k=upper_k)

    # PaCA: Registra hook para interromper backward pass nas camadas inferiores
    if upper_k is not None:
        _apply_paca_gradient_boundary(model, upper_k)

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

        sparse_lr = sora_config.get("sparse_lr") or optimizer_config["lr"]

        # --- OTIMIZAÇÃO FL: Otimizador Unificado ---
        # Antes: 2 otimizadores separados (AdamW + SparseAdamW)
        #   → 2× zero_grad(), 2× step(), 2× estado interno (momentum/variância)
        # Agora: 1 AdamW com 2 param_groups + GateSparsifier leve
        #   → 1× zero_grad(), 1× step(), 1× estado + softshrink
        #
        # A matemática do SoRA (Proximal Gradient Descent) é preservada:
        #   Passo 1: AdamW atualiza TODOS os params (adapters + gates) → grupo unificado
        #   Passo 2: GateSparsifier aplica softshrink nos gates → proximal step do L1
        unified_optimizer = AdamW([
            {'params': other_params, 'lr': optimizer_config["lr"],
             'weight_decay': optimizer_config["weight_decay"]},
            {'params': gate_params, 'lr': sparse_lr, 'weight_decay': 0.0},
        ])

        if mode == "with_sora_schedule":
            lambda_schedule = sora_config.get("lambda_schedule")
            max_lambda = sora_config.get("max_lambda")
            lambda_num = sora_config.get("lambda_num")
        else:
            lambda_schedule = None
            max_lambda = None
            lambda_num = None

        # GateSparsifier: aplica APENAS o proximal step (softshrink) nos gates.
        # Substitui o SparseAdamW completo, que era um AdamW inteiro + softshrink.
        gate_sparsifier = GateSparsifier(
            gate_params=gate_params,
            sparse_lambda=sora_config.get("sparse_lambda_2", 3e-4),
            gate_lr=sparse_lr,
            lambda_schedule=lambda_schedule,
            max_lambda=max_lambda,
            lambda_num=lambda_num,
        )

        # print(f"Gate params: {sum(p.numel() for p in gate_params):,}")
        # print(f"Other trainable params: {sum(p.numel() for p in other_params):,}")
        return unified_optimizer, gate_sparsifier

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
    
    device = next(model.parameters()).device

    # Ativa o schedule linear de lambda (ex: 3e-4 → 7e-4) para aumentar a pressão
    # de esparsidade. No FL, o otimizador é recriado a cada rodada, então avançamos
    # o schedule imediatamente para usar o lambda mais forte.
    if sparse_optimizer is not None and hasattr(sparse_optimizer, 'step_lambda'):
        sparse_optimizer.step_lambda()

    # Cache dos gate_params: evita scan de named_parameters() a cada batch
    # Os objetos Parameter não mudam de identidade durante o treino, apenas seus .data
    if sparse_optimizer is not None and sparse_lambda > 0:
        gate_params = [p for n, p in model.named_parameters() if "sora" in n and "gate" in n]
        gate_params_total = sum(p.numel() for p in gate_params)
    else:
        gate_params = []
        gate_params_total = 0

    # Cache: evita chamada a next(model.parameters()) a cada batch
    model_dtype = next(model.parameters()).dtype

    # Identifica o AdaLoRA do PEFT, que envolve apenas o vision_model.
    is_adalora = (
        hasattr(model, "vision_model")
        and hasattr(model.vision_model, "base_model")
        and hasattr(model.vision_model.base_model, "update_and_allocate")
    )
    if is_adalora and not hasattr(model, "adalora_global_step"):
        model.adalora_global_step = 0

    for pixel_values, labels in tqdm(loader, desc="train", leave=False, disable=True):
        pixel_values = pixel_values.to(device, dtype=model_dtype)
        labels = labels.to(device)
        
        # set_to_none=True: mais rápido que zerar tensores (evita alocação de zeros na GPU)
        optimizer.zero_grad(set_to_none=True)
        # GateSparsifier.zero_grad() é no-op (gates estão no otimizador unificado).
        # Mantido para compatibilidade com SparseAdamW no trainer standalone (main.py).
        if sparse_optimizer is not None:
            sparse_optimizer.zero_grad(set_to_none=True)

        outputs = model(pixel_values=pixel_values, labels=labels)
        ce_loss = outputs["loss"]
        loss = ce_loss

        loss.backward()
        optimizer.step()

        if is_adalora:
            model.vision_model.base_model.update_and_allocate(model.adalora_global_step)
            model.adalora_global_step += 1
        # GateSparsifier.step(): aplica APENAS softshrink (proximal step)
        # SparseAdamW.step(): AdamW completo + softshrink (trainer standalone)
        if sparse_optimizer is not None:
            sparse_optimizer.step()
            
        # # Poda dinâmica do AdaLoRA (PEFT)
        # if is_adalora:
        #     model.base_model.update_and_allocate(model.adalora_global_step)
        #     model.adalora_global_step += 1

        total_ce_loss += ce_loss.item()
        total_loss += loss.item()

    # Calcula a esparsidade APENAS 1 vez no final da época (Remove o gargalo de I/O da GPU)
    sparse_loss_val = 0.0
    if gate_params_total > 0:
        with torch.no_grad():
            sparse_loss = sum(torch.sum(torch.abs(p)) for p in gate_params)
            sparse_loss_val = sparse_loss.item() / gate_params_total

    n = max(len(loader), 1)
    return {
        "ce_loss": total_ce_loss / n,
        "sparse_loss": sparse_loss_val,
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
        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        
        for pixel_values, labels in tqdm(loader, desc="eval", leave=False, disable=True):
            pixel_values = pixel_values.to(device, dtype=model_dtype)
            outputs = model(pixel_values=pixel_values)
            logits = outputs["logits"]
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pred)
            true_labels.extend(labels.cpu().numpy())

    return accuracy_score(true_labels, preds)
