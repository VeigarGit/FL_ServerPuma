# Responsabilidade: Define o classificador e constrói o modelo com os adaptadores
# selecionados. Família CLIP.
#
# Principais funções/classes: CLIPForClassification, build_model.

"""CLIP model support — model."""

import torch
import torch.nn as nn
from peft import LoraConfig
from peft import AdaLoraConfig
from peft import get_peft_model
from transformers import CLIPModel
from fl_serverpuma.models.clip.adaptation import apply_sora
from fl_serverpuma.models.clip.adaptation import _apply_paca_gradient_boundary
from fl_serverpuma.models.clip.config import SORA_MODES

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
            dtype=torch.float32
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
