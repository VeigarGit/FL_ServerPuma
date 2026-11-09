# Responsabilidade: Define o classificador e constrói o modelo com os adaptadores
# selecionados. Família SLM.
#
# Principais funções/classes: SLMForClassification, build_model.

"""SLM model support — model."""

import torch
import torch.nn as nn
from peft import LoraConfig
from peft import get_peft_model
from transformers import AutoModel
from fl_serverpuma.models.slm.adaptation import apply_sora
from fl_serverpuma.models.slm.adaptation import _get_encoder_layers
from fl_serverpuma.models.slm.adaptation import _apply_paca_gradient_boundary
from fl_serverpuma.models.slm.config import SORA_MODES

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
            layers_pattern="layers" if layers_to_transform is not None else None,
        )
        model.slm_model = get_peft_model(model.slm_model, peft_config)
    elif mode in SORA_MODES:
        apply_sora(model, lora_config, upper_k=None)  # Allows dynamic expansion of PaCA in runtime

    if upper_k is not None:
        _apply_paca_gradient_boundary(model, upper_k)

    model.to(device)
    return model
