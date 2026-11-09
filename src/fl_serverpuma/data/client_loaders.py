# Responsabilidade: Constrói os DataLoaders locais conforme dataset e família do modelo.
#
# Principais funções: load_data.

"""Client loaders."""

import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from fl_serverpuma.data.client_data import read_client_data
import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger(__name__)

def load_data(dataset, client_idx, device, is_train=True, batch_size=32, is_clip=False, is_slm=False, slm_config=None):
    train_data = read_client_data(dataset, client_idx, is_train)
    X, y = zip(*train_data)
    X = torch.stack(X)

    if is_slm:
        from fl_serverpuma.models.slm.data import CustomCollator
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(slm_config["model"]["name"])
        label_to_idx = {i: i for i in range(1000)}
        label_to_idx.update({str(i): i for i in range(1000)})
        collate_fn = CustomCollator(processor, label_to_idx)

        slm_data = []
        for x_tensor, y_tensor in zip(X, y):
            x_tensor = x_tensor.cpu().float()

            # Se o tensor vier com normalização do CLIP (valores negativos < -0.1)
            if x_tensor.min() < -0.1:
                clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1)
                clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1)
                if x_tensor.shape[0] == 3:
                    x_tensor = x_tensor * clip_std + clip_mean
                else:
                    x_tensor = x_tensor * 0.5 + 0.5
            elif x_tensor.min() < 0:
                # Normalização padrão [-1, 1]
                x_tensor = x_tensor * 0.5 + 0.5

            if x_tensor.max() <= 1.0:
                x_tensor = x_tensor * 255.0

            # Clamp to [0, 255] range
            x_tensor = torch.clamp(x_tensor, 0, 255)
            x_numpy = x_tensor.byte().numpy()

            if x_numpy.shape[0] in [1, 3]: # (C, H, W) -> (H, W, C)
                x_numpy = np.transpose(x_numpy, (1, 2, 0))
            if x_numpy.shape[-1] == 1:
                x_numpy = np.squeeze(x_numpy, axis=-1)

            pil_img = Image.fromarray(x_numpy)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            slm_data.append({"image": pil_img, "label": y_tensor.item()})

        pin_mem = True if hasattr(device, 'type') and device.type == 'cuda' else False
        return DataLoader(slm_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=pin_mem)

    # Redimensionamento e normalização vitais para o CLIP
    if is_clip:
        if X.shape[-2:] != (224, 224):
            resize = T.Resize((224, 224), antialias=True)
            X = resize(X)
        if not X.is_floating_point():
            X = X.float() / 255.0
        elif X.max() > 1.0:
            X = X / 255.0
        # Se os dados não tiverem normalização do CLIP, aplica a normalização oficial do CLIP
        if X.min() >= 0.0:
            normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
            X = normalize(X)

    # Garante que o rótulo é inteiro longo e move os dados INTEIROS para a GPU
    X = X.to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)

    dataset = torch.utils.data.TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
