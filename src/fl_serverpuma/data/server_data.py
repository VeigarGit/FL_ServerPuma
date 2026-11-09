# Responsabilidade: Carrega e prepara os dados de teste globais, incluindo tokenização para
# SLM.
#
# Principais funções: load_test_data.

"""Server server data."""

import logging
import torch
from torch.utils.data import DataLoader
import numpy as np
from fl_serverpuma.data.client_data import read_client_data

logger = logging.getLogger(__name__)

def _prepare_slm_test_loader(state, X, y, batch_size=32):
    from PIL import Image
    import numpy as np

    if not hasattr(state, 'slm_collator') or state.slm_collator is None:
        from fl_serverpuma.models.slm.data import CustomCollator
        from transformers import AutoProcessor
        model_name = getattr(state, 'config', {}).get("model", {}).get("name", "Qwen/Qwen3-VL-2B-Instruct") if getattr(state, 'config', None) else "Qwen/Qwen3-VL-2B-Instruct"
        processor = AutoProcessor.from_pretrained(model_name)
        label_to_idx = {i: i for i in range(1000)}
        label_to_idx.update({str(i): i for i in range(1000)})
        state.slm_collator = CustomCollator(processor, label_to_idx)

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

    pin_mem = True if hasattr(state.args, 'device') and torch.device(state.args.device).type == 'cuda' else False
    return DataLoader(slm_data, batch_size=batch_size, shuffle=False, collate_fn=state.slm_collator, pin_memory=pin_mem)


def load_test_data(state, dataset, client_idx=0, batch_size=32):
    try:
        test_data = read_client_data(dataset, client_idx if client_idx is not None else 0, is_train=False)
        X, y = zip(*test_data)
        X = torch.stack(X)
        y = torch.tensor(y)

        if getattr(state.args, 'model', 'cnn') == 'slm':
            return _prepare_slm_test_loader(state, X, y, batch_size)

        if getattr(state.args, 'model', 'cnn') == 'clip':
            import torchvision.transforms as T
            if X.shape[-2:] != (224, 224):
                X = T.Resize((224, 224), antialias=True)(X)
            if not X.is_floating_point():
                X = X.float() / 255.0
            elif X.max() > 1.0:
                X = X / 255.0
            if X.min() >= 0.0:
                normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
                X = normalize(X)
        dataset = torch.utils.data.TensorDataset(X, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    except Exception as e:
        logger.exception("Error loading test data")
        return None
