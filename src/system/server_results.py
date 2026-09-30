"""Test-data loading and HDF5 output methods for the server."""
import logging
from pathlib import Path

import h5py
import torch
from torch.utils.data import DataLoader

from utils.data_utils import read_client_data

logger = logging.getLogger(__name__)

class ServerResultsMixin:
    def _prepare_slm_test_loader(self, X, y, batch_size=32):
        from PIL import Image
        import numpy as np

        if not hasattr(self, 'slm_collator') or self.slm_collator is None:
            from lora_slm.slm_setup import CustomCollator
            from transformers import AutoProcessor
            model_name = getattr(self, 'config', {}).get("model", {}).get("name", "Qwen/Qwen3-VL-2B-Instruct") if getattr(self, 'config', None) else "Qwen/Qwen3-VL-2B-Instruct"
            processor = AutoProcessor.from_pretrained(model_name)
            label_to_idx = {i: i for i in range(1000)}
            label_to_idx.update({str(i): i for i in range(1000)})
            self.slm_collator = CustomCollator(processor, label_to_idx)

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

        pin_mem = True if hasattr(self.args, 'device') and torch.device(self.args.device).type == 'cuda' else False
        return DataLoader(slm_data, batch_size=batch_size, shuffle=False, collate_fn=self.slm_collator, pin_memory=pin_mem)

    def load_test_data(self, dataset, client_idx=0, batch_size=32):
        try:
            test_data = read_client_data(dataset, client_idx if client_idx is not None else 0, is_train=False)
            X, y = zip(*test_data)
            X = torch.stack(X)
            y = torch.tensor(y)

            if getattr(self.args, 'model', 'cnn') == 'slm':
                return self._prepare_slm_test_loader(X, y, batch_size)

            if getattr(self.args, 'model', 'cnn') == 'clip':
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
    def save_results(self, i: int):
        i_str = str(i)
        a = "prune" if self.args.prune == 1 else "withou_Prune"
        b = "FedALA" if getattr(self, 'argalgo', 0) == 0 else "FedAVG"

        paca_val = self.args.paca if (isinstance(self.args.paca, (int, float)) and not isinstance(self.args.paca, bool) and self.args.paca > 0) else 0
        rank_val = self.args.rank if (isinstance(self.args.rank, (int, float)) and not isinstance(self.args.rank, bool) and self.args.rank > 0) else 8
        ts_suffix = f"_{self.args.timestamp}" if hasattr(self.args, 'timestamp') and self.args.timestamp else ""
        algo = f"{self.args.dataset}_{self.args.strategy}_rank{rank_val}_paca{paca_val}_{a}_freq{self.args.prune_freq}_{b}_run{self.args.run_id}{ts_suffix}"

        current_dir = Path(__file__).resolve().parent
        result_path = current_dir.parent / "results" / self.args.exp_name
        result_path.mkdir(parents=True, exist_ok=True)

        # Padronizando com o prefixo 'server_' para não misturar com o client
        file_path = result_path / f"server_{algo}.h5"

        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
            hf.create_dataset('rs_test_loss', data=self.rs_test_loss)
            hf.create_dataset('sended_model_Mb', data=self.sended_ammount)
            hf.create_dataset('Sended_without_quant', data=self.sended_withouquant)
            hf.create_dataset('Aggregated_clients', data=self.aggregated_clients)
            hf.create_dataset('Round_time', data=self.round_time)
            hf.create_dataset('Model_size_per_round_Mb', data=self.model_size_per_round)
            hf.create_dataset('Trainable_params', data=self.trainable_params_per_round)
            hf.create_dataset('rs_client_paca', data=self.rs_client_paca)
            hf.create_dataset('rs_client_training_time', data=self.rs_client_training_time)
            hf.create_dataset('rs_client_comm_time', data=self.rs_client_comm_time)
            hf.create_dataset('rs_client_eval_time', data=self.rs_client_eval_time)
            hf.create_dataset('rs_server_processing_time', data=self.rs_server_processing_time)
            hf.create_dataset('server_pruning_time', data=self.server_pruning_time)
            hf.create_dataset('received_from_clients_Mb', data=self.received_ammount)
            hf.create_dataset('total_transmitted_per_round_Mb', data=self.total_transmitted_per_round)
