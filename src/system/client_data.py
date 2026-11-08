"""Carregamento de dados, pré-processamento (CNN/CLIP/SLM) e persistência HDF5 do cliente FL."""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from utils.data_utils import read_client_data  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class ClientHistory:
    """Acumula o histórico de métricas por rodada e por encontro do cliente FL."""

    global_acc: list[float] = field(default_factory=list)
    global_loss: list[float] = field(default_factory=list)
    ala_acc: list[float] = field(default_factory=list)
    local_acc: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    post_p2p_acc: list[float] = field(default_factory=list)
    post_p2p_encounter_ids: list[int] = field(default_factory=list)
    p2p_data_sent_mb: list[float] = field(default_factory=list)
    p2p_data_sent_enc_ids: list[int] = field(default_factory=list)
    ota_encounter_epochs: list[int] = field(default_factory=list)
    ota_mean_distances: list[float] = field(default_factory=list)
    ota_is_aggregator: list[int] = field(default_factory=list)

    def save(self, args: argparse.Namespace) -> None:
        """Persiste todas as séries históricas do cliente no arquivo HDF5."""
        save_results(
            args,
            self.global_acc,
            self.global_loss,
            self.ala_acc,
            self.local_acc,
            self.train_acc,
            idx=args.client_idx,
            argalgo=args.ala,
            rs_post_p2p_acc=self.post_p2p_acc,
            rs_post_p2p_encounter_ids=self.post_p2p_encounter_ids,
            rs_p2p_data_sent_mb=self.p2p_data_sent_mb,
            rs_p2p_data_sent_enc_ids=self.p2p_data_sent_enc_ids,
            rs_ota_encounter_epochs=self.ota_encounter_epochs,
            rs_ota_mean_distances=self.ota_mean_distances,
            rs_ota_is_aggregator=self.ota_is_aggregator,
        )


def _tensor_to_pil_rgb(x_tensor: torch.Tensor) -> Image.Image:
    """Reverte normalização (CLIP ou [-1, 1]) e converte tensor CHW para imagem PIL RGB."""
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
        x_tensor = x_tensor * 0.5 + 0.5

    if x_tensor.max() <= 1.0:
        x_tensor = x_tensor * 255.0

    x_numpy = torch.clamp(x_tensor, 0, 255).byte().numpy()
    if x_numpy.shape[0] in [1, 3]:  # (C, H, W) -> (H, W, C)
        x_numpy = np.transpose(x_numpy, (1, 2, 0))
    if x_numpy.shape[-1] == 1:
        x_numpy = np.squeeze(x_numpy, axis=-1)

    pil_img = Image.fromarray(x_numpy)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return pil_img


def load_data(
    dataset,
    client_idx,
    device,
    is_train=True,
    batch_size=32,
    is_clip=False,
    is_slm=False,
    slm_config=None,
    dataset_dir=None,
):
    """Carrega a partição local .npz e prepara o DataLoader para CNN, CLIP ou SLM."""
    train_data = read_client_data(dataset, client_idx, is_train, dataset_dir=dataset_dir)
    X, y = zip(*train_data, strict=False)
    X = torch.stack(X)

    if is_slm:
        from lora_slm.slm_setup import CustomCollator
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(slm_config["model"]["name"])
        label_to_idx = {i: i for i in range(1000)}
        label_to_idx.update({str(i): i for i in range(1000)})
        collate_fn = CustomCollator(processor, label_to_idx)

        slm_data = [
            {"image": _tensor_to_pil_rgb(x_tensor), "label": y_tensor.item()}
            for x_tensor, y_tensor in zip(X, y, strict=False)
        ]
        pin_mem = hasattr(device, "type") and device.type == "cuda"
        return DataLoader(
            slm_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=pin_mem,
        )

    # Redimensionamento e normalização vitais para o CLIP
    if is_clip:
        if X.shape[-2:] != (224, 224):
            resize = T.Resize((224, 224), antialias=True)
            X = resize(X)
        if not X.is_floating_point():
            X = X.float() / 255.0
        elif X.min() >= 0.0 and X.max() > 1.0:
            X = X / 255.0
        if X.min() >= 0.0:
            normalize = T.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            )
            X = normalize(X)

    X = X.to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)

    tensor_dataset = torch.utils.data.TensorDataset(X, y)
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)


def build_client_dataloaders(
    args: argparse.Namespace, device: torch.device, run_config: dict | None
) -> tuple[DataLoader, DataLoader, float | None]:
    """Carrega train_loader, test_loader e calcula o baseline da classe majoritária."""
    try:
        is_clip_flag = args.model == "clip"
        is_slm_flag = args.model == "slm"
        slm_cfg = run_config if is_slm_flag else None
        train_loader = load_data(
            args.dataset,
            args.client_idx,
            device,
            is_train=True,
            batch_size=args.batch_size,
            is_clip=is_clip_flag,
            is_slm=is_slm_flag,
            slm_config=slm_cfg,
            dataset_dir=args.dataset_dir,
        )
        test_loader = load_data(
            args.dataset,
            args.client_idx,
            device,
            is_train=False,
            batch_size=args.batch_size,
            is_clip=is_clip_flag,
            is_slm=is_slm_flag,
            slm_config=slm_cfg,
            dataset_dir=args.dataset_dir,
        )
        test_majority_baseline = None
        if hasattr(test_loader.dataset, "tensors"):
            labels = test_loader.dataset.tensors[1]
            counts = torch.unique(labels, return_counts=True)[1]
            test_majority_baseline = 100.0 * counts.max().item() / len(labels)
        logger.info(
            f"Data loaded successfully - Train batches: {len(train_loader)}, "
            f"Test batches: {len(test_loader)}"
        )
        return train_loader, test_loader, test_majority_baseline
    except Exception:
        logger.exception("Error loading data")
        sys.exit(1)


def save_results(
    args,
    rs_global_acc,
    rs_global_loss,
    rs_ala_acc,
    rs_local_acc,
    rs_train_acc,
    idx=0,
    argalgo=0,
    rs_post_p2p_acc=None,
    rs_post_p2p_encounter_ids=None,
    rs_p2p_data_sent_mb=None,
    rs_p2p_data_sent_enc_ids=None,
    rs_ota_encounter_epochs=None,
    rs_ota_mean_distances=None,
    rs_ota_is_aggregator=None,
):
    """Salva o histórico de métricas e encontros do cliente em formato HDF5 (.h5)."""
    b = "FedALA" if argalgo == 0 else "FedAVG"

    paca_val = (
        args.paca
        if (
            isinstance(args.paca, (int, float))
            and not isinstance(args.paca, bool)
            and args.paca > 0
        )
        else 0
    )
    rank_val = (
        args.rank
        if (
            isinstance(args.rank, (int, float))
            and not isinstance(args.rank, bool)
            and args.rank > 0
        )
        else 8
    )
    ts_suffix = f"_{args.timestamp}" if hasattr(args, "timestamp") and args.timestamp else ""
    algo = (
        f"client_{idx}_{args.dataset}_{args.strategy}_rank{rank_val}_"
        f"paca{paca_val}_{b}_run{args.run_id}{ts_suffix}"
    )

    result_path = (
        getattr(args, "results_dir", None) or current_dir.parent / "results" / args.exp_name
    )
    result_path.mkdir(parents=True, exist_ok=True)

    file_path = result_path / f"{algo}.h5"
    logger.info(f"Salvando resultados em: {file_path}")

    with h5py.File(file_path, "w") as hf:
        hf.create_dataset("rs_global_acc", data=rs_global_acc)
        hf.create_dataset("rs_global_loss", data=rs_global_loss)
        if rs_ala_acc is not None and len(rs_ala_acc) > 0:
            hf.create_dataset("rs_ala_acc", data=rs_ala_acc)
        hf.create_dataset("rs_local_acc", data=rs_local_acc)
        hf.create_dataset("rs_train_acc", data=rs_train_acc)
        if rs_post_p2p_acc is not None and len(rs_post_p2p_acc) > 0:
            hf.create_dataset("rs_post_p2p_acc", data=rs_post_p2p_acc)
        if rs_post_p2p_encounter_ids is not None and len(rs_post_p2p_encounter_ids) > 0:
            hf.create_dataset("rs_post_p2p_encounter_ids", data=rs_post_p2p_encounter_ids)
        if rs_p2p_data_sent_mb is not None and len(rs_p2p_data_sent_mb) > 0:
            hf.create_dataset("rs_p2p_data_sent_mb", data=rs_p2p_data_sent_mb)
            hf["rs_p2p_data_sent_mb"].attrs["unit"] = "MiB (2**20 bytes)"
        if rs_p2p_data_sent_enc_ids is not None and len(rs_p2p_data_sent_enc_ids) > 0:
            hf.create_dataset("rs_p2p_data_sent_enc_ids", data=rs_p2p_data_sent_enc_ids)
        if rs_ota_encounter_epochs is not None and len(rs_ota_encounter_epochs) > 0:
            hf.create_dataset("rs_ota_encounter_epochs", data=rs_ota_encounter_epochs)
        if rs_ota_mean_distances is not None and len(rs_ota_mean_distances) > 0:
            hf.create_dataset("rs_ota_mean_distances", data=rs_ota_mean_distances)
        if rs_ota_is_aggregator is not None and len(rs_ota_is_aggregator) > 0:
            hf.create_dataset("rs_ota_is_aggregator", data=rs_ota_is_aggregator)
        hf.attrs["paca_used"] = paca_val
        hf.attrs["execution_id"] = getattr(args, "execution_id", "")
        hf.attrs["seed"] = getattr(args, "seed", 42)
        if getattr(args, "dataset_dir", None) is not None:
            hf.attrs["dataset_dir"] = str(args.dataset_dir)
        if getattr(args, "initial_model_sha256", None) is not None:
            hf.attrs["initial_model_sha256"] = args.initial_model_sha256
            hf.attrs["initialization_seed"] = args.seed
            hf.attrs["training_seed"] = args.training_seed
        records = getattr(args, "encounter_metrics", [])
        if records:
            group = hf.create_group("encounters")
            group.attrs["bytes_unit"] = "bytes of completed application frames"
            group.attrs["time_unit"] = "seconds"
            group.attrs["accuracy_unit"] = "percent"
            group.create_dataset(
                "records",
                data=[json.dumps(r, allow_nan=False) for r in records],
                dtype=h5py.string_dtype("utf-8"),
            )
            for key in (
                "encounter_id",
                "participant_count",
                "upload_bytes",
                "download_bytes",
                "control_upload_bytes",
                "control_download_bytes",
                "aggregation_time",
                "protocol_time",
                "evaluation_time",
                "accuracy_before",
                "accuracy_after",
                "majority_class_baseline",
                "test_samples",
                "epoch",
                "mean_distance",
                "is_aggregator",
            ):
                group.create_dataset(
                    key, data=[r.get(key) if r.get(key) is not None else np.nan for r in records]
                )
