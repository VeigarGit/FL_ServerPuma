# Responsabilidade: Define collator, codificação de rótulos e construção dos DataLoaders.
# Família CLIP.
#
# Principais funções/classes: CustomCollator, resolve_label_metadata, encode_label,
# build_dataloaders.

"""CLIP model support — data."""

import torch
from datasets import ClassLabel
from datasets import load_dataset
from torch.utils.data import DataLoader

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
