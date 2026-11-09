# Responsabilidade: Define collator, codificação de rótulos e construção dos DataLoaders.
# Família SLM.
#
# Principais funções/classes: CustomCollator, resolve_label_metadata, encode_label,
# build_dataloaders.

"""SLM model support — data."""

import torch
from datasets import ClassLabel
from datasets import load_dataset
from torch.utils.data import DataLoader
from PIL import Image

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)
    return train_loader, eval_loader, class_names
