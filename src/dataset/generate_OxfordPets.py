import numpy as np
import os
import sys
import random
import torch
from datasets import load_dataset
from PIL import Image
from dataset.utils.dataset_utils import check, separate_data, split_data, save_file
from pathlib import Path

random.seed(1)
np.random.seed(1)
# Você pode aumentar isso aqui se quiser simular mais clientes futuramente
num_clients = 2
dir_path = Path(__file__).parent / "OxfordPets"

def generate_dataset(dir_path: Path, num_clients, niid, balance, partition):
    dir_path.mkdir(parents=True, exist_ok=True)
        
    config_path = dir_path / "config.json"
    train_path = dir_path / "train"
    test_path = dir_path / "test"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        print("Dataset Oxford Pets já está preparado!")
        return

    print("Baixando dataset Oxford Pets do Hugging Face...")
    dataset = load_dataset("enterprise-explorers/oxford-pets")
    
    unique_labels = sorted(set(dataset["train"]["label"]))
    label_to_idx = {name: idx for idx, name in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    print(f'Number of classes: {num_classes}')

    dataset_image = []
    dataset_label = []

    print("Processando imagens (Resize 224x224 RGB)... Isso pode levar alguns instantes.")
    for item in dataset["train"]:
        img = item['image'].convert('RGB').resize((224, 224), Image.BICUBIC)
        # Salva como float32 em [0.0, 1.0] no formato (C, H, W)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1)) # (3, 224, 224)
        dataset_image.append(arr)
        dataset_label.append(label_to_idx[item['label']])

    dataset_image = np.array(dataset_image, dtype=np.float32)
    dataset_label = np.array(dataset_label, dtype=np.int64)

    # Como o Oxford Pets tem 37 classes, ajusta classes por cliente proporcionalmente
    cpc = min(num_classes, max(1, 4 if num_clients <= 10 else int(np.ceil(num_classes / num_clients))))
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition, class_per_client=cpc)
    
    train_data, test_data = split_data(X, y)
    
    
    print("Salvando arquivos .npz nas pastas train/ e test/ ...")
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)
    print("Finalizado com sucesso!")

if __name__ == "__main__":
    # Mantém o mesmo padrão de recebimento de argumentos do seu run.sh
    niid = True if len(sys.argv) > 1 and sys.argv[1] == "noniid" else False
    balance = True if len(sys.argv) > 2 and sys.argv[2] == "balance" else False
    partition = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else None
    if len(sys.argv) > 4 and sys.argv[4].isdigit():
        num_clients = int(sys.argv[4])

    generate_dataset(dir_path, num_clients, niid, balance, partition)