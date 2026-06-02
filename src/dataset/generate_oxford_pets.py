import numpy as np
import os
import sys
import random
import torch
from datasets import load_dataset
from transformers import CLIPProcessor
from dataset.utils.dataset_utils import check, separate_data, split_data, save_file
from pathlib import Path

random.seed(1)
np.random.seed(1)
# Você pode aumentar isso aqui se quiser simular mais clientes futuramente
num_clients = 5
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
    
    # Carregar o processador oficial do CLIP para garantir resize (224x224) e normalização
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    unique_labels = sorted(set(dataset["train"]["label"]))
    label_to_idx = {name: idx for idx, name in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    print(f'Number of classes: {num_classes}')

    dataset_image = []
    dataset_label = []

    print("Processando imagens (Resize e Normalização do CLIP)... Isso pode levar alguns instantes.")
    # Extrai todas as imagens de treino e processa para o formato de tensores
    for item in dataset["train"]:
        img = item['image'].convert('RGB')
        processed = processor(images=img, return_tensors="np")
        dataset_image.append(processed["pixel_values"][0])
        dataset_label.append(label_to_idx[item['label']])

    dataset_image = np.array(dataset_image, dtype=np.float32)
    dataset_label = np.array(dataset_label, dtype=np.int64)

    print("Distribuindo os dados para os clientes (IID/Non-IID)...")
    # Como o Oxford Pets tem 37 classes, se for non-IID, podemos dar ~4 classes diferentes para cada cliente
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition, class_per_client=4)
    
    train_data, test_data = split_data(X, y)
    
    
    print("Salvando arquivos .npz nas pastas train/ e test/ ...")
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)
    print("Finalizado com sucesso!")

if __name__ == "__main__":
    # Mantém o mesmo padrão de recebimento de argumentos do seu run.sh
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)