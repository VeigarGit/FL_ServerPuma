import pandas as pd
import numpy as np
import os
from utils.dataset_utils import check, separate_data, split_data, save_file
import random
random.seed(1)
np.random.seed(1)

# Configurações do dataset
num_clients = 20
dir_path = "ECG/"
train_csv_path = dir_path + "rawdata/train_dataset.csv"
test_csv_path = dir_path + "rawdata/teste_dataset.csv"

def load_ecg_datasets(train_csv_path, test_csv_path):
    """
    Carrega os datasets de treino e teste dos arquivos CSV.
    Espera-se que os CSVs tenham as colunas 'ECG' (dados) e 'label' (classes).
    """
    # Carregar datasets
    train_df = pd.read_csv(train_csv_path, sep=" ")
    test_df = pd.read_csv(test_csv_path, sep=" ")
    
    # Garantir que as colunas necessárias estão presentes
    for df, name in [(train_df, "train_dataset.csv"), (test_df, "teste_dataset.csv")]:
        print(df.columns)
        if 'ECG' not in df.columns or 'label' not in df.columns:
            raise ValueError(f"O arquivo {name} deve conter as colunas 'ECG' e 'label'.")
    
    # Extrair dados e rótulos de treino
    train_image = train_df['ECG'].values
    train_label = train_df['label'].values
    
    # Extrair dados e rótulos de teste
    test_image = test_df['ECG'].values
    test_label = test_df['label'].values
    
    # Transformar os dados do ECG (se necessário)
    # Exemplo: Normalizar ou converter em listas de valores numéricos
    train_image = np.array([np.fromstring(ecg, sep=' ') for ecg in train_image])
    test_image = np.array([np.fromstring(ecg, sep=' ') for ecg in test_image])
    
    return train_image, train_label, test_image, test_label

def generate_ecg_dataset(dir_path, train_csv_path, test_csv_path, num_clients, niid, balance, partition):
    """
    Gera o dataset ECG particionado para aprendizado federado.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # Setup dos caminhos para salvar os arquivos
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    # Verifica se o dataset já foi gerado
    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    # Carrega os datasets
    train_image, train_label, test_image, test_label = load_ecg_datasets(train_csv_path, test_csv_path)
    
    # Concatenar treino e teste para particionamento entre os clientes
    dataset_image = np.concatenate((train_image, test_image), axis=0)
    dataset_label = np.concatenate((train_label, test_label), axis=0)

    num_classes = len(set(dataset_label))
    print(f'Número de classes: {num_classes}')

    # Particiona os dados entre os clientes
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition, class_per_client=2)
    
    # Divide em treino e teste (baseado na divisão original)
    train_data, test_data = split_data(X, y)

    # Salva os dados particionados
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)

if __name__ == "__main__":
    # Argumentos para o gerador
    niid = True  # Não IID
    balance = True  # Dados balanceados
    partition = "dir"  # Partição usando Dirichlet

    # Gera o dataset
    generate_ecg_dataset(dir_path, train_csv_path, test_csv_path, num_clients, niid, balance, partition)
