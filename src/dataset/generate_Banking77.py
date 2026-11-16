import argparse
import json
import random
from pathlib import Path

import numpy as np
from datasets import load_dataset

from dataset.utils.dataset_utils import check, save_file, separate_data, split_data

random.seed(1)
np.random.seed(1)

dir_path = Path(__file__).resolve().parent / "Banking77"


def generate_dataset(
    dir_path: Path, num_clients: int, niid: bool, balance: bool, partition: str | None
):
    dir_path.mkdir(parents=True, exist_ok=True)

    config_path = dir_path / "config.json"
    train_path = dir_path / "train"
    test_path = dir_path / "test"
    classes_path = dir_path / "classes.json"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        print("Dataset Banking77 já está preparado!")
        return

    print("Baixando dataset PolyAI/banking77 do Hugging Face...")
    raw_dataset = load_dataset("PolyAI/banking77")

    raw_classes = raw_dataset["train"].features["label"].names
    num_classes = len(raw_classes)
    print(f"Número de classes bancárias: {num_classes}")

    # Salva descrições limpas das classes (underscores substituídos por espaços)
    class_descriptions = {
        i: name.replace("_", " ").replace("?", "") for i, name in enumerate(raw_classes)
    }
    with classes_path.open("w", encoding="utf-8") as f:
        json.dump(class_descriptions, f, indent=2, ensure_ascii=False)

    # Junta train e test para redistribuição federada controlada
    texts = []
    labels = []
    for split in ["train", "test"]:
        for item in raw_dataset[split]:
            texts.append(item["text"])
            labels.append(item["label"])

    texts = np.array(texts, dtype=object)
    labels = np.array(labels, dtype=np.int64)
    print(f"Total de amostras: {len(labels)}")

    print(
        f"Distribuindo dados para {num_clients} clientes "
        f"(Non-IID: {niid}, Partição: {partition})..."
    )
    X, y, statistic = separate_data(
        (texts, labels),
        num_clients,
        num_classes,
        niid=niid,
        balance=balance,
        partition=partition,
        class_per_client=10 if niid and partition == "pat" else None,
    )

    train_data, test_data = split_data(X, y)

    print("Salvando arquivos .npz nas pastas train/ e test/ ...")
    save_file(
        config_path,
        train_path,
        test_path,
        train_data,
        test_data,
        num_clients,
        num_classes,
        statistic,
        niid=niid,
        balance=balance,
        partition=partition,
    )
    print("Dataset Banking77 particionado com sucesso!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gerador Federado do Dataset Banking77")
    parser.add_argument("niid_pos", nargs="?", default=None, help="noniid ou iid")
    parser.add_argument("balance_pos", nargs="?", default=None, help="balance ou -")
    parser.add_argument("partition_pos", nargs="?", default=None, help="dir ou pat")
    parser.add_argument("clients_pos", nargs="?", default=None, help="Número de clientes")
    parser.add_argument("--num-clients", type=int, default=10, help="Número de clientes")
    parser.add_argument("--niid", action="store_true", help="Non-IID")
    parser.add_argument("--balance", action="store_true", help="Balanceado")
    parser.add_argument(
        "--partition", type=str, default="dir", choices=["dir", "pat"], help="Tipo de partição"
    )

    args = parser.parse_args()

    # Suporte posicional padrão de run.sh: <iid|noniid> <balance|-> <pat|dir> <num_clients>
    niid = bool(args.niid_pos == "noniid" or args.niid)
    balance = bool(args.balance_pos == "balance" or args.balance)
    partition = (
        args.partition_pos if args.partition_pos and args.partition_pos != "-" else args.partition
    )
    num_clients = (
        int(args.clients_pos)
        if args.clients_pos and args.clients_pos.isdigit()
        else args.num_clients
    )

    generate_dataset(dir_path, num_clients, niid, balance, partition)
