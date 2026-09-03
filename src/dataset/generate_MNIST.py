# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import numpy as np
import os
import sys
import argparse
import random
import torch
import torchvision
import torchvision.transforms as transforms
from dataset.utils.dataset_utils import check, separate_data, split_data, save_file
from pathlib import Path


random.seed(1)
np.random.seed(1)
dir_path = Path(__file__).parent / "MNIST"


# Allocate data to users
def generate_dataset(dir_path: Path, num_clients, niid, balance, partition):
    # Substitui o os.path.exists e os.makedirs
    dir_path.mkdir(parents=True, exist_ok=True)
        
    # Setup directory for train/test data usando o operador /
    config_path = dir_path / "config.json"
    train_path = dir_path / "train"
    test_path = dir_path / "test"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    # # FIX HTTP Error 403: Forbidden
    # from six.moves import urllib
    # opener = urllib.request.build_opener()
    # opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    # urllib.request.install_opener(opener)

    # Get MNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.MNIST(
        root=dir_path / "rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=dir_path / "rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("niid", type=str)
    parser.add_argument("balance", type=str)
    parser.add_argument("partition", type=str)
    parser.add_argument("--num-clients", type=int, default=10, help="Number of clients to partition the data for")
    args = parser.parse_args()

    niid = True if args.niid == "noniid" else False
    balance = True if args.balance == "balance" else False
    partition = args.partition if args.partition != "-" else None

    generate_dataset(dir_path, args.num_clients, niid, balance, partition)