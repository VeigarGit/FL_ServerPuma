import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import copy
import psutil
import sys

# Import internal modules
from .model import SimpleModel
from .data_utils import read_client_data
from .prunning import prune_and_restructure
from .ALA import ALA
from ..utils.logger import setup_logger

# Initialize logger for this module
logger = setup_logger("client.trainer")

class Trainer:
    def __init__(self, args, client_id):
        self.args = args
        self.client_id = client_id
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()
        
        # Load Data
        self.train_loader, self.test_loader = self._load_data()
        
        # Initialize Local Model
        self.model = self._init_model()
        self.model.to(self.device)
        
        # Initialize ALA (Adaptive Local Aggregation)
        self.ala = ALA(
            cid=client_id,
            loss=self.criterion,
            train_data=self.train_loader, 
            batch_size=args.batch_size, 
            rand_percent=80, 
            layer_idx=2, 
            eta=1.0, 
            device=self.device
        )
        
        logger.info(f"Trainer initialized for Client {client_id}")

    def _init_model(self):
        """Initializes the model architecture based on dataset."""
        if self.args.dataset == 'MNIST':
            return SimpleModel(in_features=1, num_classes=10, dim=1024)
        elif self.args.dataset == 'Cifar10':
            return SimpleModel(in_features=self.args.in_features, num_classes=10, dim=self.args.dim)
        elif self.args.dataset == 'Cifar100':
            return SimpleModel(in_features=self.args.in_features, num_classes=self.args.num_classes, dim=self.args.dim)
        else:
            return SimpleModel(in_features=self.args.in_features, num_classes=self.args.num_classes, dim=self.args.dim)

    def _load_data(self):
        """Loads train and test data."""
        try:
            train_data = read_client_data(self.args.dataset, self.client_id, is_train=True)
            X_train, y_train = zip(*train_data)
            train_dataset = torch.utils.data.TensorDataset(torch.stack(X_train), torch.tensor(y_train))
            train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)

            test_data = read_client_data(self.args.dataset, self.client_id, is_train=False)
            X_test, y_test = zip(*test_data)
            test_dataset = torch.utils.data.TensorDataset(torch.stack(X_test), torch.tensor(y_test))
            test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)
            
            return train_loader, test_loader
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            sys.exit(1)

    def _adapt_model_structure(self, state_dict):
        """
        Dynamically rebuilds self.model as a nn.Sequential to match the 
        shapes found in the incoming global_state_dict.
        """
        # 1. Detect if the state_dict implies a Pruned Sequential model (keys like '0.weight')
        # If not (keys like 'conv1.weight'), we assume SimpleModel is fine.
        is_sequential = any(k.startswith('0.') for k in state_dict.keys())
        if not is_sequential:
            return self.model

        logger.info("Adapting local model architecture to match Global Pruned Model...")

        # 2. Extract shapes from the global state dict
        shapes = {k: v.shape for k, v in state_dict.items()}
        
        try:
            
            # Layer 0: Conv2d
            w0 = shapes['0.weight'] # [out, in, k, k]
            layer0 = nn.Conv2d(in_channels=w0[1], out_channels=w0[0], kernel_size=w0[2])
            
            # Layer 3: Conv2d
            w3 = shapes['3.weight']
            layer3 = nn.Conv2d(in_channels=w3[1], out_channels=w3[0], kernel_size=w3[2])
            
            # Layer 7: Linear
            w7 = shapes['7.weight']
            layer7 = nn.Linear(in_features=w7[1], out_features=w7[0])
            
            # Layer 9: Linear
            w9 = shapes['9.weight']
            layer9 = nn.Linear(in_features=w9[1], out_features=w9[0])

            # Rebuild Sequential
            new_model = nn.Sequential(
                layer0,             # 0
                nn.ReLU(),          # 1
                nn.MaxPool2d(2),    # 2
                layer3,             # 3
                nn.ReLU(),          # 4
                nn.MaxPool2d(2),    # 5
                nn.Flatten(),       # 6
                layer7,             # 7
                nn.ReLU(),          # 8
                layer9              # 9
            )
            
            new_model.to(self.device)
            return new_model

        except KeyError as e:
            logger.error(f"Failed to adapt model. Missing key in state_dict: {e}")
            raise e

    def train(self, global_state_dict, prune_rate, round_num, use_ala=True):
        """
        Executes local training.
        """
        start_time = time.time()
        
        # 1. Structure Adaptation
        # If we received a pruned model (sequential keys), we must SNAP our architecture to it.
        # We ignore 'prune_rate' here because the global_state_dict implicitly defines the rate/structure.
        self.model = self._adapt_model_structure(global_state_dict)
        self.model.to(self.device)

        status_msg = "PRUNED model" if prune_rate > 0 else "FULL model"
        logger.info(f"Training {status_msg} with PUMA+FedALA on {self.device}")

        # 2. Prepare for FedALA
        temp_global_model = copy.deepcopy(self.model)
        
        try:
            temp_global_model.load_state_dict(global_state_dict)
        except RuntimeError as e:
            logger.error(f"State dict mismatch after adaptation!")
            raise e
        
        # 3. Apply FedALA
        if use_ala and round_num > 1:
            # logger.info("Performing Adaptive Local Aggregation...")
            self.ala.adaptive_local_aggregation(temp_global_model, self.model)
        else:
            self.model.load_state_dict(global_state_dict)

        # 4. Local Training Loop
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y)
            loss.backward()
            optimizer.step()

        training_time = time.time() - start_time
        
        # 5. Log Telemetry
        self._log_metrics(len(self.train_loader.dataset), training_time)
        
        return self.model.state_dict(), len(self.train_loader.dataset)

    def _log_metrics(self, samples, duration):
        """Logs resource usage and training stats."""
        ram_usage = psutil.Process().memory_info().rss / (1024 ** 2)
        gpu_usage = 0
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            
        logger.info(f"Training completed: {samples} samples in {duration:.2f}s")
        logger.info(f"Resource usage - GPU: {gpu_usage:.2f}MB, RAM: {ram_usage:.2f}MB")