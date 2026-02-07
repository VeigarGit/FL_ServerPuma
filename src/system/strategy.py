import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import pickle
import numpy as np

# Import internal modules
from .model import SimpleModel
from .data_utils import read_client_data
from .prunning import prune_and_restructure
from .size_mode import get_model_size
from ..utils.logger import setup_logger
from ..utils.model_utils import quantization, dequantization

logger = setup_logger("server.strategies.puma")

class PUMAStrategy:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        
        # Initialize Global Model
        logger.info(f"Initializing model for dataset: {args.dataset}")
        if args.dataset == 'MNIST':
            self.global_model = SimpleModel(in_features=1, num_classes=10, dim=1024)
        elif args.dataset == 'Cifar10':
            self.global_model = SimpleModel(in_features=args.in_features, num_classes=10, dim=args.dim)
        elif args.dataset == 'Cifar100':
            self.global_model = SimpleModel(in_features=args.in_features, num_classes=args.num_classes, dim=args.dim)
        else:
            self.global_model = SimpleModel(in_features=args.in_features, num_classes=args.num_classes, dim=args.dim)

        self.global_model.to(self.device)
        self.global_state = self.global_model.state_dict()
        
        # PUMA specific state
        self.prune_rate = 0
        self.prune_triggered = False
        self.size_fc = 25 
        
        # Metrics history
        self.rs_test_acc = []
        self.rs_test_loss = []

        # Load Test Data
        self.test_loader = self._load_test_data()
        
        size_mb = get_model_size(self.global_model)
        logger.info(f"PUMA Strategy initialized. Model size: {size_mb:.2f} MB")

    def _load_test_data(self):
        """Internal helper to load test dataset."""
        try:
            logger.info(f"Loading {self.args.dataset} test dataset...")
            test_data = read_client_data(self.args.dataset, self.args.test_client_idx, is_train=False)
            X, y = zip(*test_data)
            X = torch.stack(X)
            y = torch.tensor(y)
            dataset = torch.utils.data.TensorDataset(X, y)
            logger.info("Test dataset loaded.")
            return DataLoader(dataset, batch_size=self.args.batch_size)
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            return None

    def aggregate_updates(self, updates):
        """Aggregates a list of model state dictionaries."""
        if not updates:
            logger.warning("No updates to aggregate.")
            return

        logger.info(f"Aggregating {len(updates)} client updates...")
        
        # Initialize aggregate state with the first update
        agg_state = copy.deepcopy(updates[0])
        
        for key in agg_state.keys():
            # Sum the rest
            for i in range(1, len(updates)):
                agg_state[key] += updates[i][key]
            # Average
            agg_state[key] = agg_state[key] / len(updates)
            
        self.global_state = agg_state
        self.global_model.load_state_dict(self.global_state)
        return self.global_state

    def evaluate_global_model(self, round_num):
        """Evaluates the current global model on the test set."""
        if self.test_loader is None:
            logger.warning("Skipping evaluation (no test data).")
            return 0.0, 0.0

        self.global_model.eval()
        self.global_model.to(self.device)
        
        correct = 0
        total = 0
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0.0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.global_model(x)
                loss = loss_fn(output, y)
                total_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(self.test_loader)
        
        self.rs_test_acc.append(accuracy)
        self.rs_test_loss.append(avg_loss)
        
        logger.info(f"--- Round {round_num} Complete. Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f} ---")
        return accuracy, avg_loss

    def attempt_pruning(self, client_times, client_data_volumes):
        """
        Calculates pruning rate based on client training times (PUMA logic).
        Returns the pruning amount.
        """
        # Logic extracted from original set_amount_prune
        values = list(client_data_volumes.values()) # Using volumes/metrics as proxy for calculation basis
        
        # Identify stragglers based on training time (metrics from logs)
        times = [t for t in client_times.values() if t is not None]
        
        if not times:
            logger.warning("No client times available for pruning calculation. Defaulting to 0.")
            return 0

        min_time = min(times)
        max_time = max(times)
        
        logger.info(f"Client times - min: {min_time:.2f}s, max: {max_time:.2f}s")
        
        # Original logic used client_data values to calculate ratio
        if not values:
            return 0
            
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            calculated_amount = 0.5 # Default fallback
        else:
            # Calculate average pruning amount
            amounts = []
            for v in values:
                if v == 0: continue
                # Inverse proportion: more data/slower -> more pruning? 
                amount = 0.9 * (1 - (v - min_val) / (max_val - min_val))
                amount = max(0, min(amount, 0.9))
                amounts.append(amount)
            
            if amounts:
                calculated_amount = sum(amounts) / len(amounts)
            else:
                calculated_amount = 0

        logger.info(f"Calculated pruning rate: {calculated_amount:.4f}")
        
        # Apply pruning to global model
        logger.info("--- PRUNING START ---")
        self.global_model, _ = prune_and_restructure(
            model=self.global_model, 
            pruning_rate=calculated_amount, 
            size_fc=self.size_fc, 
            data=self.args.dataset
        )
        self.global_model.to(self.device)
        self.global_state = self.global_model.state_dict()
        self.prune_rate = calculated_amount
        self.prune_triggered = True
        
        size_mb = get_model_size(self.global_model)
        logger.info(f"--- PRUNING COMPLETE. New model size: {size_mb:.2f} MB ---")
        
        return calculated_amount

    def get_model_package(self, client_id=None):
        """
        Returns the payload to send to a client.
        Applies quantization.
        """
        # 1. Get current state (pruned or not)
        state_to_send = copy.deepcopy(self.global_state)
        
        # 2. Calculate size before quantization
        size_before = sys.getsizeof(pickle.dumps(state_to_send)) / (1024 * 1024)
        
        # 3. Quantize
        quantized_state = quantization(state_to_send)
        size_after = sys.getsizeof(pickle.dumps(quantized_state)) / (1024 * 1024)
        
        logger.debug(f"Serving model to {client_id}: {size_before:.2f}MB -> {size_after:.2f}MB")
        
        return quantized_state, self.prune_rate