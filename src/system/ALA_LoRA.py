import numpy as np
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader, Subset
from typing import List, Tuple

class ALA_LoRA:
    def __init__(self,
                 cid: int,
                 loss: nn.Module,
                 train_data: DataLoader, 
                 batch_size: int, 
                 rand_percent: int, 
                 eta: float = 1.0,
                 device: str = 'cpu', 
                 threshold: float = 0.1,
                 num_pre_loss: int = 10) -> None:
        """
        Initialize ALA_LoRA module specifically for Foundation Models with LoRA/SoRA.

        Args:
            cid: Client ID. 
            loss: The loss function. 
            train_data: The reference of the local training data (DataLoader).
            batch_size: Weight learning batch size.
            rand_percent: The percent of the local training data to sample.
            eta: Weight learning rate. Default: 1.0
            device: Using cuda or cpu. Default: 'cpu'
            threshold: Train the weight until the standard deviation of the recorded losses is less than a given threshold. Default: 0.1
            num_pre_loss: The number of the recorded losses to be considered to calculate the standard deviation. Default: 10
        """

        self.cid = cid
        self.loss = loss
        self.train_data = train_data
        self.batch_size = batch_size
        self.rand_percent = rand_percent
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.device = device

        self.weights = None # Learnable local aggregation weights
        self.start_phase = True

    def adaptive_local_aggregation(self, 
                                   global_model: nn.Module,
                                   local_model: nn.Module,
                                   mask = None) -> None:
        """
        Adapts the global model dynamically locally without doing full deepcopy
        to avoid OOM in large models. It targets explicitly trainable tensors.
        """
        rand_ratio = self.rand_percent / 100
        dataset = self.train_data.dataset
        dataset_len = len(dataset)
        rand_num = max(1, int(rand_ratio * dataset_len))
        rand_idx = random.randint(0, max(0, dataset_len - rand_num))
        
        subset = Subset(dataset, range(rand_idx, rand_idx + rand_num))
        rand_loader = DataLoader(subset, self.batch_size, drop_last=False)

        # Retrieve parameters as dicts
        params_g_dict = dict(global_model.named_parameters())
        params_dict = dict(local_model.named_parameters())
        
        # Filter explicitly trainable parameters (LoRA, SoRA, Classifier Heads)
        trainable_names = [n for n, p in params_dict.items() if p.requires_grad]

        if not trainable_names:
            return

        # Deactivate ALA at the 1st communication iteration 
        # (check if difference between ALL trainable parameters is roughly zero)
        diff_sum = 0.0
        for n in trainable_names:
            diff_sum += torch.sum(torch.abs(params_g_dict[n].data - params_dict[n].data)).item()

        print(f"\n[DEBUG FedALA] Rodada do Cliente {self.cid} | Diferença Local-Global: {diff_sum:.4f}")

        # Sobre aconselhamento de rafael veiga, o if estava comentado, 
        # MAS isso causa um loop de 11 épocas inúteis porque os pesos nunca mudam.
        # Desativar o ALA na rodada 1 é o comportamento padrão e correto do paper.
        if diff_sum == 0.0:
            return

        # Verifica se as dimensões do modelo mudaram (ex: devido a pruning pelo SoRA).
        # Se os pesos do ALA já foram inicializados, mas o shape atual da camada é 
        # diferente do shape guardado no FedALA, significa que o modelo foi podado.
        # Nesse caso, resetamos os pesos de agregação para reiniciar o FedALA corretamente.
        if self.weights is not None:
            for n in trainable_names:
                if n in self.weights and self.weights[n].shape != params_dict[n].data.shape:
                    print(f"\n[ALA_LoRA] Pruning detectado em {n}. Reinicializando pesos do FedALA.")
                    self.weights = None
                    self.start_phase = True
                    break

        # Initialize the weights to all ones in the beginning
        if self.weights is None:
            self.weights = {n: torch.ones_like(params_dict[n].data).to(self.device) for n in trainable_names}

        # Save backups of the original weights to RAM/GPU-RAM without doing deepcopy of whole model
        local_backup = {n: params_dict[n].data.clone() for n in trainable_names}
        global_backup = {n: params_g_dict[n].data.clone() for n in trainable_names}
        
        # Backup the requires_grad state to restore later
        requires_grad_backup = {n: p.requires_grad for n, p in params_dict.items()}
        
        # Ensure only trainable parameters compute gradients for ALA optimization
        for n, p in params_dict.items():
            if n in trainable_names:
                p.requires_grad = True
            else:
                p.requires_grad = False

        # Initialize the higher layers in the temporary local model simulation
        for n in trainable_names:
            params_dict[n].data = local_backup[n] + (global_backup[n] - local_backup[n]) * self.weights[n]

        # Weight learning
        losses = [] 
        cnt = 0  
        while True:
            for x, y in rand_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                # Zero out gradients manually since we don't have an explicit optimizer for model params
                for n in trainable_names:
                    if params_dict[n].grad is not None:
                        params_dict[n].grad.zero_()

                output = local_model(x)
                
                # Check for dict output (Crucial for CLIP/Foundation Models)
                if isinstance(output, dict):
                    output = output.get("logits", output)
                    
                loss_value = self.loss(output, y)
                loss_value.backward()

                # Update weights in this batch
                for n in trainable_names:
                    param_t_grad = params_dict[n].grad
                    if param_t_grad is not None:
                        param_g = global_backup[n]
                        param_l = local_backup[n]
                        weight = self.weights[n]
                        
                        weight.data = torch.clamp(
                            weight - self.eta * (param_t_grad * (param_g - param_l)), 0, 1)

                # Update local_model in this batch to keep the loop valid
                for n in trainable_names:
                    param_g = global_backup[n]
                    param_l = local_backup[n]
                    weight = self.weights[n]
                    params_dict[n].data = param_l + (param_g - param_l) * weight

            losses.append(loss_value.item())
            cnt += 1

            if not self.start_phase:
                break

            # Train the weight until convergence
            if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                print('Client:', self.cid, '\tStd:', np.std(losses[-self.num_pre_loss:]),
                      '\tALA_LoRA epochs:', cnt)
                break

        self.start_phase = False

        # The local model is now correctly set with the optimized combination.
        # Restore the original requires_grad statuses so the main train loop behaves correctly
        for n, p in params_dict.items():
            p.requires_grad = requires_grad_backup[n]
