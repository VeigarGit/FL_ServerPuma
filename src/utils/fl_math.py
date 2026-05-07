import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def dequantization(global_state: dict) -> dict:
    """Reverte a quantização int8 para tensores float normais."""
    dequantized_state_dict = {}
    for k, v in global_state.items():
        if isinstance(v, dict) and v.get('dtype') == 'quantized_int8':
            scale = v['scale']
            dequantized_state_dict[k] = v['weights'].float() * scale
        else:
            dequantized_state_dict[k] = v
    return dequantized_state_dict

def quantization(state_dict: dict) -> dict:
    """Comprime tensores float para int8 a fim de economizar banda."""
    quantized_state_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            scale = torch.max(torch.abs(v)) / 127.0
            quantized_weights = torch.clamp((v / scale).round(), -128, 127).to(torch.int8)
            quantized_state_dict[k] = {
                'dtype': 'quantized_int8',
                'scale': scale,
                'weights': quantized_weights
            }
        else:
            quantized_state_dict[k] = v
    return quantized_state_dict

def evaluate_model(model: nn.Module, data_loader: DataLoader):
    """Avalia o modelo e retorna (Acurácia, Loss Média)."""
    model.eval()
    correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    device = next(model.parameters()).device

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            if isinstance(output, dict):
                output = output["logits"]
            loss = loss_fn(output, y)
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = 100 * correct / total
    average_loss = total_loss / len(data_loader)
    return accuracy, average_loss

def set_parameters(model: nn.Module, state_new: nn.Module):
    """Copia os parâmetros de um modelo para outro in-place."""
    for new_param, old_param in zip(state_new.parameters(), model.parameters()):
        old_param.data = new_param.data.clone()