from collections.abc import Mapping

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
            v_f = v.float()
            max_val = torch.max(torch.abs(v_f))
            scale = (
                max_val / 127.0
                if max_val > 1e-8
                else torch.tensor(1.0, device=v.device, dtype=torch.float32)
            )
            quantized_weights = torch.clamp((v_f / scale).round(), -128, 127).to(torch.int8)
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
    
    #print(f"\n[DIAGNÓSTICO] Iniciando evaluate_model no device: {device}")
    
    with torch.no_grad():
        model_dtype = getattr(model, "compute_dtype", None) or next(model.parameters()).dtype
        for batch_idx, batch_data in enumerate(data_loader):
            if isinstance(batch_data, (dict, Mapping)):
                x = batch_data
                y = batch_data.get("targets", batch_data.get("labels", None))
            else:
                x, y = batch_data

            y = y.to(device)

            if isinstance(x, torch.Tensor):
                if x.is_floating_point():
                    x = x.to(device, dtype=model_dtype)
                else:
                    x = x.to(device)
            elif isinstance(x, (dict, Mapping)):
                # É um dicionário ou BatchFeature / UserDict (SLM / LFM)
                for k in list(x.keys()):
                    if isinstance(x[k], torch.Tensor):
                        if getattr(x[k], "is_floating_point", lambda: False)():
                            x[k] = x[k].to(device, dtype=model_dtype)
                        else:
                            x[k] = x[k].to(device)

            autocast_dtype = (
                model_dtype
                if model_dtype in (torch.float16, torch.bfloat16)
                else (torch.float16 if device.type == "cuda" else torch.bfloat16)
            )
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                if isinstance(x, (dict, Mapping)):
                    output = model(x) if hasattr(model, "head") else model(**x)
                else:
                    output = model(x)
            
            if isinstance(output, (dict, Mapping)):
                output = output.get("logits", output)
                
            y = y.view(-1)
            
            # TESTE C: Verificando valores corrompidos (A Teoria da Divisão por Zero)
            if torch.isnan(output).any():
                print(
                    "[Teste C] ALERTA: A saída do modelo gerou NaNs "
                    "(valores infinitos ou inválidos)!"
                )
                
            loss = loss_fn(output, y)
            
            # TESTE D: Isolando a sincronização com a CPU
            # print(f"[Teste D] Chamando loss.item()...")
            total_loss += loss.item()
            #print(f"[Teste D] loss.item() extraído!")
            
            _, predicted = torch.max(output, 1)
            total += y.size(0)
            
            #print(f"[Teste D] Chamando sum().item()...")
            correct += (predicted == y).sum().item()
            #print(f"[Teste D] sum().item() extraído! Batch concluído.\n")

    accuracy = 100 * correct / total
    average_loss = total_loss / len(data_loader)
    return accuracy, average_loss

def set_parameters(model: nn.Module, state_new: nn.Module):
    """Copia os parâmetros de um modelo para outro in-place."""
    for new_param, old_param in zip(state_new.parameters(), model.parameters()):
        old_param.data = new_param.data.clone().to(device=old_param.device, dtype=old_param.dtype)
        
def resize_model_to_pruned(model: nn.Module, pruned_dict: dict) -> nn.Module:
    """ Redimensiona o modelo existente para as dimensões podadas preservando device, dtype e requires_grad """
    with torch.no_grad():
        for name, param in list(model.named_parameters()):
            if name in pruned_dict:
                pruned_weight = pruned_dict[name]
                if not isinstance(pruned_weight, torch.Tensor):
                    pruned_weight = torch.as_tensor(pruned_weight)
                
                target_weight = pruned_weight.to(device=param.device, dtype=param.dtype)
                
                if param.shape != target_weight.shape:
                    # print(f"Redimensionando {name}: {param.shape} -> {target_weight.shape}", flush=True)
                    new_param = nn.Parameter(target_weight, requires_grad=param.requires_grad)
                    
                    if '.' in name:
                        parts = name.split('.')
                        module = model
                        for part in parts[:-1]:
                            module = getattr(module, part)
                        setattr(module, parts[-1], new_param)
                        if hasattr(module, 'r') and 'lora_A' in name:
                            module.r = target_weight.shape[0]
                            if hasattr(module, 'lora_alpha') and hasattr(module, 'scaling'):
                                base_r = getattr(module, 'base_r', 8)
                                module.scaling = module.lora_alpha / base_r
                        
                        # PEFT LoraLayer support
                        if 'lora_A.default.weight' in name:
                            lora_layer = model
                            for part in parts[:-3]:
                                lora_layer = getattr(lora_layer, part)
                            if hasattr(lora_layer, 'r') and isinstance(lora_layer.r, dict) and 'default' in lora_layer.r:
                                lora_layer.r['default'] = target_weight.shape[0]
                                if hasattr(lora_layer, 'lora_alpha') and 'default' in lora_layer.lora_alpha:
                                    base_r = getattr(lora_layer, 'base_r', {}).get('default', 8) if hasattr(lora_layer, 'base_r') and isinstance(lora_layer.base_r, dict) else 8
                                    lora_layer.scaling['default'] = lora_layer.lora_alpha['default'] / base_r
                    else:
                        setattr(model, name, new_param)
                else:
                    param.data.copy_(target_weight)
    return model