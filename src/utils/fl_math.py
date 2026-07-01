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
    
    #print(f"\n[DIAGNÓSTICO] Iniciando evaluate_model no device: {device}")
    
    with torch.no_grad():
        model_dtype = next(model.parameters()).dtype
        for batch_idx, (x, y) in enumerate(data_loader):
            #print(f"--- Processando Batch {batch_idx} ---")
            if x.is_floating_point():
                x = x.to(device, dtype=model_dtype)
            else:
                x = x.to(device)
            
            y = y.to(device)
            
            # TESTE A: Verificando os Tipos de Dados (A Teoria do Conflito de Tipos)
            #print(f"[Teste A] Tipo da imagem (x): {x.dtype} | Shape: {x.shape}")
            #print(f"[Teste A] Tipo do parâmetro do modelo: {next(model.parameters()).dtype}")
            
            # TESTE B: Isolando o Forward Pass
            with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16):
                output = model(x)
            
            if isinstance(output, dict):
                output = output.get("logits", output)
                
            y = y.squeeze()
            
            # TESTE C: Verificando valores corrompidos (A Teoria da Divisão por Zero)
            if torch.isnan(output).any():
                print(f"[Teste C] ALERTA: A saída do modelo gerou NaNs (valores infinitos ou inválidos)!")
                
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
        old_param.data = new_param.data.clone()
        
def resize_model_to_pruned(model, pruned_dict):
    """ Redimensiona o modelo existente para as dimensões podadas """
    with torch.no_grad():
        for name, param in list(model.named_parameters()):
            if name in pruned_dict:
                pruned_weight = pruned_dict[name]
                
                if param.shape != pruned_weight.shape:
                    print(f"Redimensionando {name}: {param.shape} -> {pruned_weight.shape}")
                    new_param = nn.Parameter(pruned_weight.to(param.device))
                    
                    if '.' in name:
                        parts = name.split('.')
                        module = model
                        for part in parts[:-1]:
                            module = getattr(module, part)
                        setattr(module, parts[-1], new_param)
                        if hasattr(module, 'r') and 'lora_A' in name:
                            module.r = pruned_weight.shape[0]
                    else:
                        setattr(model, name, new_param)
                else:
                    param.data.copy_(pruned_weight.to(param.device))
    return model