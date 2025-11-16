import torch

def dequantization(global_state):
    """Dequantizes a state dictionary."""
    dequantized_state_dict = {}
    for k, v in global_state.items():
        if isinstance(v, dict) and v.get('dtype') == 'quantized_int8':
            # Recupera tensores quantizados
            scale = v['scale']
            dequantized_state_dict[k] = v['weights'].float() * scale
        else:
            # Mantém tensores normais
            dequantized_state_dict[k] = v
    return dequantized_state_dict

def quantization(state_dict):
    """Quantizes a state dictionary to int8."""
    quantized_state_dict = {}
    keys = list(state_dict.keys())
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