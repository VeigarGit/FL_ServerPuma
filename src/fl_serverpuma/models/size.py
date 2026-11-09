# Responsabilidade: Mede tamanho total/treinável, quantidade de parâmetros e tamanho de
# tensores de um state_dict.
#
# Principais funções: get_model_size, get_trainable_size_and_params, get_state_dict_size_mb.

import torch

def get_model_size(model):
    total_params = 0
    for param in model.parameters():
        total_params += param.numel() * param.element_size()  # número de elementos * tamanho de cada elemento em bytes
    return total_params / (1024 ** 2)  # converte para megabytes (MB)

def get_trainable_size_and_params(model):
    """Retorna o tamanho (em MB) e a quantidade apenas dos parâmetros treináveis."""
    total_size = 0
    total_params = 0
    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()
            total_size += param.numel() * param.element_size()
    return total_size / (1024 ** 2), total_params

# Exemplo de uso
# model = ...  # seu modelo PyTorch
# print(f"Tamanho do modelo: {get_model_size(model):.2f} MB")

def get_state_dict_size_mb(state):
    """Calcula matematicamente o tamanho em MB sem serializar em pickle, evitando travar o GIL e a CPU."""
    total_bytes = 0
    if hasattr(state, 'state_dict'):
        state = state.state_dict()

    if isinstance(state, dict) and state.get('__lhdq__'):
        for k, v in state['keys'].items():
            if 'raw' in v:
                total_bytes += v['raw'].numel() * v['raw'].element_size()
            else:
                total_bytes += len(v['packed']) + 4  # 4 bytes for mu/sigma (float16)
    elif isinstance(state, dict):
        for v in state.values():
            if isinstance(v, torch.Tensor):
                total_bytes += v.numel() * v.element_size()
            elif isinstance(v, dict) and v.get('dtype') == 'quantized_int8':
                if 'weights' in v and isinstance(v['weights'], torch.Tensor):
                    total_bytes += v['weights'].numel() * v['weights'].element_size()
                if 'scale' in v and isinstance(v['scale'], torch.Tensor):
                    total_bytes += v['scale'].numel() * v['scale'].element_size()
            elif isinstance(v, (int, float)):
                total_bytes += 8
    return total_bytes / (1024 * 1024)
