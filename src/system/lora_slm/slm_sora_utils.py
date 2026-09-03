import torch
import copy

def reduce_sora_state_dict_rank(state_dict, threshold=1e-4, min_rank=2):
    """
    Analisa o state_dict e reduz iterativamente o rank das matrizes SoRA
    removendo as dimensões onde o gate está zerado, mantendo-as treináveis.
    Respeita um rank mínimo (min_rank) por módulo para evitar over-pruning.
    """
    new_state_dict = copy.deepcopy(state_dict)
    gate_keys = [k for k in new_state_dict.keys() if "sora.gate" in k]
    
    total_params_before = 0
    total_params_after = 0
    ranks_info = []
    
    for gate_key in gate_keys:
        base_key = gate_key.rsplit(".gate", 1)[0]
        A_key = f"{base_key}.lora_A"
        B_key = f"{base_key}.lora_B"
        
        gate_tensor = new_state_dict[gate_key].squeeze(0)
        A_tensor = new_state_dict[A_key]
        B_tensor = new_state_dict[B_key]
        
        r_original = len(gate_tensor)
        total_params_before += A_tensor.numel() + B_tensor.numel() + gate_tensor.numel()
        
        mask = torch.abs(gate_tensor) > threshold
        keep_idx = torch.where(mask)[0]
        
        # Garante que o rank não caia abaixo de min_rank
        effective_min = min(min_rank, r_original)
        if len(keep_idx) < effective_min:
            keep_idx = torch.topk(gate_tensor.abs(), k=effective_min).indices
            keep_idx, _ = torch.sort(keep_idx)  # Preserva ordem ordinal para alinhamento na agregação
            
        r_new = len(keep_idx)
        
        new_state_dict[A_key] = A_tensor[keep_idx, :]
        new_state_dict[B_key] = B_tensor[:, keep_idx]
        new_state_dict[gate_key] = gate_tensor[keep_idx].unsqueeze(0)
        
        total_params_after += new_state_dict[A_key].numel() + new_state_dict[B_key].numel() + new_state_dict[gate_key].numel()
        if r_new < r_original:
            ranks_info.append(f"{base_key}: {r_original} -> {r_new}")
        
    return new_state_dict, total_params_before, total_params_after, ranks_info