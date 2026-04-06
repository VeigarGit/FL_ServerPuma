import torch
import torch.nn as nn

def snip_pruning(model, dataloader, criterion, pruning_ratio, device):
    model.train()
    images, labels = next(iter(dataloader))
    images = images.to(device)
    labels = labels.to(device)
    
    # Calcular la pérdida y retropropagación
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward() # Calcular gradientes para todos los parámetros
         
    # Calcular la importancia de cada peso: |gradiente * peso|
    importance_scores = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Pode-se substituir por apenas 'param.grad' ou 'param' para diferentes critérios de importância (|gradiente| ou |gradiente * peso|).
            #importance_scores[name] = torch.abs(param.grad * param)
            importance_scores[name] = torch.abs(param.grad)
    
    # Concatenar todas las importancias y determinar el umbral de poda
    all_scores = torch.cat([torch.flatten(score) for score in importance_scores.values()])
    
    num_params_to_keep = int(len(all_scores) * (1 - pruning_ratio))
    threshold = torch.topk(all_scores, num_params_to_keep, largest=True).values[-1]
    
    # Crear máscaras
    masks = {name: score >= threshold for name, score in importance_scores.items()}
    
    return masks

def apply_mask(model, masks):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.mul_(masks[name])
    return model