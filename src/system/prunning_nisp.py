import torch
import torch.nn as nn
from tqdm import tqdm

# Función para obtener la salida de la FRL (penúltima capa)
def get_frl_output(model, dataloader, device):
    frl_outputs = []
 
    def hook(module, input, output):
        frl_outputs.append(output.detach())

    # Registrar un hook en la capa fc2

    hook_handle = model.fc.register_forward_hook(hook)

    # Realizar una pasada hacia adelante con las imágenes del dataloader
    print("Obteniendo salida de la FRL para el dataset MNIST...")
    for images, _ in tqdm(dataloader, desc="Procesando imágenes"):
        with torch.no_grad():
            model(images.to(device))

    # Eliminar el hook
    hook_handle.remove()

    return frl_outputs

def calculate_neuron_importance(frl_outputs):
    # Convertir la lista de tensores a un solo tensor
    frl_outputs_tensor = torch.cat(frl_outputs, dim=0)  # Dimensión (num_images, 84)
    
    # Calcular la suma de los valores absolutos para cada neurona en la FRL
    importance_scores = torch.sum(torch.abs(frl_outputs_tensor), dim=0)
    
    return importance_scores

def propagate_importance_fc(model, importance_scores):
    # Obtener los pesos absolutos de la capa fc2
    weights_fc2 = torch.abs(model.fc.weight.data)  # Dimensión (84, 120)
    
    # Propagar la importancia a la capa fc1
    importance_fc1 = torch.matmul(weights_fc2.T, importance_scores)  # Dimensión (120,)
    
    return importance_fc1

def prune_fc1(model, dataloader, device, pruning_ratio=0.5):

    frl_outputs = get_frl_output(model, dataloader, device)
    importance_scores = calculate_neuron_importance(frl_outputs)
    importance_fc1 = propagate_importance_fc(model, importance_scores)

    # Determinar el número de neuronas a mantener
    num_neurons = len(importance_fc1)
    num_to_keep = int(num_neurons * (1 - pruning_ratio))
    
    # Obtener los índices de las neuronas más importantes
    _, indices_to_keep = torch.topk(importance_fc1, num_to_keep)
    indices_to_keep = indices_to_keep.sort()[0]  # Ordenar los índices

    # Prunar la capa fc1
    old_fc1 = model.fc1
    new_fc1 = nn.Sequential(nn.Linear(old_fc1[0].in_features, num_to_keep), nn.ReLU(inplace=True))
    new_fc1[0].weight.data = old_fc1[0].weight.data[indices_to_keep]
    new_fc1[0].bias.data = old_fc1[0].bias.data[indices_to_keep]

    # Actualizar fc2 para reflejar los cambios en fc1
    old_fc2 = model.fc
    new_fc2 = nn.Linear(num_to_keep, old_fc2.out_features)
    new_fc2.weight.data = old_fc2.weight.data[:, indices_to_keep]
    new_fc2.bias.data = old_fc2.bias.data
    print(model)
    # Reemplazar las capas en el modelo
    model.fc1 = new_fc1
    model.fc = new_fc2
    print(model)
    
    # Verificar las nuevas dimensiones
    #print(f"Nuevo tamaño de fc1: {model.fc1.in_features} -> {model.fc1.out_features}")
    #print(f"Nuevo tamaño de fc2: {model.fc.in_features} -> {model.fc.out_features}")

    return model, indices_to_keep