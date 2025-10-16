import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy
import numpy as np

def prune_and_restructure(model, pruning_rate = 0.5, n_in = 1, size_fc = 25, data='Cifar100'):
    model_copy = copy.deepcopy(model)
    layers = []
    indices_not_remove_weight = []
    first_module = True
    first_linear = True
    index_last_layer = 0
    masks = []
    a = 0
    for n, module in enumerate(model_copy.modules()):
        if isinstance(module, nn.Linear):
            index_last_layer = n

    for n, module in enumerate(model_copy.modules()):
        if isinstance(module, nn.Conv2d):
            
            # realiza o prunning e substitui os pesos por 0
            prune.ln_structured(module, name='weight', amount=pruning_rate, n=2, dim=0)
            masks.append(list(model_copy.buffers())[0])
            prune.remove(module, 'weight')

            # filtra onde não é 0
            indices_not_remove_perceptron = module.weight.abs().sum(dim=(1, 2, 3)) != 0

            # aplica o filtragem dos valores 0
            weight_prunned = module.weight[indices_not_remove_perceptron]

            # filtra os pesos da camada
            if first_module:
                first_module = False
            else:
                weight_prunned = weight_prunned[:, indices_not_remove_weight, :, :]
                
            if module.bias is not None:
                bias_prunned = module.bias[indices_not_remove_perceptron]
            
            # determina tamanho da saída
            n_out = sum(indices_not_remove_perceptron)

            # cria nova cadamada com os parametros filtrados
            new_layer = nn.Conv2d(
                        in_channels=n_in,
                        out_channels=n_out,
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                        bias=(module.bias is not None)
                        )
            new_layer.bias = nn.Parameter(bias_prunned)
            new_layer.weight = nn.Parameter(weight_prunned)
            
            layers.append(new_layer)
            indices_not_remove_weight = indices_not_remove_perceptron
            n_in = n_out

        if isinstance(module, nn.Linear):            
            
            # realiza o prunning e substitui os pesos por 0
            prune.ln_structured(module, name='weight', amount=pruning_rate, n=2, dim=0)
            masks.append(list(model_copy.buffers())[0])
            prune.remove(module, 'weight')

            # filtra onde não é 0
            indices_not_remove_perceptron = module.weight.abs().sum(dim=1) != 0

            # filtra os pesos da camada
            if first_linear:
                if data=='Cifar100' or data=='Cifar10':
                    print('oi')
                    indices_not_remove_weight = indices_not_remove_weight.repeat(25)
                if data=='MNIST':
                    print('xau')
                    indices_not_remove_weight = indices_not_remove_weight.repeat(16)
                first_linear = False
                
                n_in = sum(indices_not_remove_weight)
                layers.append(nn.Flatten())
            
            # remove os pesos que não serão utilizados
            weight_prunned = module.weight[:, indices_not_remove_weight]
            # aplica o filtragem nos neuronios com valores 0
            if data=='Cifar100':
                ind=100
            else:
                ind=10
            if n == index_last_layer: # não reestrutura a ultima camada
                indices_not_remove_perceptron = torch.tensor([True] * ind)
                masks[-1] = masks[-1] + 1
            
            weight_prunned = weight_prunned[indices_not_remove_perceptron]
                
            if module.bias is not None:
                bias_prunned = module.bias[indices_not_remove_perceptron]
                
            # determina tamanho da saída
            n_out = sum(indices_not_remove_perceptron)

            # cria nova cadamada com os parametros filtrados
            new_layer = nn.Linear(n_in, n_out)
            new_layer.bias = nn.Parameter(bias_prunned)
            new_layer.weight = nn.Parameter(weight_prunned)
            
            layers.append(new_layer)
            indices_not_remove_weight = indices_not_remove_perceptron
            n_in = n_out

        if isinstance(module, (nn.ReLU, nn.MaxPool2d)):
            #adiciona as camadas de funções de ativação 
            layers.append(module)
            
        
        new_model = nn.Sequential(*layers)
    #print("rapaz ne que foi")
    return new_model, masks

def restore_to_original_size(model, masks, size_fc=25):
    model = copy.deepcopy(model)
    index_mask = 0 # indice atual da mascara
    layers = [] # armazena as layers ja reconstruidas
    first_layer = True
    first_linear = True

    for module in model.modules():
        if isinstance(module, nn.Linear):
            mask = masks[index_mask] # mascara para a camada atual
            
            if first_linear:
                # ajusta de acordo com as dimensões alteradas pelo flatten
                bool_mask_weight = np.repeat(bool_mask_weight, 25)
                bool_mask_weight = torch.tensor(bool_mask_weight).clone().detach()
                first_linear = False
            
            #filtra a posição dos pesos
            new_weight = torch.zeros((module.weight.shape[0], len(bool_mask_weight)))
            new_weight[:,bool_mask_weight] = module.weight

            new_layer_weight = torch.zeros(mask.shape) # novos pesos da layer 
            new_layer_bias = torch.zeros(mask.shape[0]) # novos bias da layer

            bool_mask_perceptron = mask.sum(dim=1) > 0 # mascara

            new_layer_weight[bool_mask_perceptron] = new_weight # substitui nas respectivas posições
            new_layer_bias[bool_mask_perceptron] = module.bias # substitui nas respectivas posições

            n_in = mask.shape[1]
            n_out = mask.shape[0]

            layer = nn.Linear(n_in, n_out)
            
            layer.weight = nn.Parameter(new_layer_weight)
            layer.bias = nn.Parameter(new_layer_bias)

            bool_mask_weight = bool_mask_perceptron # atualiza a mascara de pesos
            layers.append(layer)
            index_mask += 1 # atualiza indice da mascara
            
        elif isinstance(module, nn.Conv2d):
            new_weight = module.weight 
            if not first_layer:
                # no caso de não ser a primeira layer, é necessario adicionar os pesos 
                # que foram removidos da camda

                #cria tensor com dimensão correta dos pesos
                n_in = module.weight.shape[0]
                n_out = len(bool_mask_weight)
                k1 = module.kernel_size[0]
                k2 = module.kernel_size[1]
                new_weight = torch.zeros((n_in, n_out, k1, k2))

                #filtra a posição dos pesos
                new_weight[:,bool_mask_weight] = module.weight
                
            else:
                first_layer = False
        
            mask = masks[index_mask] # mascara para a camada atual
            
            new_layer_weight = torch.zeros(mask.shape) # novos pesos da layer
            new_layer_bias = torch.zeros(mask.shape[0]) # novos bias da layer

            bool_mask_perceptron = mask.sum(dim=(1, 2, 3)) > 0 # mascara

            new_layer_weight[bool_mask_perceptron] = new_weight # substitui nas respectivas posições
            new_layer_bias[bool_mask_perceptron] = module.bias # substitui nas respectivas posições
           
            bool_mask_weight = bool_mask_perceptron # atualiza a mascara de pesos

            n_in = mask.shape[1]
            n_out = mask.shape[0]

            layer = nn.Conv2d(in_channels=n_in,
                              out_channels=n_out,
                              kernel_size=module.kernel_size,
                              padding=module.padding)
            
            layer.weight = nn.Parameter(new_layer_weight)
            layer.bias = nn.Parameter(new_layer_bias)
            
            layers.append(layer)
            index_mask += 1 #atual indice da mascara
        elif isinstance(module, (nn.ReLU, nn.MaxPool2d, nn.Flatten)):
            layers.append(module) # adiciona o modulo as layers
    
    model = nn.Sequential(*layers)
    return model
