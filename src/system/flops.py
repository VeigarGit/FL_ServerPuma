import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import math

def count_flops(model, input_size=(1, 3, 32, 32), verbose=False):
    """
    Calcula o número de FLOPs para qualquer variação dos modelos.
    
    Args:
        model: Modelo a ser analisado
        input_size: Tamanho da entrada (batch, channels, height, width)
        verbose: Se True, imprime detalhes de cada camada
    
    Returns:
        total_flops: Total de FLOPs (operações de ponto flutuante)
        breakdown: Dicionário com FLOPs por camada
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(input_size).to(device)
    
    # Dicionários para armazenar informações
    flops_breakdown = defaultdict(float)
    
    def conv2d_flops(module, input_size, output_size):
        """Calcula FLOPs para Conv2d"""
        batch_size, in_channels, h_in, w_in = input_size
        out_channels, h_out, w_out = output_size[1:]
        
        # Parâmetros da convolução
        kernel_size = module.kernel_size
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        
        # Cada operação de convolução: multiplicação + adição
        # Para cada posição de saída: in_channels * kernel_h * kernel_w * 2 FLOPs
        # Multiplicado por todas as posições de saída
        flops_per_position = in_channels * kernel_size[0] * kernel_size[1] * 2
        total_positions = out_channels * h_out * w_out
        flops = batch_size * flops_per_position * total_positions
        
        # Adicionar bias (se existir)
        if module.bias is not None:
            flops += batch_size * out_channels * h_out * w_out
        
        return flops
    
    def linear_flops(module, input_size, output_size):
        """Calcula FLOPs para Linear"""
        batch_size = input_size[0]
        in_features = input_size[-1]
        out_features = output_size[-1]
        
        # Cada saída: in_features multiplicações + (in_features-1) adições + bias
        flops_per_sample = (2 * in_features - 1) * out_features
        if module.bias is not None:
            flops_per_sample += out_features  # adições do bias
        
        return batch_size * flops_per_sample
    
    def attention_flops(q, k, v, dropout_p=0.0):
        """Calcula FLOPs para scaled_dot_product_attention"""
        batch_size, n_heads, seq_len, head_dim = q.shape
        
        # 1. Q @ K^T: [batch, heads, seq_len, head_dim] @ [batch, heads, head_dim, seq_len]
        # FLOPs: batch * heads * seq_len * seq_len * head_dim * 2
        flops_qk = batch_size * n_heads * seq_len * seq_len * head_dim * 2
        
        # 2. Softmax (aproximação: 3 operações por elemento)
        flops_softmax = batch_size * n_heads * seq_len * seq_len * 3
        
        # 3. Attention @ V: [batch, heads, seq_len, seq_len] @ [batch, heads, seq_len, head_dim]
        # FLOPs: batch * heads * seq_len * seq_len * head_dim * 2
        flops_av = batch_size * n_heads * seq_len * seq_len * head_dim * 2
        
        total = flops_qk + flops_softmax + flops_av
        
        return total
    
    def maxpool2d_flops(module, input_size, output_size):
        """Calcula FLOPs para MaxPool2d (aproximação)"""
        batch_size, channels, h_in, w_in = input_size
        h_out, w_out = output_size[2:]
        
        # Cada operação de max pooling: comparações
        # kernel_size^2 - 1 comparações por posição
        kernel_size = module.kernel_size
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        
        comparisons_per_position = kernel_size[0] * kernel_size[1] - 1
        total_positions = batch_size * channels * h_out * w_out
        
        return total_positions * comparisons_per_position
    
    # Hooks para capturar tamanhos de entrada/saída
    hooks = []
    layer_info = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(input, tuple):
                input = input[0]
            if isinstance(output, tuple):
                output = output[0]
            layer_info[name] = {
                'module': module,
                'input_size': input.shape,
                'output_size': output.shape
            }
        return hook
    
    # Registrar hooks para todas as camadas
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d)):
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)
    
    # Forward pass para coletar informações
    with torch.no_grad():
        _ = model(x)
    
    # Remover hooks
    for hook in hooks:
        hook.remove()
    
    # Calcular FLOPs para cada camada
    total_flops = 0
    
    for name, info in layer_info.items():
        module = info['module']
        input_size = info['input_size']
        output_size = info['output_size']
        
        if isinstance(module, nn.Conv2d):
            flops = conv2d_flops(module, input_size, output_size)
            flops_breakdown[f'Conv2d_{name}'] = flops
            
        elif isinstance(module, nn.Linear):
            flops = linear_flops(module, input_size, output_size)
            flops_breakdown[f'Linear_{name}'] = flops
            
        elif isinstance(module, nn.MaxPool2d):
            flops = maxpool2d_flops(module, input_size, output_size)
            flops_breakdown[f'MaxPool2d_{name}'] = flops
        
        total_flops += flops
        
        if verbose:
            print(f"{name}: {input_size} -> {output_size}, FLOPs: {flops:,}")
    
    # Calcular FLOPs para operações de atenção (se existirem)
    # Para o modelo SimpleModelWithFlashAttention, precisamos calcular manualmente
    if hasattr(model, 'n_heads'):
        # Obter tamanhos da sequência após convoluções
        with torch.no_grad():
            out = model.conv1(x)
            out = model.conv2(out)
            out = out.flatten(2).transpose(1, 2)  # (batch, seq_len, features)
            
            batch_size, seq_len, features = out.shape
            n_heads = model.n_heads
            head_dim = features // n_heads
            
            # Criar tensores Q, K, V para cálculo
            q = torch.randn(batch_size, n_heads, seq_len, head_dim)
            k = torch.randn(batch_size, n_heads, seq_len, head_dim)
            v = torch.randn(batch_size, n_heads, seq_len, head_dim)
            
            # FLOPs da atenção
            flops_attention = attention_flops(q, k, v)
            flops_breakdown['Attention'] = flops_attention
            total_flops += flops_attention
            
            if verbose:
                print(f"Attention: seq_len={seq_len}, heads={n_heads}, head_dim={head_dim}")
                print(f"  FLOPs: {flops_attention:,}")
    
    # FLOPs para operações de ativação (ReLU)
    # Cada ReLU: 1 operação por elemento
    if verbose:
        # Estimar FLOPs para ReLUs
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                # Encontrar o tamanho da saída desta camada
                for layer_name, info in layer_info.items():
                    if layer_name in name or name in layer_name:
                        output_size = info['output_size']
                        flops_relu = output_size.numel()  # 1 FLOP por elemento
                        flops_breakdown[f'ReLU_{name}'] = flops_relu
                        total_flops += flops_relu
                        
                        if verbose:
                            print(f"ReLU_{name}: {output_size}, FLOPs: {flops_relu:,}")
                        break
    
    # Converter para GigaFLOPs
    gflops = total_flops / 1e9
    
    if verbose:
        print("\n" + "="*50)
        print(f"TOTAL FLOPs: {total_flops:,} ({gflops:.2f} GFLOPs)")
        print("="*50)
    
    return total_flops, dict(flops_breakdown), gflops


# Versão otimizada para uso rápido
def estimate_flops(model, input_size=(1, 3, 32, 32)):
    """
    Estimativa rápida de FLOPs sem hooks detalhados.
    Útil para comparações rápidas entre modelos.
    """
    # Configurações padrão baseadas no modelo
    batch_size, channels, height, width = input_size
    
    flops = 0
    
    # Contar camadas convolucionais
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Calcular tamanho de saída
            h_out = (height + 2*module.padding[0] - module.dilation[0]*(module.kernel_size[0]-1)-1)//module.stride[0] + 1
            w_out = (width + 2*module.padding[1] - module.dilation[1]*(module.kernel_size[1]-1)-1)//module.stride[1] + 1
            
            # FLOPs da convolução
            flops += batch_size * module.out_channels * h_out * w_out * \
                    module.in_channels * module.kernel_size[0] * module.kernel_size[1] * 2
            
            # Atualizar dimensões para próxima camada
            height, width = h_out, w_out
            channels = module.out_channels
            
        elif isinstance(module, nn.Linear):
            # FLOPs da camada linear
            flops += batch_size * (2 * module.in_features - 1) * module.out_features
    
    # Adicionar FLOPs da atenção se existir
    if hasattr(model, 'n_heads'):
        # Estimar dimensões após convoluções
        # Para entrada 32x32 após duas convoluções 5x5 com stride 1 e maxpool 2x2
        h = ((32 - 4) // 2 - 4) // 2  # = 5
        w = h  # = 5
        seq_len = h * w  # = 25
        
        batch_size = input_size[0]
        n_heads = model.n_heads
        head_dim = model.feature_dim // n_heads
        
        # FLOPs da atenção
        flops += batch_size * n_heads * seq_len * seq_len * head_dim * 4  # QK^T + AV
    
    return flops, flops / 1e9
