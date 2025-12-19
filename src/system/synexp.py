import torch
import torch.nn as nn
import numpy as np

class LayerComplexityCalculator:
    """
    Classe auxiliar que calcula α_l (parâmetros) e β_l (FLOPs) por camada.
    """
    
    def __init__(self, model, input_size=(1, 3, 32, 32)):
        """
        Args:
            model: Modelo PyTorch (ex: VGG11)
            input_size: Tamanho da entrada para calcular FLOPs
        """
        self.model = model
        self.input_size = input_size
        
    def calculate_alpha_beta(self):
        """
        Retorna duas listas:
        - alpha_list: [α_1, α_2, ...]  # parâmetros por camada
        - beta_list:  [β_1, β_2, ...]  # FLOPs por camada
        """
        # 1. Calcular α_l (parâmetros) - FÁCIL
        alpha_list = self._calculate_parameters_per_layer()
        
        # 2. Calcular β_l (FLOPs) - usando seu código
        beta_list = self._calculate_flops_per_layer()
        
        return alpha_list, beta_list
    
    def _calculate_parameters_per_layer(self):
        """Calcula α_l: número de parâmetros em cada camada"""
        parameters_per_layer = []
        
        # Percorrer cada camada do modelo
        for name, module in self.model.named_modules():
            # Só nos interessam camadas que têm parâmetros
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Contar parâmetros desta camada
                num_params = sum(p.numel() for p in module.parameters())
                parameters_per_layer.append(num_params)
                
                # Mostrar (opcional)
                print(f"Camada: {name:30} | Parâmetros: {num_params:8,}")
        
        return parameters_per_layer
    
    def _calculate_flops_per_layer(self):
        """Calcula β_l: FLOPs por camada usando seu código"""
        # Importar sua função
        from flops import count_flops
        
        # Chamar sua função com verbose=True para ver detalhes
        total_flops, breakdown, gflops = count_flops(
            self.model, 
            self.input_size, 
            verbose=True  # Mostra cada camada
        )
        
        # Extrair só os valores dos FLOPs por camada
        flops_values = []
        for key, value in breakdown.items():
            if "Conv2d" in key or "Linear" in key:  # Só camadas importantes
                flops_values.append(value)
        
        return flops_values
    
class SynExpSimple:
    """
    Implementação SIMPLES da equação (3) do artigo.
    Não precisa ser perfeito agora - vamos fazer funcionar primeiro.
    """
    
    def __init__(self, alpha_list, beta_list):
        self.alpha = np.array(alpha_list)
        self.beta = np.array(beta_list)
    
    def compute_densities(self, target_sparsity=0.1):
        """
        Calcula p_l (densidade por camada) para manter target_sparsity dos parâmetros.
        
        Exemplo: target_sparsity=0.1 significa manter apenas 10% dos parâmetros.
        """
        # Total de parâmetros
        total_params = self.alpha.sum()
        target_params = total_params * target_sparsity
        
        print(f"\nSynExp - Cálculo de densidades:")
        print(f"Parâmetros totais: {total_params:,}")
        print(f"Parâmetros alvo (10%): {target_params:,.0f}")
        
        # SOLUÇÃO SIMPLIFICADA (para começar):
        # Distribuir proporcionalmente ao inverso de α_l
        # Isso não é exatamente a equação (3), mas é um bom começo
        
        # Quanto cada camada deve contribuir?
        # Camadas com mais parâmetros (α grande) devem ter p menor
        weights = 1 / (self.alpha + 1e-8)  # +1e-8 para evitar divisão por zero
        weights = weights / weights.sum()   # Normalizar para soma = 1
        
        # Distribuir o "orçamento" de parâmetros
        densities = (target_params * weights) / self.alpha
        
        # Garantir que 0 < p_l ≤ 1
        densities = np.clip(densities, 0.01, 1.0)  # Mínimo 1%
        
        # Verificação
        params_after = (self.alpha * densities).sum()
        print(f"Parâmetros após cálculo: {params_after:,.0f}")
        print(f"Razão alcançada: {params_after/total_params:.3f}")
        
        # Mostrar primeiras 5 camadas
        print("\nPrimeiras 5 camadas:")
        for i in range(min(5, len(densities))):
            print(f"  Camada {i}: α={self.alpha[i]:8,} | p={densities[i]:.4f} | "
                  f"params={self.alpha[i]*densities[i]:,.0f}")
        
        return densities.tolist()