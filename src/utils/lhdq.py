"""
LHDQ — Low Huffman-coded Delta Quantization
=============================================
Comprime atualizações de modelo para Aprendizado Federado usando:
1. Delta Quantization: transmite ΔP = P_t − P_{t-1} em vez dos pesos brutos.
2. 3-Level Quantization: mapeia ΔP para {−1, 0, +1} usando limites estatísticos.
3. Huffman Bit-Packing: codifica os 3 símbolos em ~1.67 bits/parâmetro.

Uso:
    from utils.lhdq import lhdq_encode, lhdq_decode

    encoded = lhdq_encode(current_state, previous_state)
    recovered = lhdq_decode(encoded, previous_state)
"""

import math
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ─── Códigos de Huffman para 3 símbolos equiprováveis ────────────────────────
# Símbolo  | Código | Bits
# ---------|--------|-----
#   0      |   1    |  1
#  -1      |  01    |  2
#  +1      |  00    |  2
# Média teórica: (1/3)*1 + (1/3)*2 + (1/3)*2 = 5/3 ≈ 1.67 bits/param
#
# Mapeamento interno: -1 → 0,  0 → 1,  +1 → 2
HUFFMAN_CODES = {
    1: (0b1, 1),    # símbolo 0  → '1'    (1 bit)
    0: (0b01, 2),   # símbolo -1 → '01'   (2 bits)
    2: (0b00, 2),   # símbolo +1 → '00'   (2 bits)
}

# Tabela de decodificação (prefixo → símbolo)
# Lemos bit a bit: se o primeiro bit é 1 → símbolo 0
# Se o primeiro bit é 0, lemos o próximo: 1 → símbolo -1, 0 → símbolo +1
DECODE_FIRST_BIT_IS_1 = 1   # → símbolo 0
DECODE_SECOND_BIT_IS_1 = 0  # → símbolo -1
DECODE_SECOND_BIT_IS_0 = 2  # → símbolo +1


def _huffman_pack(symbols: np.ndarray) -> tuple:
    """
    Empacota um array de símbolos {0, 1, 2} em bytes usando Huffman.
    Implementação 100% vetorizada com Numpy para máxima performance.
    
    Returns:
        (packed_bytes: bytes, total_bits: int)
    """
    # Define os tamanhos: símbolo 1 usa 1 bit, os demais usam 2 bits
    lens = np.full(symbols.shape, 2, dtype=np.int32)
    lens[symbols == 1] = 1
    
    total_bits = int(lens.sum())
    
    # Aloca um array de bits preenchido com zeros
    bits = np.zeros(total_bits, dtype=np.uint8)
    
    # Calcula os índices iniciais de cada símbolo no array flat
    starts = np.empty_like(lens)
    starts[0] = 0
    np.cumsum(lens[:-1], out=starts[1:])
    
    # O primeiro bit é 1 para sym==1, e 0 para os demais
    bits[starts] = (symbols == 1)
    
    # O segundo bit só existe para símbolos != 1
    mask_2bits = (symbols != 1)
    starts_2bits = starts[mask_2bits]
    
    # Símbolo 0 vira '01' (segundo bit = 1)
    # Símbolo 2 vira '00' (segundo bit = 0)
    bits[starts_2bits + 1] = (symbols[mask_2bits] == 0)
    
    # O np.packbits automaticamente agrupa os bits em bytes MSB e faz o padding do último byte com zeros
    packed_bytes = np.packbits(bits).tobytes()
    return packed_bytes, total_bits


def _huffman_unpack(packed_bytes: bytes, total_bits: int, n_elements: int) -> np.ndarray:
    """
    Desempacota bytes Huffman de volta para um array de símbolos {0, 1, 2}.
    """
    symbols = np.empty(n_elements, dtype=np.int8)
    
    byte_idx = 0
    bit_idx = 7  # posição do bit no byte atual (7 = MSB)
    sym_idx = 0
    bits_read = 0
    
    current_byte = packed_bytes[0] if packed_bytes else 0
    
    while sym_idx < n_elements and bits_read < total_bits:
        # Lê o primeiro bit
        first_bit = (current_byte >> bit_idx) & 1
        bits_read += 1
        bit_idx -= 1
        if bit_idx < 0:
            byte_idx += 1
            bit_idx = 7
            current_byte = packed_bytes[byte_idx] if byte_idx < len(packed_bytes) else 0
        
        if first_bit == 1:
            # Código '1' → símbolo 0 (sem mudança)
            symbols[sym_idx] = DECODE_FIRST_BIT_IS_1
        else:
            # Primeiro bit é 0, precisa ler o segundo
            second_bit = (current_byte >> bit_idx) & 1
            bits_read += 1
            bit_idx -= 1
            if bit_idx < 0:
                byte_idx += 1
                bit_idx = 7
                current_byte = packed_bytes[byte_idx] if byte_idx < len(packed_bytes) else 0
            
            if second_bit == 1:
                symbols[sym_idx] = DECODE_SECOND_BIT_IS_1  # '01' → símbolo -1
            else:
                symbols[sym_idx] = DECODE_SECOND_BIT_IS_0  # '00' → símbolo +1
        
        sym_idx += 1
    
    return symbols


def _quantize_delta_3level(delta: torch.Tensor) -> tuple:
    """
    Quantiza um tensor delta em 3 níveis baseado em percentis reais (quantiles).
    Garante que a distribuição será dividida em terços independentemente 
    do formato (normal, cauda longa, concentrada, etc.).
    
    Returns:
        (symbols, mu, sigma) — o p_delta é recalculado pelo receptor.
    """
    delta_flat = delta.float().flatten()
    
    mu = delta_flat.mean().item()
    sigma = delta_flat.std().item()
    
    if sigma < 1e-10:
        n = delta_flat.numel()
        symbols = np.ones(n, dtype=np.int8)
        return symbols, mu, sigma
    
    # Usa percentis reais para definir os limites de cada terço
    quantiles = torch.quantile(delta_flat, torch.tensor([0.3333, 0.6667], device=delta_flat.device))
    lower_bound = quantiles[0].item()
    upper_bound = quantiles[1].item()
    
    # Previne limites idênticos em distribuições extremamente concentradas
    if lower_bound >= upper_bound:
        lower_bound = mu - 1e-8
        upper_bound = mu + 1e-8
        
    # Computa símbolos diretamente na GPU (muito mais rápido e transfere 4x menos dados)
    symbols_pt = torch.ones_like(delta_flat, dtype=torch.uint8)
    symbols_pt[delta_flat < lower_bound] = 0
    symbols_pt[delta_flat > upper_bound] = 2
    
    # Transfere apenas o array uint8 final para a CPU
    symbols = symbols_pt.cpu().numpy()
        
    return symbols, mu, sigma


def _compute_p_delta(mu: float, sigma: float) -> float:
    """
    Calcula o valor de reconstrução p_delta conforme a literatura LHDQ:
        p_delta = mu + 0.88 * sigma * sqrt(2)
    
    Este valor divide a PDF normal em 3 terços equiprováveis.
    """
    return mu + 0.88 * sigma * math.sqrt(2)


def _dequantize_3level(symbols: np.ndarray, mu: float, sigma: float, 
                        shape: tuple, device, dtype) -> torch.Tensor:
    """
    Reconstrói o tensor delta a partir dos símbolos quantizados.
    
    O p_delta é recalculado a partir de μ e σ (recebidos em float16).
    Mapeamento: 0 → -p_delta,  1 → 0,  2 → +p_delta
    """
    p_delta = _compute_p_delta(mu, sigma)
    
    # Transfere os símbolos (uint8) para a GPU antes de mapear
    symbols_pt = torch.from_numpy(symbols).to(device=device)
    
    # Cria o tensor final diretamente na GPU
    delta_values = torch.zeros(len(symbols), device=device, dtype=dtype)
    delta_values[symbols_pt == 0] = -p_delta
    delta_values[symbols_pt == 2] = +p_delta
    
    return delta_values.reshape(shape)


def lhdq_encode(current_state: dict, previous_state: dict) -> dict:
    """
    Codifica a diferença entre o estado atual e o anterior usando LHDQ.
    
    Args:
        current_state: state_dict atual (pesos float)
        previous_state: state_dict da rodada anterior (pesos float)
    
    Returns:
        dict com a estrutura:
        {
            '__lhdq__': True,  # flag de identificação
            'keys': {
                'layer_name': {
                    'packed': bytes,      # bitstream Huffman
                    'total_bits': int,
                    'n_elements': int,
                    'shape': tuple,
                    'mu': float,          # quantizado em float16
                    'sigma': float,       # quantizado em float16
                    'dtype': torch.dtype,
                    'device': str,
                }
            }
        }
    """
    encoded = {'__lhdq__': True, 'keys': {}}
    
    total_original_bytes = 0
    total_compressed_bytes = 0
    
    for key in current_state:
        current_val = current_state[key]
        
        if not isinstance(current_val, torch.Tensor):
            # Valores não-tensoriais (ex: contadores) são passados diretamente
            encoded['keys'][key] = {'raw': current_val}
            continue
        
        prev_val = previous_state.get(key)
        
        if prev_val is None or prev_val.shape != current_val.shape:
            # Chave nova ou shape mudou (ex: rank adaptativo) — fallback para raw
            encoded['keys'][key] = {'raw': current_val}
            continue
        
        # Calcula o delta
        delta = current_val.float() - prev_val.float().to(current_val.device)
        
        # Quantiza em 3 níveis
        symbols, mu, sigma = _quantize_delta_3level(delta)
        
        # Quantiza μ e σ em float16 conforme a literatura (16 bits cada)
        mu_f16 = torch.tensor(mu, dtype=torch.float16).item()
        sigma_f16 = torch.tensor(sigma, dtype=torch.float16).item()
        
        # Empacota com Huffman
        packed_bytes, total_bits = _huffman_pack(symbols)
        
        # Estatísticas de compressão
        original_bytes = current_val.numel() * current_val.element_size()
        compressed_bytes = len(packed_bytes) + 4  # +4 bytes para mu/sigma (2 × float16)
        total_original_bytes += original_bytes
        total_compressed_bytes += compressed_bytes
        
        encoded['keys'][key] = {
            'packed': packed_bytes,
            'total_bits': total_bits,
            'n_elements': current_val.numel(),
            'shape': tuple(current_val.shape),
            'mu': mu_f16,
            'sigma': sigma_f16,
            'dtype': current_val.dtype,
            'device': str(current_val.device),
        }
    
    if total_original_bytes > 0:
        ratio = total_compressed_bytes / total_original_bytes
        bpp = (total_compressed_bytes * 8) / max(1, sum(
            current_state[k].numel() for k in current_state 
            if isinstance(current_state[k], torch.Tensor) and k in previous_state
        ))
        logger.info(f"LHDQ encode: {total_original_bytes/1024/1024:.2f} MB → "
                     f"{total_compressed_bytes/1024/1024:.2f} MB "
                     f"(ratio={ratio:.3f}, ~{bpp:.2f} bits/param)")
    
    return encoded


def lhdq_decode(encoded_state: dict, previous_state: dict) -> dict:
    """
    Decodifica um estado LHDQ de volta para um state_dict de tensores.
    
    Args:
        encoded_state: dicionário produzido por lhdq_encode()
        previous_state: state_dict da rodada anterior (mesmos pesos usados no encode)
    
    Returns:
        state_dict reconstruído com tensores float
    """
    decoded = {}
    
    for key, entry in encoded_state['keys'].items():
        # Valores raw (não-tensoriais ou fallback)
        if 'raw' in entry:
            decoded[key] = entry['raw']
            continue
        
        # Desempacota Huffman
        symbols = _huffman_unpack(
            entry['packed'], 
            entry['total_bits'], 
            entry['n_elements']
        )
        
        # Determina o device correto: prefere o do previous_state
        prev_val = previous_state.get(key)
        if prev_val is not None:
            device = prev_val.device
            dtype = prev_val.dtype
        else:
            device = torch.device(entry.get('device', 'cpu'))
            dtype = entry.get('dtype', torch.float32)
        
        # Reconstrói o delta (p_delta é recalculado a partir de μ e σ)
        reconstructed_delta = _dequantize_3level(
            symbols, 
            entry['mu'], 
            entry['sigma'], 
            entry['shape'],
            device,
            dtype
        )
        
        # Soma ao estado anterior para recuperar P_t
        if prev_val is not None:
            decoded[key] = prev_val.to(dtype=dtype) + reconstructed_delta
        else:
            decoded[key] = reconstructed_delta
    
    return decoded


def is_lhdq_encoded(state: dict) -> bool:
    """Verifica se um state_dict é um payload LHDQ codificado."""
    return isinstance(state, dict) and state.get('__lhdq__', False)
