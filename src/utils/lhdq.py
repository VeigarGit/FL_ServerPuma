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
    
    Returns:
        (packed_bytes: bytes, total_bits: int)
    """
    buf = bytearray()
    current_byte = 0
    bit_pos = 0  # quantos bits já foram escritos no byte atual (0-7)
    total_bits = 0
    
    for sym in symbols:
        code, n_bits = HUFFMAN_CODES[sym]
        total_bits += n_bits
        
        # Escreve os bits do código no buffer, MSB first
        for i in range(n_bits - 1, -1, -1):
            bit = (code >> i) & 1
            current_byte = (current_byte << 1) | bit
            bit_pos += 1
            
            if bit_pos == 8:
                buf.append(current_byte)
                current_byte = 0
                bit_pos = 0
    
    # Flush do último byte parcial (padding com zeros à direita)
    if bit_pos > 0:
        current_byte <<= (8 - bit_pos)
        buf.append(current_byte)
    
    return bytes(buf), total_bits


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
    Quantiza um tensor delta em 3 níveis baseados na distribuição normal.
    
    Divide a distribuição em 3 regiões de igual probabilidade:
    - Região inferior (prob 1/3): atribui representante -p_delta
    - Região central  (prob 1/3): atribui representante 0 (sem mudança relativa)
    - Região superior (prob 1/3): atribui representante +p_delta
    
    Os limites são: mu ± 0.4307 * sigma (inversa da CDF normal para 1/3 e 2/3).
    O representante p_delta é a média condicional da região extrema.
    
    Returns:
        (symbols: np.ndarray de {0,1,2}, mu: float, sigma: float, p_delta: float)
        onde 0→negativo, 1→zero, 2→positivo
    """
    delta_flat = delta.float().flatten()
    
    mu = delta_flat.mean().item()
    sigma = delta_flat.std().item()
    
    # Proteção contra sigma zero (todos os deltas iguais)
    if sigma < 1e-10:
        n = delta_flat.numel()
        symbols = np.ones(n, dtype=np.int8)  # tudo é "sem mudança"
        return symbols, mu, sigma, 0.0
    
    # Limites para dividir em 3 regiões equiprováveis
    # InvCDF(1/3) ≈ -0.4307, InvCDF(2/3) ≈ +0.4307
    TERCILE_Z = 0.4307
    lower_bound = mu - TERCILE_Z * sigma
    upper_bound = mu + TERCILE_Z * sigma
    
    # p_delta: representante da região extrema (média condicional)
    # Para distribuição normal, E[X | X > upper] ≈ mu + sigma * φ(z) / (1 - Φ(z))
    # onde z = TERCILE_Z, φ(z) ≈ 0.3637, 1-Φ(z) = 1/3
    # Simplificado: p_delta ≈ sigma * 0.3637 / (1/3) = sigma * 1.0911
    p_delta = sigma * 1.0911
    
    # Classificação vetorizada
    delta_np = delta_flat.cpu().numpy()
    symbols = np.ones(len(delta_np), dtype=np.int8)  # default: 0 (região central)
    symbols[delta_np < lower_bound] = 0   # região inferior → -1
    symbols[delta_np > upper_bound] = 2   # região superior → +1
    
    return symbols, mu, sigma, p_delta


def _dequantize_3level(symbols: np.ndarray, mu: float, sigma: float, 
                        p_delta: float, shape: tuple, device, dtype) -> torch.Tensor:
    """
    Reconstrói o tensor delta a partir dos símbolos quantizados.
    
    Mapeamento: 0 → -p_delta,  1 → 0,  2 → +p_delta
    (todos relativos à média mu)
    """
    # Mapeia símbolos para valores de delta
    delta_values = np.zeros(len(symbols), dtype=np.float32)
    delta_values[symbols == 0] = mu - p_delta   # região inferior
    delta_values[symbols == 1] = mu              # região central
    delta_values[symbols == 2] = mu + p_delta   # região superior
    
    return torch.from_numpy(delta_values).reshape(shape).to(device=device, dtype=dtype)


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
                    'mu': float,
                    'sigma': float,
                    'p_delta': float,
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
        symbols, mu, sigma, p_delta = _quantize_delta_3level(delta)
        
        # Empacota com Huffman
        packed_bytes, total_bits = _huffman_pack(symbols)
        
        # Estatísticas de compressão
        original_bytes = current_val.numel() * current_val.element_size()
        compressed_bytes = len(packed_bytes) + 12  # +12 bytes para mu/sigma/p_delta (3 floats)
        total_original_bytes += original_bytes
        total_compressed_bytes += compressed_bytes
        
        encoded['keys'][key] = {
            'packed': packed_bytes,
            'total_bits': total_bits,
            'n_elements': current_val.numel(),
            'shape': tuple(current_val.shape),
            'mu': mu,
            'sigma': sigma,
            'p_delta': p_delta,
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
        
        # Reconstrói o delta
        reconstructed_delta = _dequantize_3level(
            symbols, 
            entry['mu'], 
            entry['sigma'], 
            entry['p_delta'],
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
