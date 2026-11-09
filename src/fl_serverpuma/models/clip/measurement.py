# Responsabilidade: Executa benchmark de inferência e quantiza pesos para exportação. Família
# CLIP.
#
# Principais funções: benchmark_attention, quantize_weights.

"""CLIP model support — measurement."""

import torch

@torch.no_grad()
def benchmark_attention(model, loader, device, num_batches=10):
    """
    Benchmarking de Infraestrutura.
    Mede a latência por batch e o consumo de VRAM, validando o impacto do SDPA na performance.
    """
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for i, (pixel_values, _) in enumerate(loader):
        if i >= num_batches: break
        _ = model(pixel_values=pixel_values.to(device))
    end.record()

    torch.cuda.synchronize()
    time_ms = start.elapsed_time(end)
    vram_mb = torch.cuda.max_memory_allocated(device) / 1024**2

    print(f"\nBENCHMARK SDPA: {time_ms/num_batches:.1f}ms/batch | VRAM: {vram_mb:.1f}MB\n")


def quantize_weights(state_dict):
    """
    Otimização de Armazenamento.
    Converte tensores de ponto flutuante para INT8 linear, reduzindo o tamanho do 
    arquivo final em ~4x com mínima perda de precisão.
    """
    quantized_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            scale = torch.max(torch.abs(v)) / 127.0
            q_weight = torch.clamp((v / scale).round(), -128, 127).to(torch.int8)
            quantized_dict[k] = {'dtype': 'int8', 'scale': scale, 'weights': q_weight}
        else:
            quantized_dict[k] = v
    return quantized_dict
