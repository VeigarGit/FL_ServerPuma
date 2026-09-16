"""
ota_aggregation.py — Pipeline de Agregação Over-the-Air (Fase 2).

Aplica a degradação do canal OTA sobre os ``state_dict`` (tensores PyTorch)
dos modelos recebidos pelo aggregator.

Implementa os 4 passos do pipeline OTA:
  1. Normalização dos pesos pelo escalar √σ²_k de cada transmissor.
  2. Cálculo do beamforming de recepção (f') e escalares de transmissão (b_k).
  3. Superposição dos sinais no ar: soma ponderada pelo canal verdadeiro.
  4. Injeção do ruído AWGN processado pelo beamformer.

Todas as operações são realizadas sobre os tensores do modelo PyTorch,
usando as matrizes de canal geradas por ``ota_channel.py``.
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ── Passo 1: Variância dos Pesos ─────────────────────────────────────────────

def compute_weight_variance(
    state_dicts: dict[int, dict[str, torch.Tensor]],
) -> tuple[dict[int, float], int]:
    """Calcula a variância L2 normalizada (potência média) dos pesos de cada cliente.

    Equivalente ao ``GradNormSquared`` do código de referência.
    Usado como escalar de normalização no Passo 1 do pipeline OTA:
    cada peso é dividido por √σ²_k antes da transmissão.

    Args:
        state_dicts: Dicionário ``{client_id: state_dict}``.

    Returns:
        Tupla ``(variances, model_size)`` onde:
          - ``variances``: Dicionário ``{client_id: σ²_k}`` (float).
          - ``model_size``: Número total de parâmetros do modelo (int).
    """
    variances: dict[int, float] = {}
    model_size = 0

    for client_id, state in state_dicts.items():
        total_squared_sum = 0.0
        total_elements = 0
        for tensor in state.values():
            t = tensor.float()
            total_squared_sum += torch.sum(t ** 2).item()
            total_elements += t.numel()

        if total_elements > 0:
            variances[client_id] = total_squared_sum / total_elements
        else:
            variances[client_id] = 1e-9  # Prevenção de divisão por zero

        model_size = total_elements  # Todos os modelos têm o mesmo tamanho

        logger.debug(
            "Cliente %d: %d parâmetros, variância=%.4e",
            client_id, total_elements, variances[client_id],
        )

    return variances, model_size


# ── Passo 2: Beamforming MMSE ────────────────────────────────────────────────

def mmse_beamforming(
    H_est: np.ndarray,
    weight_vars: np.ndarray,
    n_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula o vetor de beamforming de recepção (f') e os escalares de transmissão (b_k).

    Usa SVD sobre a matriz de canal estimada para encontrar a direção
    espacial que maximiza o ganho conjunto (Maximum Ratio Combining).
    Os escalares de transmissão são calculados para que o sinal chegue
    com a escala esperada pela agregação ponderada por amostras.

    Fórmula dos escalares:
        b_k = (n_k · √σ²_k) / (f'ᴴ · h_k)

    Isso garante que, num canal perfeito (H_true = H_est), a superposição
    resulte exatamente na média ponderada por amostras.

    Args:
        H_est: Matriz de estimação de canal. Shape ``(n_clients, n_antennas)``.
        weight_vars: Variâncias dos pesos dos transmissores. Shape ``(n_clients,)``.
        n_samples: Número de amostras de cada transmissor. Shape ``(n_clients,)``.

    Returns:
        Tupla ``(f_prime, transmit_scalars)``:
          - ``f_prime``: Vetor de beamforming. Shape ``(n_antennas,)``. Complexo, norma 1.
          - ``transmit_scalars``: Escalares b_k. Shape ``(n_clients,)``. Complexos.
    """
    n_clients, n_antennas = H_est.shape

    # SVD para encontrar a direção principal do canal
    # O primeiro vetor singular à direita (associado ao maior valor singular)
    # maximiza o ganho conjunto — heurística MRC
    _, _, Vh = np.linalg.svd(H_est, full_matrices=False)
    f_prime = np.conjugate(Vh[0, :])  # Shape: (n_antennas,)
    f_prime = f_prime / np.linalg.norm(f_prime)  # Normalizar

    # Ganho efetivo do canal estimado após beamforming: f'ᴴ · h_k
    fh = H_est @ np.conjugate(f_prime)  # Shape: (n_clients,)

    # Escalares de transmissão: b_k = (n_k · √σ²_k) / (f'ᴴ · h_k)
    # Evitar divisão por zero em canais extremamente atenuados
    fh_safe = np.where(np.abs(fh) < 1e-12, 1e-12, fh)
    transmit_scalars = (n_samples * np.sqrt(weight_vars)) / fh_safe

    logger.debug(
        "Beamforming: ||f'||=%.4f, |b_k|=[%s]",
        np.linalg.norm(f_prime),
        ", ".join(f"{np.abs(b):.4f}" for b in transmit_scalars),
    )

    return f_prime, transmit_scalars


# ── Passos 3-4: Superposição OTA + Ruído ─────────────────────────────────────

def apply_ota_channel(
    state_dicts: dict[int, dict[str, torch.Tensor]],
    transmitter_ids: list[int],
    n_samples: np.ndarray,
    H_true: np.ndarray,
    transmit_scalars: np.ndarray,
    f_prime: np.ndarray,
    noise_variance: float,
    weight_variances: np.ndarray,
) -> dict[str, torch.Tensor]:
    """Simula a transmissão Over-the-Air e retorna o modelo agregado degradado.

    Executa os 4 passos do pipeline OTA para cada tensor do modelo:

    **Passo 1 — Normalização**: Cada peso w_k[j] é dividido por √σ²_k.
    **Passo 2 — Pré-codificação**: Multiplicado pelo escalar b_k.
    **Passo 3 — Canal + Superposição**: Multiplicado pelo ganho real do
        canal verdadeiro (f'ᴴ · h_true_k) e somado para todos os clientes.
    **Passo 4 — Ruído**: Ruído AWGN efetivo é adicionado ao sinal agregado.

    O fator combinado aplicado a cada peso w_k[j] é:
        gain_k = Re(b_k · (f'ᴴ · h_true_k)) / √σ²_k

    E o resultado final é:
        y[j] = (1/K) · Σ_k gain_k · w_k[j]  +  noise_eff[j]

    Onde K = Σ n_k (total de amostras).

    Args:
        state_dicts: Dicionário ``{client_id: state_dict}`` dos **transmissores**
                     (exclui o aggregator). Cada state_dict é um dict de tensores.
        transmitter_ids: Lista ordenada dos IDs dos transmissores,
                         correspondendo às linhas de H_true.
        n_samples: Número de amostras dos transmissores. Shape ``(n_tx,)``.
        H_true: Canal verdadeiro. Shape ``(n_tx, n_antennas)``.
        transmit_scalars: Escalares b_k. Shape ``(n_tx,)``. Complexos.
        f_prime: Vetor de beamforming. Shape ``(n_antennas,)``. Complexo.
        noise_variance: Variância do ruído térmico (escala linear).
        weight_variances: Variâncias σ²_k dos transmissores. Shape ``(n_tx,)``.

    Returns:
        ``aggregated_state``: state_dict com a mesma estrutura dos originais,
        contendo a média OTA degradada pelo canal e ruído.
    """
    n_tx = len(transmitter_ids)
    K = float(np.sum(n_samples))
    device = next(iter(state_dicts[transmitter_ids[0]].values())).device

    # Ganho efetivo real: Re(b_k · (f'ᴴ · h_true_k)) / √σ²_k
    # f'ᴴ · h_true_k = H_true @ conj(f_prime)
    fh_true = H_true @ np.conjugate(f_prime)  # Shape: (n_tx,)
    effective_gains = np.real(transmit_scalars * fh_true) / np.sqrt(weight_variances)

    logger.debug(
        "Ganhos efetivos: [%s]",
        ", ".join(f"{g:.4f}" for g in effective_gains),
    )

    # Preparar o state_dict agregado vazio
    ref_client = transmitter_ids[0]
    aggregated_state: dict[str, torch.Tensor] = {}
    for key in state_dicts[ref_client]:
        aggregated_state[key] = torch.zeros_like(
            state_dicts[ref_client][key], dtype=torch.float32, device=device,
        )

    # ── Passo 3: Superposição dos sinais ──────────────────────────────────
    for i, tx_id in enumerate(transmitter_ids):
        gain = effective_gains[i]
        for key in aggregated_state:
            t = state_dicts[tx_id][key].float()
            aggregated_state[key] += (gain * t) / K

    # ── Passo 4: Ruído do receptor ────────────────────────────────────────
    # O ruído bruto (model_size, n_antennas) é projetado pelo beamformer:
    #   n_eff = n_bruto @ conj(f')
    # A variância efetiva escalada por K² é: noise_var · ||f'||² / K²
    f_norm_sq = float(np.linalg.norm(f_prime) ** 2)
    effective_noise_std = np.sqrt(noise_variance * f_norm_sq) / K

    if effective_noise_std > 1e-20:
        for key in aggregated_state:
            noise = torch.randn_like(
                aggregated_state[key], dtype=torch.float32, device=device,
            ) * effective_noise_std
            aggregated_state[key] += noise

        logger.debug(
            "Ruído efetivo injetado: std=%.4e (noise_var=%.4e, ||f'||²=%.4f, K=%.0f)",
            effective_noise_std, noise_variance, f_norm_sq, K,
        )
    else:
        logger.debug("Ruído efetivo desprezível (std=%.4e), não injetado.", effective_noise_std)

    # Restaurar dtype original
    for key in aggregated_state:
        orig_dtype = state_dicts[ref_client][key].dtype
        if aggregated_state[key].dtype != orig_dtype:
            aggregated_state[key] = aggregated_state[key].to(orig_dtype)

    logger.info(
        "Agregação OTA concluída: %d transmissores, K=%d amostras, "
        "noise_std=%.4e",
        n_tx, int(K), effective_noise_std,
    )

    return aggregated_state
