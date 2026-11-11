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
import math

import numpy as np
import torch

from utils.network_utils import validate_train_samples
from v2v_fl.over_the_air.ota_channel import build_channel_matrices, compute_noise_variance

logger = logging.getLogger(__name__)

DEFAULT_OTA_EPSILON = 0.1
DEFAULT_OTA_BANDWIDTH_HZ = 100e3


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
        n_samples: Pesos nominais positivos (q_k no pipeline normalizado).
                   Shape ``(n_clients,)``.

    Returns:
        Tupla ``(f_prime, transmit_scalars)``:
          - ``f_prime``: Vetor de beamforming. Shape ``(n_antennas,)``. Complexo, norma 1.
          - ``transmit_scalars``: Escalares b_k. Shape ``(n_clients,)``. Complexos.
    """
    n_clients, n_antennas = H_est.shape

    # SVD para encontrar a direção principal do canal
    # Vh[0, :] já é o conjugado do primeiro vetor singular à direita.
    # Como a recepção usa H @ conj(f'), esta escolha maximiza o ganho conjunto.
    _, _, Vh = np.linalg.svd(H_est, full_matrices=False)
    f_prime = Vh[0, :]  # Shape: (n_antennas,)
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
    *,
    diagnostics: dict | None = None,
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

    Onde K = Σ n_k (soma dos pesos nominais dos transmissores).

    Args:
        state_dicts: Dicionário ``{client_id: state_dict}`` dos **transmissores**
                     (exclui o aggregator). Cada state_dict é um dict de tensores.
        transmitter_ids: Lista ordenada dos IDs dos transmissores,
                         correspondendo às linhas de H_true.
        n_samples: Pesos nominais dos transmissores (q_k). Shape ``(n_tx,)``.
        H_true: Canal verdadeiro. Shape ``(n_tx, n_antennas)``.
        transmit_scalars: Escalares b_k. Shape ``(n_tx,)``. Complexos.
        f_prime: Vetor de beamforming. Shape ``(n_antennas,)``. Complexo.
        noise_variance: Variância do ruído térmico (escala linear).
        weight_variances: Variâncias σ²_k dos transmissores. Shape ``(n_tx,)``.
        diagnostics: Dicionário opcional para ganhos efetivos e escala de ruído.

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

    if diagnostics is not None:
        diagnostics.update(
            ota_weight_sum=K,
            ota_noise_std=float(effective_noise_std),
            transmitters=[
                {
                    "client_id": tx_id,
                    "true_gain_real": float(fh_true[i].real),
                    "true_gain_imag": float(fh_true[i].imag),
                    "true_gain_abs": float(abs(fh_true[i])),
                    "effective_gain": float(effective_gains[i]),
                    "gain_ratio": float(effective_gains[i] / n_samples[i]),
                }
                for i, tx_id in enumerate(transmitter_ids)
            ],
        )

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
        "Agregação OTA concluída: %d transmissores, K=%.6g pesos nominais, "
        "noise_std=%.4e",
        n_tx, K, effective_noise_std,
    )

    return aggregated_state


def ota_aggregate(
    client_states: dict[int, dict],
    positions: dict[int, tuple[float, float]],
    rsu: dict,
    *,
    train_samples: dict[int, int],
    local_client_id: int | None = None,
    diagnostics: dict | None = None,
    epsilon: float = DEFAULT_OTA_EPSILON,
    bandwidth_hz: float = DEFAULT_OTA_BANDWIDTH_HZ,
) -> dict:
    """FedAvg OTA por amostras de treino, com contribuição local opcional no V2V.

    M é o total de veículos, incluindo o agregador V2V. Usa q_i = M*n_i/sum(n)
    na pré-codificação. apply_ota_channel divide pela soma dos q dos transmissores;
    ao multiplicar sua saída por sum(q_tx)/M e adicionar q_local*modelo_local/M,
    obtém-se sum(n_i*modelo_i)/sum(n) no canal perfeito. O desvio padrão do ruído
    final é sqrt(noise_variance)*||f'||/M, independente das contagens de treino.
    Com contagens iguais, reproduz a escala anterior nos dois modos.

    No V2I todos transmitem e a RSU não possui modelo. No V2V local_client_id
    pertence a client_states, mas não transmite. rsu descreve o receptor em ambos.
    diagnostics recebe somente números/listas: compara o resultado entregue com
    a média ideal, por tensor, sem substituir a saída OTA nem consumir RNG.
    """
    if not client_states:
        raise ValueError("Agregação OTA requer pelo menos um veículo transmissor.")
    if set(train_samples) != set(client_states):
        raise ValueError("train_samples deve conter exatamente os participantes do encontro.")
    for count in train_samples.values():
        validate_train_samples(count)
    participant_ids = sorted(client_states)
    counts = np.array([train_samples[c] for c in participant_ids], dtype=np.float64)
    if not np.isfinite(counts).all() or not np.isfinite(counts.sum()):
        raise ValueError("Contagens de treino fora do intervalo numérico suportado.")
    M = len(participant_ids)
    coefficients = counts / counts.sum()
    q = dict(zip(participant_ids, M * coefficients, strict=True))
    if local_client_id is not None and local_client_id not in client_states:
        raise ValueError("O agregador local deve participar do encontro.")
    transmitter_ids = [c for c in participant_ids if c != local_client_id]
    if not transmitter_ids:
        raise ValueError("Agregação OTA requer pelo menos um veículo transmissor.")
    reference = client_states[participant_ids[0]]
    for state in client_states.values():
        if state.keys() != reference.keys() or any(
            state[k].shape != reference[k].shape or state[k].dtype != reference[k].dtype
            for k in reference
        ):
            raise ValueError("Modelos incompatíveis no encontro OTA.")
    n_antennas = rsu.get("n_antennas", 16)
    if type(n_antennas) is not int or n_antennas < 1:
        raise ValueError("A RSU deve ter um número inteiro positivo de antenas.")
    if not 0 <= epsilon < 1:
        raise ValueError("O erro de estimação OTA deve estar no intervalo [0, 1).")
    if not math.isfinite(bandwidth_hz) or bandwidth_hz <= 0:
        raise ValueError("A largura de banda OTA deve ser positiva e finita.")

    distances = np.array([
        math.hypot(positions[c][0] - rsu["x"], positions[c][1] - rsu["y"])
        for c in transmitter_ids
    ])
    # Mesma distância mínima do V2V para evitar singularidade no path loss.
    distances = np.maximum(distances, 1.0)
    n_samples = np.array([q[c] for c in transmitter_ids])
    H_est, H_true, _, _ = build_channel_matrices(
        distances, n_antennas, n_samples, epsilon=epsilon,
    )
    tx_states = {c: client_states[c] for c in transmitter_ids}
    variances, _ = compute_weight_variance(tx_states)
    weight_vars = np.maximum([variances[c] for c in transmitter_ids], 1e-12)
    f_prime, transmit_scalars = mmse_beamforming(H_est, weight_vars, n_samples)

    noise_variance = compute_noise_variance(bandwidth_hz)
    result = apply_ota_channel(
        state_dicts=tx_states,
        transmitter_ids=transmitter_ids,
        n_samples=n_samples,
        H_true=H_true,
        transmit_scalars=transmit_scalars,
        f_prime=f_prime,
        noise_variance=noise_variance,
        weight_variances=weight_vars,
        diagnostics=diagnostics,
    )
    tx_fraction = float(n_samples.sum() / M)
    if local_client_id is not None:
        local = client_states[local_client_id]
        for key in result:
            result[key] = (
                (q[local_client_id] / M) * local[key].float()
                + tx_fraction * result[key].float()
            ).to(local[key].dtype)

    if diagnostics is not None:
        # Acumular normas em float64 por tensor, sem reter um segundo modelo completo.
        error_squared = reference_squared = 0.0
        for key in reference:
            ideal = torch.zeros_like(reference[key], dtype=torch.float64)
            for i, client_id in enumerate(participant_ids):
                ideal.add_(client_states[client_id][key].double(), alpha=float(coefficients[i]))
            error_squared += torch.sum((result[key].double() - ideal) ** 2).item()
            reference_squared += torch.sum(ideal ** 2).item()
        error_norm = math.sqrt(error_squared)
        reference_norm = math.sqrt(reference_squared)
        fh_est = H_est @ np.conjugate(f_prime)
        for i, item in enumerate(diagnostics["transmitters"]):
            item.update(
                estimated_gain_real=float(fh_est[i].real),
                estimated_gain_imag=float(fh_est[i].imag),
                estimated_gain_abs=float(abs(fh_est[i])),
                model_weight=float(q[item["client_id"]] / M),
                effective_model_weight=float(item["effective_gain"] / M),
            )
        diagnostics.update(
            schema_version=1,
            aggregation="train_samples",
            train_samples={str(c): train_samples[c] for c in participant_ids},
            model_weights={str(c): float(q[c] / M) for c in participant_ids},
            ota_weights={str(c): float(q[c]) for c in participant_ids},
            participant_count=M,
            local_client_id=local_client_id,
            n_antennas=n_antennas,
            epsilon=float(epsilon),
            bandwidth_hz=float(bandwidth_hz),
            noise_variance=float(noise_variance),
            noise_std=float(diagnostics["ota_noise_std"] * tx_fraction),
            reference_l2_norm=reference_norm,
            error_l2_norm=error_norm,
            relative_l2_error=error_norm / reference_norm if reference_norm > 0 else None,
        )
        logger.info(
            "Diagnóstico OTA: amostras=%s, pesos=%s, erro relativo L2=%s, noise_std=%.4e",
            diagnostics["train_samples"], diagnostics["model_weights"],
            diagnostics["relative_l2_error"], diagnostics["noise_std"],
        )
    return result
