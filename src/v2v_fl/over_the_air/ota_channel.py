"""
ota_channel.py — Simulação do Canal Sem Fio para Over-the-Air FL (V2V).

Modela a propagação eletromagnética entre veículos transmissores e o
veículo aggregator (receptor com múltiplas antenas) no contexto V2V.

Baseado no modelo de canal de:
  "Power-Efficient Over-the-Air Aggregation with Receive Beamforming
   for Federated Learning"

Adaptações para o FL_ServerPuma:
  - Distâncias calculadas a partir de posições reais do SUMO (TraCI).
  - Aggregator eleito por proximidade (medoid geométrico).
  - Sem lógica de eficiência energética.
  - Type hints, docstrings e logging (Python 3.11).
"""

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


def compute_path_loss(distances: np.ndarray) -> np.ndarray:
    """Calcula o path loss (perda de percurso) para cada distância.

    Parâmetros do modelo (fixos, idênticos ao código original):
      - Frequência portadora: 2000 MHz
      - Altura da estação base (BS): 30 m
      - Altura do móvel (MS): 1 m (veículo)

    Args:
        distances: Array de distâncias em metros entre cada transmissor
                   e o aggregator. Shape ``(n_clients,)``.
                   Valores devem ser > 0.

    Returns:
        Array de path loss em escala **linear** (não dB). Shape ``(n_clients,)``.
        Valores maiores = mais atenuação = sinal mais fraco.

    Raises:
        ValueError: Se alguma distância for <= 0.

    Example:
        >>> distances = np.array([50.0, 100.0, 200.0])
        >>> pl = compute_path_loss(distances)
        >>> assert pl[0] < pl[1] < pl[2]  # path loss cresce com distância
    """
    if np.any(distances <= 0):
        raise ValueError(
            f"Todas as distâncias devem ser > 0. Recebido min={distances.min():.2f}"
        )

    # Parâmetros do modelo COST-231 Hata (idênticos ao código original)
    freq_mhz = 2000.0       # Frequência portadora (MHz)
    h_bs = 30.0              # Altura da estação base (m)
    h_ms = 1.0               # Altura do móvel (m)

    # Fator de correção para cidade grande (modelo original)
    c1 = 3.2 * (np.log10(11.75 * h_ms) ** 2) - 4.97

    # Componente dependente da distância
    # distances/1e3 converte metros → km (unidade do modelo Hata)
    c2 = (44.9 - 6.55 * np.log10(h_bs)) * np.log10(distances / 1e3)

    # Path loss em dB
    path_loss_db = (
        46.3
        + 33.9 * np.log10(freq_mhz)
        - 13.82 * np.log10(h_bs)
        - c1
        + c2
    )

    # Converter dB → escala linear
    path_loss_linear = np.power(10.0, path_loss_db / 10.0)

    logger.debug(
        "Path loss calculado para %d clientes: min=%.1f dB, max=%.1f dB",
        len(distances), path_loss_db.min(), path_loss_db.max(),
    )

    return path_loss_linear


# ── Geração de Canal (Rayleigh Fading) ───────────────────────────────────────

def generate_channel_estimation(
    n_clients: int,
    n_antennas: int,
    path_loss: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Gera a matriz de estimação de canal H̃ (Rayleigh fading complexo).

    Cada coeficiente h̃_{k,n} segue uma distribuição Gaussiana complexa
    com variância (1/PL_k) * (1 - ε), onde PL_k é o path loss do
    cliente k e ε é o nível de erro de estimação.

    Quando ε = 0, a estimação é perfeita (H̃ = H_true).

    Args:
        n_clients: Número de clientes transmissores.
        n_antennas: Número de antenas no aggregator receptor.
        path_loss: Path loss linear de cada cliente. Shape ``(n_clients,)``.
        epsilon: Nível de erro de estimação do canal (0 ≤ ε ≤ 1).
                 0 = estimação perfeita, 1 = nenhuma informação.

    Returns:
        Matriz complexa de estimação de canal. Shape ``(n_clients, n_antennas)``.
    """
    H_est = np.zeros((n_clients, n_antennas), dtype=np.complex128)

    for k in range(n_clients):
        variance = (1.0 / path_loss[k]) * (1.0 - epsilon)
        # Distribuição Gaussiana complexa simétrica: Re ~ N(0, σ²/2), Im ~ N(0, σ²/2)
        H_est[k] = np.sqrt(variance) * (
            np.random.normal(0, 1.0 / np.sqrt(2), n_antennas)
            + 1j * np.random.normal(0, 1.0 / np.sqrt(2), n_antennas)
        )

    return H_est


def generate_channel_error(
    n_clients: int,
    n_antennas: int,
    path_loss: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Gera a matriz de erro de canal ΔH.

    Cada coeficiente Δh_{k,n} segue uma distribuição Gaussiana complexa
    com variância (1/PL_k) * ε. Quando ε = 0, o erro é zero.

    O canal verdadeiro é: H_true = H_est - ΔH

    Args:
        n_clients: Número de clientes transmissores.
        n_antennas: Número de antenas no aggregator receptor.
        path_loss: Path loss linear de cada cliente. Shape ``(n_clients,)``.
        epsilon: Nível de erro de estimação do canal (0 ≤ ε ≤ 1).

    Returns:
        Matriz complexa de erro de canal. Shape ``(n_clients, n_antennas)``.
    """
    H_err = np.zeros((n_clients, n_antennas), dtype=np.complex128)

    for k in range(n_clients):
        variance = (1.0 / path_loss[k]) * epsilon
        H_err[k] = np.sqrt(variance) * (
            np.random.normal(0, 1.0 / np.sqrt(2), n_antennas)
            + 1j * np.random.normal(0, 1.0 / np.sqrt(2), n_antennas)
        )

    return H_err


def build_channel_matrices(
    distances: np.ndarray,
    n_antennas: int,
    clients_num_samples: np.ndarray,
    epsilon: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Função principal: constrói todas as matrizes de canal para um encontro.

    Orquestra ``compute_path_loss``, ``generate_channel_estimation`` e
    ``generate_channel_error`` para produzir as 4 matrizes necessárias
    para a agregação OTA.

    Args:
        distances: Distâncias em metros de cada transmissor ao aggregator.
                   Shape ``(n_clients,)``.
        n_antennas: Número de antenas no aggregator receptor.
        clients_num_samples: Número de amostras de treino de cada cliente.
                             Shape ``(n_clients,)``. Usado para escalar
                             a matriz de canal.
        epsilon: Nível de erro de estimação do canal (0 ≤ ε ≤ 1).
                 Default: 0.1.

    Returns:
        Tupla ``(H_est, H_true, H_scaled, path_loss)`` onde:
          - ``H_est``: Matriz de estimação de canal. Shape ``(n_clients, n_antennas)``.
          - ``H_true``: Canal verdadeiro = H_est - H_error. Shape ``(n_clients, n_antennas)``.
          - ``H_scaled``: H_est escalada pelo nº de amostras (transposta).
                          Shape ``(n_antennas, n_clients)``.
          - ``path_loss``: Path loss linear. Shape ``(n_clients,)``.
    """
    n_clients = len(distances)

    # 1. Path loss a partir das distâncias reais
    path_loss = compute_path_loss(distances)

    # 2. Estimação de canal e erro
    H_est = generate_channel_estimation(n_clients, n_antennas, path_loss, epsilon)
    H_err = generate_channel_error(n_clients, n_antennas, path_loss, epsilon)

    # 3. Canal verdadeiro (o que realmente afeta o sinal)
    H_true = H_est - H_err

    # 4. Canal escalado pelo número de amostras de cada cliente
    # Fator 1e7 do código original para estabilidade numérica
    H_scaled = np.zeros_like(H_est)
    for k in range(n_clients):
        H_scaled[k] = 1e7 * H_est[k] / clients_num_samples[k]
    H_scaled = H_scaled.T  # Shape: (n_antennas, n_clients)

    logger.info(
        "Canal construído: %d transmissores × %d antenas, ε=%.2f, "
        "distâncias=[%s] m",
        n_clients, n_antennas, epsilon,
        ", ".join(f"{d:.1f}" for d in distances),
    )

    return H_est, H_true, H_scaled, path_loss


# ── Ruído do Receptor ────────────────────────────────────────────────────────

def compute_noise_variance(bandwidth_hz: float = 100e3) -> float:
    """Calcula a variância do ruído térmico a partir da largura de banda.

    Usa o modelo de ruído térmico do código original:
      noise_variance_dB = 10 - 174 + 40 + 10·log10(bandwidth) - 30

    Onde:
      - -174 dBm/Hz: densidade espectral de potência do ruído térmico (kT)
      - +40 dB: figura de ruído do receptor
      - +10·log10(B): efeito da largura de banda

    Args:
        bandwidth_hz: Largura de banda do canal em Hz. Default: 100 kHz.

    Returns:
        Variância do ruído em escala linear (Watts).
    """
    noise_var_db = 10 - 174 + 40 + 10 * np.log10(bandwidth_hz) - 30
    noise_var_linear = np.power(10.0, noise_var_db / 10.0)

    logger.debug(
        "Variância do ruído: %.2f dB = %.2e linear (BW=%.0f Hz)",
        noise_var_db, noise_var_linear, bandwidth_hz,
    )

    return float(noise_var_linear)


def generate_receiver_noise(
    model_size: int,
    n_antennas: int,
    noise_variance: float,
) -> np.ndarray:
    """Gera a matriz de ruído AWGN complexo no receptor multi-antena.

    Cada elemento segue uma distribuição Gaussiana complexa simétrica
    com variância ``noise_variance``.

    Args:
        model_size: Número total de parâmetros do modelo (dimensão do
                    vetor de sinal).
        n_antennas: Número de antenas no aggregator receptor.
        noise_variance: Variância do ruído térmico (escala linear).

    Returns:
        Matriz de ruído complexo. Shape ``(model_size, n_antennas)``.
    """
    std = np.sqrt(noise_variance) / np.sqrt(2)

    noise = (
        np.random.normal(0, std, (model_size, n_antennas))
        + 1j * np.random.normal(0, std, (model_size, n_antennas))
    )

    return noise


# ── Seleção do Aggregator (Medoid Geométrico) ───────────────────────────────

def find_medoid(positions: dict[int, tuple[float, float]]) -> int:
    """Encontra o cliente mais próximo de todos os outros (medoid geométrico).

    O medoid é o ponto do conjunto que minimiza a soma das distâncias
    euclidianas para todos os outros pontos. Diferente do centroide
    (que pode não coincidir com nenhum ponto), o medoid é sempre um
    dos clientes reais.

    Args:
        positions: Dicionário ``{client_id: (x, y)}`` com as posições
                   de cada cliente no cluster (coordenadas SUMO em metros).

    Returns:
        ``client_id`` do medoid (o aggregator eleito).

    Raises:
        ValueError: Se ``positions`` tiver menos de 2 clientes.

    Example:
        >>> positions = {0: (0.0, 0.0), 1: (10.0, 0.0), 2: (5.0, 0.0)}
        >>> find_medoid(positions)  # Cliente 2 está no meio
        2
    """
    if len(positions) < 2:
        raise ValueError(
            f"Medoid requer pelo menos 2 clientes. Recebido: {len(positions)}"
        )

    client_ids = list(positions.keys())
    coords = np.array([positions[cid] for cid in client_ids])  # Shape: (n, 2)

    # Calcular a soma das distâncias de cada cliente para todos os outros
    min_total_dist = float('inf')
    medoid_id = client_ids[0]

    for i, cid in enumerate(client_ids):
        # Distância euclidiana de cid para todos os outros
        diffs = coords - coords[i]  # Broadcasting: (n, 2)
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))  # (n,)
        total_dist = np.sum(dists)

        if total_dist < min_total_dist:
            min_total_dist = total_dist
            medoid_id = cid

    logger.debug(
        "Medoid: cliente %d (soma dist=%.1f m) dentre %d candidatos",
        medoid_id, min_total_dist, len(positions),
    )

    return medoid_id


# ── Utilitário: Distâncias a partir de Posições ─────────────────────────────

def distances_from_positions(
    positions: dict[int, tuple[float, float]],
    aggregator_id: int,
) -> tuple[list[int], np.ndarray]:
    """Calcula as distâncias de cada transmissor ao aggregator.

    Args:
        positions: Dicionário ``{client_id: (x, y)}`` com posições SUMO.
        aggregator_id: ID do cliente aggregator (receptor).

    Returns:
        Tupla ``(transmitter_ids, distances)`` onde:
          - ``transmitter_ids``: Lista ordenada dos IDs dos transmissores
            (todos exceto o aggregator).
          - ``distances``: Array de distâncias em metros de cada
            transmissor ao aggregator. Shape ``(n_transmitters,)``.

    Raises:
        ValueError: Se ``aggregator_id`` não estiver em ``positions``.
    """
    if aggregator_id not in positions:
        raise ValueError(
            f"Aggregator {aggregator_id} não encontrado nas posições: "
            f"{list(positions.keys())}"
        )

    agg_pos = np.array(positions[aggregator_id])
    transmitter_ids = sorted(cid for cid in positions if cid != aggregator_id)

    distances = np.array([
        math.sqrt(
            (positions[cid][0] - agg_pos[0]) ** 2
            + (positions[cid][1] - agg_pos[1]) ** 2
        )
        for cid in transmitter_ids
    ])

    # Garantir distância mínima de 1m para evitar path loss zero
    distances = np.maximum(distances, 1.0)

    logger.debug(
        "Distâncias ao aggregator %d: %s",
        aggregator_id,
        {cid: f"{d:.1f}m" for cid, d in zip(transmitter_ids, distances)},
    )

    return transmitter_ids, distances
