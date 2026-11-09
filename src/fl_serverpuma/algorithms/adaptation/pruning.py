# Responsabilidade: Calcula a taxa de poda usando custo computacional, comunicação e histórico
# de tempo; inclui fallback.
#
# Principais funções: new_set_amount_prune, set_amount_prune.

"""Server pruning."""

import logging

logger = logging.getLogger(__name__)

def new_set_amount_prune(state, client_id):
    # Filosofia EAFP: Tenta acessar direto e falha limpo
    try:
        client_info = state.clients_info[client_id]
    except KeyError:
        logger.warning(f"Informações do cliente {client_id} ausentes. Usando pruning rate padrão.")
        return 0.5

    bandwidth = client_info.get('bandwidth', 1.0)
    training_time = client_info.get('training_time', 60.0)
    dataset_size = max(client_info.get('dataset_size', 1.0), 1.0)

    if not hasattr(state, 'alpha_list') or not hasattr(state, 'beta_list'):
        return _fallback_prune_calculation(state, client_id)

    total_alpha = sum(state.alpha_list)
    total_beta = sum(state.beta_list)

    client_info.setdefault('last_flops', total_beta)
    client_info.setdefault('last_params', total_alpha)

    target_latency = getattr(state, 'target_latency', getattr(state, 'time_threshold', 60.0))

    time_per_flop = (training_time / (dataset_size * total_beta)) if (training_time > 0 and total_beta > 0) else 1e-6
    bytes_per_param = 1
    bandwidth = bandwidth or 10.0  # Fallback se None
    bandwidth_bytes = bandwidth * 1024 * 1024

    computational_component = dataset_size * total_beta * time_per_flop
    communication_component = total_alpha * bytes_per_param / bandwidth_bytes
    total_factor = computational_component + communication_component

    if total_factor <= 0:
        return 0.5

    required_one_minus_p = target_latency / total_factor

    if required_one_minus_p >= 1.0:
        pruning_rate = 0.0
    elif required_one_minus_p <= 0.15:
        pruning_rate = 0.85
    else:
        pruning_rate = 1.0 - required_one_minus_p

    pruning_rate = max(0.2, min(0.85, pruning_rate))
    pruning_rate = _adjust_pruning_with_history(state, client_id, pruning_rate)

    logger.debug(f"Calculated pruning rate para {client_id}: {pruning_rate:.3f}")
    return pruning_rate


def _fallback_prune_calculation(state, client_id):
    try:
        client_info = state.clients_info[client_id]
    except KeyError:
        return 0.5

    bandwidth = client_info.get('bandwidth', 1.0)
    training_time = client_info.get('training_time', 60.0)

    if bandwidth < 0.5:
        bw_factor = 0.8
    elif bandwidth < 5.0:
        bw_factor = 0.6
    elif bandwidth < 20.0:
        bw_factor = 0.4
    else:
        bw_factor = 0.2

    if training_time < 30:
        time_factor = 0.2
    elif training_time < 60:
        time_factor = 0.4
    elif training_time < 120:
        time_factor = 0.6
    else:
        time_factor = 0.8

    pruning_rate = (bw_factor + time_factor) / 2
    return max(0.2, min(0.85, pruning_rate))


def _adjust_pruning_with_history(state, client_id, base_pruning_rate):
    try:
        client_info = state.clients_info[client_id]
    except KeyError:
        return base_pruning_rate

    training_time = client_info.get('training_time', 0)
    last_training_time = client_info.get('last_training_time', training_time)
    client_info['last_training_time'] = training_time

    if training_time > last_training_time * 1.5:
        adjustment = 0.05
    elif training_time > state.time_threshold * 1.2:
        adjustment = 0.03
    else:
        adjustment = 0.0
    return max(0.2, min(0.85, base_pruning_rate + adjustment))


def set_amount_prune(state):
    values = [v for v in state.client_data.values() if v is not None]
    if not values:
        return 0

    if len(values) == 1:
        data = values[0]
        non_null_times = [info['training_time'] for info in state.clients_info.values() if info.get('training_time')]
        return non_null_times[0] / data if non_null_times else 0.0
    else:
        sorted_values = sorted(values, reverse=True)
        maior_valor = sorted_values[0]
        penultimo_maior = sorted_values[1]

        distancia_normalizada = (1 - ((maior_valor - penultimo_maior) / maior_valor)) if maior_valor != 0 else 0
        return 0.85 if distancia_normalizada > 0.9 else distancia_normalizada
