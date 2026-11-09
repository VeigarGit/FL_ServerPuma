# Responsabilidade: Calcula a meta de latência e o timeout amortecido a partir das medições
# dos clientes.
#
# Principais funções: set_threshold.

"""Server latency."""

import logging

logger = logging.getLogger(__name__)

def set_threshold(state):
    tot_time = 0
    clients_with_time = 0
    max_client_time = 0.0
    for info in state.clients_info.values():
        t_train = info.get('training_time', 0.0) or 0.0
        t_eval = info.get('client_eval_time', 0.0) or 0.0
        t_proc = info.get('server_processing_time', 0.0) or 0.0
        t_comm = info.get('comm_time', 0.0) or 0.0
        total_client_time = t_train + t_eval + t_proc + t_comm

        if total_client_time > 0:
            tot_time += total_client_time
            clients_with_time += 1
            if total_client_time > max_client_time:
                max_client_time = total_client_time

    if clients_with_time == 0:
        logger.warning("No clients reported training time. Sticking to default threshold.")
        return

    mean_time = tot_time / clients_with_time
    state.mean_training_time = mean_time

    # Meta de latência adaptativa do cluster: média dos clientes + 15% de tolerância
    # Utilizada pelo PaCA e Rank adaptativos para identificar stragglers relativos
    state.target_latency = mean_time * 1.15

    # Pisos de segurança por tipo de modelo para evitar colapso de timeout em rodadas pontualmente rápidas
    if state.args.model == 'slm':
        min_floor = 1200.0
        multiplier = 2.5
    elif state.args.model == 'clip':
        min_floor = 300.0
        multiplier = 2.0
    else:
        min_floor = 60.0
        multiplier = 1.5

    calculated_threshold = max(min_floor, multiplier * mean_time, 1.5 * max_client_time)

    # Amortecimento: impede quedas bruscas de timeout entre rodadas
    if hasattr(state, 'time_threshold') and state.time_threshold is not None:
        state.time_threshold = max(calculated_threshold, state.time_threshold * 0.8, min_floor)
    else:
        state.time_threshold = calculated_threshold

    logger.info(f"Limiares definidos: timeout_socket={state.time_threshold:.2f}s | target_latency={state.target_latency:.2f}s (mean_time={mean_time:.2f}s)")
