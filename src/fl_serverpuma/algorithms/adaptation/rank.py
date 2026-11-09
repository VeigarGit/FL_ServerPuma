# Responsabilidade: Decide rank adaptativo por latência, perda e acurácia, preservando limites
# e histórico.
#
# Principais funções: calculate_adaptive_rank.

"""Server rank."""

import logging

logger = logging.getLogger(__name__)

def calculate_adaptive_rank(state, client_id):
    try:
        client_info = state.clients_info[client_id]
    except KeyError:
        return state.args.rank

    bandwidth = client_info.get('bandwidth')
    t_train = client_info.get('training_time', 0.0) or 0.0
    t_eval = client_info.get('client_eval_time', 0.0) or 0.0
    t_proc = client_info.get('server_processing_time', 0.0) or 0.0
    t_comm = client_info.get('comm_time', 0.0) or 0.0
    total_time = t_train + t_eval + t_proc + t_comm

    current_rank = client_info.get('current_rank', state.args.rank)
    max_rank = getattr(state.args, 'adaptive_rank_max', 8)
    min_rank = getattr(state.args, 'adaptive_rank_min', 2)

    # Sem dados suficientes, mantém o atual
    if bandwidth is None or t_train == 0.0:
        return current_rank

    target_latency = getattr(state, 'target_latency', getattr(state, 'time_threshold', 60.0))
    mean_time = getattr(state, 'mean_training_time', target_latency / 1.15)

    # --- 1. Sinal da Acurácia Global ---
    last_reduction_round = client_info.get('last_rank_reduction_round', None)
    current_round = len(state.rs_test_acc)
    cooldown_rounds = 5
    in_cooldown = (last_reduction_round is not None and (current_round - last_reduction_round) < cooldown_rounds)

    acc_signal = _check_accuracy_signal_rank(state, client_id, current_rank, min_rank, max_rank, in_cooldown)

    # Acurácia despencou (Pânico! Reversão tem prioridade)
    if acc_signal is not None and acc_signal > current_rank:
        client_info['last_rank_reduction_round'] = current_round
        return acc_signal

    # --- 2. Sinal de Loss Local (Dificuldade do cliente) ---
    global_loss = state.rs_test_loss[-1] if state.rs_test_loss else 1.0
    client_loss = client_info.get('last_loss', 0.0)

    is_difficult_client = (client_loss > 0.0 and global_loss > 0.0 and client_loss > global_loss * 1.2)

    if is_difficult_client:
        if current_rank < max_rank:
            if getattr(state.args, 'allow_rank_upscale', False) or total_time <= target_latency:
                logger.info(f"Rank adaptativo {client_id}: cliente difícil (loss={client_loss:.4f} > global={global_loss:.4f}*1.2). Forçando subida {current_rank} -> {current_rank + 1}")
                return current_rank + 1
            else:
                logger.info(f"Rank adaptativo {client_id}: cliente difícil, mas sem margem de tempo ({total_time:.1f}s > {target_latency:.1f}s) e upscale bloqueado")
        return current_rank

    # --- 3. Acurácia Estável (Platô) ---
    if acc_signal is not None and acc_signal < current_rank:
        client_info['last_rank_reduction_round'] = current_round
        return acc_signal

    # --- 4. Sinal de Latência Relativa ---
    # Cliente está rápido (>20% mais rápido que a média do cluster): pode subir o rank
    if total_time < mean_time * 0.80 and current_rank < max_rank:
        if getattr(state.args, 'allow_rank_upscale', False):
            new_rank = min(current_rank + 1, max_rank)
            logger.info(f"Rank adaptativo {client_id}: subindo {current_rank} -> {new_rank} "
                        f"(tempo_total={total_time:.1f}s < 80% da média={mean_time:.1f}s)")
            return new_rank
        else:
            logger.info(f"Rank adaptativo {client_id}: subida bloqueada por padrão (use --allow-rank-upscale para permitir) (tempo_total={total_time:.1f}s < 80%)")

    # Se estiver dentro da tolerância normal, mantém
    if total_time <= target_latency:
        return current_rank

    # Cliente está lento (> target_latency): reduz o rank em 1
    if current_rank > min_rank:
        new_rank = current_rank - 1
        logger.info(f"Rank adaptativo {client_id}: reduzindo {current_rank} -> {new_rank} "
                    f"(tempo_total={total_time:.1f}s > target={target_latency:.1f}s, média={mean_time:.1f}s)")
        client_info['last_rank_reduction_round'] = current_round
        return new_rank

    return current_rank


def _check_accuracy_signal_rank(state, client_id, current_rank, rank_min, max_rank, in_cooldown=False):
    """
    Verifica se a acurácia indica que o Rank pode ser ajustado.
    - Fallback mechanism: Revert PaCA layer count if accuracy drops by more than 2% after a reduction.
    - Plateau: APENAS fora do cooldown. Se estabilizou, reduz -1.
    """
    acc_history = state.rs_test_acc
    window = 5

    if len(acc_history) < window:
        return None

    recent = acc_history[-window:]
    mean_acc = sum(recent) / len(recent)
    std_acc = (sum((x - mean_acc) ** 2 for x in recent) / len(recent)) ** 0.5

    rank_before_last_change = state.clients_info[client_id].get('rank_before_change')
    if rank_before_last_change is not None and rank_before_last_change > current_rank:
        acc_before_window = acc_history[-(window + 1):-1] if len(acc_history) > window else acc_history[:window]
        mean_before = sum(acc_before_window) / len(acc_before_window)

        if mean_acc < mean_before - 2.0:
            new_rank = min(current_rank + 2, max_rank)
            logger.info(f"Rank adaptativo {client_id}: REVERTENDO {current_rank} -> {new_rank} "
                        f"(acurácia caiu: {mean_before:.1f}% -> {mean_acc:.1f}%)")
            state.clients_info[client_id]['rank_before_change'] = current_rank
            return new_rank

    if in_cooldown:
        return None

    best_acc = max(acc_history) if acc_history else 0.0
    min_acc_for_plateau = best_acc * 0.90

    is_plateau = (std_acc < 2.0 or (mean_acc > 0 and (std_acc / mean_acc) < 0.03))
    if (is_plateau 
            and current_rank > rank_min 
            and mean_acc >= min_acc_for_plateau):
        new_rank = current_rank - 1
        logger.info(f"Rank adaptativo {client_id}: reduzindo {current_rank} -> {new_rank} "
                    f"(acurácia estável: {mean_acc:.1f}% ± {std_acc:.2f}%)")
        state.clients_info[client_id]['rank_before_change'] = current_rank
        return new_rank

    return None
