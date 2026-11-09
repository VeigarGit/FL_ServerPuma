# Responsabilidade: Decide PaCA por latência, perda e acurácia; aplica cooldown, limites e
# reversão, atualizando o histórico da política.
#
# Principais funções: calculate_adaptive_paca.

"""Server paca."""

import logging

logger = logging.getLogger(__name__)

def calculate_adaptive_paca(state, client_id):
    """
    Calcula o PaCA ideal para um cliente baseado em dois critérios:

    1. LATÊNCIA RELATIVA: Se o cliente é gargalo (training_time > target_latency), reduz PaCA.
    2. ACURÁCIA: Se a acurácia estabilizou (plateau), reduz PaCA porque as camadas
       extras não estão contribuindo — menos camadas = menos comunicação de graça.

    Proteções:
    - Se a acurácia caiu após redução de PaCA, reverte (sobe de volta).
    - Mínimo configurável (padrão: 2).
    - Máximo ±2 camadas de mudança por round.
    """
    try:
        client_info = state.clients_info[client_id]
    except KeyError:
        return state.args.paca

    bandwidth = client_info.get('bandwidth')
    t_train = client_info.get('training_time', 0.0) or 0.0
    t_eval = client_info.get('client_eval_time', 0.0) or 0.0
    t_proc = client_info.get('server_processing_time', 0.0) or 0.0
    t_comm = client_info.get('comm_time', 0.0) or 0.0
    total_time = t_train + t_eval + t_proc + t_comm

    current_paca = client_info.get('current_paca', state.args.paca)
    max_paca = state.args.paca  # O máximo possível (definido na CLI)
    paca_min = getattr(state.args, 'adaptive_paca_min', 2)  # Mínimo configurável (padrão: 2)

    # Sem dados suficientes, mantém o atual
    if bandwidth is None or t_train == 0.0:
        return current_paca

    target_latency = getattr(state, 'target_latency', getattr(state, 'time_threshold', 60.0))
    mean_time = getattr(state, 'mean_training_time', target_latency / 1.15)

    # --- COOLDOWN UNIFICADO: nenhuma redução dentro de 5 rounds ---
    # Alinhado com a janela de acurácia (window=5) para garantir que o
    # impacto de uma redução seja observável antes de reduzir novamente.
    last_reduction_round = client_info.get('last_paca_any_reduction_round', None)
    current_round = len(state.rs_test_acc)
    cooldown_rounds = 5
    in_cooldown = (last_reduction_round is not None and (current_round - last_reduction_round) < cooldown_rounds)

    # --- CRITÉRIO 2: ACURÁCIA (verificado primeiro, é global) ---
    # Reversão por queda de acurácia SEMPRE permitida (mesmo em cooldown).
    # Redução por plateau APENAS fora do cooldown.
    accuracy_signal = _check_accuracy_signal(state, 
        client_id, current_paca, paca_min, max_paca, in_cooldown
    )
    if accuracy_signal is not None and accuracy_signal > current_paca:
        return accuracy_signal

    # --- SINAL DE LOSS LOCAL (Dificuldade do cliente) ---
    global_loss = state.rs_test_loss[-1] if state.rs_test_loss else 1.0
    client_loss = client_info.get('last_loss', 0.0)
    is_difficult_client = (client_loss > 0.0 and global_loss > 0.0 and client_loss > global_loss * 1.2)

    if is_difficult_client:
        if current_paca < max_paca:
            if getattr(state.args, 'allow_paca_upscale', False) or total_time <= target_latency:
                new_paca = current_paca + 1
                logger.info(f"PaCA adaptativo {client_id}: cliente difícil (loss={client_loss:.4f} > global={global_loss:.4f}*1.2) com margem de tempo ({total_time:.1f}s <= {target_latency:.1f}s). Subindo PaCA {current_paca} -> {new_paca}")
                return new_paca
            else:
                logger.info(f"PaCA adaptativo {client_id}: cliente difícil (loss={client_loss:.4f}), mas sem margem de tempo ({total_time:.1f}s > {target_latency:.1f}s). Mantendo {current_paca}")
        return current_paca

    if accuracy_signal is not None and accuracy_signal < current_paca:
        client_info['last_paca_any_reduction_round'] = current_round
        return accuracy_signal

    # --- CRITÉRIO 1: LATÊNCIA (per-client em relação à média do cluster) ---
    # Increase PaCA if client is significantly faster than the cluster average (>20% faster)
    if total_time < mean_time * 0.80 and current_paca < max_paca:
        if getattr(state.args, 'allow_paca_upscale', False):
            new_paca = min(current_paca + 1, max_paca)
            logger.info(f"PaCA adaptativo {client_id}: subindo {current_paca} -> {new_paca} "
                        f"(tempo_total={total_time:.1f}s < 80% da média={mean_time:.1f}s)")
            return new_paca
        else:
            logger.info(f"PaCA adaptativo {client_id}: subida bloqueada por padrão (use --allow-paca-upscale para permitir) (tempo_total={total_time:.1f}s < 80%)")

    # Se estiver dentro da tolerância do cluster, mantém
    if total_time <= target_latency:
        return current_paca

    # Em cooldown: não reduzir por latência
    if in_cooldown:
        return current_paca

    # Cliente está lento (straggler relativo): calcular quantas camadas cabem no budget
    # Budget real para treinamento é o target_latency subtraindo os tempos fixos
    fixed_overhead = t_eval + t_proc + t_comm
    available_train_time = max(0.1, target_latency - fixed_overhead)
    time_per_paca_layer = t_train / max(current_paca, 1)
    ideal_paca = int(available_train_time / time_per_paca_layer) if time_per_paca_layer > 0 else current_paca
    ideal_paca = max(paca_min, min(max_paca, ideal_paca))

    # Suavização: não reduzir mais que 1 camada por round (evita cascata destrutiva)
    if ideal_paca < current_paca:
        ideal_paca = max(ideal_paca, current_paca - 1)
        client_info['last_paca_any_reduction_round'] = current_round

    logger.info(f"PaCA adaptativo {client_id}: {current_paca} -> {ideal_paca} "
                f"(tempo_total={total_time:.1f}s > target={target_latency:.1f}s, média={mean_time:.1f}s)")
    return ideal_paca


def _check_accuracy_signal(state, client_id, current_paca, paca_min, max_paca, in_cooldown=False):
    """
    Verifica se a acurácia indica que o PaCA pode ser ajustado.

    - Reversão: SEMPRE ativa (mesmo em cooldown). Se a acurácia caiu >2pp
      após uma redução de PaCA, reverte +2 como rede de segurança.
    - Plateau: APENAS fora do cooldown. Se a acurácia estabilizou, reduz -1.

    Retorna:
        int: novo PaCA se a acurácia motivou uma mudança
        None: se a acurácia não é conclusiva (deixa a latência decidir)
    """
    acc_history = state.rs_test_acc
    window = 5  # Janela de rounds para análise

    # Precisa de pelo menos 'window' rounds de histórico
    if len(acc_history) < window:
        return None

    recent = acc_history[-window:]
    mean_acc = sum(recent) / len(recent)
    std_acc = (sum((x - mean_acc) ** 2 for x in recent) / len(recent)) ** 0.5

    # --- Safety check: Revert PaCA reduction if accuracy drops significantly ---
    # (Always active, even during cooldown periods)
    paca_before_last_change = state.clients_info[client_id].get('paca_before_change')
    if paca_before_last_change is not None and paca_before_last_change > current_paca:
        # PaCA foi reduzido. Verificar se a acurácia caiu.
        acc_before_window = acc_history[-(window + 1):-1] if len(acc_history) > window else acc_history[:window]
        mean_before = sum(acc_before_window) / len(acc_before_window)

        if mean_acc < mean_before - 2.0:  # Caiu mais de 2 pontos percentuais
            new_paca = min(current_paca + 2, max_paca)  # Reversão forte: +2
            logger.info(f"PaCA adaptativo {client_id}: REVERTENDO {current_paca} -> {new_paca} "
                        f"(acurácia caiu: {mean_before:.1f}% -> {mean_acc:.1f}%)")
            state.clients_info[client_id]['paca_before_change'] = current_paca
            return new_paca

    # --- PLATEAU: acurácia estabilizou? Tenta reduzir. ---
    # (APENAS fora do cooldown — respeita o tempo de observação)
    if in_cooldown:
        return None

    best_acc = max(acc_history) if acc_history else 0.0
    min_acc_for_plateau = best_acc * 0.90  # 90% da melhor acurácia observada
    is_plateau = (std_acc < 2.0 or (mean_acc > 0 and (std_acc / mean_acc) < 0.03))
    if (is_plateau 
            and current_paca > paca_min 
            and mean_acc >= min_acc_for_plateau):
        new_paca = current_paca - 1
        logger.info(f"PaCA adaptativo {client_id}: reduzindo {current_paca} -> {new_paca} "
                    f"(acurácia estável: {mean_acc:.1f}% ± {std_acc:.2f}%, camadas extras desnecessárias)")
        state.clients_info[client_id]['paca_before_change'] = current_paca
        return new_paca

    return None  # Acurácia não é conclusiva, deixa latência decidir


def _is_excluded_by_paca(state, key, min_layer_idx):
    """Retorna True se a chave pertence a uma camada ABAIXO do PaCA mínimo."""
    if "encoder.layers." not in key and ".layers." not in key:
        return False  # Head de classificação etc. sempre incluída
    try:
        if "encoder.layers." in key:
            layer_idx = int(key.split("encoder.layers.")[1].split(".")[0])
        elif ".layers." in key:
            layer_idx = int(key.split(".layers.")[1].split(".")[0])
        else:
            return False
        return layer_idx < min_layer_idx
    except (ValueError, IndexError):
        return False
