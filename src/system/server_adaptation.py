"""Adaptive pruning, PaCA, rank, and timeout policies for the server."""
import logging

logger = logging.getLogger(__name__)

class ServerAdaptationMixin:
    def set_threshold(self):
        tot_time = 0
        clients_with_time = 0
        max_client_time = 0.0
        for info in self.clients_info.values():
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
        self.mean_training_time = mean_time

        # Meta de latência adaptativa do cluster: média dos clientes + 15% de tolerância
        # Utilizada pelo PaCA e Rank adaptativos para identificar stragglers relativos
        self.target_latency = mean_time * 1.15

        # Pisos de segurança por tipo de modelo para evitar colapso de timeout em rodadas pontualmente rápidas
        if self.args.model == 'slm':
            min_floor = 1200.0
            multiplier = 2.5
        elif self.args.model == 'clip':
            min_floor = 300.0
            multiplier = 2.0
        else:
            min_floor = 60.0
            multiplier = 1.5

        calculated_threshold = max(min_floor, multiplier * mean_time, 1.5 * max_client_time)

        # Amortecimento: impede quedas bruscas de timeout entre rodadas
        if hasattr(self, 'time_threshold') and self.time_threshold is not None:
            self.time_threshold = max(calculated_threshold, self.time_threshold * 0.8, min_floor)
        else:
            self.time_threshold = calculated_threshold

        logger.info(f"Limiares definidos: timeout_socket={self.time_threshold:.2f}s | target_latency={self.target_latency:.2f}s (mean_time={mean_time:.2f}s)")

    def new_set_amount_prune(self, client_id):
        # Filosofia EAFP: Tenta acessar direto e falha limpo
        try:
            client_info = self.clients_info[client_id]
        except KeyError:
            logger.warning(f"Informações do cliente {client_id} ausentes. Usando pruning rate padrão.")
            return 0.5

        bandwidth = client_info.get('bandwidth', 1.0)
        training_time = client_info.get('training_time', 60.0)
        dataset_size = max(client_info.get('dataset_size', 1.0), 1.0)

        if not hasattr(self, 'alpha_list') or not hasattr(self, 'beta_list'):
            return self._fallback_prune_calculation(client_id)

        total_alpha = sum(self.alpha_list)
        total_beta = sum(self.beta_list)

        client_info.setdefault('last_flops', total_beta)
        client_info.setdefault('last_params', total_alpha)

        target_latency = getattr(self, 'target_latency', getattr(self, 'time_threshold', 60.0))

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
        pruning_rate = self._adjust_pruning_with_history(client_id, pruning_rate)

        logger.debug(f"Calculated pruning rate para {client_id}: {pruning_rate:.3f}")
        return pruning_rate

    def calculate_adaptive_paca(self, client_id):
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
            client_info = self.clients_info[client_id]
        except KeyError:
            return self.args.paca

        bandwidth = client_info.get('bandwidth')
        t_train = client_info.get('training_time', 0.0) or 0.0
        t_eval = client_info.get('client_eval_time', 0.0) or 0.0
        t_proc = client_info.get('server_processing_time', 0.0) or 0.0
        t_comm = client_info.get('comm_time', 0.0) or 0.0
        total_time = t_train + t_eval + t_proc + t_comm

        current_paca = client_info.get('current_paca', self.args.paca)
        max_paca = self.args.paca  # O máximo possível (definido na CLI)
        paca_min = getattr(self.args, 'adaptive_paca_min', 2)  # Mínimo configurável (padrão: 2)

        # Sem dados suficientes, mantém o atual
        if bandwidth is None or t_train == 0.0:
            return current_paca

        target_latency = getattr(self, 'target_latency', getattr(self, 'time_threshold', 60.0))
        mean_time = getattr(self, 'mean_training_time', target_latency / 1.15)

        # --- COOLDOWN UNIFICADO: nenhuma redução dentro de 5 rounds ---
        # Alinhado com a janela de acurácia (window=5) para garantir que o
        # impacto de uma redução seja observável antes de reduzir novamente.
        last_reduction_round = client_info.get('last_paca_any_reduction_round', None)
        current_round = len(self.rs_test_acc)
        cooldown_rounds = 5
        in_cooldown = (last_reduction_round is not None and (current_round - last_reduction_round) < cooldown_rounds)

        # --- CRITÉRIO 2: ACURÁCIA (verificado primeiro, é global) ---
        # Reversão por queda de acurácia SEMPRE permitida (mesmo em cooldown).
        # Redução por plateau APENAS fora do cooldown.
        accuracy_signal = self._check_accuracy_signal(
            client_id, current_paca, paca_min, max_paca, in_cooldown
        )
        if accuracy_signal is not None and accuracy_signal > current_paca:
            return accuracy_signal

        # --- SINAL DE LOSS LOCAL (Dificuldade do cliente) ---
        global_loss = self.rs_test_loss[-1] if self.rs_test_loss else 1.0
        client_loss = client_info.get('last_loss', 0.0)
        is_difficult_client = (client_loss > 0.0 and global_loss > 0.0 and client_loss > global_loss * 1.2)

        if is_difficult_client:
            if current_paca < max_paca:
                if getattr(self.args, 'allow_paca_upscale', False) or total_time <= target_latency:
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
            if getattr(self.args, 'allow_paca_upscale', False):
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

    def calculate_adaptive_rank(self, client_id):
        try:
            client_info = self.clients_info[client_id]
        except KeyError:
            return self.args.rank

        bandwidth = client_info.get('bandwidth')
        t_train = client_info.get('training_time', 0.0) or 0.0
        t_eval = client_info.get('client_eval_time', 0.0) or 0.0
        t_proc = client_info.get('server_processing_time', 0.0) or 0.0
        t_comm = client_info.get('comm_time', 0.0) or 0.0
        total_time = t_train + t_eval + t_proc + t_comm

        current_rank = client_info.get('current_rank', self.args.rank)
        max_rank = getattr(self.args, 'adaptive_rank_max', 8)
        min_rank = getattr(self.args, 'adaptive_rank_min', 2)

        # Sem dados suficientes, mantém o atual
        if bandwidth is None or t_train == 0.0:
            return current_rank

        target_latency = getattr(self, 'target_latency', getattr(self, 'time_threshold', 60.0))
        mean_time = getattr(self, 'mean_training_time', target_latency / 1.15)

        # --- 1. Sinal da Acurácia Global ---
        last_reduction_round = client_info.get('last_rank_reduction_round', None)
        current_round = len(self.rs_test_acc)
        cooldown_rounds = 5
        in_cooldown = (last_reduction_round is not None and (current_round - last_reduction_round) < cooldown_rounds)

        acc_signal = self._check_accuracy_signal_rank(client_id, current_rank, min_rank, max_rank, in_cooldown)

        # Acurácia despencou (Pânico! Reversão tem prioridade)
        if acc_signal is not None and acc_signal > current_rank:
            client_info['last_rank_reduction_round'] = current_round
            return acc_signal

        # --- 2. Sinal de Loss Local (Dificuldade do cliente) ---
        global_loss = self.rs_test_loss[-1] if self.rs_test_loss else 1.0
        client_loss = client_info.get('last_loss', 0.0)

        is_difficult_client = (client_loss > 0.0 and global_loss > 0.0 and client_loss > global_loss * 1.2)

        if is_difficult_client:
            if current_rank < max_rank:
                if getattr(self.args, 'allow_rank_upscale', False) or total_time <= target_latency:
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
            if getattr(self.args, 'allow_rank_upscale', False):
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

    def _check_accuracy_signal_rank(self, client_id, current_rank, rank_min, max_rank, in_cooldown=False):
        """
        Verifica se a acurácia indica que o Rank pode ser ajustado.
        - Fallback mechanism: Revert PaCA layer count if accuracy drops by more than 2% after a reduction.
        - Plateau: APENAS fora do cooldown. Se estabilizou, reduz -1.
        """
        acc_history = self.rs_test_acc
        window = 5

        if len(acc_history) < window:
            return None

        recent = acc_history[-window:]
        mean_acc = sum(recent) / len(recent)
        std_acc = (sum((x - mean_acc) ** 2 for x in recent) / len(recent)) ** 0.5

        rank_before_last_change = self.clients_info[client_id].get('rank_before_change')
        if rank_before_last_change is not None and rank_before_last_change > current_rank:
            acc_before_window = acc_history[-(window + 1):-1] if len(acc_history) > window else acc_history[:window]
            mean_before = sum(acc_before_window) / len(acc_before_window)

            if mean_acc < mean_before - 2.0:
                new_rank = min(current_rank + 2, max_rank)
                logger.info(f"Rank adaptativo {client_id}: REVERTENDO {current_rank} -> {new_rank} "
                            f"(acurácia caiu: {mean_before:.1f}% -> {mean_acc:.1f}%)")
                self.clients_info[client_id]['rank_before_change'] = current_rank
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
            self.clients_info[client_id]['rank_before_change'] = current_rank
            return new_rank

        return None

    def _check_accuracy_signal(self, client_id, current_paca, paca_min, max_paca, in_cooldown=False):
        """
        Verifica se a acurácia indica que o PaCA pode ser ajustado.

        - Reversão: SEMPRE ativa (mesmo em cooldown). Se a acurácia caiu >2pp
          após uma redução de PaCA, reverte +2 como rede de segurança.
        - Plateau: APENAS fora do cooldown. Se a acurácia estabilizou, reduz -1.

        Retorna:
            int: novo PaCA se a acurácia motivou uma mudança
            None: se a acurácia não é conclusiva (deixa a latência decidir)
        """
        acc_history = self.rs_test_acc
        window = 5  # Janela de rounds para análise

        # Precisa de pelo menos 'window' rounds de histórico
        if len(acc_history) < window:
            return None

        recent = acc_history[-window:]
        mean_acc = sum(recent) / len(recent)
        std_acc = (sum((x - mean_acc) ** 2 for x in recent) / len(recent)) ** 0.5

        # --- Safety check: Revert PaCA reduction if accuracy drops significantly ---
        # (Always active, even during cooldown periods)
        paca_before_last_change = self.clients_info[client_id].get('paca_before_change')
        if paca_before_last_change is not None and paca_before_last_change > current_paca:
            # PaCA foi reduzido. Verificar se a acurácia caiu.
            acc_before_window = acc_history[-(window + 1):-1] if len(acc_history) > window else acc_history[:window]
            mean_before = sum(acc_before_window) / len(acc_before_window)

            if mean_acc < mean_before - 2.0:  # Caiu mais de 2 pontos percentuais
                new_paca = min(current_paca + 2, max_paca)  # Reversão forte: +2
                logger.info(f"PaCA adaptativo {client_id}: REVERTENDO {current_paca} -> {new_paca} "
                            f"(acurácia caiu: {mean_before:.1f}% -> {mean_acc:.1f}%)")
                self.clients_info[client_id]['paca_before_change'] = current_paca
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
            self.clients_info[client_id]['paca_before_change'] = current_paca
            return new_paca

        return None  # Acurácia não é conclusiva, deixa latência decidir


    def _is_excluded_by_paca(self, key, min_layer_idx):
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

    def _fallback_prune_calculation(self, client_id):
        try:
            client_info = self.clients_info[client_id]
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

    def _adjust_pruning_with_history(self, client_id, base_pruning_rate):
        try:
            client_info = self.clients_info[client_id]
        except KeyError:
            return base_pruning_rate

        training_time = client_info.get('training_time', 0)
        last_training_time = client_info.get('last_training_time', training_time)
        client_info['last_training_time'] = training_time

        if training_time > last_training_time * 1.5:
            adjustment = 0.05
        elif training_time > self.time_threshold * 1.2:
            adjustment = 0.03
        else:
            adjustment = 0.0
        return max(0.2, min(0.85, base_pruning_rate + adjustment))

    def set_amount_prune(self):
        values = [v for v in self.client_data.values() if v is not None]
        if not values:
            return 0

        if len(values) == 1:
            data = values[0]
            non_null_times = [info['training_time'] for info in self.clients_info.values() if info.get('training_time')]
            return non_null_times[0] / data if non_null_times else 0.0
        else:
            sorted_values = sorted(values, reverse=True)
            maior_valor = sorted_values[0]
            penultimo_maior = sorted_values[1]

            distancia_normalizada = (1 - ((maior_valor - penultimo_maior) / maior_valor)) if maior_valor != 0 else 0
            return 0.85 if distancia_normalizada > 0.9 else distancia_normalizada
