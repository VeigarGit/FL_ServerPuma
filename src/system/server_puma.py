# =============================================================================
# server_puma.py — Motor Analítico PUMA-GT Plus (Índice γ^c, PaCA e Rank Adaptativos)
# =============================================================================

import logging

from server_data import get_state_dict_size_mb

logger = logging.getLogger(__name__)


class PumaEngineMixin:
    """Mixin que provê o motor analítico do PUMA-GT Plus para o servidor FL.

    Calcula o Índice de Capacidade γ^c (combinando latência sistêmica η^c,
    dificuldade estatística Non-IID ψ^c e decaimento por platô ϕ^c_plateau),
    além de alocar dinamicamente PaCA e Rank por cliente com mecanismos de segurança
    (Panic Reversal, Cooldown de transições e Freio de Acurácia Global).
    """

    def new_set_amount_prune(self, client_id):
        """Calcula a taxa de poda estruturada para CNNs usando o Índice de Capacidade γ^c.

        p^c = 1 - γ^c, clamped em [0.20, 0.85]
        Mantém o ajuste histórico da Equação (5) do artigo PUMA (ISCC 2026)
        para correção de instabilidades temporais.
        """
        gamma = self.calculate_capacity_index(client_id)
        pruning_rate = 1.0 - gamma  # γ=1.0 → 0% poda, γ=0.2 → 80% poda
        pruning_rate = max(0.2, min(0.85, pruning_rate))
        pruning_rate = self._adjust_pruning_with_history(client_id, pruning_rate)
        logger.debug(f"Pruning rate γ-driven {client_id}: {pruning_rate:.3f} (γ={gamma:.3f})")
        return pruning_rate

    def calculate_capacity_index(self, client_id):
        """Calcula o Índice de Capacidade γ^c ∈ [γ_min, 1.0] para um cliente.

        Fórmula:
            γ^c = clamp(η^c × ψ^c × ϕ^c, γ_min, 1.0)

        Onde:
            η^c = T_target / T_full^c   (viabilidade de tempo e rede, Eq. 4 do PUMA ISCC 2026)
            ψ^c = (Acc_avg / Acc^c)^δ   (fator de necessidade de qualidade/acurácia Non-IID)
            ϕ^c = decaimento gradual após consolidação em platô de acurácia
        """
        gamma_min = 0.15  # Piso de segurança (evita colapso total de capacidade)

        try:
            client_info = self.clients_info[client_id]
        except KeyError:
            return 1.0

        training_time = client_info.get("training_time")
        bandwidth = client_info.get("bandwidth")
        dataset_size = max(client_info.get("dataset_size", 1.0), 1.0)
        acc_history = client_info.get("acc_history", [])
        comm_time = client_info.get("comm_time")
        eval_time = client_info.get("client_eval_time", 0.0) or 0.0
        proc_time = client_info.get("server_processing_time", 0.0) or 0.0

        # Sem dados suficientes na primeira rodada, retorna capacidade máxima
        if training_time is None or training_time <= 0:
            self.clients_info[client_id]["capacity_index"] = 1.0
            return 1.0

        # ── TERMO 1: Restrição Física de Tempo e Rede (η^c) ──────────────
        target_latency = getattr(self, "target_latency", getattr(self, "time_threshold", 60.0))

        if self.model_size_per_round:
            s_model = self.model_size_per_round[-1]
        else:
            s_model = get_state_dict_size_mb(self.global_model)

        bw_safe = max(bandwidth, 0.01) if bandwidth is not None else 10.0

        t_comm = comm_time if comm_time is not None and comm_time >= 0 else s_model / bw_safe

        t_full = training_time + t_comm + eval_time + proc_time
        eta = 1.0 if t_full <= 0 else min(1.0, target_latency / t_full)

        # ── TERMO 2: Fator de Necessidade / Qualidade (ψ^c) ─────────────
        delta = getattr(self.args, "gamma_delta", 0.5)

        if acc_history and self.rs_aux_simple_avg_acc:
            acc_c = acc_history[-1]
            acc_avg = self.rs_aux_simple_avg_acc[-1] if self.rs_aux_simple_avg_acc[-1] > 0 else 1.0

            if acc_c > 0 and acc_avg > 0:
                need_ratio = acc_avg / acc_c
                need_ratio = max(0.8, min(1.35, need_ratio))
                psi = need_ratio**delta
            else:
                psi = 1.0
        else:
            psi = 1.0

        client_loss = client_info.get("last_loss", 0.0) or 0.0
        global_loss = self.rs_test_loss[-1] if self.rs_test_loss else 0.0
        if client_loss > 0 and global_loss > 0 and client_loss > global_loss * 1.2:
            psi = max(psi, 1.15)

        # ── TERMO 3: Fator de Platô e Maturação (ϕ^c) ─────────────────────
        window = 5
        current_round = len(self.rs_aux_simple_avg_acc)
        eval_history = acc_history if len(acc_history) >= window else self.rs_aux_simple_avg_acc

        if current_round >= 12 and len(eval_history) >= window:
            recent_accs = eval_history[-window:]
            mean_acc = sum(recent_accs) / window
            std_acc = (sum((x - mean_acc) ** 2 for x in recent_accs) / window) ** 0.5
            best_acc = max(eval_history) if eval_history else 0.0

            is_plateau = (mean_acc >= best_acc * 0.90) and (
                std_acc < 1.8 or (mean_acc > 0 and (std_acc / mean_acc) < 0.025)
            )

            plateau_rounds_current = client_info.get("plateau_rounds", 0)
            if plateau_rounds_current >= 8 and len(self.rs_test_acc) >= 10:
                global_recent = self.rs_test_acc[-5:]
                global_mean = sum(global_recent) / len(global_recent)
                global_peak = max(self.rs_test_acc[-10:])
                if global_mean < global_peak - 3.0:
                    is_plateau = False
                    client_info["plateau_rounds"] = max(0, plateau_rounds_current - 3)
                    logger.debug(
                        f"PLATEAU BRAKE {client_id}: global acc {global_mean:.1f}% < "
                        f"peak {global_peak:.1f}% - 3pp. plateau_rounds "
                        f"{plateau_rounds_current} → {client_info['plateau_rounds']}"
                    )

            last_checked = client_info.get("last_plateau_check_round", -1)
            plateau_rounds = client_info.get("plateau_rounds", 0)
            if last_checked != current_round:
                client_info["last_plateau_check_round"] = current_round
                if is_plateau:
                    plateau_rounds += 1
                else:
                    plateau_rounds = max(0, plateau_rounds - 1)
                client_info["plateau_rounds"] = plateau_rounds

            phi_min = getattr(self.args, "plateau_phi_min", 0.45)
            plateau_decay_rate = getattr(self.args, "plateau_decay_rate", 0.025)
            phi_plateau = max(phi_min, 1.0 - (plateau_rounds * plateau_decay_rate))
        else:
            phi_plateau = 1.0

        # ── COMPOSIÇÃO FINAL ─────────────────────────────────────────────
        gamma = max(gamma_min, min(1.0, eta * psi * phi_plateau))

        self.clients_info[client_id]["capacity_index"] = gamma
        self.clients_info[client_id]["eta"] = eta
        self.clients_info[client_id]["psi"] = psi
        self.clients_info[client_id]["phi_plateau"] = phi_plateau

        logger.debug(
            f"γ({client_id}): η={eta:.3f} × ψ={psi:.3f} × ϕ={phi_plateau:.3f} = γ={gamma:.3f} "
            f"(T={training_time:.1f}s, |D|={dataset_size}, B={bw_safe:.2f}MB/s, "
            f"T_target={target_latency:.1f}s, S={s_model:.2f}MB)"
        )

        return gamma

    def calculate_adaptive_paca(self, client_id):
        """Calcula o PaCA ideal para um cliente usando o Índice de Capacidade γ^c

        com rede de segurança (Panic Reversal) e Cooldown de reduções.
        """
        client_info = self.clients_info.get(client_id, {})
        max_paca = self.args.paca
        paca_min = getattr(self.args, "adaptive_paca_min", 2)
        current_paca = client_info.get("current_paca", max_paca)
        acc_history = client_info.get("acc_history", [])
        current_round = len(self.rs_aux_simple_avg_acc)
        cooldown_rounds = getattr(self.args, "adaptive_paca_cooldown", 2)
        last_reduction_round = client_info.get("last_paca_reduction_round", -999)

        # ── 1. REDE DE SEGURANÇA: PANIC REVERSAL ─────────────────────────
        paca_before = client_info.get("paca_before_change")
        paca_acc_at_reduction = client_info.get("paca_acc_at_reduction")
        if (
            paca_before is not None
            and paca_before > current_paca
            and acc_history
            and paca_acc_at_reduction is not None
        ):
            recent_acc = acc_history[-1]
            if recent_acc < paca_acc_at_reduction - 2.0:
                reverted_paca = min(current_paca + 2, max_paca)
                logger.warning(
                    f"PANIC REVERSAL PaCA {client_id}: queda de "
                    f"{paca_acc_at_reduction:.1f}% para {recent_acc:.1f}%. "
                    f"Revertendo {current_paca} → {reverted_paca}!"
                )
                client_info["plateau_rounds"] = max(0, client_info.get("plateau_rounds", 0) - 3)
                client_info["paca_before_change"] = None
                client_info["paca_acc_at_reduction"] = None
                client_info["last_paca_reduction_round"] = current_round
                return reverted_paca
            elif (current_round - last_reduction_round) >= cooldown_rounds:
                client_info["paca_before_change"] = None
                client_info["paca_acc_at_reduction"] = None

        # ── 2. CÁLCULO BASE VIA ÍNDICE DE CAPACIDADE γ ─────────────────────
        gamma = self.calculate_capacity_index(client_id)
        ideal_paca = round(gamma * max_paca)
        ideal_paca = max(paca_min, min(max_paca, ideal_paca))

        # ── 3. COOLDOWN & SUAVIZAÇÃO DE TRANSIÇÃO ─────────────────────────
        in_cooldown = (current_round - last_reduction_round) < cooldown_rounds

        if ideal_paca < current_paca:
            if in_cooldown:
                logger.debug(
                    f"PaCA {client_id} em cooldown (round {current_round}, "
                    f"última redução: {last_reduction_round}). Mantendo {current_paca}"
                )
                return current_paca
            ideal_paca = max(current_paca - 2, ideal_paca)
            client_info["paca_before_change"] = current_paca
            client_info["paca_acc_at_reduction"] = acc_history[-1] if acc_history else 0.0
            client_info["last_paca_reduction_round"] = current_round
            logger.info(
                f"PaCA γ-driven {client_id}: {current_paca} → {ideal_paca} (γ={gamma:.3f})"
            )
        elif ideal_paca > current_paca:
            ideal_paca = min(current_paca + 2, ideal_paca)
            client_info["paca_before_change"] = None
            client_info["paca_acc_at_reduction"] = None
            logger.info(
                f"PaCA γ-driven {client_id}: subindo {current_paca} → {ideal_paca} (γ={gamma:.3f})"
            )

        return ideal_paca

    def calculate_adaptive_rank(self, client_id):
        """Calcula o Rank ideal para um cliente usando o Índice de Capacidade γ^c

        com rede de segurança (Panic Reversal) e Cooldown de reduções.
        """
        client_info = self.clients_info.get(client_id, {})
        max_rank = getattr(self.args, "adaptive_rank_max", 8)
        min_rank = getattr(self.args, "adaptive_rank_min", 4)
        current_rank = client_info.get("current_rank", max_rank)
        acc_history = client_info.get("acc_history", [])
        current_round = len(self.rs_aux_simple_avg_acc)
        cooldown_rounds = getattr(self.args, "adaptive_rank_cooldown", 3)
        last_reduction_round = client_info.get("last_rank_reduction_round", -999)

        # ── 1. REDE DE SEGURANÇA: PANIC REVERSAL ─────────────────────────
        rank_before = client_info.get("rank_before_change")
        rank_acc_at_reduction = client_info.get("rank_acc_at_reduction")
        if (
            rank_before is not None
            and rank_before > current_rank
            and acc_history
            and rank_acc_at_reduction is not None
        ):
            recent_acc = acc_history[-1]
            if recent_acc < rank_acc_at_reduction - 2.0:
                reverted_rank = min(current_rank + 2, max_rank)
                logger.warning(
                    f"PANIC REVERSAL Rank {client_id}: queda de "
                    f"{rank_acc_at_reduction:.1f}% para {recent_acc:.1f}%. "
                    f"Revertendo {current_rank} → {reverted_rank}!"
                )
                client_info["plateau_rounds"] = max(0, client_info.get("plateau_rounds", 0) - 3)
                client_info["rank_before_change"] = None
                client_info["rank_acc_at_reduction"] = None
                client_info["last_rank_reduction_round"] = current_round
                return reverted_rank
            elif (current_round - last_reduction_round) >= cooldown_rounds:
                client_info["rank_before_change"] = None
                client_info["rank_acc_at_reduction"] = None

        # ── 2. CÁLCULO BASE VIA ÍNDICE DE CAPACIDADE γ ─────────────────────
        gamma = self.calculate_capacity_index(client_id)
        ideal_rank = round(gamma * max_rank)
        ideal_rank = max(min_rank, min(max_rank, ideal_rank))

        # ── 3. COOLDOWN & SUAVIZAÇÃO DE TRANSIÇÃO ─────────────────────────
        in_cooldown = (current_round - last_reduction_round) < cooldown_rounds
        allow_upscale = getattr(self.args, "allow_rank_upscale", True)

        if ideal_rank < current_rank:
            if in_cooldown:
                logger.debug(
                    f"Rank {client_id} em cooldown (round {current_round}, "
                    f"última redução: {last_reduction_round}). Mantendo {current_rank}"
                )
                return current_rank
            ideal_rank = max(current_rank - 1, ideal_rank)
            client_info["rank_before_change"] = current_rank
            client_info["rank_acc_at_reduction"] = acc_history[-1] if acc_history else 0.0
            client_info["last_rank_reduction_round"] = current_round
            logger.info(
                f"Rank γ-driven {client_id}: {current_rank} → {ideal_rank} (γ={gamma:.3f})"
            )
        elif ideal_rank > current_rank:
            if not allow_upscale:
                return current_rank
            ideal_rank = min(current_rank + 2, ideal_rank)
            client_info["rank_before_change"] = None
            client_info["rank_acc_at_reduction"] = None
            logger.info(
                f"Rank γ-driven {client_id}: subindo {current_rank} → {ideal_rank} (γ={gamma:.3f})"
            )

        return ideal_rank

    def _is_excluded_by_paca(self, key: str, min_layer_idx: int) -> bool:
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

    def _adjust_pruning_with_history(self, client_id, base_pruning_rate):
        try:
            client_info = self.clients_info[client_id]
        except KeyError:
            return base_pruning_rate

        training_time = client_info.get("training_time", 0)
        last_training_time = client_info.get("last_training_time", training_time)
        client_info["last_training_time"] = training_time

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
            non_null_times = [
                info["training_time"]
                for info in self.clients_info.values()
                if info.get("training_time")
            ]
            return non_null_times[0] / data if non_null_times else 0.0
        else:
            sorted_values = sorted(values, reverse=True)
            maior_valor = sorted_values[0]
            penultimo_maior = sorted_values[1]

            distancia_normalizada = (
                (1 - ((maior_valor - penultimo_maior) / maior_valor)) if maior_valor != 0 else 0
            )
            return 0.85 if distancia_normalizada > 0.9 else distancia_normalizada
