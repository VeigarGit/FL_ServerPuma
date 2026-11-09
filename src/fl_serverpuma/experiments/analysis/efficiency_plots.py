# Responsabilidade: Relaciona acurácia, comunicação e tempo para visualizar eficiência.
#
# Principais funções: plot_13_acuracia_vs_tempo, plot_14_resumo_comparativo.

"""Experiment plotting — efficiency_plots."""

import numpy as np
import matplotlib.pyplot as plt
from fl_serverpuma.experiments.analysis.plot_style import _mark_every
from fl_serverpuma.experiments.analysis.plot_style import _set_title
from fl_serverpuma.experiments.analysis.plot_style import _save_and_close
from fl_serverpuma.experiments.analysis.result_loading import _compute_mb_per_round
from fl_serverpuma.experiments.analysis.result_loading import _get_train_time_per_round
from fl_serverpuma.experiments.analysis.plot_style import STYLES

def plot_13_acuracia_vs_tempo(experiments, output_dir):
    """Mostra a eficiência temporal de cada estratégia em dois gráficos separados:
    (a) Acurácia vs tempo real (wall-clock acumulado)
    (b) Acurácia vs tempo estimado (treino real + rede simulada a 100 Mbps)
    """
    BANDWIDTH_MBPS = 100
    BANDWIDTH_MBs = BANDWIDTH_MBPS / 8  # MB/s

    # --- Gráfico (a): Tempo real ---
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    for i, exp in enumerate(experiments):
        s = STYLES[i % len(STYLES)]
        d = exp['server']
        accumulated_time = np.cumsum(d['time_mean'])
        ax1.plot(accumulated_time, d['acc_mean'], color=s['color'], marker=s['marker'],
                 markevery=_mark_every(len(d['acc_mean'])), linestyle=s['ls'],
                 label=exp['label'])
        ax1.fill_between(accumulated_time, d['acc_mean'] - d['acc_std'],
                          d['acc_mean'] + d['acc_std'], color=s['color'], alpha=0.1)

    _set_title(ax1, "Accuracy vs Accumulated Real Time")
    ax1.set_xlabel("Accumulated Time (s)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend()
    ax1.grid(True, ls='--', alpha=0.4)
    fig1.tight_layout()
    _save_and_close(fig1, output_dir, "13a_acuracia_vs_tempo_real.pdf")

    # --- Gráfico (b): Tempo estimado (treino real + rede simulada) ---
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    for i, exp in enumerate(experiments):
        s = STYLES[i % len(STYLES)]
        d = exp['server']

        num_rounds = len(d['time_mean'])
        result = _compute_mb_per_round(d)
        if result is None:
            continue
        mb_discrete, _, _ = result

        train_per_round = _get_train_time_per_round(d)

        estimated_round_time = []
        for r in range(num_rounds):
            t_net = mb_discrete[r] / BANDWIDTH_MBs if r < len(mb_discrete) else 0
            t_tr = train_per_round[r] if r < len(train_per_round) else np.mean(train_per_round)
            estimated_round_time.append(t_tr + t_net)

        accumulated_time = np.cumsum(estimated_round_time)

        ax2.plot(accumulated_time, d['acc_mean'], color=s['color'], marker=s['marker'],
                 markevery=_mark_every(len(d['acc_mean'])), linestyle=s['ls'],
                 label=exp['label'])
        ax2.fill_between(accumulated_time, d['acc_mean'] - d['acc_std'],
                          d['acc_mean'] + d['acc_std'], color=s['color'], alpha=0.1)

    _set_title(ax2, f"Accuracy vs Estimated Time ({BANDWIDTH_MBPS} Mbps Network)")
    ax2.set_xlabel(f"Accumulated Estimated Time (s)")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, ls='--', alpha=0.4)
    fig2.tight_layout()
    _save_and_close(fig2, output_dir, "13b_acuracia_vs_tempo_simulado.pdf")


def plot_14_resumo_comparativo(experiments, output_dir):
    """Painel com 4 métricas finais de cada estratégia: acurácia, loss, tempo médio, banda total."""
    labels = [e['label'] for e in experiments]
    final_acc = [e['server']['acc_mean'][-1] for e in experiments]
    final_loss = [e['server']['loss_mean'][-1] for e in experiments]
    avg_time = [np.mean(e['server']['time_mean']) for e in experiments]
    total_mb = [e['server']['mb_mean'][-1] for e in experiments]

    x = np.arange(len(labels))
    width = 0.6

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Acurácia final
    ax = axes[0, 0]
    bars = ax.bar(x, final_acc, width, color=[STYLES[i % len(STYLES)]['color'] for i in range(len(experiments))])
    _set_title(ax, "Final Accuracy (%)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    for bar, val in zip(bars, final_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}', ha='center', fontsize=9)
    ax.grid(True, axis='y', ls='--', alpha=0.4)

    # Loss final
    ax = axes[0, 1]
    bars = ax.bar(x, final_loss, width, color=[STYLES[i % len(STYLES)]['color'] for i in range(len(experiments))])
    _set_title(ax, "Final Loss")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    for bar, val in zip(bars, final_loss):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01, f'{val:.3f}', ha='center', fontsize=9)
    ax.grid(True, axis='y', ls='--', alpha=0.4)

    # Tempo médio por rodada
    ax = axes[1, 0]
    bars = ax.bar(x, avg_time, width, color=[STYLES[i % len(STYLES)]['color'] for i in range(len(experiments))])
    _set_title(ax, "Average Time per Round (s)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    for bar, val in zip(bars, avg_time):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.1f}', ha='center', fontsize=9)
    ax.grid(True, axis='y', ls='--', alpha=0.4)

    # Total MB trafegado
    ax = axes[1, 1]
    bars = ax.bar(x, total_mb, width, color=[STYLES[i % len(STYLES)]['color'] for i in range(len(experiments))])
    _set_title(ax, "Total Transferred MB")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    for bar, val in zip(bars, total_mb):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01, f'{val:.1f}', ha='center', fontsize=9)
    ax.grid(True, axis='y', ls='--', alpha=0.4)

    fig.suptitle("Comparative Summary of Performance and Cost", fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_and_close(fig, output_dir, "14_resumo_comparativo.pdf")
