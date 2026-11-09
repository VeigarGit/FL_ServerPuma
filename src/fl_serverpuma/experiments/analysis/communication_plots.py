# Responsabilidade: Gera gráficos do volume de dados transferido por rodada, acumulado e em
# função do tempo.
#
# Principais funções: plot_04_banda_acumulada, plot_05_banda_por_rodada, plot_15_mb_vs_tempo.

"""Experiment plotting — communication_plots."""

import numpy as np
import matplotlib.pyplot as plt
from fl_serverpuma.experiments.analysis.plot_style import _mark_every
from fl_serverpuma.experiments.analysis.plot_style import _set_title
from fl_serverpuma.experiments.analysis.plot_style import _save_and_close
from fl_serverpuma.experiments.analysis.result_loading import _compute_mb_per_round
from fl_serverpuma.experiments.analysis.plot_style import STYLES

def plot_04_banda_acumulada(experiments, output_dir):
    """Mostra o total de dados (MB) trafegados na rede de forma acumulada ao longo das rodadas."""
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, exp in enumerate(experiments):
        s = STYLES[i % len(STYLES)]
        d = exp['server']
        result = _compute_mb_per_round(d)
        if result is None:
            continue
        _, mb_accumulated, _ = result
        x = range(1, len(mb_accumulated) + 1)
        ax.plot(x, mb_accumulated, color=s['color'], marker=s['marker'],
                markevery=_mark_every(len(mb_accumulated)), linestyle=s['ls'],
                label=exp['label'])

    _set_title(ax, "Accumulated Network Bandwidth per Round")
    ax.set_xlabel("Rounds"); ax.set_ylabel("Accumulated MB")
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    _save_and_close(fig, output_dir, "04_banda_acumulada.pdf")


def plot_05_banda_por_rodada(experiments, output_dir):
    """Mostra quanto de dados (MB) foi trafegado (ida + volta) em cada rodada individual."""
    fig, ax = plt.subplots(figsize=(12, 7))
    has_valid_data = False

    for i, exp in enumerate(experiments):
        s = STYLES[i % len(STYLES)]
        d = exp['server']
        result = _compute_mb_per_round(d)
        if result is None:
            continue
        mb_discrete, _, _ = result
        x = range(1, len(mb_discrete) + 1)
        ax.plot(x, mb_discrete, color=s['color'], marker=s['marker'],
                markevery=_mark_every(len(mb_discrete)), linestyle=s['ls'],
                label=exp['label'])
        has_valid_data = True

    if not has_valid_data:
        plt.close(fig)
        return

    _set_title(ax, "Bandwidth Consumption per Round (Upload + Download)")
    ax.set_xlabel("Rounds"); ax.set_ylabel("Transferred MB per Round")
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    _save_and_close(fig, output_dir, "05_banda_por_rodada.pdf")


def plot_15_mb_vs_tempo(experiments, output_dir):
    """Mostra o total de dados (MB) trafegados na rede em função do tempo real acumulado (wall-clock)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    has_valid_data = False

    for i, exp in enumerate(experiments):
        s = STYLES[i % len(STYLES)]
        d = exp['server']
        result = _compute_mb_per_round(d)
        if result is None:
            continue
        _, mb_accumulated, _ = result
        accumulated_time = np.cumsum(d['time_mean'])
        # Alinhar tamanhos (usar o mínimo entre os dois)
        n = min(len(accumulated_time), len(mb_accumulated))
        ax.plot(accumulated_time[:n], mb_accumulated[:n], color=s['color'], marker=s['marker'],
                markevery=_mark_every(n), linestyle=s['ls'],
                label=exp['label'])
        has_valid_data = True

    if not has_valid_data:
        plt.close(fig)
        return

    _set_title(ax, "Accumulated MB vs Real Time (Wall-Clock)")
    ax.set_xlabel("Accumulated Time (s)")
    ax.set_ylabel("Accumulated MB")
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    _save_and_close(fig, output_dir, "15_mb_vs_tempo.pdf")
