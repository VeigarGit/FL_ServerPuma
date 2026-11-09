# Responsabilidade: Gera gráficos do tamanho dos adaptadores, quantidade de parâmetros
# treináveis e evolução do PaCA.
#
# Principais funções: plot_06_tamanho_modelo, plot_07_parametros_treinaveis,
# plot_08_paca_evolucao.

"""Experiment plotting — model_plots."""

import numpy as np
import matplotlib.pyplot as plt
from fl_serverpuma.experiments.analysis.plot_style import _mark_every
from fl_serverpuma.experiments.analysis.plot_style import _set_title
from fl_serverpuma.experiments.analysis.plot_style import _save_and_close
from fl_serverpuma.experiments.analysis.plot_style import STYLES

def plot_06_tamanho_modelo(experiments, output_dir):
    """Evolução do tamanho dos adaptadores treináveis (em MB) ao longo das rodadas."""
    has_data = any(not np.all(e['server']['size_mb_mean'] == 0) for e in experiments)
    if not has_data:
        print("  ⚠️ Sem dados de tamanho de modelo, pulando.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, exp in enumerate(experiments):
        d = exp['server']
        if np.all(d['size_mb_mean'] == 0):
            continue
        s = STYLES[i % len(STYLES)]
        x = range(1, len(d['size_mb_mean']) + 1)
        ax.plot(x, d['size_mb_mean'], color=s['color'], marker=s['marker'],
                markevery=_mark_every(len(d['size_mb_mean'])), linestyle=s['ls'],
                label=exp['label'])
        ax.fill_between(x, d['size_mb_mean'] - d['size_mb_std'],
                         d['size_mb_mean'] + d['size_mb_std'], color=s['color'], alpha=0.1)

    _set_title(ax, "Trainable Adapters Size Evolution (MB)")
    ax.set_xlabel("Round"); ax.set_ylabel("MB")
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    _save_and_close(fig, output_dir, "06_tamanho_modelo.pdf")


def plot_07_parametros_treinaveis(experiments, output_dir):
    """Evolução da quantidade de parâmetros treináveis do modelo ao longo das rodadas."""
    has_data = any(not np.all(e['server']['params_mean'] == 0) for e in experiments)
    if not has_data:
        print("  ⚠️ Sem dados de parâmetros treináveis, pulando.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, exp in enumerate(experiments):
        d = exp['server']
        if np.all(d['params_mean'] == 0):
            continue
        s = STYLES[i % len(STYLES)]
        x = range(1, len(d['params_mean']) + 1)
        ax.plot(x, d['params_mean'], color=s['color'], marker=s['marker'],
                markevery=_mark_every(len(d['params_mean'])), linestyle=s['ls'],
                label=exp['label'])

    _set_title(ax, "Trainable Parameters per Round")
    ax.set_xlabel("Round"); ax.set_ylabel("Quantity")
    ax.ticklabel_format(style='plain', axis='y')
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    _save_and_close(fig, output_dir, "07_parametros_treinaveis.pdf")


def plot_08_paca_evolucao(experiments, output_dir):
    """Mostra como o valor médio do PaCA (adaptadores por camada) evolui nos clientes."""
    has_data = any(e['server'].get('paca_mean') is not None for e in experiments)
    if not has_data:
        print("  ⚠️ Sem dados de evolução do PaCA, pulando.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, exp in enumerate(experiments):
        d = exp['server']
        if d.get('paca_mean') is None:
            continue
        s = STYLES[i % len(STYLES)]
        # d['paca_mean'] tem shape (Rodadas, Clientes)
        paca_per_round = np.mean(d['paca_mean'], axis=1)
        paca_std_per_round = np.std(d['paca_mean'], axis=1)
        x = range(1, len(paca_per_round) + 1)
        ax.plot(x, paca_per_round, color=s['color'], marker=s['marker'],
                markevery=_mark_every(len(paca_per_round)), linestyle=s['ls'],
                label=exp['label'])
        ax.fill_between(x, paca_per_round - paca_std_per_round, paca_per_round + paca_std_per_round,
                         color=s['color'], alpha=0.1)

    _set_title(ax, "Average Adaptive PaCA Dynamics across Clients")
    ax.set_xlabel("Round"); ax.set_ylabel("PaCA Value")
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    _save_and_close(fig, output_dir, "08_paca_evolucao.pdf")
