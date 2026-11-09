# Responsabilidade: Gera gráficos de acurácia global, perda de treinamento e comparação entre
# acurácia local de teste e de treino.
#
# Principais funções: plot_01_acuracia_global, plot_02_train_loss, plot_03_local_vs_treino.

"""Experiment plotting — accuracy_plots."""

import matplotlib.pyplot as plt
from fl_serverpuma.experiments.analysis.plot_style import _mark_every
from fl_serverpuma.experiments.analysis.plot_style import _set_title
from fl_serverpuma.experiments.analysis.plot_style import _save_and_close
from fl_serverpuma.experiments.analysis.plot_style import STYLES

def plot_01_acuracia_global(experiments, output_dir):
    """Evolução da acurácia global do modelo avaliada pelo servidor ao longo das rodadas."""
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, exp in enumerate(experiments):
        s = STYLES[i % len(STYLES)]
        d = exp['server']
        x = range(1, len(d['acc_mean']) + 1)
        ax.plot(x, d['acc_mean'], color=s['color'], marker=s['marker'],
                markevery=_mark_every(len(d['acc_mean'])), linestyle=s['ls'],
                label=exp['label'])
        ax.fill_between(x, d['acc_mean'] - d['acc_std'], d['acc_mean'] + d['acc_std'],
                         color=s['color'], alpha=0.1)

    _set_title(ax, "Global Model Accuracy Evolution (Server)")
    ax.set_xlabel("Rounds"); ax.set_ylabel("Accuracy (%)")
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    _save_and_close(fig, output_dir, "01_acuracia_global.pdf")


def plot_02_train_loss(experiments, output_dir):
    """Evolução da função de custo (loss) de treinamento ao longo das rodadas."""
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, exp in enumerate(experiments):
        s = STYLES[i % len(STYLES)]
        d = exp['server']
        x = range(1, len(d['loss_mean']) + 1)
        ax.plot(x, d['loss_mean'], color=s['color'], marker=s['marker'],
                markevery=_mark_every(len(d['loss_mean'])), linestyle=s['ls'],
                label=exp['label'])
        ax.fill_between(x, d['loss_mean'] - d['loss_std'], d['loss_mean'] + d['loss_std'],
                         color=s['color'], alpha=0.1)

    _set_title(ax, "Training Loss Evolution")
    ax.set_xlabel("Rounds"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    _save_and_close(fig, output_dir, "02_train_loss.pdf")


def plot_03_local_vs_treino(experiments, output_dir):
    """Compara acurácia de teste local e de treino nos clientes para detectar overfitting."""
    has_data = any(e['client'] is not None for e in experiments)
    if not has_data:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, exp in enumerate(experiments):
        if exp['client'] is None:
            continue
        s = STYLES[i % len(STYLES)]
        d = exp['client']
        x = range(1, len(d['local_acc_mean']) + 1)
        ax.plot(x, d['local_acc_mean'], color=s['color'], marker=s['marker'],
                markevery=_mark_every(len(d['local_acc_mean'])), linestyle=s['ls'],
                label=f"{exp['label']} (local test)")
        ax.plot(x, d['train_acc_mean'], color=s['color'], marker=s['marker'],
                markevery=_mark_every(len(d['train_acc_mean'])), linestyle=':',
                alpha=0.6, label=f"{exp['label']} (train)")

    _set_title(ax, "Overfitting Analysis: Train vs. Local Test Accuracy")
    ax.set_xlabel("Rounds"); ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=9); ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    _save_and_close(fig, output_dir, "03_acuracia_local_vs_treino.pdf")
