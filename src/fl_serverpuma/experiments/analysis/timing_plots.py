# Responsabilidade: Gera gráficos de duração das rodadas, treino, comunicação, processamento e
# avaliação.
#
# Principais funções: plot_09_tempo_por_rodada, plot_10_tempo_treinamento_clientes,
# plot_11_tempo_comunicacao_clientes, plot_12_decomposicao_temporal.

"""Experiment plotting — timing_plots."""

import numpy as np
import matplotlib.pyplot as plt
from fl_serverpuma.experiments.analysis.plot_style import _mark_every
from fl_serverpuma.experiments.analysis.plot_style import _set_title
from fl_serverpuma.experiments.analysis.plot_style import _save_and_close
from fl_serverpuma.experiments.analysis.plot_style import STYLES

def plot_09_tempo_por_rodada(experiments, output_dir):
    """Tempo real de execução (wall-clock) de cada rodada de comunicação federada."""
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, exp in enumerate(experiments):
        s = STYLES[i % len(STYLES)]
        d = exp['server']
        x = range(1, len(d['time_mean']) + 1)
        ax.plot(x, d['time_mean'], color=s['color'], marker=s['marker'],
                markevery=_mark_every(len(d['time_mean'])), linestyle=s['ls'],
                label=f"{exp['label']} (μ={np.mean(d['time_mean']):.1f}s)")
        ax.fill_between(x, d['time_mean'] - d['time_std'], d['time_mean'] + d['time_std'],
                         color=s['color'], alpha=0.1)

    _set_title(ax, "Execution Time per Communication Round")
    ax.set_xlabel("Rounds"); ax.set_ylabel("Time (s)")
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    _save_and_close(fig, output_dir, "09_tempo_por_rodada.pdf")


def plot_10_tempo_treinamento_clientes(experiments, output_dir):
    """Tempo médio gasto pelos clientes no treinamento local por rodada."""
    has_data = any(e['server'].get('client_training_time_mean') is not None for e in experiments)
    if not has_data:
        print("  ⚠️ Sem dados de tempo de treinamento dos clientes, pulando.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    has_valid_data = False
    for i, exp in enumerate(experiments):
        d = exp['server']
        t_mean = d.get('client_training_time_mean')
        if t_mean is None:
            continue
        has_valid_data = True
        s = STYLES[i % len(STYLES)]

        if t_mean.ndim > 1:
            time_per_round = np.mean(t_mean, axis=1)
            time_std_per_round = np.std(t_mean, axis=1)
        else:
            time_per_round = t_mean
            time_std_per_round = np.zeros_like(t_mean)

        x = range(1, len(time_per_round) + 1)
        ax.plot(x, time_per_round, color=s['color'], marker=s['marker'],
                markevery=_mark_every(len(time_per_round)), linestyle=s['ls'],
                label=f"{exp['label']} (μ={np.mean(time_per_round):.1f}s)")

        if np.any(time_std_per_round > 0):
            ax.fill_between(x, time_per_round - time_std_per_round, time_per_round + time_std_per_round,
                             color=s['color'], alpha=0.1)

    if not has_valid_data:
        plt.close(fig)
        return

    _set_title(ax, "Average Local Training Time across Clients")
    ax.set_xlabel("Rounds"); ax.set_ylabel("Time (s)")
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    _save_and_close(fig, output_dir, "10_tempo_treinamento_clientes.pdf")


def plot_11_tempo_comunicacao_clientes(experiments, output_dir):
    """Tempo médio de comunicação de rede dos clientes, suavizado com média móvel de 10 rodadas."""
    import pandas as pd

    has_data = any(e['server'].get('client_comm_time_mean') is not None for e in experiments)
    if not has_data:
        print("  ⚠️ Sem dados de tempo de comunicação dos clientes, pulando.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    has_valid_data = False
    for i, exp in enumerate(experiments):
        d = exp['server']
        t_mean = d.get('client_comm_time_mean')
        if t_mean is None:
            continue
        has_valid_data = True
        s = STYLES[i % len(STYLES)]

        if t_mean.ndim > 1:
            time_per_round = np.mean(t_mean, axis=1)
            time_std_per_round = np.std(t_mean, axis=1)
        else:
            time_per_round = t_mean
            time_std_per_round = np.zeros_like(t_mean)

        # Aplica média móvel para estabilizar o gráfico
        time_per_round = pd.Series(time_per_round).rolling(window=10, min_periods=1).mean().values

        x = range(1, len(time_per_round) + 1)
        ax.plot(x, time_per_round, color=s['color'], marker=s['marker'],
                markevery=_mark_every(len(time_per_round)), linestyle=s['ls'],
                label=f"{exp['label']} (μ={np.mean(time_per_round):.1f}s)")

        if np.any(time_std_per_round > 0):
            ax.fill_between(x, time_per_round - time_std_per_round, time_per_round + time_std_per_round,
                             color=s['color'], alpha=0.1)

    if not has_valid_data:
        plt.close(fig)
        return

    _set_title(ax, "Average Client Communication Time (Moving Average=10)")
    ax.set_xlabel("Rounds"); ax.set_ylabel("Time (s)")
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    _save_and_close(fig, output_dir, "11_tempo_comunicacao_clientes.pdf")


def plot_12_decomposicao_temporal(experiments, output_dir):
    """Mostra onde cada estratégia gasta tempo: treino, rede, eval, servidor, overhead."""
    labels = [e['label'] for e in experiments]

    training_times = []
    comm_times = []
    eval_times = []
    server_proc_times = []
    other_times = []

    for exp in experiments:
        d = exp['server']
        avg_round = np.mean(d['time_mean'])

        # Treinamento
        t_train = d.get('client_training_time_mean')
        if t_train is not None:
            if t_train.ndim > 1:
                avg_train = np.mean(np.mean(t_train, axis=1))
            else:
                avg_train = np.mean(t_train)
        else:
            avg_train = 0.0

        # Comunicação de rede
        t_comm = d.get('client_comm_time_mean')
        if t_comm is not None:
            if t_comm.ndim > 1:
                avg_comm = np.mean(np.mean(t_comm, axis=1))
            else:
                avg_comm = np.mean(t_comm)
        else:
            avg_comm = 0.0

        # Processamento do cliente (eval + quant + dequant)
        t_eval = d.get('client_eval_time_mean')
        if t_eval is not None:
            if t_eval.ndim > 1:
                avg_eval = np.mean(np.mean(t_eval, axis=1))
            else:
                avg_eval = np.mean(t_eval)
        else:
            avg_eval = 0.0

        # Processamento do servidor (quant + dequant)
        t_sproc = d.get('server_processing_time_mean')
        if t_sproc is not None:
            if t_sproc.ndim > 1:
                avg_sproc = np.mean(np.mean(t_sproc, axis=1))
            else:
                avg_sproc = np.mean(t_sproc)
        else:
            avg_sproc = 0.0

        # Overhead residual (agregação, sleep, etc.)
        avg_other = max(0.0, avg_round - avg_train - avg_comm - avg_eval - avg_sproc)

        training_times.append(avg_train)
        comm_times.append(avg_comm)
        eval_times.append(avg_eval)
        server_proc_times.append(avg_sproc)
        other_times.append(avg_other)

    x = np.arange(len(labels))
    width = 0.5

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.bar(x, training_times, width, label='Local Training', color='#2196F3')
    ax.bar(x, comm_times, width, bottom=training_times, label='Network Communication', color='#FF9800')

    bottom2 = [t + c for t, c in zip(training_times, comm_times)]
    ax.bar(x, eval_times, width, bottom=bottom2, label='Client Processing (eval+quant)', color='#4CAF50')

    bottom3 = [b + e for b, e in zip(bottom2, eval_times)]
    ax.bar(x, server_proc_times, width, bottom=bottom3, label='Server Processing (quant/dequant)', color='#9C27B0')

    bottom4 = [b + s for b, s in zip(bottom3, server_proc_times)]
    ax.bar(x, other_times, width, bottom=bottom4, label='Overhead (agg+misc)', color='#607D8B')

    ax.set_xlabel('Strategy')
    ax.set_ylabel('Average Time per Round (s)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, axis='y', ls='--', alpha=0.4)

    # Adiciona valores totais no topo de cada barra
    for i in range(len(labels)):
        total = training_times[i] + comm_times[i] + eval_times[i] + server_proc_times[i] + other_times[i]
        ax.text(x[i], total + 0.2, f'{total:.1f}s', ha='center', fontsize=11, fontweight='bold')

    _set_title(ax, "Average Round Time Breakdown")
    fig.tight_layout()
    _save_and_close(fig, output_dir, "12_decomposicao_temporal.pdf")
