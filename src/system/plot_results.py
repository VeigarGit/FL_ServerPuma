#!/usr/bin/env python3
"""
Plot Results - FL_ServerPuma (Comparativo)
Gera gráficos PDF comparando múltiplos experimentos de Aprendizado Federado.

Uso:
    uv run plot_results.py

Gráficos gerados (14 no total):
    01 - Acurácia global do modelo (servidor) vs rodadas
    02 - Loss de treinamento vs rodadas
    03 - Acurácia local vs treino (clientes) — detecção de overfitting
    04 - Banda de rede acumulada vs rodadas
    05 - Banda de rede por rodada individual (upload + download)
    06 - Tamanho dos adaptadores (MB) vs rodadas
    07 - Quantidade de parâmetros treináveis vs rodadas
    08 - Evolução do PaCA adaptativo médio
    09 - Tempo wall-clock por rodada
    10 - Tempo de treinamento local nos clientes
    11 - Tempo de comunicação dos clientes (média móvel)
    12 - Decomposição temporal (barras empilhadas)
    13 - Acurácia vs tempo: (a) tempo real, (b) tempo simulado @100Mbps
    14 - Resumo comparativo (barras: acc, loss, tempo, banda)
    15 - MB acumulados vs tempo real (wall-clock)

Os gráficos são salvos em: ../results/graficos_comparativos/
"""

import os
import sys
import glob
import numpy as np
import h5py
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==============================================================================
# Configurações globais
# ==============================================================================

# Flag para ativar/desativar títulos explicativos nos gráficos.
# True  = títulos aparecem (útil para apresentações e revisões)
# False = títulos ocultos (útil para artigos/papers onde a legenda da figura é suficiente)
SHOW_TITLES = False

plt.rcParams.update({
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
})

# ==============================================================================
# Paleta de cores e estilos para comparação
# ==============================================================================

STYLES = [
    {'color': '#1f77b4', 'marker': 'o',  'ls': '-'},
    {'color': '#d62728', 'marker': 's',  'ls': '--'},
    {'color': '#2ca02c', 'marker': 'D',  'ls': '-.'},
    {'color': '#ff7f0e', 'marker': '^',  'ls': ':'},
    {'color': '#9467bd', 'marker': 'v',  'ls': '-'},
    {'color': '#8c564b', 'marker': 'P',  'ls': '--'},
    {'color': '#e377c2', 'marker': 'X',  'ls': '-.'},
    {'color': '#17becf', 'marker': 'h',  'ls': ':'},
    {'color': '#bcbd22', 'marker': '*',  'ls': '-'},
    {'color': '#7f7f7f', 'marker': 'd',  'ls': '--'},
]


# ==============================================================================
# Funções auxiliares
# ==============================================================================

def _mark_every(n):
    return max(1, n // 10)


def _set_title(ax, title):
    """Aplica título ao eixo apenas se SHOW_TITLES estiver ativado."""
    if SHOW_TITLES:
        ax.set_title(title, fontsize=18, fontweight='bold', pad=12)


def _save_and_close(fig, output_dir, filename):
    """Salva a figura como PDF e fecha."""
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  📊 {path}")


# ==============================================================================
# Funções de carregamento
# ==============================================================================

def parse_experiment(exp_name, results_base="../results"):
    """Analisa um diretório de experimento e retorna seus metadados + dados."""
    results_dir = os.path.join(results_base, exp_name)
    if not os.path.isdir(results_dir):
        print(f"⚠️ Diretório não encontrado: {results_dir}")
        return None

    server_files = sorted(glob.glob(os.path.join(results_dir, "server_*.h5")))
    if not server_files:
        print(f"⚠️ Nenhum arquivo de servidor em: {results_dir}")
        return None

    # Extrair metadados do nome do arquivo
    sample = os.path.basename(server_files[0])
    
    # Exemplo: server_OxfordPets_sora_with_schedule_rank8_paca12_prune_freq1_FedAVG_run1.h5
    # Ou: server_OxfordPets_lora_rank8_paca12_withou_Prune_freq1_FedAVG_run1.h5
    pattern = r"server_(?P<dataset>[^_]+)_(?P<strategy>.+)_rank(?P<rank>\d+)_paca(?P<paca>\d+)_(?P<prune>.*?)_freq(?P<freq>\d+)_(?P<algo>[^_]+)_run\d+\.h5"
    
    match = re.match(pattern, sample)
    if not match:
        print(f"⚠️ Não foi possível extrair metadados do arquivo (padrão regex falhou): {sample}")
        return None

    dataset   = match.group('dataset')
    strategy  = match.group('strategy')
    rank      = int(match.group('rank'))
    paca      = int(match.group('paca'))
    prune_str = match.group('prune')
    freq      = int(match.group('freq'))
    algo      = match.group('algo')

    # Label legível para os gráficos
    if "adalora" in exp_name.lower():
        label = f"AdaLoRA r={rank}"
    elif "sora" in exp_name.lower():
        if "adaptpaca" in exp_name.lower():
            label = f"PUMA-GT r={rank}"
        else:
            label = f"Static SoRa r={rank}"
    elif "lora" in exp_name.lower():
        label = f"LoRA r={rank}"
    else:
        label = f"{strategy.upper()} r={rank} p={paca} ({algo})"
        if "withou" not in prune_str.lower():
            label += " +prune"

    # Carregar dados do servidor
    server_data = _load_server_h5(server_files)

    # Carregar dados dos clientes
    client_pattern = os.path.join(results_dir, f"client_*_{dataset}_{strategy}_rank{rank}_paca*_{algo}_run*.h5")
    client_files = sorted(glob.glob(client_pattern))
    client_data = _load_client_h5(client_files) if client_files else None

    return {
        'exp_name':    exp_name,
        'label':       label,
        'dataset':     dataset,
        'strategy':    strategy,
        'rank':        rank,
        'paca':        paca,
        'prune':       prune_str,
        'freq':        freq,
        'algo':        algo,
        'server':      server_data,
        'client':      client_data,
        'results_dir': results_dir,
    }


def _load_server_h5(files):
    acc, loss, mb, mb_bruto, time_, params, size_, paca = [], [], [], [], [], [], [], []
    client_training_time = []
    client_comm_time = []
    server_pruning_time = []
    client_eval_time = []
    server_processing_time = []
    for f in files:
        with h5py.File(f, 'r') as hf:
            acc.append(np.array(hf['rs_test_acc']))
            loss.append(np.array(hf['rs_train_loss']))
            mb.append(np.array(hf['sended_model_Mb']))
            mb_bruto.append(np.array(hf['Sended_without_quant']))
            time_.append(np.array(hf['Round_time']))
            params.append(np.array(hf.get('Trainable_params', np.zeros_like(np.array(hf['Round_time'])))))
            size_.append(np.array(hf.get('Model_size_per_round_Mb', np.zeros_like(np.array(hf['Round_time'])))))
            
            if 'rs_client_paca' in hf:
                paca.append(np.array(hf['rs_client_paca']))
            else:
                paca.append(None)
                
            if 'rs_client_training_time' in hf:
                client_training_time.append(np.array(hf['rs_client_training_time']))
            else:
                client_training_time.append(None)
                
            if 'rs_client_comm_time' in hf:
                client_comm_time.append(np.array(hf['rs_client_comm_time']))
            else:
                client_comm_time.append(None)
                
            if 'server_pruning_time' in hf:
                server_pruning_time.append(np.array(hf['server_pruning_time']))
            else:
                server_pruning_time.append(None)
            
            if 'rs_client_eval_time' in hf:
                client_eval_time.append(np.array(hf['rs_client_eval_time']))
            else:
                client_eval_time.append(None)
            
            if 'rs_server_processing_time' in hf:
                server_processing_time.append(np.array(hf['rs_server_processing_time']))
            else:
                server_processing_time.append(None)

    paca_mean = None
    if all(p is not None for p in paca):
        paca_mean = np.mean(paca, axis=0)

    valid_client_training_time = [t for t in client_training_time if t is not None]
    client_training_time_mean = None
    if valid_client_training_time:
        min_len = min(len(t) for t in valid_client_training_time)
        valid_client_training_time = [t[:min_len] for t in valid_client_training_time]
        client_training_time_mean = np.mean(valid_client_training_time, axis=0)

    valid_client_comm_time = [t for t in client_comm_time if t is not None]
    client_comm_time_mean = None
    if valid_client_comm_time:
        min_len = min(len(t) for t in valid_client_comm_time)
        valid_client_comm_time = [t[:min_len] for t in valid_client_comm_time]
        client_comm_time_mean = np.mean(valid_client_comm_time, axis=0)

    valid_server_pruning_time = [t for t in server_pruning_time if t is not None]
    server_pruning_time_mean = None
    if valid_server_pruning_time:
        min_len = min(len(t) for t in valid_server_pruning_time)
        valid_server_pruning_time = [t[:min_len] for t in valid_server_pruning_time]
        server_pruning_time_mean = np.mean(valid_server_pruning_time, axis=0)

    valid_client_eval_time = [t for t in client_eval_time if t is not None]
    client_eval_time_mean = None
    if valid_client_eval_time:
        min_len = min(len(t) for t in valid_client_eval_time)
        valid_client_eval_time = [t[:min_len] for t in valid_client_eval_time]
        client_eval_time_mean = np.mean(valid_client_eval_time, axis=0)

    valid_server_processing_time = [t for t in server_processing_time if t is not None]
    server_processing_time_mean = None
    if valid_server_processing_time:
        min_len = min(len(t) for t in valid_server_processing_time)
        valid_server_processing_time = [t[:min_len] for t in valid_server_processing_time]
        server_processing_time_mean = np.mean(valid_server_processing_time, axis=0)

    return {
        'acc_mean': np.mean(acc, axis=0), 'acc_std': np.std(acc, axis=0),
        'loss_mean': np.mean(loss, axis=0), 'loss_std': np.std(loss, axis=0),
        'mb_mean': np.mean(mb, axis=0), 'mb_std': np.std(mb, axis=0),
        'mb_bruto_mean': np.mean(mb_bruto, axis=0),
        'time_mean': np.mean(time_, axis=0), 'time_std': np.std(time_, axis=0),
        'params_mean': np.mean(params, axis=0),
        'size_mb_mean': np.mean(size_, axis=0), 'size_mb_std': np.std(size_, axis=0),
        'paca_mean': paca_mean,
        'client_training_time_mean': client_training_time_mean,
        'client_comm_time_mean': client_comm_time_mean,
        'server_pruning_time_mean': server_pruning_time_mean,
        'client_eval_time_mean': client_eval_time_mean,
        'server_processing_time_mean': server_processing_time_mean,
        'num_runs': len(files),
    }


def _load_client_h5(files):
    g_acc, g_loss, l_acc, t_acc = [], [], [], []
    clients_set = set()
    for f in files:
        clients_set.add(os.path.basename(f).split('_')[1])
        with h5py.File(f, 'r') as hf:
            g_acc.append(np.array(hf['rs_global_acc']))
            g_loss.append(np.array(hf['rs_global_loss']))
            l_acc.append(np.array(hf['rs_local_acc']))
            t_acc.append(np.array(hf['rs_train_acc']))

    return {
        'global_acc_mean': np.mean(g_acc, axis=0), 'global_acc_std': np.std(g_acc, axis=0),
        'global_loss_mean': np.mean(g_loss, axis=0), 'global_loss_std': np.std(g_loss, axis=0),
        'local_acc_mean': np.mean(l_acc, axis=0), 'local_acc_std': np.std(l_acc, axis=0),
        'train_acc_mean': np.mean(t_acc, axis=0), 'train_acc_std': np.std(t_acc, axis=0),
        'num_clients': len(clients_set),
    }


# ==============================================================================
# Funções auxiliares de cálculo (usadas em múltiplos gráficos)
# ==============================================================================

def _compute_mb_per_round(d):
    """Calcula MB trafegados por rodada individual (ida + volta) a partir do acumulado."""
    num_rounds = len(d['time_mean'])
    total_envios = len(d['mb_mean'])
    if num_rounds == 0 or total_envios == 0:
        return None
    clients_per_round = max(1, total_envios // num_rounds)
    indices = [min(total_envios - 1, r * clients_per_round) for r in range(1, num_rounds + 1)]
    mb_accumulated = [d['mb_mean'][idx] for idx in indices]
    mb_discrete = []
    for r in range(len(mb_accumulated)):
        if r == 0:
            mb_round = mb_accumulated[r]
        else:
            mb_round = max(0, mb_accumulated[r] - mb_accumulated[r-1])
        # Multiplicamos por 2: o .h5 salva apenas envio servidor→cliente,
        # mas o cliente devolve os mesmos tensores (upload ≈ download).
        mb_discrete.append(mb_round * 2.0)
    return mb_discrete, mb_accumulated, indices


def _get_train_time_per_round(d):
    """Extrai o tempo médio de treinamento por rodada."""
    t_train = d.get('client_training_time_mean')
    if t_train is not None:
        if t_train.ndim > 1:
            return np.mean(t_train, axis=1)
        return t_train
    # Fallback: estima como 50% do tempo total da rodada
    return d['time_mean'] * 0.5


# ==============================================================================
# 01 - Acurácia Global do Modelo (Servidor)
# ==============================================================================

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


# ==============================================================================
# 02 - Loss de Treinamento
# ==============================================================================

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


# ==============================================================================
# 03 - Acurácia Local vs Treino (Clientes) — Detecção de Overfitting
# ==============================================================================

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


# ==============================================================================
# 04 - Banda de Rede Acumulada por Rodada
# ==============================================================================

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


# ==============================================================================
# 05 - Banda de Rede por Rodada Individual (Upload + Download)
# ==============================================================================

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


# ==============================================================================
# 06 - Tamanho dos Adaptadores (MB)
# ==============================================================================

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


# ==============================================================================
# 07 - Quantidade de Parâmetros Treináveis
# ==============================================================================

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


# ==============================================================================
# 08 - Evolução do PaCA Adaptativo
# ==============================================================================

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


# ==============================================================================
# 09 - Tempo Wall-Clock por Rodada
# ==============================================================================

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


# ==============================================================================
# 10 - Tempo de Treinamento Local nos Clientes
# ==============================================================================

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


# ==============================================================================
# 11 - Tempo de Comunicação dos Clientes (Média Móvel)
# ==============================================================================

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


# ==============================================================================
# 12 - Decomposição Temporal (Barras Empilhadas)
# ==============================================================================

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


# ==============================================================================
# 13 - Acurácia vs Tempo (2 Gráficos separados: tempo real e tempo simulado)
# ==============================================================================

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


# ==============================================================================
# 14 - Resumo Comparativo (Barras)
# ==============================================================================

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


# ==============================================================================
# 15 - MB Acumulados vs Tempo Real (Wall-Clock)
# ==============================================================================

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


# ==============================================================================
# Main
# ==============================================================================

# ==============================================================================
# Experimentos para plotar (Adicione ou remova itens desta lista)
# ==============================================================================
EXPERIMENTOS_PARA_PLOTAR = [
    "lora_rank2",
    "lora_rank8",
    "sora_estatico_rank2",
    "sora_estatico_rank8"
]

def main():
    exp_names = EXPERIMENTOS_PARA_PLOTAR

    if not exp_names:
        print("❌ Nenhum experimento definido na lista EXPERIMENTOS_PARA_PLOTAR.")
        sys.exit(1)


    print(f"{'='*60}")
    print(f"📊 Carregando {len(exp_names)} experimento(s)...")
    print(f"{'='*60}")

    experiments = []
    for name in exp_names:
        print(f"\n→ {name}")
        exp = parse_experiment(name)
        if exp and exp['server']:
            experiments.append(exp)
            sv = exp['server']
            print(f"  ✅ Servidor: {sv['num_runs']} runs, {len(sv['acc_mean'])} rounds")
            if exp['client']:
                print(f"  ✅ Clientes: {exp['client']['num_clients']} clientes")

    if not experiments:
        print("\n❌ Nenhum experimento válido encontrado.")
        sys.exit(1)

    # Diretório de saída
    if len(experiments) == 1:
        output_dir = os.path.join(experiments[0]['results_dir'], "graficos")
    else:
        output_dir = os.path.join("..", "results", "graficos_comparativos")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"📂 Salvando gráficos em: {output_dir}")
    print(f"{'='*60}")
    print(f"📝 Títulos nos gráficos: {'ATIVADOS' if SHOW_TITLES else 'DESATIVADOS'}\n")

    # Gerar todos os gráficos (em ordem sequencial 01-15)
    # --- Desempenho do Modelo ---
    plot_01_acuracia_global(experiments, output_dir)
    plot_02_train_loss(experiments, output_dir)
    plot_03_local_vs_treino(experiments, output_dir)

    # --- Custos de Comunicação ---
    plot_04_banda_acumulada(experiments, output_dir)
    plot_05_banda_por_rodada(experiments, output_dir)

    # --- Estrutura do Modelo ---
    plot_06_tamanho_modelo(experiments, output_dir)
    plot_07_parametros_treinaveis(experiments, output_dir)
    plot_08_paca_evolucao(experiments, output_dir)

    # --- Análise Temporal ---
    plot_09_tempo_por_rodada(experiments, output_dir)
    plot_10_tempo_treinamento_clientes(experiments, output_dir)
    plot_11_tempo_comunicacao_clientes(experiments, output_dir)
    plot_12_decomposicao_temporal(experiments, output_dir)

    # --- Eficiência (Acurácia vs Tempo) ---
    plot_13_acuracia_vs_tempo(experiments, output_dir)

    # --- Resumo ---
    plot_14_resumo_comparativo(experiments, output_dir)

    # --- MB vs Tempo ---
    plot_15_mb_vs_tempo(experiments, output_dir)

    print(f"\n✅ {len(experiments)} experimento(s) plotados com sucesso em: {output_dir}")


if __name__ == "__main__":
    main()