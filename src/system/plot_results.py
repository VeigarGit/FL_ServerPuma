#!/usr/bin/env python3
"""
Plot Results - FL_ServerPuma (Comparativo)
Gera gráficos PDF comparando múltiplos experimentos.

Uso:
    uv run plot_results.py <EXP_1> <EXP_2> ... <EXP_N>

Exemplo:
    uv run plot_results.py \
        david_clip_lora_prune1_ala1_20260714_134821 \
        david_clip_sora_prune0_ala0_20260715_100000

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
    if "sora" in exp_name.lower():
        if "adaptpaca" in exp_name.lower():
            label = "SORA (Com Pruning e PACA Adaptativo)"
        else:
            label = "SORA (Com Pruning e Sem PACA Adaptativo)"
    elif "lora" in exp_name.lower():
        label = "LoRA"
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

    paca_mean = None
    if all(p is not None for p in paca):
        paca_mean = np.mean(paca, axis=0)

    return {
        'acc_mean': np.mean(acc, axis=0), 'acc_std': np.std(acc, axis=0),
        'loss_mean': np.mean(loss, axis=0), 'loss_std': np.std(loss, axis=0),
        'mb_mean': np.mean(mb, axis=0), 'mb_std': np.std(mb, axis=0),
        'mb_bruto_mean': np.mean(mb_bruto, axis=0),
        'time_mean': np.mean(time_, axis=0), 'time_std': np.std(time_, axis=0),
        'params_mean': np.mean(params, axis=0),
        'size_mb_mean': np.mean(size_, axis=0), 'size_mb_std': np.std(size_, axis=0),
        'paca_mean': paca_mean,
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
# Funções de plotagem comparativa
# ==============================================================================

def _mark_every(n):
    return max(1, n // 10)


def plot_comparative_accuracy(experiments, output_dir):
    """Acurácia Global do Servidor - comparativo."""
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

    ax.set_title("Acurácia Global do Servidor")
    ax.set_xlabel("Rodadas"); ax.set_ylabel("Acurácia (%)")
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    path = os.path.join(output_dir, "01_acuracia_global.pdf")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  📊 {path}")


def plot_comparative_loss(experiments, output_dir):
    """Loss de Treinamento - comparativo."""
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

    ax.set_title("Loss de Treinamento")
    ax.set_xlabel("Rodadas"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    path = os.path.join(output_dir, "02_train_loss.pdf")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  📊 {path}")


def plot_comparative_comm_cost(experiments, output_dir):
    """Custo de Comunicação Acumulado - comparativo."""
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, exp in enumerate(experiments):
        s = STYLES[i % len(STYLES)]
        d = exp['server']
        x = range(len(d['mb_mean']))
        ax.plot(x, d['mb_bruto_mean'], color=s['color'], linestyle=':', alpha=0.4)
        ax.plot(x, d['mb_mean'], color=s['color'], marker=s['marker'],
                markevery=_mark_every(len(d['mb_mean'])), linestyle=s['ls'],
                label=f"{exp['label']} (trafegado)")
        ax.fill_between(x, d['mb_mean'] - d['mb_std'], d['mb_mean'] + d['mb_std'],
                         color=s['color'], alpha=0.1)

    ax.set_title("Custo de Rede Acumulado")
    ax.set_xlabel("Envios (Clientes × Rodadas)"); ax.set_ylabel("MB Acumulados")
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    path = os.path.join(output_dir, "03_custo_comunicacao.pdf")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  📊 {path}")


def plot_comparative_round_time(experiments, output_dir):
    """Tempo por Rodada - comparativo."""
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

    ax.set_title("Tempo de Execução por Rodada")
    ax.set_xlabel("Rodadas"); ax.set_ylabel("Tempo (s)")
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    path = os.path.join(output_dir, "04_tempo_por_rodada.pdf")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  📊 {path}")


def plot_comparative_model_size(experiments, output_dir):
    """Tamanho do Modelo por Rodada - comparativo."""
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

    ax.set_title("Tamanho dos Adaptadores por Rodada")
    ax.set_xlabel("Rodada"); ax.set_ylabel("MB")
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    path = os.path.join(output_dir, "05_tamanho_modelo.pdf")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  📊 {path}")


def plot_comparative_paca_evolution(experiments, output_dir):
    """Evolução do PaCA no Adaptativo - comparativo."""
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
        # Vamos calcular a média de PaCA por rodada (média entre todos os clientes selecionados)
        paca_per_round = np.mean(d['paca_mean'], axis=1)
        paca_std_per_round = np.std(d['paca_mean'], axis=1)
        
        x = range(1, len(paca_per_round) + 1)
        ax.plot(x, paca_per_round, color=s['color'], marker=s['marker'],
                markevery=_mark_every(len(paca_per_round)), linestyle=s['ls'],
                label=exp['label'])
        ax.fill_between(x, paca_per_round - paca_std_per_round, paca_per_round + paca_std_per_round,
                         color=s['color'], alpha=0.1)

    ax.set_title("Evolução do PaCA Médio por Rodada (Adaptativo)")
    ax.set_xlabel("Rodada"); ax.set_ylabel("Valor do PaCA")
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    path = os.path.join(output_dir, "06_paca_evolucao.pdf")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  📊 {path}")


def plot_comparative_comm_cost_per_round(experiments, output_dir):
    """Custo de Comunicação Acumulado por Rodada - comparativo."""
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, exp in enumerate(experiments):
        s = STYLES[i % len(STYLES)]
        d = exp['server']
        
        # d['mb_mean'] tem um registro por cliente (acumulado globalmente)
        # d['time_mean'] tem um registro por rodada
        # Para extrair o valor acumulado no final de cada rodada:
        num_rounds = len(d['time_mean'])
        total_envios = len(d['mb_mean'])
        
        if num_rounds == 0 or total_envios == 0:
            continue
            
        # Estimamos clientes por rodada:
        clients_per_round = max(1, total_envios // num_rounds)
        
        # Coleta o acumulado no final de cada rodada
        indices = [min(total_envios - 1, (r * clients_per_round) - 1) for r in range(1, num_rounds + 1)]
        mb_per_round = [d['mb_mean'][idx] for idx in indices]
        
        x = range(1, num_rounds + 1)
        ax.plot(x, mb_per_round, color=s['color'], marker=s['marker'],
                markevery=_mark_every(len(mb_per_round)), linestyle=s['ls'],
                label=exp['label'])

    ax.set_title("Banda Acumulada por Rodada")
    ax.set_xlabel("Rodadas"); ax.set_ylabel("MB Acumulados")
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    path = os.path.join(output_dir, "07_banda_acumulada_rodada.pdf")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  📊 {path}")


def plot_comparative_discrete_comm_cost_per_round(experiments, output_dir):
    """Custo de Comunicação por Rodada (Não Acumulado) - comparativo."""
    fig, ax = plt.subplots(figsize=(12, 7))
    has_valid_data = False
    
    for i, exp in enumerate(experiments):
        s = STYLES[i % len(STYLES)]
        d = exp['server']
        
        num_rounds = len(d['time_mean'])
        total_envios = len(d['mb_mean'])
        
        if num_rounds == 0 or total_envios == 0:
            continue
            
        clients_per_round = max(1, total_envios // num_rounds)
        
        # Coleta o acumulado no final de cada rodada
        # Como o array inicia com 0 (len = 501 para 50 rounds e 10 clientes), 
        # o fim do round 1 está no índice 10.
        indices = [min(total_envios - 1, r * clients_per_round) for r in range(1, num_rounds + 1)]
        mb_accumulated = [d['mb_mean'][idx] for idx in indices]
        
        # Calcula a diferença para obter o custo de cada rodada individual (ida e volta = x 2)
        mb_discrete = []
        for r in range(len(mb_accumulated)):
            if r == 0:
                mb_round = mb_accumulated[r]
            else:
                mb_round = max(0, mb_accumulated[r] - mb_accumulated[r-1])
            
            # Multiplicamos por 2 porque o .h5 só salva o envio do servidor para o cliente.
            # Como o cliente devolve exatamente os mesmos tensores de volta, o upload é igual ao download.
            mb_discrete.append(mb_round * 2.0)
                
        x = range(1, num_rounds + 1)
        ax.plot(x, mb_discrete, color=s['color'], marker=s['marker'],
                markevery=_mark_every(len(mb_discrete)), linestyle=s['ls'],
                label=exp['label'])
        has_valid_data = True

    if not has_valid_data:
        plt.close(fig)
        return

    ax.set_title("Banda Trafegada por Rodada (Upload + Download)")
    ax.set_xlabel("Rodadas"); ax.set_ylabel("MB Trafegados na Rodada")
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    path = os.path.join(output_dir, "09_banda_por_rodada.pdf")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  📊 {path}")



def plot_comparative_params(experiments, output_dir):
    """Parâmetros Treináveis - comparativo."""
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

    ax.set_title("Parâmetros Treináveis no Servidor")
    ax.set_xlabel("Rodada"); ax.set_ylabel("Quantidade")
    ax.ticklabel_format(style='plain', axis='y')
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    path = os.path.join(output_dir, "06_parametros_treinaveis.pdf")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  📊 {path}")


def plot_comparative_client_global_acc(experiments, output_dir):
    """Acurácia Global (Clientes) - comparativo."""
    has_data = any(e['client'] is not None for e in experiments)
    if not has_data:
        print("  ⚠️ Sem dados de clientes, pulando.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, exp in enumerate(experiments):
        if exp['client'] is None:
            continue
        s = STYLES[i % len(STYLES)]
        d = exp['client']
        x = range(1, len(d['global_acc_mean']) + 1)
        ax.plot(x, d['global_acc_mean'], color=s['color'], marker=s['marker'],
                markevery=_mark_every(len(d['global_acc_mean'])), linestyle=s['ls'],
                label=exp['label'])
        ax.fill_between(x, d['global_acc_mean'] - d['global_acc_std'],
                         d['global_acc_mean'] + d['global_acc_std'], color=s['color'], alpha=0.1)

    ax.set_title("Acurácia Global (Vista pelos Clientes)")
    ax.set_xlabel("Rodadas"); ax.set_ylabel("Acurácia (%)")
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    path = os.path.join(output_dir, "07_acuracia_global_clientes.pdf")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  📊 {path}")


def plot_comparative_client_local_vs_train(experiments, output_dir):
    """Acurácia Local vs Treino (Clientes) - comparativo."""
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
                label=f"{exp['label']} (local)")
        ax.plot(x, d['train_acc_mean'], color=s['color'], marker=s['marker'],
                markevery=_mark_every(len(d['train_acc_mean'])), linestyle=':',
                alpha=0.6, label=f"{exp['label']} (treino)")

    ax.set_title("Efeito Overfitting: Acurácia de Treino vs Teste Local (Clientes)")
    ax.set_xlabel("Rodadas"); ax.set_ylabel("Acurácia (%)")
    ax.legend(fontsize=9); ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    path = os.path.join(output_dir, "08_local_vs_treino_clientes.pdf")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  📊 {path}")


def plot_comparative_client_loss(experiments, output_dir):
    """Loss Global (Clientes) - comparativo."""
    has_data = any(e['client'] is not None for e in experiments)
    if not has_data:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, exp in enumerate(experiments):
        if exp['client'] is None:
            continue
        s = STYLES[i % len(STYLES)]
        d = exp['client']
        x = range(1, len(d['global_loss_mean']) + 1)
        ax.plot(x, d['global_loss_mean'], color=s['color'], marker=s['marker'],
                markevery=_mark_every(len(d['global_loss_mean'])), linestyle=s['ls'],
                label=exp['label'])
        ax.fill_between(x, d['global_loss_mean'] - d['global_loss_std'],
                         d['global_loss_mean'] + d['global_loss_std'], color=s['color'], alpha=0.1)

    ax.set_title("Loss Global (Vista pelos Clientes)")
    ax.set_xlabel("Rodadas"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    path = os.path.join(output_dir, "09_loss_global_clientes.pdf")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  📊 {path}")


def plot_comparative_summary_bar(experiments, output_dir):
    """Gráfico de barras comparativo: métricas finais de cada experimento."""
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
    ax.set_title("Acurácia Final (%)"); ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    for bar, val in zip(bars, final_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}', ha='center', fontsize=9)
    ax.grid(True, axis='y', ls='--', alpha=0.4)

    # Loss final
    ax = axes[0, 1]
    bars = ax.bar(x, final_loss, width, color=[STYLES[i % len(STYLES)]['color'] for i in range(len(experiments))])
    ax.set_title("Loss Final"); ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    for bar, val in zip(bars, final_loss):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01, f'{val:.3f}', ha='center', fontsize=9)
    ax.grid(True, axis='y', ls='--', alpha=0.4)

    # Tempo médio por rodada
    ax = axes[1, 0]
    bars = ax.bar(x, avg_time, width, color=[STYLES[i % len(STYLES)]['color'] for i in range(len(experiments))])
    ax.set_title("Tempo Médio por Rodada (s)"); ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    for bar, val in zip(bars, avg_time):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.1f}', ha='center', fontsize=9)
    ax.grid(True, axis='y', ls='--', alpha=0.4)

    # Total MB trafegado
    ax = axes[1, 1]
    bars = ax.bar(x, total_mb, width, color=[STYLES[i % len(STYLES)]['color'] for i in range(len(experiments))])
    ax.set_title("Total MB Trafegados"); ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    for bar, val in zip(bars, total_mb):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01, f'{val:.1f}', ha='center', fontsize=9)
    ax.grid(True, axis='y', ls='--', alpha=0.4)

    fig.suptitle("Resumo Comparativo das Execuções", fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, "10_resumo_comparativo.pdf")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  📊 {path}")


# ==============================================================================
# Main
# ==============================================================================

# ==============================================================================
# Experimentos para plotar (Adicione ou remova itens desta lista)
# ==============================================================================
EXPERIMENTOS_PARA_PLOTAR = [
    "david_clip_lora_prune1_ala1_20260714_134821",
    "david_clip_sora_with_schedule_prune0_ala1_20260714_151603",
    "david_clip_sora_with_schedule_prune0_ala1_adaptpaca_20260714_163600"
    # Adicione outras pastas de experimentos abaixo, por exemplo:
    # "david_clip_sora_prune0_ala0_20260715_100000",
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
    print(f"{'='*60}\n")

    # Gerar todos os gráficos
    plot_comparative_accuracy(experiments, output_dir)
    plot_comparative_loss(experiments, output_dir)
    plot_comparative_comm_cost(experiments, output_dir)
    plot_comparative_round_time(experiments, output_dir)
    plot_comparative_model_size(experiments, output_dir)
    plot_comparative_params(experiments, output_dir)
    plot_comparative_paca_evolution(experiments, output_dir)
    plot_comparative_comm_cost_per_round(experiments, output_dir)
    plot_comparative_client_global_acc(experiments, output_dir)
    plot_comparative_client_local_vs_train(experiments, output_dir)
    plot_comparative_client_loss(experiments, output_dir)
    plot_comparative_summary_bar(experiments, output_dir)
    plot_comparative_discrete_comm_cost_per_round(experiments, output_dir)

    print(f"\n✅ {len(experiments)} experimento(s) plotados com sucesso em: {output_dir}")


if __name__ == "__main__":
    main()
