#!/usr/bin/env python3
"""
Gera o gráfico de Acurácia Auxiliar Ponderada dos Clientes vs Rodadas.
Ponderação: w_k = n_k / N, onde n_k é o número de amostras de teste do cliente k (OxfordPets, 22 clientes).
"""

import os
import glob
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil

plt.rcParams.update({
    'font.family': 'sans-serif',
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
})

test_counts = np.array([90, 124, 72, 116, 103, 85, 86, 90, 96, 87, 43, 100, 80, 86, 97, 98, 92, 49, 49, 86, 72, 53])
weights = test_counts / np.sum(test_counts)

experiments = [
    {
        'key': 'OxfordPets__LoRA_Padrao_22clients',
        'label': 'LoRA Padrão (22 clientes)',
        'color': '#1f77b4',
        'marker': 'o',
        'ls': '-'
    },
    {
        'key': 'fl_puma_v2_OxfordPets_clip_sora_with_schedule_prune1_ala1_adaptpaca',
        'label': 'PUMA-GT v2 (PruneFreq=1)',
        'color': '#2ca02c',
        'marker': 'D',
        'ls': '-.'
    },
    {
        'key': 'OxfordPets__PUMA-GT_v2_PruneFreq3_LocalAcc',
        'label': 'PUMA-GT (PruneFreq=3, Platô Local)',
        'color': '#d62728',
        'marker': 's',
        'ls': '--'
    }
]

base_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'results'))
output_dir = os.path.join(base_dir, 'graficos_comparativos')
os.makedirs(output_dir, exist_ok=True)

# -------------------------------------------------------------
# Figura 1: Acurácia Auxiliar Ponderada (Comparativo Direto)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 7))

results_summary = {}

for exp in experiments:
    exp_path = os.path.join(base_dir, exp['key'])
    client_files = sorted(
        glob.glob(os.path.join(exp_path, 'client_*.h5')),
        key=lambda x: int(os.path.basename(x).split('client_')[1].split('_')[0])
    )
    
    server_files = glob.glob(os.path.join(exp_path, 'server_*.h5'))
    with h5py.File(server_files[0], 'r') as h:
        srv_acc = np.array(h['rs_test_acc'])
        aux_simple = np.array(h['rs_aux_simple_avg_acc'])
    
    client_accs = []
    for cf in client_files:
        with h5py.File(cf, 'r') as h:
            client_accs.append(np.array(h['rs_global_acc']))
    client_accs = np.array(client_accs)
    
    weighted_acc = np.sum(client_accs * weights[:, None], axis=0)
    rounds = np.arange(1, len(weighted_acc) + 1)
    
    mark_every = max(1, len(rounds) // 12)
    ax.plot(
        rounds, weighted_acc,
        label=f"{exp['label']} (Final: {weighted_acc[-1]:.2f}%)",
        color=exp['color'],
        linestyle=exp['ls'],
        marker=exp['marker'],
        markevery=mark_every,
        linewidth=2.5,
        markersize=8,
        alpha=0.9
    )
    
    results_summary[exp['label']] = {
        'final_weighted': weighted_acc[-1],
        'max_weighted': np.max(weighted_acc),
        'max_round': np.argmax(weighted_acc) + 1,
        'final_simple': aux_simple[-1],
        'final_server': srv_acc[-1]
    }

ax.set_xlabel('Rodada de Comunicação')
ax.set_ylabel('Acurácia Auxiliar Ponderada (%)')
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xlim(1, 150)
ax.set_ylim(bottom=0, top=100)
ax.legend(loc='lower right', framealpha=0.95)
fig.tight_layout()

out_pdf = os.path.join(output_dir, '03b_acuracia_auxiliar_ponderada.pdf')
out_png = os.path.join(output_dir, '03b_acuracia_auxiliar_ponderada.png')
fig.savefig(out_pdf, dpi=200)
fig.savefig(out_png, dpi=200)
plt.close(fig)

# -------------------------------------------------------------
# Figura 2: Comparativo Dual Panel (Ponderada vs Simples vs Global Centralizada)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7), sharey=True)

# Panel 1: Acurácia Ponderada dos Clientes vs Global Centralizada
for exp in experiments:
    exp_path = os.path.join(base_dir, exp['key'])
    client_files = sorted(
        glob.glob(os.path.join(exp_path, 'client_*.h5')),
        key=lambda x: int(os.path.basename(x).split('client_')[1].split('_')[0])
    )
    server_files = glob.glob(os.path.join(exp_path, 'server_*.h5'))
    with h5py.File(server_files[0], 'r') as h:
        srv_acc = np.array(h['rs_test_acc'])
        aux_simple = np.array(h['rs_aux_simple_avg_acc'])
    client_accs = [np.array(h5py.File(cf, 'r')['rs_global_acc']) for cf in client_files]
    weighted_acc = np.sum(np.array(client_accs) * weights[:, None], axis=0)
    rounds = np.arange(1, len(weighted_acc) + 1)
    mark_every = max(1, len(rounds) // 12)
    
    # Linha contínua: Ponderada
    ax1.plot(rounds, weighted_acc, label=f"{exp['label']} (Ponderada: {weighted_acc[-1]:.1f}%)",
             color=exp['color'], linestyle='-', linewidth=2.5, marker=exp['marker'], markevery=mark_every)
    # Linha pontilhada: Global Servidor
    ax1.plot(rounds, srv_acc, label=f"{exp['label']} (Global Servidor: {srv_acc[-1]:.1f}%)",
             color=exp['color'], linestyle=':', linewidth=1.8, alpha=0.7)

ax1.set_title('Acurácia Ponderada Clientes vs Global Servidor', fontsize=18, fontweight='bold', pad=10)
ax1.set_xlabel('Rodada de Comunicação')
ax1.set_ylabel('Acurácia (%)')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_xlim(1, 150)
ax1.set_ylim(70, 97)
ax1.legend(loc='lower right', fontsize=12, framealpha=0.9)

# Panel 2: Ponderada vs Média Simples (Mostra a diferença da distribuição Non-IID)
for exp in experiments:
    exp_path = os.path.join(base_dir, exp['key'])
    client_files = sorted(
        glob.glob(os.path.join(exp_path, 'client_*.h5')),
        key=lambda x: int(os.path.basename(x).split('client_')[1].split('_')[0])
    )
    server_files = glob.glob(os.path.join(exp_path, 'server_*.h5'))
    with h5py.File(server_files[0], 'r') as h:
        aux_simple = np.array(h['rs_aux_simple_avg_acc'])
    client_accs = [np.array(h5py.File(cf, 'r')['rs_global_acc']) for cf in client_files]
    weighted_acc = np.sum(np.array(client_accs) * weights[:, None], axis=0)
    rounds = np.arange(1, len(weighted_acc) + 1)
    mark_every = max(1, len(rounds) // 12)
    
    ax2.plot(rounds, weighted_acc, label=f"{exp['label']} (Ponderada: {weighted_acc[-1]:.1f}%)",
             color=exp['color'], linestyle='-', linewidth=2.5, marker=exp['marker'], markevery=mark_every)
    ax2.plot(rounds, aux_simple, label=f"{exp['label']} (Média Simples: {aux_simple[-1]:.1f}%)",
             color=exp['color'], linestyle='--', linewidth=1.8, alpha=0.7)

ax2.set_title('Acurácia Ponderada vs Média Simples dos Clientes', fontsize=18, fontweight='bold', pad=10)
ax2.set_xlabel('Rodada de Comunicação')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.set_xlim(1, 150)
ax2.legend(loc='lower right', fontsize=12, framealpha=0.9)

fig.tight_layout()
out_dual_pdf = os.path.join(output_dir, '03c_acuracia_ponderada_comparativo_detalhado.pdf')
out_dual_png = os.path.join(output_dir, '03c_acuracia_ponderada_comparativo_detalhado.png')
fig.savefig(out_dual_pdf, dpi=200)
fig.savefig(out_dual_png, dpi=200)
plt.close(fig)

# Copiar para artifacts
artifact_dir = '/home/rafael.teixeira.silva/.gemini/antigravity-ide/brain/2eec4104-d681-4c94-9129-6300a2ac895e'
shutil.copy(out_png, os.path.join(artifact_dir, '03b_acuracia_auxiliar_ponderada.png'))
shutil.copy(out_dual_png, os.path.join(artifact_dir, '03c_acuracia_ponderada_comparativo_detalhado.png'))

print("Sucesso! Gráficos gerados:")
print(f"1) {out_png}")
print(f"2) {out_dual_png}")
for k, v in results_summary.items():
    print(f"{k}: Final Ponderada={v['final_weighted']:.2f}%, Max Ponderada={v['max_weighted']:.2f}% (R{v['max_round']}), Simples={v['final_simple']:.2f}%, Global={v['final_server']:.2f}%")
