#!/usr/bin/env python3
"""
Gera gráficos comparativos entre as 3 fases do PUMA-GT Plus no OxfordPets:
1. Anterior (Heurística de Platô/Latência) - em backup_pumagt_pre_gamma/
2. Anomalia de Medição (Colapso Prematuro do γ) - em backup_pumagt_anomalia_medicao/
3. Corrigido (Medição Real de Rede e Estabilidade) - em results/
"""

import os
import glob
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 16
})

BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'results'))
OUT_DIR = os.path.join(BASE_DIR, 'graficos_comparativos_gamma')
os.makedirs(OUT_DIR, exist_ok=True)

def load_data(h5_path):
    if not h5_path or not os.path.exists(h5_path):
        return None
    with h5py.File(h5_path, 'r') as h:
        return {
            'acc': np.array(h['rs_test_acc']),
            'loss': np.array(h['rs_test_loss']),
            'paca': np.array(h['rs_client_paca']) if 'rs_client_paca' in h else None,
            'rank': np.array(h['rs_client_rank']) if 'rs_client_rank' in h else None,
            'gamma': np.array(h['rs_client_gamma']) if 'rs_client_gamma' in h else None,
            'tpr': np.array(h['total_transmitted_per_round_Mb']) if 'total_transmitted_per_round_Mb' in h else None,
        }

# 1. Simulação Anterior (Heurística)
prev_dir = os.path.join(BASE_DIR, 'backup_pumagt_pre_gamma', 'fl_pumagt_plus_OxfordPets_clip_sora_with_schedule_prune1_ala1_adaptpaca_adaptrank')
prev_files = glob.glob(os.path.join(prev_dir, 'server_*.h5'))
prev_h5 = prev_files[0] if prev_files else None
d_prev = load_data(prev_h5)

# 2. Simulação com Anomalia de Medição
anom_dir = os.path.join(BASE_DIR, 'backup_pumagt_anomalia_medicao')
anom_files = glob.glob(os.path.join(anom_dir, 'server_*.h5'))
anom_h5 = anom_files[0] if anom_files else None
d_anom = load_data(anom_h5)

# 3. Simulação Atual (Corrigida)
curr_dir = os.path.join(BASE_DIR, 'fl_pumagt_plus_OxfordPets_clip_sora_with_schedule_prune1_ala1_adaptpaca_adaptrank')
curr_files = glob.glob(os.path.join(curr_dir, 'server_*.h5'))
curr_h5 = curr_files[0] if curr_files else None
d_curr = load_data(curr_h5)

print(f"[INFO] Anterior:  {os.path.basename(prev_h5) if prev_h5 else 'Não encontrado'}")
print(f"[INFO] Anomalia:  {os.path.basename(anom_h5) if anom_h5 else 'Não encontrado'}")
print(f"[INFO] Corrigido: {os.path.basename(curr_h5) if curr_h5 else 'Não encontrado'}")

fig, axes = plt.subplots(2, 2, figsize=(17, 12))

# Configurações de estilo
styles = {
    'prev': {'color': '#1f77b4', 'ls': '-',  'lw': 2.2, 'label': '1. Heurística Anterior'},
    'anom': {'color': '#d62728', 'ls': ':',  'lw': 2.0, 'label': '2. Com Anomalia de Medição'},
    'curr': {'color': '#2ca02c', 'ls': '--', 'lw': 2.4, 'label': '3. Medição Corrigida (Atual)'},
}

rounds = np.arange(1, 151)

# Painel 1: Acurácia Global de Teste
ax1 = axes[0, 0]
if d_prev:
    ax1.plot(rounds[:len(d_prev['acc'])], d_prev['acc'], color=styles['prev']['color'], 
             ls=styles['prev']['ls'], lw=styles['prev']['lw'],
             label=f"{styles['prev']['label']} (Final: {d_prev['acc'][-1]:.2f}%, Máx: {np.max(d_prev['acc']):.2f}%)")
if d_anom:
    ax1.plot(rounds[:len(d_anom['acc'])], d_anom['acc'], color=styles['anom']['color'], 
             ls=styles['anom']['ls'], lw=styles['anom']['lw'],
             label=f"{styles['anom']['label']} (Final: {d_anom['acc'][-1]:.2f}%, Máx: {np.max(d_anom['acc']):.2f}%)")
if d_curr:
    ax1.plot(rounds[:len(d_curr['acc'])], d_curr['acc'], color=styles['curr']['color'], 
             ls=styles['curr']['ls'], lw=styles['curr']['lw'],
             label=f"{styles['curr']['label']} (Final: {d_curr['acc'][-1]:.2f}%, Máx: {np.max(d_curr['acc']):.2f}%)")
ax1.set_title('Acurácia Global Centralizada de Teste (%)', fontweight='bold')
ax1.set_xlabel('Rodada de Comunicação')
ax1.set_ylabel('Acurácia (%)')
ax1.set_xlim(1, 150); ax1.set_ylim(0, 100)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='lower right', framealpha=0.95)

# Painel 2: Perda Global de Teste
ax2 = axes[0, 1]
if d_prev:
    ax2.plot(rounds[:len(d_prev['loss'])], d_prev['loss'], color=styles['prev']['color'], 
             ls=styles['prev']['ls'], lw=styles['prev']['lw'],
             label=f"{styles['prev']['label']} (Final: {d_prev['loss'][-1]:.3f})")
if d_anom:
    ax2.plot(rounds[:len(d_anom['loss'])], d_anom['loss'], color=styles['anom']['color'], 
             ls=styles['anom']['ls'], lw=styles['anom']['lw'],
             label=f"{styles['anom']['label']} (Final: {d_anom['loss'][-1]:.3f})")
if d_curr:
    ax2.plot(rounds[:len(d_curr['loss'])], d_curr['loss'], color=styles['curr']['color'], 
             ls=styles['curr']['ls'], lw=styles['curr']['lw'],
             label=f"{styles['curr']['label']} (Final: {d_curr['loss'][-1]:.3f})")
ax2.set_title('Perda Global (Test Loss)', fontweight='bold')
ax2.set_xlabel('Rodada de Comunicação')
ax2.set_ylabel('Cross-Entropy Loss')
ax2.set_xlim(1, 150)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right', framealpha=0.95)

# Painel 3: Dinâmica Média do PaCA
ax3 = axes[1, 0]
if d_prev and d_prev['paca'] is not None:
    m = np.mean(d_prev['paca'], axis=1)
    ax3.plot(rounds[:len(m)], m, color=styles['prev']['color'], ls=styles['prev']['ls'], lw=styles['prev']['lw'],
             label=f"{styles['prev']['label']} (Média Final: {m[-1]:.1f})")
if d_anom and d_anom['paca'] is not None:
    m = np.mean(d_anom['paca'], axis=1)
    ax3.plot(rounds[:len(m)], m, color=styles['anom']['color'], ls=styles['anom']['ls'], lw=styles['anom']['lw'],
             label=f"{styles['anom']['label']} (Média Final: {m[-1]:.1f})")
if d_curr and d_curr['paca'] is not None:
    m = np.mean(d_curr['paca'], axis=1)
    ax3.plot(rounds[:len(m)], m, color=styles['curr']['color'], ls=styles['curr']['ls'], lw=styles['curr']['lw'],
             label=f"{styles['curr']['label']} (Média Final: {m[-1]:.1f})")
ax3.set_title('Dinâmica do PaCA Médio (Camadas Treináveis)', fontweight='bold')
ax3.set_xlabel('Rodada de Comunicação')
ax3.set_ylabel('Camadas PaCA (2 a 12)')
ax3.set_xlim(1, 150); ax3.set_ylim(1, 13)
ax3.grid(True, linestyle=':', alpha=0.6)
ax3.legend(loc='center right', framealpha=0.95)

# Painel 4: Dinâmica Média do Rank
ax4 = axes[1, 1]
if d_prev and d_prev['rank'] is not None:
    m = np.mean(d_prev['rank'], axis=1)
    ax4.plot(rounds[:len(m)], m, color=styles['prev']['color'], ls=styles['prev']['ls'], lw=styles['prev']['lw'],
             label=f"{styles['prev']['label']} (Média Final: {m[-1]:.1f})")
if d_anom and d_anom['rank'] is not None:
    m = np.mean(d_anom['rank'], axis=1)
    ax4.plot(rounds[:len(m)], m, color=styles['anom']['color'], ls=styles['anom']['ls'], lw=styles['anom']['lw'],
             label=f"{styles['anom']['label']} (Média Final: {m[-1]:.1f})")
if d_curr and d_curr['rank'] is not None:
    m = np.mean(d_curr['rank'], axis=1)
    ax4.plot(rounds[:len(m)], m, color=styles['curr']['color'], ls=styles['curr']['ls'], lw=styles['curr']['lw'],
             label=f"{styles['curr']['label']} (Média Final: {m[-1]:.1f})")
ax4.set_title('Dinâmica do Rank Médio (SoRA/LoRA)', fontweight='bold')
ax4.set_xlabel('Rodada de Comunicação')
ax4.set_ylabel('Rank (2 a 8)')
ax4.set_xlim(1, 150); ax4.set_ylim(1, 9)
ax4.grid(True, linestyle=':', alpha=0.6)
ax4.legend(loc='center right', framealpha=0.95)

fig.suptitle('Comparativo Evolutivo PUMA-GT Plus (OxfordPets - CLIP ViT-B/16)', fontsize=17, fontweight='bold', y=0.995)
fig.tight_layout()

out_png = os.path.join(OUT_DIR, 'comparativo_completo_pre_pos_gamma.png')
out_pdf = os.path.join(OUT_DIR, 'comparativo_completo_pre_pos_gamma.pdf')
fig.savefig(out_png, dpi=200, bbox_inches='tight')
fig.savefig(out_pdf, bbox_inches='tight')
plt.close(fig)

print(f"[SUCESSO] Gráfico comparativo gerado em:\n  - {out_png}\n  - {out_pdf}")
