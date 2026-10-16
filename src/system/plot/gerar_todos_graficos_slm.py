#!/usr/bin/env python3
"""
gerar_todos_graficos_slm.py
===========================
Gera automaticamente a suíte completa de gráficos para os 4 datasets (SLM)
comparando PUMA-GT Plus vs LoRA Padrão:
  1. OxfordPets
  2. DTD
  3. FGVCAircraft
  4. Flowers102
  5. Gráficos Consolidados Multi-Dataset (4 Painéis: Acurácia Auxiliar Ponderada, Banda e Tempo)
"""

import os
import sys
import subprocess
import glob
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_RESULTS = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "results"))
DATASET_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "dataset"))

DATASETS = ["OxfordPets", "DTD", "FGVCAircraft", "Flowers102"]

DATASET_CONFIGS = {
    "OxfordPets": {
        "puma": "fl_pumagt_plus_slm_OxfordPets_slm_sora_with_schedule_prune1_ala1_adaptpaca_adaptrank",
        "lora": "fl_lora_slm_OxfordPets_slm_lora_prune1_ala1_paca28",
        "title": "Oxford-IIIT Pet (37 classes)"
    },
    "DTD": {
        "puma": "fl_pumagt_plus_slm_DTD_slm_sora_with_schedule_prune1_ala1_adaptpaca_adaptrank",
        "lora": "fl_lora_slm_DTD_slm_lora_prune1_ala1_paca28",
        "title": "Describable Textures (DTD - 47 classes)"
    },
    "FGVCAircraft": {
        "puma": "fl_pumagt_plus_slm_FGVCAircraft_slm_sora_with_schedule_prune1_ala1_adaptpaca_adaptrank",
        "lora": "fl_lora_slm_FGVCAircraft_slm_lora_prune1_ala1_paca28",
        "title": "FGVC Aircraft (100 classes)"
    },
    "Flowers102": {
        "puma": "fl_pumagt_plus_slm_Flowers102_slm_sora_with_schedule_prune1_ala1_adaptpaca_adaptrank",
        "lora": "fl_lora_slm_Flowers102_slm_lora_prune1_ala1_paca28",
        "title": "Oxford Flowers-102 (102 classes)"
    }
}

def run_dataset_plots():
    plot_script = os.path.join(SCRIPT_DIR, "plot_results.py")
    python_bin = sys.executable

    for ds in DATASETS:
        cfg = DATASET_CONFIGS[ds]
        puma_dir = os.path.join(BASE_RESULTS, cfg["puma"])
        lora_dir = os.path.join(BASE_RESULTS, cfg["lora"])

        if not os.path.exists(puma_dir) or not os.path.exists(lora_dir):
            print(f"⚠️ Pastas incompletas para {ds}. Pulando.")
            continue

        out_dir = os.path.join(BASE_RESULTS, f"graficos_{ds}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n{'='*70}")
        print(f"🎨 Gerando 15 gráficos comparativos para: {ds}")
        print(f"📂 Diretório: {out_dir}")
        print(f"{'='*70}")

        # Chama plot_results.py passando os dois experimentos
        cmd = [python_bin, plot_script, cfg["lora"], cfg["puma"]]
        env = os.environ.copy()
        subprocess.run(cmd, cwd=os.path.join(SCRIPT_DIR, ".."), check=False, env=env)

        # Move/copia os gráficos de graficos_comparativos para a pasta específica
        default_out = os.path.join(BASE_RESULTS, "graficos_comparativos")
        if os.path.exists(default_out):
            for f in glob.glob(os.path.join(default_out, "*.*")):
                dest = os.path.join(out_dir, os.path.basename(f))
                import shutil
                shutil.copy2(f, dest)

def get_weighted_acc_series(exp_dir, dataset_name):
    test_dir = os.path.join(DATASET_DIR, dataset_name, "test")
    counts = []
    for cid in range(12):
        npz_path = os.path.join(test_dir, f"{cid}.npz")
        d = np.load(npz_path, allow_pickle=True)
        counts.append(len(d['data'].item()['y']))
    weights = np.array(counts, dtype=float) / sum(counts)

    runs_w_series = []
    for r in range(1, 11):
        c_accs = []
        for cid in range(12):
            cf = glob.glob(os.path.join(exp_dir, f"client_{cid}_*_run{r}_*.h5"))
            if not cf: break
            with h5py.File(cf[0], 'r') as h:
                c_accs.append(np.array(h['rs_global_acc']))
        if len(c_accs) == 12:
            # c_accs is (12, rounds)
            w_run = np.sum(np.array(c_accs) * weights[:, None], axis=0)
            runs_w_series.append(w_run)
    if not runs_w_series:
        return None, None
    mean_w = np.mean(runs_w_series, axis=0)
    std_w = np.std(runs_w_series, axis=0)
    return mean_w, std_w

def get_server_bandwidth_series(exp_dir):
    mb_runs = []
    for r in range(1, 11):
        sf = glob.glob(os.path.join(exp_dir, f"server_*_run{r}_*.h5"))
        if not sf: continue
        with h5py.File(sf[0], 'r') as h:
            key = 'total_transmitted_per_round_Mb' if 'total_transmitted_per_round_Mb' in h else 'total_transmitted_per_round'
            if key in h:
                mb_runs.append(np.cumsum(np.array(h[key])))
    if not mb_runs:
        return None
    return np.mean(mb_runs, axis=0)

def generate_multi_dataset_dashboard():
    """Gera um painel elegante 2x2 com a Acurácia Auxiliar Ponderada dos 4 datasets."""
    out_dir = os.path.join(BASE_RESULTS, "graficos_consolidados_paper")
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        cfg = DATASET_CONFIGS[ds]
        puma_dir = os.path.join(BASE_RESULTS, cfg["puma"])
        lora_dir = os.path.join(BASE_RESULTS, cfg["lora"])

        puma_mean, puma_std = get_weighted_acc_series(puma_dir, ds)
        lora_mean, lora_std = get_weighted_acc_series(lora_dir, ds)

        if puma_mean is not None and lora_mean is not None:
            rounds = np.arange(1, len(puma_mean) + 1)
            # LoRA Baseline (Vermelho)
            ax.plot(rounds, lora_mean, label=f"LoRA Padrão ({lora_mean[-1]:.2f}%)",
                    color="#d62728", linestyle="--", linewidth=2.5)
            ax.fill_between(rounds, lora_mean - lora_std, lora_mean + lora_std,
                            color="#d62728", alpha=0.15)

            # PUMA-GT Plus (Azul)
            ax.plot(rounds, puma_mean, label=f"PUMA-GT Plus ({puma_mean[-1]:.2f}%)",
                    color="#1f77b4", linestyle="-", linewidth=3.0)
            ax.fill_between(rounds, puma_mean - puma_std, puma_mean + puma_std,
                            color="#1f77b4", alpha=0.20)

            ax.set_title(cfg["title"], fontsize=16, fontweight="bold", pad=10)
            ax.set_xlabel("Rodadas de Comunicação", fontsize=14)
            ax.set_ylabel("Acurácia Auxiliar Ponderada (%)", fontsize=14)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="lower right", fontsize=13, framealpha=0.95)
            ax.tick_params(labelsize=12)
            ax.set_xlim(1, 100)

    fig.suptitle("Evolução da Acurácia Auxiliar Ponderada: PUMA-GT Plus vs. LoRA Padrão (SLM)",
                 fontsize=20, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    pdf_path = os.path.join(out_dir, "painel_4datasets_acuracia_ponderada.pdf")
    png_path = os.path.join(out_dir, "painel_4datasets_acuracia_ponderada.png")
    fig.savefig(pdf_path, dpi=300)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"\n📊 Painel 2x2 salvo em: {pdf_path} e {png_path}")

    # Painel 2x2 de Banda de Rede Acumulada (MB)
    fig_b, axes_b = plt.subplots(2, 2, figsize=(18, 12))
    axes_b = axes_b.flatten()

    for idx, ds in enumerate(DATASETS):
        ax = axes_b[idx]
        cfg = DATASET_CONFIGS[ds]
        puma_dir = os.path.join(BASE_RESULTS, cfg["puma"])
        lora_dir = os.path.join(BASE_RESULTS, cfg["lora"])

        puma_mb = get_server_bandwidth_series(puma_dir)
        lora_mb = get_server_bandwidth_series(lora_dir)

        if puma_mb is not None and lora_mb is not None:
            rounds = np.arange(1, len(puma_mb) + 1)
            # LoRA Baseline (Vermelho)
            ax.plot(rounds, lora_mb, label=f"LoRA ({lora_mb[-1]:,.0f} MB)",
                    color="#d62728", linestyle="--", linewidth=2.5)

            # PUMA-GT Plus (Azul)
            ax.plot(rounds, puma_mb, label=f"PUMA-GT Plus ({puma_mb[-1]:,.0f} MB)",
                    color="#1f77b4", linestyle="-", linewidth=3.0)

            savings = (1 - puma_mb[-1] / lora_mb[-1]) * 100
            ax.set_title(f"{cfg['title']} — Economia de {savings:.1f}%", fontsize=15, fontweight="bold", pad=10)
            ax.set_xlabel("Rodadas de Comunicação", fontsize=14)
            ax.set_ylabel("Banda Acumulada (MB)", fontsize=14)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper left", fontsize=13, framealpha=0.95)
            ax.tick_params(labelsize=12)
            ax.set_xlim(1, 100)

    fig_b.suptitle("Tráfego de Rede Acumulado (MB): PUMA-GT Plus vs. LoRA Padrão (SLM)",
                   fontsize=20, fontweight="bold", y=0.98)
    fig_b.tight_layout(rect=[0, 0, 1, 0.96])

    pdf_b_path = os.path.join(out_dir, "painel_4datasets_banda_acumulada.pdf")
    png_b_path = os.path.join(out_dir, "painel_4datasets_banda_acumulada.png")
    fig_b.savefig(pdf_b_path, dpi=300)
    fig_b.savefig(png_b_path, dpi=300)
    plt.close(fig_b)
    print(f"📊 Painel de Banda salvo em: {pdf_b_path} e {png_b_path}")

if __name__ == "__main__":
    print("🚀 Iniciando geração de gráficos para todas as execuções SLM...")
    run_dataset_plots()
    generate_multi_dataset_dashboard()
    print("\n🎉 TODOS OS GRÁFICOS FORAM GERADOS COM SUCESSO!")
