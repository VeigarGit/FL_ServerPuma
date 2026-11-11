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
        "sora": "fl_sora_slm_OxfordPets_slm_sora_with_schedule_prune1_ala1_paca28",
        "onlypaca": "fl_pumagt_plus_slm_OxfordPets_ONLYPACA_norank",
        "title": "Oxford-IIIT Pet (37 classes)"
    },
    "DTD": {
        "puma": "fl_pumagt_plus_slm_DTD_slm_sora_with_schedule_prune1_ala1_adaptpaca_adaptrank",
        "lora": "fl_lora_slm_DTD_slm_lora_prune1_ala1_paca28",
        "sora": "fl_sora_slm_DTD_slm_sora_with_schedule_prune1_ala1_paca28",
        "title": "Describable Textures (DTD - 47 classes)"
    },
    "FGVCAircraft": {
        "puma": "fl_pumagt_plus_slm_FGVCAircraft_slm_sora_with_schedule_prune1_ala1_adaptpaca_adaptrank",
        "lora": "fl_lora_slm_FGVCAircraft_slm_lora_prune1_ala1_paca28",
        "sora": "fl_sora_slm_FGVCAircraft_slm_sora_with_schedule_prune1_ala1_paca28",
        "title": "FGVC Aircraft (100 classes)"
    },
    "Flowers102": {
        "puma": "fl_pumagt_plus_slm_Flowers102_slm_sora_with_schedule_prune1_ala1_adaptpaca_adaptrank",
        "lora": "fl_lora_slm_Flowers102_slm_lora_prune1_ala1_paca28",
        "sora": "fl_sora_slm_Flowers102_slm_sora_with_schedule_prune1_ala1_paca28",
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

        out_dir = os.path.join(BASE_RESULTS, "graficos", ds)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n{'='*70}")
        print(f"🎨 Gerando 15 gráficos comparativos para: {ds}")
        print(f"📂 Diretório: {out_dir}")
        print(f"{'='*70}")

        # Chama plot_results.py passando todos os experimentos disponíveis
        exp_list = [cfg["lora"]]
        if "sora" in cfg and os.path.exists(os.path.join(BASE_RESULTS, cfg["sora"])):
            exp_list.append(cfg["sora"])
        exp_list.append(cfg["puma"])
        if "onlypaca" in cfg and os.path.exists(os.path.join(BASE_RESULTS, cfg["onlypaca"])):
            exp_list.append(cfg["onlypaca"])

        cmd = [python_bin, plot_script] + exp_list
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

def get_paca_rank_series(exp_dir):
    sfiles = sorted(glob.glob(os.path.join(exp_dir, "server_*.h5")))
    paca_runs, rank_runs = [], []
    for sf in sfiles:
        with h5py.File(sf, "r") as h:
            if "rs_client_paca" in h:
                paca_runs.append(np.mean(np.array(h["rs_client_paca"]), axis=1))
            if "rs_client_rank" in h:
                rank_runs.append(np.mean(np.array(h["rs_client_rank"]), axis=1))
    paca_m = np.mean(paca_runs, axis=0) if paca_runs else None
    rank_m = np.mean(rank_runs, axis=0) if rank_runs else None
    return paca_m, rank_m

def generate_multi_dataset_dashboard():
    """Gera um painel elegante 2x2 com a Acurácia Auxiliar Ponderada dos 4 datasets."""
    out_dir = os.path.join(BASE_RESULTS, "graficos", "paper")
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        cfg = DATASET_CONFIGS[ds]
        puma_dir = os.path.join(BASE_RESULTS, cfg["puma"])
        lora_dir = os.path.join(BASE_RESULTS, cfg["lora"])
        sora_dir = os.path.join(BASE_RESULTS, cfg.get("sora", ""))

        puma_mean, puma_std = get_weighted_acc_series(puma_dir, ds)
        lora_mean, lora_std = get_weighted_acc_series(lora_dir, ds)
        sora_mean, sora_std = get_weighted_acc_series(sora_dir, ds) if os.path.exists(sora_dir) else (None, None)

        if puma_mean is not None and lora_mean is not None:
            rounds = np.arange(1, len(puma_mean) + 1)
            # LoRA Baseline (Vermelho)
            ax.plot(rounds, lora_mean, label=f"LoRA ({lora_mean[-1]:.2f}%)",
                    color="#d62728", linestyle="--", linewidth=2.5)
            ax.fill_between(rounds, lora_mean - lora_std, lora_mean + lora_std,
                            color="#d62728", alpha=0.10)

            # SoRA (Laranja)
            if sora_mean is not None:
                ax.plot(rounds, sora_mean, label=f"SoRA ({sora_mean[-1]:.2f}%)",
                        color="#ff7f0e", linestyle=":", linewidth=2.5)
                ax.fill_between(rounds, sora_mean - sora_std, sora_mean + sora_std,
                                color="#ff7f0e", alpha=0.10)

            # PUMA-GT (apenas PaCA, se disponível, ex: OxfordPets)
            if "onlypaca" in cfg:
                op_dir = os.path.join(BASE_RESULTS, cfg["onlypaca"])
                if os.path.exists(op_dir):
                    op_mean, _ = get_weighted_acc_series(op_dir, ds)
                    if op_mean is not None:
                        ax.plot(rounds, op_mean, label=f"PUMA-GT ({op_mean[-1]:.2f}%)",
                                color="#9467bd", linestyle="-.", linewidth=2.5)

            # PUMA-GT Evolution (Azul)
            ax.plot(rounds, puma_mean, label=f"PUMA-GT Evolution ({puma_mean[-1]:.2f}%)",
                    color="#1f77b4", linestyle="-", linewidth=3.0)
            ax.fill_between(rounds, puma_mean - puma_std, puma_mean + puma_std,
                            color="#1f77b4", alpha=0.15)

            ax.set_title(cfg["title"], fontsize=16, fontweight="bold", pad=10)
            ax.set_xlabel("Rodadas de Comunicação", fontsize=14)
            ax.set_ylabel("Acurácia Auxiliar Ponderada (%)", fontsize=14)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="lower right", fontsize=12, framealpha=0.95)
            ax.tick_params(labelsize=12)
            ax.set_xlim(1, 100)

    fig.suptitle("Evolução da Acurácia Auxiliar Ponderada nos 4 Datasets (SLM Qwen-VL)",
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
        sora_dir = os.path.join(BASE_RESULTS, cfg.get("sora", ""))

        puma_mb = get_server_bandwidth_series(puma_dir)
        lora_mb = get_server_bandwidth_series(lora_dir)
        sora_mb = get_server_bandwidth_series(sora_dir) if os.path.exists(sora_dir) else None

        if puma_mb is not None and lora_mb is not None:
            rounds = np.arange(1, len(puma_mb) + 1)
            # LoRA Baseline (Vermelho)
            ax.plot(rounds, lora_mb, label=f"LoRA ({lora_mb[-1]:,.0f} MB)",
                    color="#d62728", linestyle="--", linewidth=2.5)

            # SoRA (Laranja)
            if sora_mb is not None:
                sora_sav = (1 - sora_mb[-1] / lora_mb[-1]) * 100
                ax.plot(rounds, sora_mb, label=f"SoRA ({sora_mb[-1]:,.0f} MB, -{sora_sav:.1f}%)",
                        color="#ff7f0e", linestyle=":", linewidth=2.5)

            # PUMA-GT (Apenas PaCA)
            if "onlypaca" in cfg:
                op_dir = os.path.join(BASE_RESULTS, cfg["onlypaca"])
                if os.path.exists(op_dir):
                    op_mb = get_server_bandwidth_series(op_dir)
                    if op_mb is not None:
                        op_sav = (1 - op_mb[-1] / lora_mb[-1]) * 100
                        ax.plot(rounds, op_mb, label=f"PUMA-GT ({op_mb[-1]:,.0f} MB, -{op_sav:.1f}%)",
                                color="#9467bd", linestyle="-.", linewidth=2.5)

            # PUMA-GT Evolution (Azul)
            puma_sav = (1 - puma_mb[-1] / lora_mb[-1]) * 100
            ax.plot(rounds, puma_mb, label=f"PUMA-GT Evolution ({puma_mb[-1]:,.0f} MB, -{puma_sav:.1f}%)",
                    color="#1f77b4", linestyle="-", linewidth=3.0)

            ax.set_title(f"{cfg['title']} — Economia PUMA: {puma_sav:.1f}%", fontsize=15, fontweight="bold", pad=10)
            ax.set_xlabel("Rodadas de Comunicação", fontsize=14)
            ax.set_ylabel("Banda Acumulada (MB)", fontsize=14)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper left", fontsize=12, framealpha=0.95)
            ax.tick_params(labelsize=12)
            ax.set_xlim(1, 100)

    fig_b.suptitle("Tráfego de Rede Acumulado (MB) nos 4 Datasets (SLM Qwen-VL)",
                   fontsize=20, fontweight="bold", y=0.98)
    fig_b.tight_layout(rect=[0, 0, 1, 0.96])

    pdf_b_path = os.path.join(out_dir, "painel_4datasets_banda_acumulada.pdf")
    png_b_path = os.path.join(out_dir, "painel_4datasets_banda_acumulada.png")
    fig_b.savefig(pdf_b_path, dpi=300)
    fig_b.savefig(png_b_path, dpi=300)
    plt.close(fig_b)
    print(f"📊 Painel de Banda salvo em: {pdf_b_path} e {png_b_path}")

def generate_oxfordpets_ablation_dashboard():
    """Gera um painel 2x2 elegante focado na ablação das 4 estratégias no OxfordPets."""
    out_dir = os.path.join(BASE_RESULTS, "graficos", "paper")
    os.makedirs(out_dir, exist_ok=True)

    cfg = DATASET_CONFIGS["OxfordPets"]
    lora_dir = os.path.join(BASE_RESULTS, cfg["lora"])
    sora_dir = os.path.join(BASE_RESULTS, cfg["sora"])
    puma_dir = os.path.join(BASE_RESULTS, cfg["puma"])
    onlypaca_dir = os.path.join(BASE_RESULTS, cfg["onlypaca"])

    l_acc, _ = get_weighted_acc_series(lora_dir, "OxfordPets")
    s_acc, _ = get_weighted_acc_series(sora_dir, "OxfordPets")
    p_acc, _ = get_weighted_acc_series(puma_dir, "OxfordPets")
    op_acc, _ = get_weighted_acc_series(onlypaca_dir, "OxfordPets")

    l_mb = get_server_bandwidth_series(lora_dir)
    s_mb = get_server_bandwidth_series(sora_dir)
    p_mb = get_server_bandwidth_series(puma_dir)
    op_mb = get_server_bandwidth_series(onlypaca_dir)

    l_paca, l_rank = get_paca_rank_series(lora_dir)
    s_paca, s_rank = get_paca_rank_series(sora_dir)
    p_paca, p_rank = get_paca_rank_series(puma_dir)
    op_paca, op_rank = get_paca_rank_series(onlypaca_dir)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    rounds = np.arange(1, 101)

    # 1. Acurácia Auxiliar Ponderada
    ax1 = axes[0, 0]
    ax1.plot(rounds, l_acc, label=f"LoRA Padrão ({l_acc[-1]:.2f}%)", color="#d62728", linestyle="--", linewidth=2.2)
    ax1.plot(rounds, s_acc, label=f"SoRA Estático ({s_acc[-1]:.2f}%)", color="#ff7f0e", linestyle=":", linewidth=2.5)
    ax1.plot(rounds, op_acc, label=f"PUMA-GT ({op_acc[-1]:.2f}%)", color="#9467bd", linestyle="-.", linewidth=2.5)
    ax1.plot(rounds, p_acc, label=f"PUMA-GT Evolution ({p_acc[-1]:.2f}%)", color="#1f77b4", linestyle="-", linewidth=3.0)
    ax1.set_title("Acurácia Auxiliar Ponderada dos Clientes", fontsize=15, fontweight="bold", pad=10)
    ax1.set_xlabel("Rodadas de Comunicação", fontsize=13)
    ax1.set_ylabel("Acurácia (%)", fontsize=13)
    ax1.set_xlim(1, 100); ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=12, framealpha=0.95)

    # 2. Banda de Rede Acumulada
    ax2 = axes[0, 1]
    ax2.plot(rounds, l_mb, label=f"LoRA Padrão ({l_mb[-1]:,.0f} MB)", color="#d62728", linestyle="--", linewidth=2.2)
    ax2.plot(rounds, s_mb, label=f"SoRA Estático ({s_mb[-1]:,.0f} MB, -{(1-s_mb[-1]/l_mb[-1])*100:.1f}%)", color="#ff7f0e", linestyle=":", linewidth=2.5)
    ax2.plot(rounds, op_mb, label=f"PUMA-GT ({op_mb[-1]:,.0f} MB, -{(1-op_mb[-1]/l_mb[-1])*100:.1f}%)", color="#9467bd", linestyle="-.", linewidth=2.5)
    ax2.plot(rounds, p_mb, label=f"PUMA-GT Evolution ({p_mb[-1]:,.0f} MB, -{(1-p_mb[-1]/l_mb[-1])*100:.1f}%)", color="#1f77b4", linestyle="-", linewidth=3.0)
    ax2.set_title("Tráfego de Rede Acumulado (MB)", fontsize=15, fontweight="bold", pad=10)
    ax2.set_xlabel("Rodadas de Comunicação", fontsize=13)
    ax2.set_ylabel("Banda Acumulada (MB)", fontsize=13)
    ax2.set_xlim(1, 100); ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=12, framealpha=0.95)

    # 3. Evolução do PaCA Adaptativo (Camadas)
    ax3 = axes[1, 0]
    ax3.plot(rounds, l_paca, label="LoRA (p=28 fixo)", color="#d62728", linestyle="--", linewidth=2.2)
    ax3.plot(rounds, s_paca, label="SoRA (p=28 fixo)", color="#ff7f0e", linestyle=":", linewidth=2.5)
    ax3.plot(rounds, op_paca, label=f"PUMA-GT (28 → {op_paca[-1]:.1f})", color="#9467bd", linestyle="-.", linewidth=2.5)
    ax3.plot(rounds, p_paca, label=f"PUMA-GT Evolution (28 → {p_paca[-1]:.1f})", color="#1f77b4", linestyle="-", linewidth=3.0)
    ax3.set_title("Evolução das Camadas Ativas Médias (PaCA)", fontsize=15, fontweight="bold", pad=10)
    ax3.set_xlabel("Rodadas de Comunicação", fontsize=13)
    ax3.set_ylabel("Número de Camadas Ativas", fontsize=13)
    ax3.set_xlim(1, 100); ax3.set_ylim(0, 30); ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="lower left", fontsize=12, framealpha=0.95)

    # 4. Evolução do Rank Adaptativo
    ax4 = axes[1, 1]
    ax4.plot(rounds, l_rank, label="LoRA (r=8 fixo)", color="#d62728", linestyle="--", linewidth=2.2)
    ax4.plot(rounds, s_rank, label="SoRA (r=8 fixo)", color="#ff7f0e", linestyle=":", linewidth=2.5)
    ax4.plot(rounds, op_rank, label="PUMA-GT (r=8 fixo)", color="#9467bd", linestyle="-.", linewidth=2.5)
    ax4.plot(rounds, p_rank, label=f"PUMA-GT Evolution (8 → {p_rank[-1]:.1f})", color="#1f77b4", linestyle="-", linewidth=3.0)
    ax4.set_title("Evolução do Rank Efetivo Médio", fontsize=15, fontweight="bold", pad=10)
    ax4.set_xlabel("Rodadas de Comunicação", fontsize=13)
    ax4.set_ylabel("Rank Efetivo", fontsize=13)
    ax4.set_xlim(1, 100); ax4.set_ylim(1, 9); ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="lower left", fontsize=12, framealpha=0.95)

    fig.suptitle("Estudo de Ablação: LoRA vs. SoRA vs. PUMA-GT vs. PUMA-GT Evolution (OxfordPets / Qwen-VL)",
                 fontsize=18, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    pdf_out = os.path.join(out_dir, "painel_oxfordpets_ablacao_4estrategias.pdf")
    png_out = os.path.join(out_dir, "painel_oxfordpets_ablacao_4estrategias.png")
    fig.savefig(pdf_out, dpi=300)
    fig.savefig(png_out, dpi=300)
    plt.close(fig)
    print(f"📊 Painel de Ablação OxfordPets salvo em: {pdf_out} e {png_out}")

    # Copia também para graficos/OxfordPets
    oxf_dir = os.path.join(BASE_RESULTS, "graficos", "OxfordPets")
    if os.path.exists(oxf_dir):
        import shutil
        shutil.copy2(png_out, os.path.join(oxf_dir, "painel_oxfordpets_ablacao_4estrategias.png"))
        shutil.copy2(pdf_out, os.path.join(oxf_dir, "painel_oxfordpets_ablacao_4estrategias.pdf"))

if __name__ == "__main__":
    print("🚀 Iniciando geração de gráficos para todas as execuções SLM...")
    run_dataset_plots()
    generate_multi_dataset_dashboard()
    generate_oxfordpets_ablation_dashboard()
    print("\n🎉 TODOS OS GRÁFICOS FORAM GERADOS COM SUCESSO!")
