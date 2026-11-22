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

import argparse
import glob
import os
import shutil
import subprocess
import sys

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_RESULTS = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "results"))
DATASET_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "dataset"))

DATASETS = ["OxfordPets", "DTD", "FGVCAircraft", "Flowers102"]

DATASET_CONFIGS = {
    "OxfordPets": {
        "puma": "fl_pumagt_plus_slm_OxfordPets_slm_sora_with_schedule_prune1_ala1_adaptpaca_adaptrank",
        "puma_delta": "fl_pumagt_plus_slm_OxfordPets_slm_sora_with_schedule_prune1_ala1_adaptpaca_adaptrank_deltacoding",
        "lora": "fl_lora_slm_OxfordPets_slm_lora_prune1_ala1_paca28",
        "sora": "fl_sora_slm_OxfordPets_slm_sora_with_schedule_prune1_ala1_paca28",
        "onlypaca": "fl_pumagt_plus_slm_OxfordPets_ONLYPACA_norank",
        "title": "Oxford-IIIT Pet (37 classes)",
    },
    "DTD": {
        "puma": "fl_pumagt_plus_slm_DTD_slm_sora_with_schedule_prune1_ala1_adaptpaca_adaptrank",
        "puma_delta": "fl_pumagt_plus_slm_DTD_slm_sora_with_schedule_prune1_ala1_adaptpaca_adaptrank_deltacoding",
        "lora": "fl_lora_slm_DTD_slm_lora_prune1_ala1_paca28",
        "sora": "fl_sora_slm_DTD_slm_sora_with_schedule_prune1_ala1_paca28",
        "title": "Describable Textures (DTD - 47 classes)",
    },
    "FGVCAircraft": {
        "puma": "fl_pumagt_plus_slm_FGVCAircraft_slm_sora_with_schedule_prune1_ala1_adaptpaca_adaptrank",
        "puma_delta": "fl_pumagt_plus_slm_FGVCAircraft_slm_sora_with_schedule_prune1_ala1_adaptpaca_adaptrank_deltacoding",
        "lora": "fl_lora_slm_FGVCAircraft_slm_lora_prune1_ala1_paca28",
        "sora": "fl_sora_slm_FGVCAircraft_slm_sora_with_schedule_prune1_ala1_paca28",
        "title": "FGVC Aircraft (100 classes)",
    },
    "Flowers102": {
        "puma": "fl_pumagt_plus_slm_Flowers102_slm_sora_with_schedule_prune1_ala1_adaptpaca_adaptrank",
        "puma_delta": "fl_pumagt_plus_slm_Flowers102_slm_sora_with_schedule_prune1_ala1_adaptpaca_adaptrank_deltacoding",
        "lora": "fl_lora_slm_Flowers102_slm_lora_prune1_ala1_paca28",
        "sora": "fl_sora_slm_Flowers102_slm_sora_with_schedule_prune1_ala1_paca28",
        "title": "Oxford Flowers-102 (102 classes)",
    },
}


def run_dataset_plots(
    out_root: str,
    *,
    include_delta: bool = False,
    exclude_onlypaca: bool = False,
    normalize_lora_bf16: bool = False,
):
    plot_script = os.path.join(SCRIPT_DIR, "plot_results.py")
    python_bin = sys.executable

    for ds in DATASETS:
        cfg = DATASET_CONFIGS[ds]
        puma_dir = os.path.join(BASE_RESULTS, cfg["puma"])
        lora_dir = os.path.join(BASE_RESULTS, cfg["lora"])

        if not os.path.exists(puma_dir) or not os.path.exists(lora_dir):
            print(f"⚠️ Pastas incompletas para {ds}. Pulando.")
            continue

        out_dir = os.path.join(out_root, ds)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n{'=' * 70}")
        print(f"🎨 Gerando gráficos comparativos para: {ds}")
        print(f"📂 Diretório: {out_dir}")
        print(f"{'=' * 70}")

        exp_list = [cfg["lora"]]
        if "sora" in cfg and os.path.exists(os.path.join(BASE_RESULTS, cfg["sora"])):
            exp_list.append(cfg["sora"])
        exp_list.append(cfg["puma"])
        if (
            include_delta
            and "puma_delta" in cfg
            and os.path.exists(os.path.join(BASE_RESULTS, cfg["puma_delta"]))
        ):
            exp_list.append(cfg["puma_delta"])
        if (
            not exclude_onlypaca
            and "onlypaca" in cfg
            and os.path.exists(os.path.join(BASE_RESULTS, cfg["onlypaca"]))
        ):
            exp_list.append(cfg["onlypaca"])

        cmd = [python_bin, plot_script, "--out-dir", out_dir]
        if normalize_lora_bf16:
            cmd.append("--normalize-lora-bf16")
        cmd.extend(exp_list)
        env = os.environ.copy()
        subprocess.run(cmd, cwd=os.path.join(SCRIPT_DIR, ".."), check=False, env=env)

def _latest_run_file(exp_dir: str, pattern_prefix: str, run_id: int) -> str | None:
    """Seleciona o arquivo mais recente de um run específico."""
    matches = sorted(glob.glob(os.path.join(exp_dir, f"{pattern_prefix}_run{run_id}*.h5")))
    return matches[-1] if matches else None


def get_weighted_acc_series(exp_dir, dataset_name):
    test_dir = os.path.join(DATASET_DIR, dataset_name, "test")
    counts = []
    for cid in range(12):
        npz_path = os.path.join(test_dir, f"{cid}.npz")
        d = np.load(npz_path, allow_pickle=True)
        counts.append(len(d["data"].item()["y"]))
    weights = np.array(counts, dtype=float) / sum(counts)

    runs_w_series = []
    for r in range(1, 11):
        c_accs = []
        for cid in range(12):
            cf = _latest_run_file(exp_dir, f"client_{cid}_*", r)
            if not cf:
                break
            with h5py.File(cf, "r") as h:
                c_accs.append(np.array(h["rs_global_acc"]))
        if len(c_accs) == 12:
            w_run = np.sum(np.array(c_accs) * weights[:, None], axis=0)
            runs_w_series.append(w_run)
    if not runs_w_series:
        return None, None
    return np.mean(runs_w_series, axis=0), np.std(runs_w_series, axis=0)


def get_server_acc_series(exp_dir: str):
    """Retorna (media, desvio_padrao) de rs_test_acc entre as runs."""
    runs = []
    for r in range(1, 11):
        sf = _latest_run_file(exp_dir, "server_*", r)
        if not sf:
            continue
        with h5py.File(sf, "r") as h:
            if "rs_test_acc" in h:
                runs.append(np.array(h["rs_test_acc"]))
    if not runs:
        return None, None
    return np.mean(runs, axis=0), np.std(runs, axis=0)


def get_server_bandwidth_series(exp_dir):
    mb_runs = []
    for r in range(1, 11):
        sf = _latest_run_file(exp_dir, "server_*", r)
        if not sf:
            continue
        with h5py.File(sf, "r") as h:
            key = (
                "total_transmitted_per_round_Mb"
                if "total_transmitted_per_round_Mb" in h
                else "total_transmitted_per_round"
            )
            if key in h:
                mb_runs.append(np.cumsum(np.array(h[key])))
    if not mb_runs:
        return None
    return np.mean(mb_runs, axis=0)


def get_server_time_breakdown(exp_dir: str) -> dict[str, float] | None:
    """Retorna média por rodada (s) de treino, proc servidor, eval cliente, rede e total."""
    totals, trains, sprocs, evals, comms = [], [], [], [], []
    for r in range(1, 11):
        sf = _latest_run_file(exp_dir, "server_*", r)
        if not sf:
            continue
        with h5py.File(sf, "r") as h:
            totals.append(float(np.mean(h["Round_time"][:])))
            trains.append(
                float(np.mean(h["rs_client_training_time"][:]))
                if "rs_client_training_time" in h
                else 0.0
            )
            sprocs.append(
                float(np.mean(h["rs_server_processing_time"][:]))
                if "rs_server_processing_time" in h
                else 0.0
            )
            evals.append(
                float(np.mean(h["rs_client_eval_time"][:]))
                if "rs_client_eval_time" in h
                else 0.0
            )
            comms.append(
                float(np.mean(h["rs_client_comm_time"][:]))
                if "rs_client_comm_time" in h
                else 0.0
            )
    if not totals:
        return None
    tot_m = float(np.mean(totals))
    tr_m = float(np.mean(trains))
    sp_m = float(np.mean(sprocs))
    ev_m = float(np.mean(evals))
    cm_m = float(np.mean(comms))
    oth_m = max(0.0, tot_m - tr_m - sp_m - ev_m - cm_m)
    return {"total": tot_m, "train": tr_m, "srv_proc": sp_m, "eval": ev_m, "other": cm_m + oth_m}


def get_paca_rank_series(exp_dir):
    paca_runs, rank_runs = [], []
    for r in range(1, 11):
        sf = _latest_run_file(exp_dir, "server_*", r)
        if not sf:
            continue
        with h5py.File(sf, "r") as h:
            if "rs_client_paca" in h:
                paca_runs.append(np.mean(np.array(h["rs_client_paca"]), axis=1))
            if "rs_client_rank" in h:
                rank_runs.append(np.mean(np.array(h["rs_client_rank"]), axis=1))
    paca_m = np.mean(paca_runs, axis=0) if paca_runs else None
    rank_m = np.mean(rank_runs, axis=0) if rank_runs else None
    return paca_m, rank_m


def _strategy_entries(cfg: dict, include_delta: bool, exclude_onlypaca: bool):
    entries = [
        ("LoRA Padrão", cfg["lora"], "#d62728", "--", 2.4),
        ("SoRA Estático", cfg.get("sora", ""), "#ff7f0e", ":", 2.4),
    ]
    if not exclude_onlypaca and "onlypaca" in cfg:
        entries.append(("PUMA-GT", cfg["onlypaca"], "#9467bd", "-.", 2.4))
    entries.append(("PUMA-GT Evolution", cfg["puma"], "#1f77b4", "-", 2.8))
    if include_delta and "puma_delta" in cfg:
        entries.append(("PUMA-GT E + Delta", cfg["puma_delta"], "#2ca02c", "-.", 2.8))
    return entries


def _plot_accuracy_grid(
    out_dir: str,
    filename: str,
    suptitle: str,
    ylabel: str,
    series_fn,
    *,
    include_delta: bool,
    exclude_onlypaca: bool,
):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    for ax, ds in zip(axes.flatten(), DATASETS, strict=False):
        cfg = DATASET_CONFIGS[ds]
        plotted = False
        for label, rel_dir, color, ls, lw in _strategy_entries(
            cfg, include_delta, exclude_onlypaca
        ):
            exp_dir = os.path.join(BASE_RESULTS, rel_dir) if rel_dir else ""
            if not exp_dir or not os.path.exists(exp_dir):
                continue
            mean_arr, std_arr = series_fn(exp_dir, ds)
            if mean_arr is None:
                continue
            rounds = np.arange(1, len(mean_arr) + 1)
            ax.plot(rounds, mean_arr, label=f"{label} ({mean_arr[-1]:.2f}%)", color=color, linestyle=ls, linewidth=lw)
            ax.fill_between(rounds, mean_arr - std_arr, mean_arr + std_arr, color=color, alpha=0.10)
            plotted = True
        if plotted:
            ax.set_title(cfg["title"], fontsize=16, fontweight="bold", pad=10)
            ax.set_xlabel("Rodadas de Comunicação", fontsize=14)
            ax.set_ylabel(ylabel, fontsize=14)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="lower right", fontsize=11.5, framealpha=0.95)
            ax.tick_params(labelsize=12)
            ax.set_xlim(1, 100)

    fig.suptitle(suptitle, fontsize=20, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{filename}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"📊 Painel salvo em: {os.path.join(out_dir, filename)}.png", flush=True)


def generate_multi_dataset_dashboard(
    out_root: str,
    *,
    include_delta: bool = False,
    exclude_onlypaca: bool = False,
):
    """Gera os painéis 2x2 consolidados (Acurácias, Banda Acumulada e Tempo)."""
    out_dir = os.path.join(out_root, "paper")
    os.makedirs(out_dir, exist_ok=True)

    _plot_accuracy_grid(
        out_dir,
        "painel_4datasets_acuracia_ponderada",
        "Evolução da Acurácia Auxiliar Ponderada nos 4 Datasets (SLM Qwen-VL)",
        "Acurácia Auxiliar Ponderada (%)",
        lambda d, ds: get_weighted_acc_series(d, ds),
        include_delta=include_delta,
        exclude_onlypaca=exclude_onlypaca,
    )
    _plot_accuracy_grid(
        out_dir,
        "painel_4datasets_acuracia_global",
        "Evolução da Acurácia Global Federada (rs_test_acc) nos 4 Datasets",
        "Acurácia Global (%)",
        lambda d, _ds: get_server_acc_series(d),
        include_delta=include_delta,
        exclude_onlypaca=exclude_onlypaca,
    )

    # Painel 2x2 de Banda de Rede Acumulada (MB)
    fig_b, axes_b = plt.subplots(2, 2, figsize=(18, 12))
    for ax, ds in zip(axes_b.flatten(), DATASETS, strict=False):
        cfg = DATASET_CONFIGS[ds]
        lora_mb = get_server_bandwidth_series(os.path.join(BASE_RESULTS, cfg["lora"]))
        if lora_mb is None:
            continue
        rounds = np.arange(1, len(lora_mb) + 1)
        eco_texts = []
        for label, rel_dir, color, ls, lw in _strategy_entries(
            cfg, include_delta, exclude_onlypaca
        ):
            exp_dir = os.path.join(BASE_RESULTS, rel_dir) if rel_dir else ""
            if not exp_dir or not os.path.exists(exp_dir):
                continue
            mb_arr = get_server_bandwidth_series(exp_dir)
            if mb_arr is None:
                continue
            if label == "LoRA Padrão":
                lbl = f"LoRA Padrão ({mb_arr[-1]:,.0f} MB)"
            else:
                sav = (1.0 - mb_arr[-1] / lora_mb[-1]) * 100.0
                lbl = f"{label} ({mb_arr[-1]:,.0f} MB, -{sav:.1f}%)"
                if "PUMA" in label:
                    eco_texts.append(f"{'Delta' if 'Delta' in label else 'PUMA'}: -{sav:.1f}%")
            ax.plot(rounds, mb_arr, label=lbl, color=color, linestyle=ls, linewidth=lw)

        subtitle = f" — {' | '.join(eco_texts)}" if eco_texts else ""
        ax.set_title(f"{cfg['title']}{subtitle}", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Rodadas de Comunicação", fontsize=14)
        ax.set_ylabel("Banda Acumulada (MB)", fontsize=14)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", fontsize=11.5, framealpha=0.95)
        ax.tick_params(labelsize=12)
        ax.set_xlim(1, 100)

    fig_b.suptitle(
        "Tráfego de Rede Acumulado (MB) nos 4 Datasets (SLM Qwen-VL)",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )
    fig_b.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("pdf", "png"):
        fig_b.savefig(os.path.join(out_dir, f"painel_4datasets_banda_acumulada.{ext}"), dpi=300)
    plt.close(fig_b)
    print(f"📊 Painel de Banda salvo em: {os.path.join(out_dir, 'painel_4datasets_banda_acumulada.png')}", flush=True)

    # Painel 2x2 de Decomposição Temporal por Rodada (s)
    fig_t, axes_t = plt.subplots(2, 2, figsize=(18, 12))
    handles_t, labels_leg_t = [], []
    for ax, ds in zip(axes_t.flatten(), DATASETS, strict=False):
        cfg = DATASET_CONFIGS[ds]
        labels_b, train_s, sproc_s, eval_s, other_s, tot_s = [], [], [], [], [], []
        for label, rel_dir, _, _, _ in _strategy_entries(cfg, include_delta, exclude_onlypaca):
            exp_dir = os.path.join(BASE_RESULTS, rel_dir) if rel_dir else ""
            if not exp_dir or not os.path.exists(exp_dir):
                continue
            tb = get_server_time_breakdown(exp_dir)
            if tb is None:
                continue
            labels_b.append(label.replace("PUMA-GT Evolution", "PUMA-GTE").replace("PUMA-GT E + Delta", "PUMA-GTE\n+ Delta"))
            train_s.append(tb["train"])
            sproc_s.append(tb["srv_proc"])
            eval_s.append(tb["eval"])
            other_s.append(tb["other"])
            tot_s.append(tb["total"])

        if not labels_b:
            continue
        x = np.arange(len(labels_b))
        w = 0.52
        b1 = np.array(train_s)
        b2 = b1 + np.array(eval_s)
        b3 = b2 + np.array(sproc_s)
        ax.bar(x, train_s, w, label="Treino Local", color="#1f77b4")
        ax.bar(x, eval_s, w, bottom=b1, label="Eval/Quant Cliente", color="#ff7f0e")
        ax.bar(x, sproc_s, w, bottom=b2, label="Proc. Servidor (LHDQ/Quant)", color="#9467bd")
        ax.bar(x, other_s, w, bottom=b3, label="Rede / Sleep / Overhead", color="#7f7f7f")
        for idx_b, tot in enumerate(tot_s):
            ax.text(idx_b, tot + max(tot_s) * 0.02, f"{tot:.1f}s", ha="center", va="bottom", fontsize=11.5, fontweight="bold")
        ax.set_title(cfg["title"], fontsize=16, fontweight="bold", pad=10)
        ax.set_ylabel("Tempo Médio por Rodada (s)", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_b, fontsize=11.5)
        ax.set_ylim(0, max(tot_s) * 1.16)
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        if not handles_t:
            handles_t, labels_leg_t = ax.get_legend_handles_labels()

    fig_t.suptitle(
        "Decomposição do Tempo Médio por Rodada (s) nos 4 Datasets (SLM Qwen-VL)",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )
    if handles_t:
        fig_t.legend(
            handles_t,
            labels_leg_t,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncol=4,
            fontsize=12,
            framealpha=0.95,
        )
    fig_t.tight_layout(rect=[0, 0, 1, 0.90])
    for ext in ("pdf", "png"):
        fig_t.savefig(os.path.join(out_dir, f"painel_4datasets_tempo_decomposicao.{ext}"), dpi=300)
    plt.close(fig_t)
    print(f"📊 Painel de Tempo salvo em: {os.path.join(out_dir, 'painel_4datasets_tempo_decomposicao.png')}", flush=True)


def generate_oxfordpets_ablation_dashboard(out_root: str):
    """Gera um painel 2x2 focado na ablação das estratégias no OxfordPets."""
    out_dir = os.path.join(out_root, "paper")
    os.makedirs(out_dir, exist_ok=True)

    cfg = DATASET_CONFIGS["OxfordPets"]
    lora_dir = os.path.join(BASE_RESULTS, cfg["lora"])
    sora_dir = os.path.join(BASE_RESULTS, cfg["sora"])
    puma_dir = os.path.join(BASE_RESULTS, cfg["puma"])
    onlypaca_dir = os.path.join(BASE_RESULTS, cfg["onlypaca"])
    if not os.path.exists(onlypaca_dir):
        return

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

    ax1 = axes[0, 0]
    ax1.plot(rounds, l_acc, label=f"LoRA Padrão ({l_acc[-1]:.2f}%)", color="#d62728", linestyle="--", linewidth=2.2)
    ax1.plot(rounds, s_acc, label=f"SoRA Estático ({s_acc[-1]:.2f}%)", color="#ff7f0e", linestyle=":", linewidth=2.5)
    ax1.plot(rounds, op_acc, label=f"PUMA-GT ({op_acc[-1]:.2f}%)", color="#9467bd", linestyle="-.", linewidth=2.5)
    ax1.plot(rounds, p_acc, label=f"PUMA-GT Evolution ({p_acc[-1]:.2f}%)", color="#1f77b4", linestyle="-", linewidth=3.0)
    ax1.set_title("Acurácia Auxiliar Ponderada dos Clientes", fontsize=15, fontweight="bold", pad=10)
    ax1.set_xlabel("Rodadas de Comunicação", fontsize=13)
    ax1.set_ylabel("Acurácia (%)", fontsize=13)
    ax1.set_xlim(1, 100)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=12, framealpha=0.95)

    ax2 = axes[0, 1]
    ax2.plot(rounds, l_mb, label=f"LoRA Padrão ({l_mb[-1]:,.0f} MB)", color="#d62728", linestyle="--", linewidth=2.2)
    ax2.plot(rounds, s_mb, label=f"SoRA Estático ({s_mb[-1]:,.0f} MB, -{(1 - s_mb[-1] / l_mb[-1]) * 100:.1f}%)", color="#ff7f0e", linestyle=":", linewidth=2.5)
    ax2.plot(rounds, op_mb, label=f"PUMA-GT ({op_mb[-1]:,.0f} MB, -{(1 - op_mb[-1] / l_mb[-1]) * 100:.1f}%)", color="#9467bd", linestyle="-.", linewidth=2.5)
    ax2.plot(rounds, p_mb, label=f"PUMA-GT Evolution ({p_mb[-1]:,.0f} MB, -{(1 - p_mb[-1] / l_mb[-1]) * 100:.1f}%)", color="#1f77b4", linestyle="-", linewidth=3.0)
    ax2.set_title("Tráfego de Rede Acumulado (MB)", fontsize=15, fontweight="bold", pad=10)
    ax2.set_xlabel("Rodadas de Comunicação", fontsize=13)
    ax2.set_ylabel("Banda Acumulada (MB)", fontsize=13)
    ax2.set_xlim(1, 100)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=12, framealpha=0.95)

    ax3 = axes[1, 0]
    ax3.plot(rounds, l_paca, label="LoRA (p=28 fixo)", color="#d62728", linestyle="--", linewidth=2.2)
    ax3.plot(rounds, s_paca, label="SoRA (p=28 fixo)", color="#ff7f0e", linestyle=":", linewidth=2.5)
    ax3.plot(rounds, op_paca, label=f"PUMA-GT (28 → {op_paca[-1]:.1f})", color="#9467bd", linestyle="-.", linewidth=2.5)
    ax3.plot(rounds, p_paca, label=f"PUMA-GT Evolution (28 → {p_paca[-1]:.1f})", color="#1f77b4", linestyle="-", linewidth=3.0)
    ax3.set_title("Evolução das Camadas Ativas Médias (PaCA)", fontsize=15, fontweight="bold", pad=10)
    ax3.set_xlabel("Rodadas de Comunicação", fontsize=13)
    ax3.set_ylabel("Número de Camadas Ativas", fontsize=13)
    ax3.set_xlim(1, 100)
    ax3.set_ylim(0, 30)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="lower left", fontsize=12, framealpha=0.95)

    ax4 = axes[1, 1]
    ax4.plot(rounds, l_rank, label="LoRA (r=8 fixo)", color="#d62728", linestyle="--", linewidth=2.2)
    ax4.plot(rounds, s_rank, label="SoRA (r=8 fixo)", color="#ff7f0e", linestyle=":", linewidth=2.5)
    ax4.plot(rounds, op_rank, label="PUMA-GT (r=8 fixo)", color="#9467bd", linestyle="-.", linewidth=2.5)
    ax4.plot(rounds, p_rank, label=f"PUMA-GT Evolution (8 → {p_rank[-1]:.1f})", color="#1f77b4", linestyle="-", linewidth=3.0)
    ax4.set_title("Evolução do Rank Efetivo Médio", fontsize=15, fontweight="bold", pad=10)
    ax4.set_xlabel("Rodadas de Comunicação", fontsize=13)
    ax4.set_ylabel("Rank Efetivo", fontsize=13)
    ax4.set_xlim(1, 100)
    ax4.set_ylim(1, 9)
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="lower left", fontsize=12, framealpha=0.95)

    fig.suptitle(
        "Estudo de Ablação: LoRA vs. SoRA vs. PUMA-GT vs. PUMA-GT Evolution (OxfordPets / Qwen-VL)",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    pdf_out = os.path.join(out_dir, "painel_oxfordpets_ablacao_4estrategias.pdf")
    png_out = os.path.join(out_dir, "painel_oxfordpets_ablacao_4estrategias.png")
    fig.savefig(pdf_out, dpi=300)
    fig.savefig(png_out, dpi=300)
    plt.close(fig)
    print(f"📊 Painel de Ablação OxfordPets salvo em: {pdf_out} e {png_out}")

    oxf_dir = os.path.join(out_root, "OxfordPets")
    if os.path.exists(oxf_dir):
        shutil.copy2(png_out, os.path.join(oxf_dir, "painel_oxfordpets_ablacao_4estrategias.png"))
        shutil.copy2(pdf_out, os.path.join(oxf_dir, "painel_oxfordpets_ablacao_4estrategias.pdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera gráficos comparativos SLM (4 datasets).")
    parser.add_argument(
        "--4estrategias",
        dest="four_strategies",
        action="store_true",
        help="Plota LoRA, SoRA Estático, PUMA-GT Evolution e PUMA-GT E + Delta Coding.",
    )
    parser.add_argument("--include-delta", action="store_true")
    parser.add_argument("--exclude-onlypaca", action="store_true")
    parser.add_argument("--normalize-lora-bf16", action="store_true")
    parser.add_argument("--out-subdir", type=str, default=None)
    parser.add_argument(
        "--delta-suffix",
        type=str,
        default="_deltacoding",
        help="Sufixo das pastas PUMA-GT E + Delta Coding (ex: _deltacoding ou _deltacoding_v2).",
    )
    args = parser.parse_args()

    if args.delta_suffix != "_deltacoding":
        for cfg in DATASET_CONFIGS.values():
            cfg["puma_delta"] = f"{cfg['puma']}{args.delta_suffix}"

    inc_delta = args.include_delta or args.four_strategies
    exc_onlypaca = args.exclude_onlypaca or args.four_strategies
    norm_bf16 = args.normalize_lora_bf16 or args.four_strategies
    subdir = args.out_subdir or ("graficos/puma_gte_delta_suite" if args.four_strategies else "graficos")
    out_root = os.path.join(BASE_RESULTS, subdir)

    print(f"🚀 Iniciando geração de gráficos SLM em: {out_root}")
    run_dataset_plots(
        out_root,
        include_delta=inc_delta,
        exclude_onlypaca=exc_onlypaca,
        normalize_lora_bf16=norm_bf16,
    )
    generate_multi_dataset_dashboard(
        out_root, include_delta=inc_delta, exclude_onlypaca=exc_onlypaca
    )
    if not exc_onlypaca:
        generate_oxfordpets_ablation_dashboard(out_root)
    print("\n🎉 TODOS OS GRÁFICOS FORAM GERADOS COM SUCESSO!")
