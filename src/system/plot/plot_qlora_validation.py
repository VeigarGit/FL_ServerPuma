#!/usr/bin/env python3
"""Gerador de gráficos científicos para validação ponta a ponta do QLoRA no FL_ServerPuma.

Lê os resultados HDF5 de servidor e clientes combinados com a telemetria contínua de GPU
(`gpu_metrics.csv` via nvidia-smi) gerada durante os experimentos e produz painéis em
alta resolução (PNG e PDF) cobrindo VRAM, uso de GPU, potência/energia, decomposição de tempo,
dados transmitidos, convergência (acurácia/loss), Pareto de eficiência e dinâmica PUMA-GT+.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

PALETTE = {
    "slm_lora": {
        "color": "#d62728",
        "ls": "--",
        "marker": "s",
        "label": "LoRA Padrão (BF16/FP32)",
    },
    "slm_qlora": {"color": "#1f77b4", "ls": "-", "marker": "o", "label": "QLoRA (4-bit NF4)"},
    "slm_qlora_puma": {
        "color": "#2ca02c",
        "ls": "-.",
        "marker": "D",
        "label": "QLoRA + PUMA-GT+ + LHDQ",
    },
    "d1_lora": {"color": "#ff7f0e", "ls": "--", "marker": "^", "label": "d1-2B LoRA (BF16)"},
    "d1_qlora": {"color": "#9467bd", "ls": "-", "marker": "v", "label": "d1-2B QLoRA (4-bit NF4)"},
}


def _save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    """Salva figura em PNG (180 DPI) e PDF com layout ajustado."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Salvo: {png_path} e {pdf_path}")


def _load_gpu_csv(csv_path: Path) -> dict[str, np.ndarray]:
    """Carrega série temporal de GPU coletada via nvidia-smi."""
    cols: dict[str, list[float]] = {
        "elapsed_s": [],
        "memory_used_mib": [],
        "memory_free_mib": [],
        "fl_vram_mib": [],
        "fl_process_count": [],
        "gpu_util_pct": [],
        "mem_util_pct": [],
        "power_draw_w": [],
        "temp_c": [],
    }
    if not csv_path.exists():
        return {k: np.array([], dtype=float) for k in cols}

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key in cols:
                cols[key].append(float(row[key]))
    return {k: np.array(v, dtype=float) for k, v in cols.items()}


def _load_latest_h5_data(exp_dir: Path) -> dict[str, Any]:
    """Carrega datasets do último arquivo server_*.h5 e dos client_*.h5."""
    server_files = sorted(glob.glob(str(exp_dir / "server_*.h5")))
    if not server_files:
        return {}
    server_path = Path(server_files[-1])

    data: dict[str, Any] = {"server_h5": str(server_path)}
    with h5py.File(server_path, "r") as handle:
        for key in handle:
            data[key] = np.array(handle[key][()])

    client_files = sorted(glob.glob(str(exp_dir / "client_*.h5")))
    local_accs = []
    global_accs = []
    for cpath in client_files:
        with h5py.File(cpath, "r") as handle:
            if "rs_local_acc" in handle:
                local_accs.append(np.array(handle["rs_local_acc"][()], dtype=float))
            if "rs_global_acc" in handle:
                global_accs.append(np.array(handle["rs_global_acc"][()], dtype=float))

    if local_accs:
        min_len = min(len(arr) for arr in local_accs)
        stacked = np.stack([arr[:min_len] for arr in local_accs], axis=0)
        data["client_local_acc_mean"] = stacked.mean(axis=0)
        data["client_local_acc_std"] = stacked.std(axis=0)
    if global_accs:
        min_len = min(len(arr) for arr in global_accs)
        stacked = np.stack([arr[:min_len] for arr in global_accs], axis=0)
        data["client_global_acc_mean"] = stacked.mean(axis=0)
    return data


def _compute_energy_wh(elapsed_s: np.ndarray, power_w: np.ndarray) -> float:
    """Integra potência elétrica (W) ao longo do tempo (s) para obter energia em Wh."""
    if len(elapsed_s) < 2:
        return 0.0
    dt = np.diff(elapsed_s)
    p_avg = 0.5 * (power_w[:-1] + power_w[1:])
    return float(np.sum(dt * p_avg) / 3600.0)


def load_experiment_bundle(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Agrega metadados, HDF5 e telemetria GPU de um experimento do manifesto."""
    exp_dir = Path(entry["results_dir"])
    h5_data = _load_latest_h5_data(exp_dir)
    if not h5_data or "rs_test_acc" not in h5_data or len(h5_data["rs_test_acc"]) == 0:
        print(f"[AVISO] Experimento incompleto ou sem H5: {entry['id']}")
        return None

    gpu = _load_gpu_csv(Path(entry["telemetry_dir"]) / "gpu_metrics.csv")
    energy_wh = _compute_energy_wh(gpu["elapsed_s"], gpu["power_draw_w"])
    fl_vram_gb = gpu["fl_vram_mib"] / 1024.0 if len(gpu["fl_vram_mib"]) else np.array([0.0])
    total_vram_gb = (
        gpu["memory_used_mib"] / 1024.0 if len(gpu["memory_used_mib"]) else np.array([0.0])
    )
    style = PALETTE.get(
        entry["id"],
        {"color": "#333333", "ls": "-", "marker": "o", "label": entry.get("label", entry["id"])},
    )

    round_time = h5_data.get("Round_time", np.array([0.0]))
    tx_round = h5_data.get("total_transmitted_per_round_Mb", np.array([0.0]))
    sended_raw = h5_data.get("Sended_without_quant", np.array([0.0]))
    acc = h5_data["rs_test_acc"]
    loss = h5_data["rs_test_loss"]

    train_t = h5_data.get("rs_client_training_time")
    eval_t = h5_data.get("rs_client_eval_time")
    comm_t = h5_data.get("rs_client_comm_time")
    srv_t = h5_data.get("rs_server_processing_time")

    summary = {
        "id": entry["id"],
        "suite": entry["suite"],
        "label": entry.get("label", style["label"]),
        "rounds": int(len(acc)),
        "initial_acc_pct": float(acc[0]),
        "final_acc_pct": float(acc[-1]),
        "best_acc_pct": float(np.max(acc)),
        "final_loss": float(loss[-1]),
        "final_local_acc_pct": float(h5_data["client_local_acc_mean"][-1])
        if "client_local_acc_mean" in h5_data
        else float("nan"),
        "total_wall_min": float(entry.get("wall_seconds", float(np.sum(round_time))) / 60.0),
        "avg_round_s": float(np.mean(round_time)),
        "avg_train_s": float(np.mean(train_t)) if train_t is not None else 0.0,
        "avg_eval_s": float(np.mean(eval_t)) if eval_t is not None else 0.0,
        "avg_comm_s": float(np.mean(comm_t)) if comm_t is not None else 0.0,
        "avg_server_s": float(np.mean(srv_t)) if srv_t is not None else 0.0,
        "model_payload_mb": float(h5_data["Model_size_per_round_Mb"][0])
        if "Model_size_per_round_Mb" in h5_data
        else 0.0,
        "total_tx_mb": float(np.sum(tx_round)),
        "raw_tx_mb": float(sended_raw[-1]) * 2.0 if len(sended_raw) else 0.0,
        "trainable_params": int(h5_data["Trainable_params"][0])
        if "Trainable_params" in h5_data
        else 0,
        "peak_fl_vram_gb": float(np.max(fl_vram_gb)),
        "mean_fl_vram_gb": float(np.mean(fl_vram_gb)),
        "peak_total_vram_gb": float(np.max(total_vram_gb)),
        "mean_gpu_util_pct": float(np.mean(gpu["gpu_util_pct"]))
        if len(gpu["gpu_util_pct"])
        else 0.0,
        "peak_gpu_util_pct": float(np.max(gpu["gpu_util_pct"]))
        if len(gpu["gpu_util_pct"])
        else 0.0,
        "mean_power_w": float(np.mean(gpu["power_draw_w"])) if len(gpu["power_draw_w"]) else 0.0,
        "peak_power_w": float(np.max(gpu["power_draw_w"])) if len(gpu["power_draw_w"]) else 0.0,
        "energy_wh": energy_wh,
        "peak_temp_c": float(np.max(gpu["temp_c"])) if len(gpu["temp_c"]) else 0.0,
    }

    return {
        "entry": entry,
        "style": {**style, "label": summary["label"]},
        "h5": h5_data,
        "gpu": gpu,
        "summary": summary,
    }


def plot_gpu_vram_timeline(
    bundles: list[dict[str, Any]], out_dir: Path, title_suffix: str
) -> None:
    """Gráfico 01: Série temporal de VRAM alocada na GPU 0 e comparação de pico/média."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.85, 1.0]}
    )

    for bundle in bundles:
        gpu = bundle["gpu"]
        st = bundle["style"]
        if len(gpu["elapsed_s"]) == 0:
            continue
        t_min = gpu["elapsed_s"] / 60.0
        vram_gb = gpu["fl_vram_mib"] / 1024.0
        ax1.plot(t_min, vram_gb, color=st["color"], ls=st["ls"], lw=2.0, label=st["label"])

    ax1.set_xlabel("Tempo decorrido (min)")
    ax1.set_ylabel("VRAM usada pelos processos FL (GB)")
    ax1.set_title(f"Uso Contínuo de VRAM na GPU 0 — {title_suffix}")
    ax1.legend(loc="best", frameon=True)

    labels = [b["style"]["label"].replace(" + ", "\n+ ") for b in bundles]
    peaks = [b["summary"]["peak_fl_vram_gb"] for b in bundles]
    means = [b["summary"]["mean_fl_vram_gb"] for b in bundles]
    colors = [b["style"]["color"] for b in bundles]
    x = np.arange(len(bundles))
    width = 0.36

    bars1 = ax2.bar(x - width / 2, peaks, width, color=colors, alpha=0.92, label="Pico (GB)")
    bars2 = ax2.bar(
        x + width / 2, means, width, color=colors, alpha=0.45, hatch="//", label="Média (GB)"
    )

    base_peak = peaks[0] if peaks else 0.0
    for idx, bar in enumerate(bars1):
        val = bar.get_height()
        pct_txt = f"\n(-{(1.0 - val / base_peak) * 100:.0f}%)" if idx > 0 and base_peak > 0 else ""
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.6,
            f"{val:.1f} GB{pct_txt}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    for bar in bars2:
        val = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.6,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("VRAM na GPU 0 (GB)")
    ax2.set_ylim(0, max(peaks) * 1.25 if peaks else 10)
    ax2.set_title("Pico vs Média de VRAM")
    ax2.legend(loc="upper right", fontsize=9)

    _save_fig(fig, out_dir, "01_gpu_vram_timeline")


def plot_gpu_compute_power_thermals(
    bundles: list[dict[str, Any]], out_dir: Path, title_suffix: str
) -> None:
    """Gráfico 02: Utilização de GPU (%), Potência (W) e Temperatura (°C) ao longo do tempo."""
    fig, axes = plt.subplots(3, 1, figsize=(13, 8.5), sharex=True)

    for bundle in bundles:
        gpu = bundle["gpu"]
        st = bundle["style"]
        sm = bundle["summary"]
        if len(gpu["elapsed_s"]) == 0:
            continue
        t_min = gpu["elapsed_s"] / 60.0

        # Média móvel curta (5s) para suavizar picos de sincronização TCP entre rodadas
        kernel = np.ones(5) / 5.0
        util_smooth = (
            np.convolve(gpu["gpu_util_pct"], kernel, mode="same")
            if len(gpu["gpu_util_pct"]) >= 5
            else gpu["gpu_util_pct"]
        )
        pwr_smooth = (
            np.convolve(gpu["power_draw_w"], kernel, mode="same")
            if len(gpu["power_draw_w"]) >= 5
            else gpu["power_draw_w"]
        )

        axes[0].plot(
            t_min,
            util_smooth,
            color=st["color"],
            ls=st["ls"],
            lw=1.7,
            label=f"{st['label']} (média {sm['mean_gpu_util_pct']:.1f}%)",
        )
        axes[1].plot(
            t_min,
            pwr_smooth,
            color=st["color"],
            ls=st["ls"],
            lw=1.7,
            label=f"{st['label']} (média {sm['mean_power_w']:.0f}W | {sm['energy_wh']:.1f} Wh)",
        )
        axes[2].plot(
            t_min,
            gpu["temp_c"],
            color=st["color"],
            ls=st["ls"],
            lw=1.7,
            label=f"{st['label']} (máx {sm['peak_temp_c']:.0f}°C)",
        )

    axes[0].set_ylabel("Utilização SM (%)")
    axes[0].set_title(f"Telemetria Computacional, Potência e Térmica da GPU 0 — {title_suffix}")
    axes[0].legend(loc="best", fontsize=9)

    axes[1].set_ylabel("Potência (W)")
    axes[1].legend(loc="best", fontsize=9)

    axes[2].set_ylabel("Temperatura (°C)")
    axes[2].set_xlabel("Tempo decorrido (min)")
    axes[2].legend(loc="best", fontsize=9)

    _save_fig(fig, out_dir, "02_gpu_compute_power_thermals")


def plot_time_and_decomposition(
    bundles: list[dict[str, Any]], out_dir: Path, title_suffix: str
) -> None:
    """Gráfico 03: Tempo por rodada e decomposição temporal (treino, eval, rede, servidor)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    for bundle in bundles:
        st = bundle["style"]
        rt = bundle["h5"].get("Round_time", np.array([]))
        if len(rt) == 0:
            continue
        rounds = np.arange(1, len(rt) + 1)
        ax1.plot(
            rounds,
            rt,
            color=st["color"],
            ls=st["ls"],
            marker=st["marker"],
            markersize=5,
            lw=2.0,
            label=f"{st['label']} (média {np.mean(rt):.1f}s)",
        )

    ax1.set_xlabel("Rodada Federada")
    ax1.set_ylabel("Tempo Wall-Clock por Rodada (s)")
    ax1.set_title(f"Latência por Rodada — {title_suffix}")
    ax1.legend(loc="best", fontsize=9)

    labels = [b["style"]["label"].replace(" + ", "\n+ ") for b in bundles]
    train_s = np.array([b["summary"]["avg_train_s"] for b in bundles])
    eval_s = np.array([b["summary"]["avg_eval_s"] for b in bundles])
    comm_s = np.array([b["summary"]["avg_comm_s"] for b in bundles])
    srv_s = np.array([b["summary"]["avg_server_s"] for b in bundles])

    x = np.arange(len(bundles))
    width = 0.5
    ax2.bar(x, train_s, width, label="Treino Local", color="#2b5c8f")
    ax2.bar(x, eval_s, width, bottom=train_s, label="Eval + Overhead Cliente", color="#f28e2b")
    ax2.bar(x, comm_s, width, bottom=train_s + eval_s, label="Comunicação TCP", color="#59a14f")
    ax2.bar(
        x,
        srv_s,
        width,
        bottom=train_s + eval_s + comm_s,
        label="Processamento Servidor",
        color="#e15759",
    )

    totals = train_s + eval_s + comm_s + srv_s
    for idx, tot in enumerate(totals):
        ax2.text(
            x[idx],
            tot + max(totals) * 0.02 if max(totals) > 0 else 1,
            f"{tot:.1f}s",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Tempo Médio por Cliente/Rodada (s)")
    ax2.set_ylim(0, max(totals) * 1.18 if len(totals) and max(totals) > 0 else 10)
    ax2.set_title("Decomposição Média por Etapa")
    ax2.legend(loc="best", fontsize=9)

    _save_fig(fig, out_dir, "03_tempo_rodada_e_decomposicao")


def plot_transmitted_data(bundles: list[dict[str, Any]], out_dir: Path, title_suffix: str) -> None:
    """Gráfico 04: Dados transmitidos acumulados e comparação bruto vs comprimido."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    for bundle in bundles:
        st = bundle["style"]
        tx_round = bundle["h5"].get("total_transmitted_per_round_Mb", np.array([]))
        if len(tx_round) == 0:
            continue
        rounds = np.arange(1, len(tx_round) + 1)
        cum_mb = np.cumsum(tx_round)
        ax1.plot(
            rounds,
            cum_mb,
            color=st["color"],
            ls=st["ls"],
            marker=st["marker"],
            markersize=5,
            lw=2.0,
            label=f"{st['label']} ({cum_mb[-1]:.1f} MB)",
        )

    ax1.set_xlabel("Rodada Federada")
    ax1.set_ylabel("Dados Transmitidos Acumulados (MB)")
    ax1.set_title("Tráfego Acumulado de Rede (Upload + Download)")
    ax1.legend(loc="best", fontsize=9)

    labels = [b["style"]["label"].replace(" + ", "\n+ ") for b in bundles]
    raw_mb = [b["summary"]["raw_tx_mb"] for b in bundles]
    tx_mb = [b["summary"]["total_tx_mb"] for b in bundles]
    x = np.arange(len(bundles))
    width = 0.36

    b_raw = ax2.bar(x - width / 2, raw_mb, width, color="#b0b0b0", label="Sem compressão (MB)")
    b_tx = ax2.bar(
        x + width / 2,
        tx_mb,
        width,
        color=[b["style"]["color"] for b in bundles],
        label="Com compressão (INT8 / LHDQ)",
    )
    for bar in b_raw:
        val = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + max(raw_mb) * 0.015 if max(raw_mb) > 0 else 1,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    for idx, bar in enumerate(b_tx):
        val = bar.get_height()
        r = raw_mb[idx]
        saving = f"\n(-{(1.0 - val / r) * 100:.1f}%)" if r > 0 else ""
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + max(raw_mb) * 0.015 if max(raw_mb) > 0 else 1,
            f"{val:.0f}{saving}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Volume Total no Experimento (MB)")
    ax2.set_ylim(0, max(raw_mb) * 1.2 if max(raw_mb) > 0 else 100)
    ax2.set_title("Economia Total de Banda (Bruto vs Transmitido)")
    ax2.legend(loc="upper right", fontsize=9)

    _save_fig(fig, out_dir, "04_dados_transmitidos_banda")


def plot_accuracy_and_loss(
    bundles: list[dict[str, Any]], out_dir: Path, title_suffix: str
) -> None:
    """Gráfico 05: Convergência de Acurácia Auxiliar/Local (%) e Loss Global."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    for bundle in bundles:
        st = bundle["style"]
        h5 = bundle["h5"]
        aux_acc = h5.get("rs_aux_simple_avg_acc")
        has_aux = aux_acc is not None and len(aux_acc) == len(h5["rs_test_acc"])
        acc = aux_acc if has_aux else h5["rs_test_acc"]
        acc_tag = "Auxiliar" if has_aux else "Global"
        loss = h5["rs_test_loss"]
        rounds = np.arange(1, len(acc) + 1)

        ax1.plot(
            rounds,
            acc,
            color=st["color"],
            ls=st["ls"],
            marker=st["marker"],
            markersize=5,
            lw=2.1,
            label=f"{st['label']} ({acc_tag}: {acc[-1]:.2f}%)",
        )
        if "client_local_acc_mean" in h5:
            loc_acc = h5["client_local_acc_mean"]
            loc_rounds = np.arange(1, len(loc_acc) + 1)
            ax1.plot(
                loc_rounds,
                loc_acc,
                color=st["color"],
                ls=":",
                lw=1.3,
                alpha=0.65,
            )

        ax2.plot(
            rounds,
            loss,
            color=st["color"],
            ls=st["ls"],
            marker=st["marker"],
            markersize=5,
            lw=2.1,
            label=f"{st['label']} (Final: {loss[-1]:.3f})",
        )

    ax1.set_xlabel("Rodada Federada")
    ax1.set_ylabel("Acurácia (%) — Principal: Auxiliar | Pontilhada: Local")
    ax1.set_title("Convergência de Acurácia — Auxiliar vs Local")
    ax1.set_ylim(0, 102)
    ax1.legend(loc="lower right", fontsize=9)

    ax2.set_xlabel("Rodada Federada")
    ax2.set_ylabel("Loss Média Global")
    ax2.set_title("Evolução da Loss Global")
    ax2.legend(loc="upper right", fontsize=9)

    _save_fig(fig, out_dir, "05_convergencia_acuracia_loss")


def plot_efficiency_pareto(
    bundles: list[dict[str, Any]], out_dir: Path, title_suffix: str
) -> None:
    """Gráfico 06: Pareto multidimensional (Acurácia × VRAM × Banda) e ganhos relativos."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.6))

    max_tx = max(b["summary"]["total_tx_mb"] for b in bundles) or 1.0
    for idx, bundle in enumerate(bundles):
        sm = bundle["summary"]
        st = bundle["style"]
        size = 180 + 750 * (sm["total_tx_mb"] / max_tx)
        ax1.scatter(
            sm["peak_fl_vram_gb"],
            sm["final_acc_pct"],
            s=size,
            color=st["color"],
            alpha=0.82,
            edgecolors="black",
            linewidths=1.2,
            label=st["label"],
        )
        annot_line2 = (
            f"{sm['total_tx_mb']:.0f} MB | {sm['total_wall_min']:.1f} min | "
            f"{sm['energy_wh']:.1f} Wh"
        )
        # Evita sobreposição quando dois métodos empatam no pico de VRAM (QLoRA vs PUMA)
        if idx == 1:
            xytext = (18, 22)
        elif idx == 2:
            xytext = (18, -36)
        else:
            xytext = (0, -34 if sm["final_acc_pct"] > 75 else 18)
        ax1.annotate(
            f"{sm['peak_fl_vram_gb']:.1f} GB | {sm['final_acc_pct']:.1f}%\n{annot_line2}",
            xy=(sm["peak_fl_vram_gb"], sm["final_acc_pct"]),
            xytext=xytext,
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "alpha": 0.90, "ec": st["color"]},
        )

    ax1.set_xlabel("Pico de VRAM FL na GPU 0 (GB)  [← Melhor]")
    ax1.set_ylabel("Acurácia Global Final (%)  [↑ Melhor]")
    ax1.set_title("Pareto: Memória GPU × Acurácia (área = MB rede)")
    ax1.set_xlim(0, max(1.0, max(b["summary"]["peak_fl_vram_gb"] for b in bundles)) * 1.28)
    ax1.set_ylim(
        max(0, min(b["summary"]["final_acc_pct"] for b in bundles) - 18),
        min(105, max(b["summary"]["final_acc_pct"] for b in bundles) + 12),
    )

    # Painel B: Barras de redução relativa contra o baseline (LoRA BF16)
    base = bundles[0]["summary"]
    metrics = ["Pico VRAM", "VRAM Média", "Banda Total", "Tempo Total", "Energia (Wh)"]
    x = np.arange(len(metrics))
    others = bundles[1:]
    all_vals: list[float] = []
    if others:
        width = 0.72 / len(others)
        for idx, bundle in enumerate(others):
            sm = bundle["summary"]
            st = bundle["style"]
            vals = [
                (1.0 - sm["peak_fl_vram_gb"] / max(base["peak_fl_vram_gb"], 1e-6)) * 100.0,
                (1.0 - sm["mean_fl_vram_gb"] / max(base["mean_fl_vram_gb"], 1e-6)) * 100.0,
                (1.0 - sm["total_tx_mb"] / max(base["total_tx_mb"], 1e-6)) * 100.0,
                (1.0 - sm["total_wall_min"] / max(base["total_wall_min"], 1e-6)) * 100.0,
                (1.0 - sm["energy_wh"] / max(base["energy_wh"], 1e-6)) * 100.0,
            ]
            all_vals.extend(vals)
            offset = (idx - (len(others) - 1) / 2.0) * width
            bars = ax2.bar(x + offset, vals, width, color=st["color"], label=st["label"])
            for bar, v in zip(bars, vals, strict=True):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + (1.4 if v >= 0 else -1.6),
                    f"{v:+.1f}%",
                    ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=8,
                    fontweight="bold",
                )
        ax2.axhline(0, color="black", lw=0.9)
        ax2.legend(loc="best", fontsize=9)

    y_min = min(-8.0, min(all_vals) - 8.0) if all_vals else -5.0
    y_max = max(55.0, max(all_vals) + 10.0) if all_vals else 55.0
    ax2.set_ylim(y_min, y_max)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=9)
    ax2.set_ylabel("Redução / Ganho vs LoRA Padrão (%)")
    ax2.set_title("Economia Percentual em Relação ao Baseline")

    _save_fig(fig, out_dir, "06_tradeoff_eficiencia_global")


def plot_puma_dynamics(bundle: dict[str, Any], out_dir: Path) -> None:
    """Gráfico 07: Dinâmica adaptativa de PaCA, Rank, γ^c e tráfego no PUMA-GT+."""
    h5 = bundle["h5"]
    if "rs_client_paca" not in h5 or "rs_client_rank" not in h5:
        return
    paca = np.array(h5["rs_client_paca"], dtype=float)
    rank = np.array(h5["rs_client_rank"], dtype=float)
    gamma = np.array(h5.get("rs_client_gamma", np.ones_like(paca)), dtype=float)
    tx = np.array(h5.get("total_transmitted_per_round_Mb", np.zeros(len(paca))), dtype=float)
    rounds = np.arange(1, len(paca) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8))

    axes[0, 0].plot(rounds, paca.mean(axis=1), "o-", color="#2ca02c", lw=2, label="Média PaCA")
    axes[0, 0].fill_between(
        rounds, paca.min(axis=1), paca.max(axis=1), color="#2ca02c", alpha=0.2, label="Min–Max"
    )
    axes[0, 0].set_title("Camadas Ativas (PaCA Adaptativo)")
    axes[0, 0].set_ylabel("Camadas")
    axes[0, 0].legend(loc="best", fontsize=9)

    axes[0, 1].plot(rounds, rank.mean(axis=1), "s-", color="#1f77b4", lw=2, label="Média Rank")
    axes[0, 1].fill_between(
        rounds, rank.min(axis=1), rank.max(axis=1), color="#1f77b4", alpha=0.2, label="Min–Max"
    )
    axes[0, 1].set_title("Rank LoRA Adaptativo por Cliente")
    axes[0, 1].set_ylabel("Rank r")
    axes[0, 1].legend(loc="best", fontsize=9)

    axes[1, 0].plot(rounds, gamma.mean(axis=1), "D-", color="#9467bd", lw=2, label="Média γ^c")
    axes[1, 0].fill_between(
        rounds, gamma.min(axis=1), gamma.max(axis=1), color="#9467bd", alpha=0.2, label="Min–Max"
    )
    axes[1, 0].axvline(12, color="gray", ls="--", lw=1.1, label="Início check platô (r=12)")
    axes[1, 0].set_title("Índice de Capacidade PUMA-GT+ (γ^c)")
    axes[1, 0].set_xlabel("Rodada Federada")
    axes[1, 0].set_ylabel("γ^c")
    axes[1, 0].set_ylim(0, 1.08)
    axes[1, 0].legend(loc="best", fontsize=9)

    axes[1, 1].bar(rounds, tx, color="#ff7f0e", alpha=0.85, label="Tráfego LHDQ / Rodada")
    axes[1, 1].set_title("Tráfego Transmitido por Rodada (LHDQ)")
    axes[1, 1].set_xlabel("Rodada Federada")
    axes[1, 1].set_ylabel("MB / Rodada")
    axes[1, 1].legend(loc="best", fontsize=9)

    _save_fig(fig, out_dir, "07_puma_adaptativo_qlora")


def plot_d1_suite(bundles_d1: list[dict[str, Any]], out_dir: Path) -> None:
    """Gráfico 08: Comparativo consolidado 2x2 para LiquidAI/d1-2B (LoRA BF16 vs QLoRA NF4)."""
    if len(bundles_d1) < 2:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))

    for bundle in bundles_d1:
        gpu = bundle["gpu"]
        h5 = bundle["h5"]
        st = bundle["style"]
        sm = bundle["summary"]

        if len(gpu["elapsed_s"]):
            axes[0, 0].plot(
                gpu["elapsed_s"] / 60.0,
                gpu["fl_vram_mib"] / 1024.0,
                color=st["color"],
                ls=st["ls"],
                lw=2.0,
                label=f"{st['label']} (pico {sm['peak_fl_vram_gb']:.1f} GB)",
            )
        rounds = np.arange(1, len(h5["rs_test_acc"]) + 1)
        axes[0, 1].plot(
            rounds,
            h5["rs_test_acc"],
            color=st["color"],
            ls=st["ls"],
            marker=st["marker"],
            lw=2.0,
            label=f"{st['label']} ({sm['final_acc_pct']:.2f}%)",
        )
        axes[1, 0].plot(
            rounds,
            h5["Round_time"],
            color=st["color"],
            ls=st["ls"],
            marker=st["marker"],
            lw=2.0,
            label=f"{st['label']} (média {sm['avg_round_s']:.1f}s)",
        )

    axes[0, 0].set_title("LiquidAI/d1-2B — VRAM FL na GPU 0")
    axes[0, 0].set_xlabel("Tempo decorrido (min)")
    axes[0, 0].set_ylabel("VRAM FL (GB)")
    axes[0, 0].legend(loc="best", fontsize=9)

    axes[0, 1].set_title("LiquidAI/d1-2B — Acurácia Global (Banking77)")
    axes[0, 1].set_xlabel("Rodada Federada")
    axes[0, 1].set_ylabel("Acurácia (%)")
    axes[0, 1].set_xticks(np.arange(1, len(bundles_d1[0]["h5"]["rs_test_acc"]) + 1))
    axes[0, 1].legend(loc="best", fontsize=9)

    axes[1, 0].set_title("LiquidAI/d1-2B — Tempo por Rodada")
    axes[1, 0].set_xlabel("Rodada Federada")
    axes[1, 0].set_ylabel("Tempo (s)")
    axes[1, 0].set_xticks(np.arange(1, len(bundles_d1[0]["h5"]["Round_time"]) + 1))
    axes[1, 0].legend(loc="best", fontsize=9)

    # Resumo quantitativo em barras normalizadas
    categories = [
        "Pico VRAM\n(GB)",
        "Tempo Total\n(min)",
        "Energia GPU\n(Wh)",
        "Tráfego Rede\n(MB)",
    ]
    b0, b1 = bundles_d1[0]["summary"], bundles_d1[1]["summary"]
    v0 = [b0["peak_fl_vram_gb"], b0["total_wall_min"], b0["energy_wh"], b0["total_tx_mb"]]
    v1 = [b1["peak_fl_vram_gb"], b1["total_wall_min"], b1["energy_wh"], b1["total_tx_mb"]]
    x = np.arange(len(categories))
    w = 0.35
    # Normaliza cada métrica pelo máximo entre LoRA e QLoRA para visualização na mesma escala
    maxs = [max(a, b, 1e-6) for a, b in zip(v0, v1, strict=True)]
    r0 = [a / m * 100.0 for a, m in zip(v0, maxs, strict=True)]
    r1 = [b / m * 100.0 for b, m in zip(v1, maxs, strict=True)]
    bars0 = axes[1, 1].bar(
        x - w / 2, r0, w, color=bundles_d1[0]["style"]["color"], label=b0["label"]
    )
    bars1 = axes[1, 1].bar(
        x + w / 2, r1, w, color=bundles_d1[1]["style"]["color"], label=b1["label"]
    )
    for bar, raw in zip(bars0, v0, strict=True):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{raw:.1f}",
            ha="center",
            fontsize=8.5,
        )
    for bar, raw in zip(bars1, v1, strict=True):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{raw:.1f}",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(categories, fontsize=9)
    axes[1, 1].set_ylim(0, 134)
    axes[1, 1].set_ylabel("% relativo ao máximo")
    axes[1, 1].set_title("d1-2B — Comparativo Quantitativo Direto")
    axes[1, 1].legend(loc="upper center", ncol=2, fontsize=8.5, frameon=True)

    _save_fig(fig, out_dir, "08_d1_lora_vs_qlora_comparativo")


def write_summary_tables(bundles: list[dict[str, Any]], suite_dir: Path) -> None:
    """Exporta CSV e tabela Markdown consolidada com todas as métricas observadas."""
    if not bundles:
        return
    csv_path = suite_dir / "validation_summary.csv"
    md_path = suite_dir / "validation_summary.md"
    fields = list(bundles[0]["summary"].keys())

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for bundle in bundles:
            writer.writerow(bundle["summary"])

    header_cols = [
        "Experimento",
        "Rodadas",
        "Acc Inic. (%)",
        "Acc Final (%)",
        "Loss Final",
        "Pico VRAM FL (GB)",
        "Média VRAM FL (GB)",
        "GPU Util (%)",
        "Energia (Wh)",
        "Tempo Total (min)",
        "Média/Rodada (s)",
        "Payload (MB)",
        "Tráfego Total (MB)",
    ]
    align_cols = [":---"] + ["---:"] * (len(header_cols) - 1)
    lines = [
        "# Resumo da Validação Experimental QLoRA (GPU 0)\n",
        "| " + " | ".join(header_cols) + " |",
        "| " + " | ".join(align_cols) + " |",
    ]
    for b in bundles:
        s = b["summary"]
        lines.append(
            f"| {s['label']} | {s['rounds']} | {s['initial_acc_pct']:.2f} | "
            f"{s['final_acc_pct']:.2f} | {s['final_loss']:.4f} | {s['peak_fl_vram_gb']:.2f} | "
            f"{s['mean_fl_vram_gb']:.2f} | {s['mean_gpu_util_pct']:.1f} | {s['energy_wh']:.2f} | "
            f"{s['total_wall_min']:.2f} | {s['avg_round_s']:.1f} | {s['model_payload_mb']:.2f} | "
            f"{s['total_tx_mb']:.2f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[TABELA] Salvo: {csv_path} e {md_path}")


def run_standard_plot_results(slm_bundles: list[dict[str, Any]], suite_dir: Path) -> None:
    """Executa o plot_results.py nativo do repositório para gerar os 15 gráficos padrão."""
    if len(slm_bundles) < 2:
        return
    plot_script = Path(__file__).resolve().parent / "plot_results.py"
    if not plot_script.exists():
        return
    exp_names = [Path(b["entry"]["results_dir"]).name for b in slm_bundles]
    subprocess.run(
        [sys.executable, str(plot_script), *exp_names],
        cwd=str(plot_script.parent),
        check=False,
    )
    default_out = suite_dir.parent / "graficos_comparativos"
    target_out = suite_dir / "plots_padrao_repo"
    if default_out.exists():
        target_out.mkdir(parents=True, exist_ok=True)
        for item in default_out.glob("*"):
            if item.is_file():
                (target_out / item.name).write_bytes(item.read_bytes())
        print(f"[PLOT] Gráficos padrão copiados para: {target_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera gráficos da suíte de validação QLoRA.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("src/results/qlora_validation_suite/validation_manifest.json"),
        help="Caminho para validation_manifest.json",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    suite_dir = manifest_path.parent
    plots_dir = suite_dir / "plots"

    bundles = []
    for entry in manifest.get("experiments", []):
        bundle = load_experiment_bundle(entry)
        if bundle is not None:
            bundles.append(bundle)

    slm_bundles = [b for b in bundles if b["summary"]["suite"] == "slm_oxfordpets"]
    d1_bundles = [b for b in bundles if b["summary"]["suite"] == "d1_banking77"]

    if slm_bundles:
        title = "SLM Qwen3-VL-2B (OxfordPets, 4 clientes)"
        plot_gpu_vram_timeline(slm_bundles, plots_dir, title)
        plot_gpu_compute_power_thermals(slm_bundles, plots_dir, title)
        plot_time_and_decomposition(slm_bundles, plots_dir, title)
        plot_transmitted_data(slm_bundles, plots_dir, title)
        plot_accuracy_and_loss(slm_bundles, plots_dir, title)
        if len(slm_bundles) >= 2:
            plot_efficiency_pareto(slm_bundles, plots_dir, title)
        for b in slm_bundles:
            if b["summary"]["id"] == "slm_qlora_puma":
                plot_puma_dynamics(b, plots_dir)
        run_standard_plot_results(slm_bundles, suite_dir)

    if d1_bundles:
        plot_d1_suite(d1_bundles, plots_dir)

    write_summary_tables(bundles, suite_dir)


if __name__ == "__main__":
    main()
