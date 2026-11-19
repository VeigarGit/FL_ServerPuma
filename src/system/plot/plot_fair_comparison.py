#!/usr/bin/env python3
"""Gera gráficos 100% pareados: BF16 LoRA vs 4-bit QLoRA vs 4-bit QSoRA."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    }
)

VARIANT_ORDER = ("BF16 LoRA", "4-bit QLoRA", "4-bit QSoRA")
VARIANT_STYLE = {
    "BF16 LoRA": {"color": "#d62728", "marker": "o", "ls": "-"},
    "4-bit QLoRA": {"color": "#ff7f0e", "marker": "s", "ls": "-"},
    "4-bit QSoRA": {"color": "#1f77b4", "marker": "^", "ls": "-"},
}
GROUPS = (
    ("slm", "SLM (OxfordPets, 4c, 15r)"),
    ("d1", "d1-2B (Banking77, 4c, 6r)"),
    ("lfm", "LFM3-VL-2B (Banking77, 4c, 6r)"),
)


def _load_h5(results_dir: Path) -> dict[str, np.ndarray]:
    srv_files = sorted(results_dir.glob("server_*.h5"))
    if not srv_files:
        raise FileNotFoundError(f"Sem server_*.h5 em {results_dir}")
    data: dict[str, np.ndarray] = {}
    with h5py.File(srv_files[0], "r") as hf:
        for key in hf:
            data[key] = hf[key][:]
    return data


def _energy_wh(telemetry_dir: Path) -> float:
    csv_path = telemetry_dir / "gpu_metrics.csv"
    if not csv_path.exists():
        return 0.0
    df = pd.read_csv(csv_path)
    t = pd.to_numeric(df["elapsed_s"], errors="coerce").to_numpy(float)
    p = pd.to_numeric(df["power_draw_w"], errors="coerce").to_numpy(float)
    mask = np.isfinite(t) & np.isfinite(p)
    if mask.sum() < 2:
        return 0.0
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(p[mask], t[mask])) / 3600.0


def _save(fig: plt.Figure, out_dir: Path, name: str, suffix: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    stem = f"{name}{suffix}"
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Salvo: {out_dir / f'{stem}.png'}", flush=True)


# Ajuste determinístico fp32 -> bfloat16 nos casos pré-correção em que peft.get_peft_model
# fez upcast automático dos adaptadores LoRA (2 bytes/param vs 4 bytes/param).
_ADAPTER_FP32_TO_BF16_MB = {
    ("slm", "BF16 LoRA"): (8_716_288 * 2) / (1024**2),
    ("d1", "BF16 LoRA"): (8_052_736 * 2) / (1024**2),
    ("d1", "4-bit QLoRA"): (8_052_736 * 2) / (1024**2),
    ("lfm", "BF16 LoRA"): (8_052_736 * 2) / (1024**2),
    ("lfm", "4-bit QLoRA"): (8_052_736 * 2) / (1024**2),
}
# Diferença medida por prepare_frame(round_message("result", ...)) para 8.052.736 params
# (16.106.120 bytes/cliente) nos runs com --timeout (d1 e LFM); no SLM tudo vai em INT8.
_TIMEOUT_RESULT_FP32_TO_BF16_MB_PER_CLIENT = 16_106_120 / (1024**2)


def _apply_bf16_normalization(
    records: list[dict[str, Any]],
    bench_by_key: dict[tuple[str, str], dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
]:
    """Retorna cópias em memória com tamanhos de adaptadores e rede normalizados para BF16."""
    norm_bench = {k: dict(v) for k, v in bench_by_key.items()}
    for (model_key, variant), delta_mb in _ADAPTER_FP32_TO_BF16_MB.items():
        if (model_key, variant) in norm_bench:
            entry = norm_bench[(model_key, variant)]
            entry["trainable_params_mb"] = round(float(entry["trainable_params_mb"]) - delta_mb, 2)
            entry["static_model_mb"] = round(float(entry["static_model_mb"]) - delta_mb, 2)

    norm_records: list[dict[str, Any]] = []
    norm_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in records:
        key = (rec["model"], rec["label"])
        h5_copy = {k: np.array(v, copy=True) for k, v in rec["h5"].items()}
        delta_mb = _ADAPTER_FP32_TO_BF16_MB.get(key, 0.0)
        if delta_mb > 0.0:
            h5_copy["Model_size_per_round_Mb"] = np.maximum(
                0.0, h5_copy["Model_size_per_round_Mb"] - delta_mb
            )
            if rec["model"] in ("d1", "lfm"):
                tx_delta_round = _TIMEOUT_RESULT_FP32_TO_BF16_MB_PER_CLIENT * int(rec["clients"])
                h5_copy["total_transmitted_per_round_Mb"] = np.maximum(
                    0.0, h5_copy["total_transmitted_per_round_Mb"] - tx_delta_round
                )
        new_rec = {**rec, "h5": h5_copy}
        norm_records.append(new_rec)
        norm_by_key[key] = new_rec
    return norm_records, norm_by_key, norm_bench


def plot_fair_aux_accuracy(
    by_key: dict[tuple[str, str], dict[str, Any]],
    out_dir: Path,
    *,
    suffix: str = "",
    normalized_bf16: bool = False,
) -> None:
    """Curvas por rodada e comparativo final de Aux Accuracy em condições 100% iguais."""
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2))
    curve_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
    ax_bar = axes[1, 1]

    final_vals: dict[str, list[float]] = {v: [] for v in VARIANT_ORDER}
    for ax, (model_key, group_title) in zip(curve_axes, GROUPS, strict=False):
        for var in VARIANT_ORDER:
            rec = by_key[(model_key, var)]
            aux = rec["h5"]["rs_aux_simple_avg_acc"]
            rounds = np.arange(1, len(aux) + 1)
            st = VARIANT_STYLE[var]
            ax.plot(
                rounds,
                aux,
                label=f"{var}: {aux[-1]:.2f}%",
                color=st["color"],
                marker=st["marker"],
                linestyle=st["ls"],
                linewidth=2.1,
                markersize=5.5,
            )
            final_vals[var].append(float(aux[-1]))
        ax.set_title(group_title)
        ax.set_xlabel("Rodada")
        ax.set_ylabel("Aux Accuracy Média (%)")
        ax.set_ylim(-2, 106)
        ax.legend(loc="lower right", framealpha=0.92)

    x = np.arange(len(GROUPS))
    w = 0.25
    for idx, var in enumerate(VARIANT_ORDER):
        vals = final_vals[var]
        bars = ax_bar.bar(
            x + (idx - 1) * w,
            vals,
            w,
            label=var,
            color=VARIANT_STYLE[var]["color"],
        )
        for bar, val in zip(bars, vals, strict=False):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                val + 1.4,
                f"{val:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8.2,
                fontweight="bold",
            )

    ax_bar.set_title("Aux Accuracy Final (%) — Condições 100% Pareadas")
    ax_bar.set_ylabel("Aux Accuracy Média (%)")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(
        ["SLM\n(OxfordPets, 15r)", "d1-2B\n(Banking77, 6r)", "LFM3-VL-2B\n(Banking77, 6r)"]
    )
    ax_bar.set_ylim(0, 122)
    ax_bar.legend(loc="upper center", ncol=3, fontsize=8.5)

    tag = (
        "\n(Acurácia mantida conforme medição original antes da padronização BF16)"
        if normalized_bf16
        else "\n(4 clientes, batch_size=4, epochs=1, r=8, PaCA cheio, FedALA ativo, sem PUMA/LHDQ)"
    )
    fig.suptitle(
        f"Comparação 100% Pareada — Aux Accuracy (rs_aux_simple_avg_acc){tag}",
        y=1.01,
    )
    _save(fig, out_dir, "01_fair_aux_accuracy", suffix=suffix)


def plot_fair_params_and_network(
    by_key: dict[tuple[str, str], dict[str, Any]],
    out_dir: Path,
    *,
    suffix: str = "",
    normalized_bf16: bool = False,
) -> None:
    """Compara tamanho final dos adaptadores (MB) e tráfego total de rede."""
    fig, (ax_mb, ax_tx) = plt.subplots(1, 2, figsize=(14.5, 5.4))

    x = np.arange(len(GROUPS))
    w = 0.25
    for idx, var in enumerate(VARIANT_ORDER):
        adapt_mb = [
            float(by_key[(m, var)]["h5"]["Model_size_per_round_Mb"][-1]) for m, _ in GROUPS
        ]
        tx_mb = [
            float(np.sum(by_key[(m, var)]["h5"]["total_transmitted_per_round_Mb"][:]))
            for m, _ in GROUPS
        ]
        c = VARIANT_STYLE[var]["color"]
        b1 = ax_mb.bar(x + (idx - 1) * w, adapt_mb, w, label=var, color=c)
        b2 = ax_tx.bar(x + (idx - 1) * w, tx_mb, w, label=var, color=c)

        for bar, val in zip(b1, adapt_mb, strict=False):
            ax_mb.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.6,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )
        for bar, val in zip(b2, tx_mb, strict=False):
            ax_tx.text(
                bar.get_x() + bar.get_width() / 2,
                val + 18,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    short_labels = [
        "SLM\n(OxfordPets, 15r)",
        "d1-2B\n(Banking77, 6r)",
        "LFM3-VL-2B\n(Banking77, 6r)",
    ]
    ax_mb.set_title("Tamanho Final dos Adaptadores na GPU (MB)")
    ax_mb.set_ylabel("Adaptadores + Head (MB)")
    ax_mb.set_xticks(x)
    ax_mb.set_xticklabels(short_labels)
    ax_mb.set_ylim(0, 25.5 if normalized_bf16 else 44)
    ax_mb.legend(loc="upper center", ncol=3, fontsize=8.5)

    ax_tx.set_title("Tráfego Total na Rede em Todas as Rodadas (MB)")
    ax_tx.set_ylabel("Total Transmitido + Recebido (MB)")
    ax_tx.set_xticks(x)
    ax_tx.set_xticklabels(short_labels)
    ax_tx.set_ylim(0, 1280 if normalized_bf16 else 1360)
    ax_tx.legend(loc="upper center", ncol=3, fontsize=8.5)

    title_mode = (
        "Economia de Parâmetros e Banda [Normalizado BF16: Adapt./Rede]"
        if normalized_bf16
        else "Economia de Parâmetros e Banda (Sem LHDQ, 100% Pareado): BF16 LoRA vs QLoRA vs QSoRA"
    )
    fig.suptitle(title_mode, y=1.02)
    _save(fig, out_dir, "02_fair_params_and_network", suffix=suffix)


def plot_fair_time_and_pareto(
    by_key: dict[tuple[str, str], dict[str, Any]],
    bench_by_key: dict[tuple[str, str], dict[str, Any]],
    out_dir: Path,
    *,
    suffix: str = "",
    normalized_bf16: bool = False,
) -> None:
    """Compara tempo total (min), pico isolado de VRAM e fronteira Pareto Aux Acc vs Rede."""
    fig, (ax_time, ax_vram, ax_pareto) = plt.subplots(1, 3, figsize=(16.5, 5.2))
    x = np.arange(len(GROUPS))
    w = 0.25
    short_labels = ["SLM\n(15r)", "d1-2B\n(6r)", "LFM3-VL-2B\n(6r)"]

    for idx, var in enumerate(VARIANT_ORDER):
        times = [float(by_key[(m, var)]["wall_seconds"]) / 60.0 for m, _ in GROUPS]
        vrams = [
            float(bench_by_key[(m, var)]["peak_train_mb"]) / 1024.0
            if (m, var) in bench_by_key
            else 0.0
            for m, _ in GROUPS
        ]
        c = VARIANT_STYLE[var]["color"]
        b_t = ax_time.bar(x + (idx - 1) * w, times, w, label=var, color=c)
        b_v = ax_vram.bar(x + (idx - 1) * w, vrams, w, label=var, color=c)
        for bar, val in zip(b_t, times, strict=False):
            ax_time.text(
                bar.get_x() + bar.get_width() / 2,
                val + 1.2,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
        for bar, val in zip(b_v, vrams, strict=False):
            ax_vram.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.12,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax_time.set_title("Tempo Total de Execução (min)")
    ax_time.set_ylabel("Minutos (4 clientes em 1 GPU)")
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(short_labels)
    ax_time.set_ylim(0, 86)
    ax_time.legend(loc="upper left", fontsize=8)

    ax_vram.set_title("Pico de VRAM por Cliente (bs=4, GB)")
    ax_vram.set_ylabel("Pico Isolado na GPU (GB)")
    ax_vram.set_xticks(x)
    ax_vram.set_xticklabels(short_labels)
    ax_vram.set_ylim(0, 7.2)
    ax_vram.legend(loc="upper right", fontsize=8)

    model_markers = {"slm": "o", "d1": "s", "lfm": "^"}
    model_names = {"slm": "SLM", "d1": "d1-2B", "lfm": "LFM3"}
    for m, _ in GROUPS:
        xs = [
            float(np.sum(by_key[(m, v)]["h5"]["total_transmitted_per_round_Mb"][:]))
            for v in VARIANT_ORDER
        ]
        ys = [float(by_key[(m, v)]["h5"]["rs_aux_simple_avg_acc"][-1]) for v in VARIANT_ORDER]
        ax_pareto.plot(xs, ys, color="#888888", linestyle=":", linewidth=1.2, zorder=1)
        for v, xv, yv in zip(VARIANT_ORDER, xs, ys, strict=False):
            ax_pareto.scatter(
                xv,
                yv,
                color=VARIANT_STYLE[v]["color"],
                marker=model_markers[m],
                s=95,
                edgecolor="black",
                linewidth=0.6,
                zorder=3,
            )
            if v == "4-bit QSoRA":
                if normalized_bf16:
                    offset = (6, 6) if m == "slm" else ((-6, -24) if m == "d1" else (-56, 6))
                else:
                    offset = (6, 6) if m in ("slm", "d1") else (6, -14)
                ax_pareto.annotate(
                    f"{model_names[m]}\nQSoRA",
                    (xv, yv),
                    textcoords="offset points",
                    xytext=offset,
                    fontsize=7.8,
                    fontweight="bold",
                    color="#1f77b4",
                )

    for v in VARIANT_ORDER:
        ax_pareto.scatter([], [], color=VARIANT_STYLE[v]["color"], label=v, s=65)
    for m in ("slm", "d1", "lfm"):
        ax_pareto.scatter(
            [], [], color="gray", marker=model_markers[m], label=model_names[m], s=65
        )
    ax_pareto.set_title("Eficiência: Aux Acc vs Tráfego de Rede")
    ax_pareto.set_xlabel("Tráfego Total na Rede (MB) ← menor é melhor")
    ax_pareto.set_ylabel("Aux Accuracy Final (%) ↑")
    ax_pareto.set_xlim(410, 1170)
    ax_pareto.set_ylim(88.0, 98.2)
    ax_pareto.legend(loc="lower right", ncol=2, fontsize=7.5, framealpha=0.9)

    fig.suptitle(
        "Trade-off Geral (Tempo, VRAM por Cliente e Eficiência de Rede) — "
        + (
            "Normalizado BF16 [Rede ajust.; Tempo/Pico/Acc medidos]"
            if normalized_bf16
            else "Bateria 100% Pareada"
        ),
        y=1.02,
    )
    _save(fig, out_dir, "03_fair_time_vram_and_pareto", suffix=suffix)


def plot_d1_acuracia_e_loss(
    by_key: dict[tuple[str, str], dict[str, Any]],
    out_dir: Path,
    *,
    suffix: str = "",
    normalized_bf16: bool = False,
) -> None:
    """Gera gráficos exclusivos do d1-2B: Aux Acc, Test Acc, Test Loss e barras finais."""
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0))
    ax_aux, ax_test, ax_loss, ax_bar = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    aux_finals, test_finals = [], []
    for var in VARIANT_ORDER:
        h5 = by_key[("d1", var)]["h5"]
        st = VARIANT_STYLE[var]
        aux = h5["rs_aux_simple_avg_acc"]
        tst = h5["rs_test_acc"]
        loss = h5["rs_test_loss"]
        rounds = np.arange(1, len(aux) + 1)

        ax_aux.plot(
            rounds,
            aux,
            label=f"{var}: {aux[-1]:.2f}%",
            color=st["color"],
            marker=st["marker"],
            linewidth=2.2,
            markersize=6,
        )
        ax_test.plot(
            rounds,
            tst,
            label=f"{var}: {tst[-1]:.2f}%",
            color=st["color"],
            marker=st["marker"],
            linewidth=2.2,
            markersize=6,
        )
        ax_loss.plot(
            rounds,
            loss,
            label=f"{var}: {loss[-1]:.4f}",
            color=st["color"],
            marker=st["marker"],
            linewidth=2.2,
            markersize=6,
        )
        aux_finals.append(float(aux[-1]))
        test_finals.append(float(tst[-1]))

    for ax, title, ylabel, ylim, loc in (
        (
            ax_aux,
            "Aux Accuracy Média dos Clientes (%)",
            "Aux Accuracy (%)",
            (84.0, 93.5),
            "lower right",
        ),
        (ax_test, "Test Accuracy Federada (%)", "Test Accuracy (%)", (84.0, 93.5), "lower right"),
        (ax_loss, "Test Loss Federada (CE + Brier)", "Loss", (0.38, 0.56), "upper right"),
    ):
        ax.set_title(title)
        ax.set_xlabel("Rodada")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(1, 7))
        ax.set_ylim(*ylim)
        ax.legend(loc=loc, framealpha=0.92)

    x = np.arange(len(VARIANT_ORDER))
    w = 0.34
    b_aux = ax_bar.bar(x - w / 2, aux_finals, w, label="Aux Accuracy Final (%)", color="#1f77b4")
    b_tst = ax_bar.bar(x + w / 2, test_finals, w, label="Test Accuracy Final (%)", color="#2ca02c")
    for bars in (b_aux, b_tst):
        for bar in bars:
            val = bar.get_height()
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                val + 1.2,
                f"{val:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8.2,
                fontweight="bold",
            )
    ax_bar.set_title("Acurácia Final por Configuração (Rodada 6)")
    ax_bar.set_ylabel("Acurácia (%)")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(VARIANT_ORDER)
    ax_bar.set_ylim(0, 114)
    ax_bar.legend(loc="upper center", ncol=2, fontsize=8.5)

    sub = (
        " — Acurácia e Loss [Mantidas conforme medição original]"
        if normalized_bf16
        else " — Acurácia e Loss"
    )
    fig.suptitle(
        f"d1-2B (Banking77, 4 clientes, 6 rodadas, PaCA=24, r=8){sub}",
        y=1.01,
    )
    _save(fig, out_dir, "04_d1_acuracia_e_loss", suffix=suffix)


def plot_d1_vram_e_parametros(
    by_key: dict[tuple[str, str], dict[str, Any]],
    bench_by_key: dict[tuple[str, str], dict[str, Any]],
    out_dir: Path,
    *,
    suffix: str = "",
    normalized_bf16: bool = False,
) -> None:
    """Gera gráficos exclusivos do d1-2B: VRAM estática/pico e poda de parâmetros."""
    fig, (ax_vram, ax_adapt, ax_curve) = plt.subplots(1, 3, figsize=(16.5, 5.2))

    x = np.arange(len(VARIANT_ORDER))
    w = 0.34
    static_mb = [float(bench_by_key[("d1", v)]["static_model_mb"]) for v in VARIANT_ORDER]
    peak_mb = [float(bench_by_key[("d1", v)]["peak_train_mb"]) for v in VARIANT_ORDER]

    static_lbl = "Modelo Estático (norm. BF16)" if normalized_bf16 else "Modelo Estático na GPU"
    peak_lbl = (
        "Pico VRAM Treino (bs=4, medido)" if normalized_bf16 else "Pico de VRAM no Treino (bs=4)"
    )
    b_st = ax_vram.bar(x - w / 2, static_mb, w, label=static_lbl, color="#4c72b0")
    b_pk = ax_vram.bar(x + w / 2, peak_mb, w, label=peak_lbl, color="#dd8452")
    for bars in (b_st, b_pk):
        for bar in bars:
            val = bar.get_height()
            ax_vram.text(
                bar.get_x() + bar.get_width() / 2,
                val + 65,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=8.2,
                fontweight="bold",
            )
    ax_vram.set_title("VRAM por Cliente no d1-2B (MB)")
    ax_vram.set_ylabel("VRAM na GPU (MB)")
    ax_vram.set_xticks(x)
    ax_vram.set_xticklabels(VARIANT_ORDER)
    ax_vram.set_ylim(0, 5350)
    ax_vram.legend(loc="upper right", fontsize=8.2)

    adapt_labels = [
        "BF16 LoRA",
        "4-bit QLoRA",
        "4-bit QSoRA\n(Inicial)",
        "4-bit QSoRA\n(Pós-Poda R6)",
    ]
    adapt_vals = [
        float(bench_by_key[("d1", "BF16 LoRA")]["trainable_params_mb"]),
        float(bench_by_key[("d1", "4-bit QLoRA")]["trainable_params_mb"]),
        float(bench_by_key[("d1", "4-bit QSoRA")]["trainable_params_mb"]),
        float(by_key[("d1", "4-bit QSoRA")]["h5"]["Model_size_per_round_Mb"][-1]),
    ]
    adapt_colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]
    b_ad = ax_adapt.bar(np.arange(len(adapt_labels)), adapt_vals, 0.56, color=adapt_colors)
    ad_pad = 0.35 if normalized_bf16 else 0.6
    for bar, val in zip(b_ad, adapt_vals, strict=False):
        ax_adapt.text(
            bar.get_x() + bar.get_width() / 2,
            val + ad_pad,
            f"{val:.2f} MB",
            ha="center",
            va="bottom",
            fontsize=8.3,
            fontweight="bold",
        )
    ax_adapt.set_title("Memória dos Adaptadores Treináveis (MB)")
    ax_adapt.set_ylabel("Adaptadores na GPU (MB)")
    ax_adapt.set_xticks(np.arange(len(adapt_labels)))
    ax_adapt.set_xticklabels(adapt_labels, fontsize=8.5)
    ax_adapt.set_ylim(0, 20.5 if normalized_bf16 else 36.5)

    for var in VARIANT_ORDER:
        tp_m = by_key[("d1", var)]["h5"]["Trainable_params"] / 1e6
        st = VARIANT_STYLE[var]
        rounds = np.arange(1, len(tp_m) + 1)
        ls = "--" if var == "BF16 LoRA" else "-"
        ax_curve.plot(
            rounds,
            tp_m,
            label=f"{var}: {tp_m[0]:.2f}M -> {tp_m[-1]:.2f}M",
            color=st["color"],
            marker=st["marker"],
            linestyle=ls,
            linewidth=2.2,
            markersize=6,
        )
    ax_curve.set_title("Parâmetros Treináveis por Rodada (Milhões)")
    ax_curve.set_xlabel("Rodada")
    ax_curve.set_ylabel("Milhões de Parâmetros")
    ax_curve.set_xticks(range(1, 7))
    ax_curve.set_ylim(3.0, 9.2)
    ax_curve.legend(loc="lower left", fontsize=8.2)

    fig.suptitle(
        "d1-2B (Banking77) — Economia de VRAM e Poda Estrutural de Parâmetros"
        + (" [Adapt./Estát. norm. BF16]" if normalized_bf16 else ""),
        y=1.02,
    )
    _save(fig, out_dir, "05_d1_vram_e_parametros", suffix=suffix)


def plot_d1_rede_tempo_e_pareto(
    by_key: dict[tuple[str, str], dict[str, Any]],
    bench_by_key: dict[tuple[str, str], dict[str, Any]],
    out_dir: Path,
    *,
    suffix: str = "",
    normalized_bf16: bool = False,
) -> None:
    """Gera gráficos exclusivos do d1-2B: rede acumulada, tempo e eficiência Pareto."""
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0))
    ax_net, ax_time, ax_pnet, ax_pvram = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    train_mins, ovh_mins, other_mins, total_mins = [], [], [], []
    for var in VARIANT_ORDER:
        rec = by_key[("d1", var)]
        h5 = rec["h5"]
        st = VARIANT_STYLE[var]
        cum_tx = np.cumsum(h5["total_transmitted_per_round_Mb"])
        rounds = np.arange(1, len(cum_tx) + 1)
        ls = "--" if var == "BF16 LoRA" else "-"
        ax_net.plot(
            rounds,
            cum_tx,
            label=f"{var}: {cum_tx[-1]:.1f} MB",
            color=st["color"],
            marker=st["marker"],
            linestyle=ls,
            linewidth=2.2,
            markersize=6,
        )

        tr_m = float(np.sum(np.mean(h5["rs_client_training_time"][:], axis=1))) / 60.0
        ev_m = float(np.sum(np.mean(h5["rs_client_eval_time"][:], axis=1))) / 60.0
        tot_m = float(rec["wall_seconds"]) / 60.0
        oth_m = max(0.0, tot_m - tr_m - ev_m)
        train_mins.append(tr_m)
        ovh_mins.append(ev_m)
        other_mins.append(oth_m)
        total_mins.append(tot_m)

    ax_net.set_title("Tráfego de Rede Acumulado (MB)")
    ax_net.set_xlabel("Rodada")
    ax_net.set_ylabel("Total Transmitido + Recebido (MB)")
    ax_net.set_xticks(range(1, 7))
    ax_net.set_ylim(0, 720 if normalized_bf16 else 1120)
    ax_net.legend(loc="upper left", fontsize=8.5)

    x = np.arange(len(VARIANT_ORDER))
    w = 0.48
    ax_time.bar(x, train_mins, w, label="Treino Local", color="#1f77b4")
    ax_time.bar(
        x, ovh_mins, w, bottom=train_mins, label="Eval + Overhead Cliente", color="#ff7f0e"
    )
    bottom2 = [a + b for a, b in zip(train_mins, ovh_mins, strict=False)]
    ax_time.bar(x, other_mins, w, bottom=bottom2, label="Init / Servidor / Rede", color="#7f7f7f")
    for idx, tot in enumerate(total_mins):
        ax_time.text(idx, tot + 1.2, f"{tot:.1f} min", ha="center", va="bottom", fontweight="bold")
    ax_time.set_title("Decomposição do Tempo Total (6 rodadas)")
    ax_time.set_ylabel("Tempo (minutos)")
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(VARIANT_ORDER)
    ax_time.set_ylim(0, 86)
    ax_time.legend(loc="upper center", ncol=2, fontsize=8.0)

    for var in VARIANT_ORDER:
        rec = by_key[("d1", var)]
        h5 = rec["h5"]
        st = VARIANT_STYLE[var]
        aux_final = float(h5["rs_aux_simple_avg_acc"][-1])
        tx_tot = float(np.sum(h5["total_transmitted_per_round_Mb"]))
        vram_gb = float(bench_by_key[("d1", var)]["peak_train_mb"]) / 1024.0

        ax_pnet.scatter(tx_tot, aux_final, color=st["color"], marker=st["marker"], s=130, zorder=3)
        if normalized_bf16:
            offset_net = (8, -4) if var == "4-bit QSoRA" else (8, 4 if var == "BF16 LoRA" else -16)
        else:
            offset_net = (8, -4) if tx_tot < 700 else (-82, 8 if var == "BF16 LoRA" else -16)
        ax_pnet.annotate(
            f"{var}\n({tx_tot:.1f} MB, {aux_final:.2f}%)",
            (tx_tot, aux_final),
            textcoords="offset points",
            xytext=offset_net,
            fontsize=8.2,
            fontweight="bold",
            color=st["color"],
        )

        ax_pvram.scatter(
            vram_gb, aux_final, color=st["color"], marker=st["marker"], s=130, zorder=3
        )
        offset_vr = (10, -14 if var == "4-bit QSoRA" else 6) if vram_gb < 3.0 else (-115, -4)
        ax_pvram.annotate(
            f"{var}\n({vram_gb:.2f} GB, {aux_final:.2f}%)",
            (vram_gb, aux_final),
            textcoords="offset points",
            xytext=offset_vr,
            fontsize=8.2,
            fontweight="bold",
            color=st["color"],
        )

    ax_pnet.set_title("Eficiência: Aux Accuracy vs Tráfego na Rede")
    ax_pnet.set_xlabel("Tráfego Total (MB) ← menor é melhor")
    ax_pnet.set_ylabel("Aux Accuracy Final (%) ↑")
    ax_pnet.set_xlim(455, 695) if normalized_bf16 else ax_pnet.set_xlim(420, 1040)
    ax_pnet.set_ylim(90.4, 92.4)

    ax_pvram.set_title("Eficiência: Aux Accuracy vs Pico de VRAM por Cliente")
    ax_pvram.set_xlabel("Pico de VRAM no Treino (GB) ← menor é melhor")
    ax_pvram.set_ylabel("Aux Accuracy Final (%) ↑")
    ax_pvram.set_xlim(1.9, 4.75)
    ax_pvram.set_ylim(90.4, 92.4)

    fig.suptitle(
        "d1-2B (Banking77, 4 clientes, 6 rodadas) — Rede, Tempo e Fronteiras de Eficiência"
        + (" [Rede norm. BF16]" if normalized_bf16 else ""),
        y=1.01,
    )
    _save(fig, out_dir, "06_d1_rede_tempo_e_pareto", suffix=suffix)


def write_summary(
    records: list[dict[str, Any]],
    bench_by_key: dict[tuple[str, str], dict[str, Any]],
    suite_dir: Path,
    *,
    suffix: str = "",
    normalized_bf16: bool = False,
) -> None:
    rows = []
    for rec in records:
        h5 = rec["h5"]
        b = bench_by_key.get((rec["model"], rec["label"]), {})
        p0 = int(h5["Trainable_params"][0])
        pf = int(h5["Trainable_params"][-1])
        mb0 = float(h5["Model_size_per_round_Mb"][0])
        mbf = float(h5["Model_size_per_round_Mb"][-1])
        rows.append(
            {
                "id": rec["id"],
                "model": rec["model"],
                "dataset": rec["dataset"],
                "variant": rec["label"],
                "gpu_id": rec["gpu_id"],
                "rounds": rec["rounds"],
                "clients": rec["clients"],
                "wall_min": round(rec["wall_seconds"] / 60.0, 2),
                "aux_acc_pct": round(float(h5["rs_aux_simple_avg_acc"][-1]), 2),
                "test_acc_pct": round(float(h5["rs_test_acc"][-1]), 2),
                "final_loss": round(float(h5["rs_test_loss"][-1]), 4),
                "params_init": p0,
                "params_final": pf,
                "params_red_pct": round((1.0 - pf / p0) * 100.0, 2) if p0 else 0.0,
                "adapter_init_mb": round(mb0, 2),
                "adapter_final_mb": round(mbf, 2),
                "total_tx_mb": round(float(np.sum(h5["total_transmitted_per_round_Mb"][:])), 2),
                "static_model_mb": b.get("static_model_mb", ""),
                "peak_train_mb": b.get("peak_train_mb", ""),
                "energy_wh": round(rec["energy_wh"], 2),
            }
        )

    csv_path = suite_dir / f"fair_summary{suffix}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Resumo — Bateria 100% Pareada (BF16 LoRA vs 4-bit QLoRA vs 4-bit QSoRA)"
        + (" [Normalizado BF16]\n" if normalized_bf16 else "\n"),
    ]
    if normalized_bf16:
        md_lines.append(
            "> Nota: `Adapt. (MB)`, `static_model_mb` e `Rede (MB)` (`d1`/`lfm` em `--timeout`) "
            "estão ajustados deterministicamente de `fp32` para `bfloat16` (2 bytes/param nos "
            "adaptadores). `Tempo`, `Aux Acc`, `Test Acc`, `Loss` e `Pico VRAM` são os valores "
            "originais medidos.\n"
        )
    md_lines.extend(
        [
            (
                "| Experimento | Modelo | Dataset | Rodadas | Tempo (min) | Aux Acc (%) | "
                "Test Acc (%) | Loss | Params (Inic -> Final) | Adapt. (MB) | Rede (MB) | "
                "Pico VRAM/cli (MB) |"
            ),
            "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- | ---: | --: |",
        ]
    )
    for r in rows:
        md_lines.append(
            f"| {r['id']} | {r['model']} | {r['dataset']} | {r['rounds']} | {r['wall_min']:.1f} | "
            f"{r['aux_acc_pct']:.2f} | {r['test_acc_pct']:.2f} | {r['final_loss']:.4f} | "
            f"{r['params_init']:,} -> {r['params_final']:,} (-{r['params_red_pct']:.1f}%) | "
            f"{r['adapter_init_mb']:.2f} -> {r['adapter_final_mb']:.2f} | "
            f"{r['total_tx_mb']:.2f} | {r['peak_train_mb']} |"
        )
    md_path = suite_dir / f"fair_summary{suffix}.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[SUMMARY] Salvo: {csv_path} e {md_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plota comparativo 100% pareado QLoRA vs QSoRA.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("src/results/fair_qlora_vs_qsora_suite/fair_manifest_complete.json"),
    )
    parser.add_argument(
        "--bench-json",
        type=Path,
        default=Path("src/results/qsora_validation_suite/qlora_vs_qsora_gpu_benchmark.json"),
    )
    parser.add_argument(
        "--normalize-bf16",
        action="store_true",
        help="Normaliza deterministicamente adaptadores/estático/rede de fp32 para bfloat16.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Sufixo nos nomes dos arquivos (padrão '_bf16_norm' com --normalize-bf16).",
    )
    args = parser.parse_args()

    suffix = args.suffix or ("_bf16_norm" if args.normalize_bf16 else "")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    suite_dir = args.manifest.parent
    out_dir = suite_dir / "plots"

    bench_items = json.loads(args.bench_json.read_text(encoding="utf-8"))
    bench_by_key = {(b["model"], b["variant"]): b for b in bench_items}

    records = []
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for item in manifest["experiments"]:
        rec = {
            **item,
            "h5": _load_h5(Path(item["results_dir"])),
            "energy_wh": _energy_wh(Path(item["telemetry_dir"])),
        }
        records.append(rec)
        by_key[(item["model"], item["label"])] = rec

    if args.normalize_bf16:
        records, by_key, bench_by_key = _apply_bf16_normalization(records, bench_by_key)

    kw = {"suffix": suffix, "normalized_bf16": args.normalize_bf16}
    plot_fair_aux_accuracy(by_key, out_dir, **kw)
    plot_fair_params_and_network(by_key, out_dir, **kw)
    plot_fair_time_and_pareto(by_key, bench_by_key, out_dir, **kw)
    plot_d1_acuracia_e_loss(by_key, out_dir, **kw)
    plot_d1_vram_e_parametros(by_key, bench_by_key, out_dir, **kw)
    plot_d1_rede_tempo_e_pareto(by_key, bench_by_key, out_dir, **kw)
    write_summary(records, bench_by_key, suite_dir, **kw)


if __name__ == "__main__":
    main()
