#!/usr/bin/env python3
"""Gerador de gráficos de validação federada QSoRA separados por dataset (justa comparação)."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

STYLES = {
    "slm_qlora": {"color": "#d62728", "marker": "s", "ls": "--", "label": "SLM QLoRA (LoRA)"},
    "slm_qsora": {"color": "#1f77b4", "marker": "o", "ls": "-", "label": "SLM QSoRA (SoRA)"},
    "d1_qsora_banking77": {
        "color": "#9467bd",
        "marker": "s",
        "ls": "--",
        "label": "d1-2B QSoRA",
    },
    "lfm_qsora": {"color": "#2ca02c", "marker": "^", "ls": "-", "label": "LFM3-VL-2B QSoRA"},
    "d1_qsora_robust": {
        "color": "#9467bd",
        "marker": "s",
        "ls": "--",
        "label": "d1-2B QSoRA (4c)",
    },
    "lfm_qsora_robust": {
        "color": "#2ca02c",
        "marker": "^",
        "ls": "-",
        "label": "LFM3-VL-2B QSoRA (4c)",
    },
}


def _save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Salvo: {png}")


def _load_h5_bundle(exp_dir: Path) -> dict[str, Any]:
    server_files = sorted(glob.glob(str(exp_dir / "server_*.h5")))
    if not server_files:
        return {}
    bundle: dict[str, Any] = {"server_h5": server_files[-1]}
    with h5py.File(server_files[-1], "r") as handle:
        for key in handle:
            bundle[key] = np.array(handle[key][()])

    client_files = sorted(glob.glob(str(exp_dir / "client_*.h5")))
    local_accs = []
    for cpath in client_files:
        with h5py.File(cpath, "r") as handle:
            if "rs_local_acc" in handle:
                local_accs.append(np.array(handle["rs_local_acc"][()], dtype=float))
    if local_accs:
        bundle["client_local_acc_mean"] = np.mean(np.stack(local_accs), axis=0)
    return bundle


def _load_gpu_csv(csv_path: Path) -> dict[int, dict[str, np.ndarray]]:
    by_gpu: dict[int, dict[str, list[float]]] = {}
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            gid = int(row["gpu_id"])
            bucket = by_gpu.setdefault(
                gid,
                {"elapsed_s": [], "memory_used_gb": [], "gpu_util_pct": []},
            )
            bucket["elapsed_s"].append(float(row["elapsed_s"]))
            bucket["memory_used_gb"].append(float(row["memory_used_mib"]) / 1024.0)
            bucket["gpu_util_pct"].append(float(row["gpu_util_pct"]))
    return {
        gid: {k: np.array(v, dtype=float) for k, v in cols.items()} for gid, cols in by_gpu.items()
    }


def _delta_vram_gb(gpu_series: dict[str, np.ndarray]) -> float:
    mem = gpu_series.get("memory_used_gb", np.array([]))
    if len(mem) == 0:
        return 0.0
    return float(max(0.0, mem.max() - mem[0]))


def _plot_suite(
    records: list[dict[str, Any]],
    suite_title: str,
    stem: str,
    out_dir: Path,
) -> None:
    """Gera painel 2x2 comparando apenas configurações dentro de um mesmo dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))
    ax_acc, ax_loss = axes[0]
    ax_params, ax_comm = axes[1]

    for rec in records:
        default_st = {"color": "#333333", "marker": "o", "ls": "-", "label": rec["label"]}
        st = STYLES.get(rec["id"], default_st)
        lbl = st["label"]
        h5 = rec["h5"]

        acc = h5.get("rs_test_acc", np.array([]))
        loss = h5.get("rs_test_loss", np.array([]))
        params = h5.get("Trainable_params", np.array([]))
        comm = h5.get("total_transmitted_per_round_Mb", np.array([]))
        if len(acc) == 0:
            continue

        rounds = np.arange(1, len(acc) + 1)
        ax_acc.plot(
            rounds, acc, color=st["color"], marker=st["marker"], ls=st["ls"], lw=2.2, label=lbl
        )
        if "client_local_acc_mean" in h5 and len(h5["client_local_acc_mean"]) == len(rounds):
            ax_acc.plot(
                rounds,
                h5["client_local_acc_mean"],
                color=st["color"],
                ls=":",
                lw=1.5,
                alpha=0.7,
                label=f"{lbl} (pós-treino local)",
            )
        ax_loss.plot(
            rounds, loss, color=st["color"], marker=st["marker"], ls=st["ls"], lw=2.2, label=lbl
        )
        if len(params) > 0:
            ax_params.plot(
                rounds,
                params / 1e6,
                color=st["color"],
                marker=st["marker"],
                ls=st["ls"],
                lw=2.2,
                label=f"{lbl} (Δ={params[0] - params[-1]:,.0f})",
            )
        if len(comm) > 0:
            ax_comm.plot(
                rounds,
                comm,
                color=st["color"],
                marker=st["marker"],
                ls=st["ls"],
                lw=2.2,
                label=f"{lbl} (total={comm.sum():.1f} MB)",
            )

    ax_acc.set_title("Acurácia Federada Ponderada (%)")
    ax_acc.set_xlabel("Rodada")
    ax_acc.set_ylabel("Acurácia (%)")
    ax_acc.set_ylim(0, 105)
    ax_acc.legend(loc="best", fontsize=8.5)

    ax_loss.set_title("Loss Média Ponderada")
    ax_loss.set_xlabel("Rodada")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(loc="best", fontsize=8.5)

    ax_params.set_title("Parâmetros Treináveis (Poda SoRA)")
    ax_params.set_xlabel("Rodada")
    ax_params.set_ylabel("Parâmetros (Milhões)")
    ax_params.legend(loc="best", fontsize=8.5)

    ax_comm.set_title("Tráfego Transmitido por Rodada (LHDQ)")
    ax_comm.set_xlabel("Rodada")
    ax_comm.set_ylabel("MB / Rodada")
    ax_comm.legend(loc="best", fontsize=8.5)

    fig.suptitle(suite_title, y=1.01)
    _save_fig(fig, out_dir, stem)


def _plot_efficiency_bars(
    pets_records: list[dict[str, Any]],
    bank_records: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    """Painel de recursos (VRAM FL e Tempo/Rodada) agrupado separadamente por dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    ax_vram, ax_time = axes

    ordered = pets_records + bank_records
    labels = [
        f"{STYLES.get(r['id'], {}).get('label', r['label'])}\n({r['dataset']})" for r in ordered
    ]
    v0s = [_delta_vram_gb(r["gpu"][0]) if 0 in r["gpu"] else 0.0 for r in ordered]
    v1s = [_delta_vram_gb(r["gpu"][1]) if 1 in r["gpu"] else 0.0 for r in ordered]
    rts = [float(np.mean(r["h5"].get("Round_time", [0.0]))) for r in ordered]
    trs = [float(np.mean(r["h5"].get("rs_client_training_time", [0.0]))) for r in ordered]

    x = np.arange(len(ordered))
    w = 0.36
    ax_vram.bar(x - w / 2, v0s, w, label="GPU 0 (Server + Client 0)", color="#ff9896")
    ax_vram.bar(x + w / 2, v1s, w, label="GPU 1 (Client 1 isolado)", color="#98df8a")
    for idx, (v0, v1) in enumerate(zip(v0s, v1s, strict=False)):
        ax_vram.text(
            idx - w / 2, v0 + 0.15, f"{v0:.1f} GB", ha="center", va="bottom", fontsize=8.5
        )
        ax_vram.text(
            idx + w / 2, v1 + 0.15, f"{v1:.1f} GB", ha="center", va="bottom", fontsize=8.5
        )
    if len(pets_records) > 0 and len(bank_records) > 0:
        ax_vram.axvline(len(pets_records) - 0.5, color="#555555", linestyle=":", linewidth=1.2)
    ax_vram.set_ylim(0, max(v0s + v1s + [1.0]) * 1.28)
    ax_vram.set_xticks(x)
    ax_vram.set_xticklabels(labels, fontsize=8.5)
    ax_vram.set_title("Pico de VRAM FL (sobre baseline t=0)")
    ax_vram.set_ylabel("VRAM Adicional (GB)")
    ax_vram.legend(loc="upper left")

    ax_time.bar(x - w / 2, trs, w, label="Treino Local Médio", color="#1f77b4")
    ax_time.bar(x + w / 2, rts, w, label="Duração Total da Rodada", color="#aec7e8")
    for idx, (tr, rt) in enumerate(zip(trs, rts, strict=False)):
        ax_time.text(idx - w / 2, tr + 2.5, f"{tr:.1f}s", ha="center", va="bottom", fontsize=8.5)
        ax_time.text(idx + w / 2, rt + 2.5, f"{rt:.1f}s", ha="center", va="bottom", fontsize=8.5)
    if len(pets_records) > 0 and len(bank_records) > 0:
        ax_time.axvline(len(pets_records) - 0.5, color="#555555", linestyle=":", linewidth=1.2)
    ax_time.set_ylim(0, max(rts + [1.0]) * 1.18)
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(labels, fontsize=8.5)
    ax_time.set_title("Tempo Médio por Rodada (por Dataset)")
    ax_time.set_ylabel("Segundos / Rodada")
    ax_time.legend(loc="upper left")

    _save_fig(fig, out_dir, "03_qsora_recursos_por_dataset")


def _write_summary(
    pets_records: list[dict[str, Any]],
    bank_records: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    rows = []
    sections = [
        ("Comparação 1 — OxfordPets (Visão, 37 classes, 5 rodadas)", pets_records),
        ("Comparação 2 — Banking77 (Texto/Decisão, 77 classes, 3 rodadas)", bank_records),
    ]
    md_lines = ["# Validação Justa do QSoRA por Dataset\n"]

    for title, recs in sections:
        md_lines.append(f"\n## {title}\n\n")
        md_lines.append(
            "| Configuração | Rodadas | Acc Inic. (%) | Acc Final (%) | Loss Final | "
            "Params Iniciais | Params Finais | Redução Params (%) | Tráfego Total (MB) | "
            "Tempo/Rodada (s) | VRAM FL GPU0/GPU1 (GB) |\n"
        )
        md_lines.append(
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |\n"
        )
        for rec in recs:
            h5 = rec["h5"]
            gpu = rec["gpu"]
            acc = h5.get("rs_test_acc", [0.0])
            loss = h5.get("rs_test_loss", [0.0])
            params = h5.get("Trainable_params", [0])
            p0, p1 = int(params[0]), int(params[-1])
            red = (100.0 * (p0 - p1) / p0) if p0 > 0 else 0.0
            sent = float(h5.get("sended_model_Mb", [0.0])[-1])
            recv = float(h5.get("received_from_clients_Mb", [0.0])[-1])
            rt = float(np.mean(h5.get("Round_time", [0.0])))
            v0 = _delta_vram_gb(gpu[0]) if 0 in gpu else 0.0
            v1 = _delta_vram_gb(gpu[1]) if 1 in gpu else 0.0
            row = {
                "id": rec["id"],
                "label": STYLES.get(rec["id"], {}).get("label", rec["label"]),
                "dataset": rec["dataset"],
                "rounds": rec["rounds"],
                "acc_init": round(float(acc[0]), 2),
                "acc_final": round(float(acc[-1]), 2),
                "loss_final": round(float(loss[-1]), 4),
                "params_init": p0,
                "params_final": p1,
                "param_reduction_pct": round(red, 2),
                "traffic_mb": round(sent + recv, 2),
                "round_time_s": round(rt, 2),
                "vram_gpu0_gb": round(v0, 2),
                "vram_gpu1_gb": round(v1, 2),
            }
            rows.append(row)
            md_lines.append(
                f"| {row['label']} | {row['rounds']} | {row['acc_init']:.2f} | "
                f"{row['acc_final']:.2f} | {row['loss_final']:.4f} | {p0:,} | {p1:,} | "
                f"{red:.2f}% | {row['traffic_mb']:.2f} | {row['round_time_s']:.1f} | "
                f"{row['vram_gpu0_gb']:.1f} / {row['vram_gpu1_gb']:.1f} |\n"
            )

    csv_path = out_dir / "validation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "validation_summary.md").write_text("".join(md_lines), encoding="utf-8")
    print(f"[SUMMARY] Salvo: {csv_path} e {out_dir / 'validation_summary.md'}")


def _plot_long_pruning(entry: dict[str, Any], out_dir: Path) -> None:
    """Gera painel 2x2 focado no efeito acumulado da poda iterativa SoRA em 15 rodadas."""
    h5 = _load_h5_bundle(Path(entry["results_dir"]))
    acc = h5.get("rs_test_acc", np.array([]))
    loss = h5.get("rs_test_loss", np.array([]))
    params = h5.get("Trainable_params", np.array([]))
    model_mb = h5.get("Model_size_per_round_Mb", np.array([]))
    comm_mb = h5.get("total_transmitted_per_round_Mb", np.array([]))
    pacas = h5.get("rs_client_paca", np.array([]))
    ranks = h5.get("rs_client_rank", np.array([]))
    if len(acc) == 0 or len(params) == 0:
        return

    rounds = np.arange(1, len(acc) + 1)
    red_pct = (1.0 - params / float(params[0])) * 100.0

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))
    ax_p, ax_acc = axes[0]
    ax_sz, ax_puma = axes[1]

    # 1) Parâmetros treináveis + Redução %
    l1 = ax_p.plot(rounds, params / 1e6, "o-", color="#1f77b4", lw=2.2, label="Parâmetros (M)")
    ax_p_r = ax_p.twinx()
    ax_p_r.spines["top"].set_visible(False)
    l2 = ax_p_r.plot(
        rounds, red_pct, "s--", color="#d62728", lw=2.0, label="Redução acumulada (%)"
    )
    ax_p.set_title("Poda Iterativa SoRA — Parâmetros Treináveis")
    ax_p.set_xlabel("Rodada Federada")
    ax_p.set_ylabel("Parâmetros Treináveis (Milhões)")
    ax_p_r.set_ylabel("Redução Acumulada (%)")
    lines = l1 + l2
    ax_p.legend(lines, [ln.get_label() for ln in lines], loc="center left")

    # 2) Acurácia + Loss
    a1 = ax_acc.plot(rounds, acc, "o-", color="#2ca02c", lw=2.2, label="Acc Global Ponderada (%)")
    if "client_local_acc_mean" in h5 and len(h5["client_local_acc_mean"]) == len(rounds):
        a1 += ax_acc.plot(
            rounds,
            h5["client_local_acc_mean"],
            ":",
            color="#2ca02c",
            lw=1.6,
            label="Acc Pós-Treino Local (%)",
        )
    ax_acc_r = ax_acc.twinx()
    ax_acc_r.spines["top"].set_visible(False)
    a2 = ax_acc_r.plot(rounds, loss, "^--", color="#ff7f0e", lw=2.0, label="Loss Global Média")
    ax_acc.set_title("Convergência Durante a Poda Contínua")
    ax_acc.set_xlabel("Rodada Federada")
    ax_acc.set_ylabel("Acurácia (%)")
    ax_acc.set_ylim(0, 105)
    ax_acc_r.set_ylabel("Loss")
    lines_a = a1 + a2
    ax_acc.legend(lines_a, [ln.get_label() for ln in lines_a], loc="center right")

    # 3) Tamanho do modelo vs tráfego LHDQ
    if len(model_mb) > 0:
        ax_sz.plot(
            rounds, model_mb, "o-", color="#9467bd", lw=2.2, label="Tamanho dos Adaptadores (MB)"
        )
    if len(comm_mb) > 0:
        ax_sz.plot(
            rounds,
            comm_mb,
            "s--",
            color="#1f77b4",
            lw=2.0,
            label=f"Tráfego LHDQ/Rodada (total={comm_mb.sum():.1f} MB)",
        )
    ax_sz.set_title("Tamanho do Modelo Treinável vs Tráfego LHDQ")
    ax_sz.set_xlabel("Rodada Federada")
    ax_sz.set_ylabel("Megabytes (MB)")
    ax_sz.legend(loc="best")

    # 4) Dinâmica PUMA-GT+ (PaCA e Rank médio)
    puma_lines = []
    if pacas.ndim == 2 and pacas.size > 0:
        puma_lines += ax_puma.plot(
            rounds[: pacas.shape[0]],
            pacas.mean(axis=1),
            "o-",
            color="#8c564b",
            lw=2.2,
            label="PaCA Médio (camadas)",
        )
    ax_puma_r = ax_puma.twinx()
    ax_puma_r.spines["top"].set_visible(False)
    if ranks.ndim == 2 and ranks.size > 0:
        puma_lines += ax_puma_r.plot(
            rounds[: ranks.shape[0]],
            ranks.mean(axis=1),
            "D--",
            color="#17becf",
            lw=2.0,
            label="Rank Médio Alocado",
        )
    ax_puma.set_title("Dinâmica Adaptativa PUMA-GT+ (PaCA & Rank)")
    ax_puma.set_xlabel("Rodada Federada")
    ax_puma.set_ylabel("Camadas Ativas (PaCA)")
    ax_puma_r.set_ylabel("Rank (r)")
    ax_puma_r.set_ylim(0, 9.5)
    if puma_lines:
        ax_puma.legend(puma_lines, [ln.get_label() for ln in puma_lines], loc="best")

    fig.suptitle(
        f"{entry['label']} — {entry['dataset']} "
        f"(Params: {int(params[0]):,} → {int(params[-1]):,}, -{red_pct[-1]:.2f}%)",
        y=1.01,
    )
    _save_fig(fig, out_dir, "04_qsora_pruning_15_rodadas")


def _plot_qlora_vs_qsora_vram(
    bench_items: list[dict[str, Any]],
    post_prune_mb: dict[str, float],
    out_dir: Path,
) -> None:
    """Painel 1x3 comparando BF16 LoRA, 4-bit QLoRA e 4-bit QSoRA em VRAM e parâmetros."""
    models = ["SLM (OxfordPets)", "d1-2B (Banking77)", "LFM3-VL-2B (Banking77)"]
    short_names = ["SLM\n(Qwen3-VL-2B)", "d1-2B\n(LiquidAI)", "LFM3-VL-2B\n(LiquidAI)"]
    variants = ["BF16 LoRA", "4-bit QLoRA", "4-bit QSoRA"]
    v_colors = {"BF16 LoRA": "#d62728", "4-bit QLoRA": "#ff7f0e", "4-bit QSoRA": "#1f77b4"}

    lookup = {(item["model_label"], item["variant"]): item for item in bench_items}
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0))
    ax_static, ax_train, ax_peak = axes
    x = np.arange(len(models))
    w = 0.25

    for idx_v, var in enumerate(variants):
        offset = (idx_v - 1) * w
        statics = [lookup[(m, var)]["static_model_mb"] for m in models]
        trains = [lookup[(m, var)]["trainable_params_mb"] for m in models]
        peaks = [lookup[(m, var)]["peak_train_mb"] for m in models]

        ax_static.bar(x + offset, statics, w, label=var, color=v_colors[var])
        ax_train.bar(x + offset, trains, w, label=var, color=v_colors[var])
        ax_peak.bar(x + offset, peaks, w, label=var, color=v_colors[var])

        for i in range(len(models)):
            ax_static.text(
                x[i] + offset,
                statics[i] + 60,
                f"{statics[i]:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            ax_train.text(
                x[i] + offset,
                trains[i] + 0.6,
                f"{trains[i]:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            ax_peak.text(
                x[i] + offset,
                peaks[i] + 80,
                f"{peaks[i]:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Sobrepõe barras do pós-poda QSoRA observados nas execuções FL
    pruned_vals = [post_prune_mb.get(m, 0.0) for m in models]
    if any(v > 0 for v in pruned_vals):
        ax_train.bar(
            x + w * 1.95,
            pruned_vals,
            w * 0.8,
            label="QSoRA Pós-Poda (FL)",
            color="#2ca02c",
        )
        for i, pv in enumerate(pruned_vals):
            if pv > 0:
                ax_train.text(
                    x[i] + w * 1.95,
                    pv + 0.6,
                    f"{pv:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#1b6e1b",
                    fontweight="bold",
                )

    ax_static.set_title("Armazenamento do Modelo na GPU (MB)")
    ax_static.set_ylabel("VRAM Estática (MB)")
    ax_static.set_xticks(x)
    ax_static.set_xticklabels(short_names)
    ax_static.set_ylim(0, max(i["static_model_mb"] for i in bench_items) * 1.18)
    ax_static.legend(loc="upper right", fontsize=8.5)

    ax_train.set_title("Memória dos Parâmetros Treináveis (MB)")
    ax_train.set_ylabel("Parâmetros na GPU (MB)")
    ax_train.set_xticks(x + 0.1)
    ax_train.set_xticklabels(short_names)
    ax_train.set_ylim(0, max(i["trainable_params_mb"] for i in bench_items) * 1.45)
    ax_train.legend(loc="upper center", ncol=2, fontsize=8)

    ax_peak.set_title("Pico Total de VRAM no Treino (bs=4, MB)")
    ax_peak.set_ylabel("Pico de VRAM (MB)")
    ax_peak.set_xticks(x)
    ax_peak.set_xticklabels(short_names)
    ax_peak.set_ylim(0, max(i["peak_train_mb"] for i in bench_items) * 1.18)
    ax_peak.legend(loc="upper right", fontsize=8.5)

    fig.suptitle(
        "Comparação Direta de Gasto de Memória (GPU 0 isolada): BF16 LoRA vs QLoRA vs QSoRA",
        y=1.02,
    )
    _save_fig(fig, out_dir, "06_qlora_vs_qsora_consumo_vram")


def _gpu0_energy_wh(csv_path: Path) -> float:
    if not csv_path.exists():
        return 0.0
    ts, pwrs = [], []
    with csv_path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["gpu_id"]) == 0:
                ts.append(float(row["elapsed_s"]))
                pwrs.append(float(row["power_draw_w"]))
    if len(ts) < 2:
        return 0.0
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz_fn(pwrs, ts) / 3600.0)


def _plot_qlora_vs_qsora_fl_cost(
    slm_qlora: dict[str, Any],
    slm_qsora: dict[str, Any],
    out_dir: Path,
) -> None:
    """Painel 2x2 comparando custo real em FL (SLM QLoRA vs SLM QSoRA, OxfordPets, 5 rodadas)."""
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0))
    ax_p, ax_rt, ax_comm, ax_bar = axes.ravel()

    pairs = [
        ("SLM QLoRA", slm_qlora, "#ff7f0e", "s", "--"),
        ("SLM QSoRA", slm_qsora, "#1f77b4", "o", "-"),
    ]
    for lbl, rec, col, mk, ls in pairs:
        h5 = rec["h5"]
        params = h5.get("Trainable_params", np.array([]))
        rt = h5.get("Round_time", np.array([]))
        comm = h5.get("total_transmitted_per_round_Mb", np.array([]))
        rounds = np.arange(1, len(params) + 1)
        ax_p.plot(rounds, params / 1e6, color=col, marker=mk, ls=ls, lw=2.2, label=lbl)
        ax_rt.plot(
            rounds,
            rt,
            color=col,
            marker=mk,
            ls=ls,
            lw=2.2,
            label=f"{lbl} (média={rt.mean():.1f}s)",
        )
        ax_comm.plot(
            rounds,
            comm,
            color=col,
            marker=mk,
            ls=ls,
            lw=2.2,
            label=f"{lbl} (total={comm.sum():.1f} MB)",
        )

    ax_p.set_title("Parâmetros Treináveis ao Longo das Rodadas")
    ax_p.set_xlabel("Rodada")
    ax_p.set_ylabel("Milhões de Parâmetros")
    ax_p.legend(loc="best")

    ax_rt.set_title("Tempo por Rodada (s)")
    ax_rt.set_xlabel("Rodada")
    ax_rt.set_ylabel("Segundos")
    ax_rt.legend(loc="best")

    ax_comm.set_title("Tráfego de Rede por Rodada (LHDQ)")
    ax_comm.set_xlabel("Rodada")
    ax_comm.set_ylabel("MB / Rodada")
    ax_comm.legend(loc="best")

    # Resumo comparativo de gasto total (Tempo min, Energia Wh, Tráfego MB, Acurácia %)
    metrics_lbls = [
        "Tempo Total\n(min)",
        "Energia GPU0\n(Wh)",
        "Tráfego Total\n(MB)",
        "Acurácia Final\n(%)",
    ]
    q_vals = [
        float(slm_qlora["wall_seconds"]) / 60.0,
        _gpu0_energy_wh(Path(slm_qlora["telemetry_dir"]) / "gpu_metrics.csv"),
        float(slm_qlora["h5"]["total_transmitted_per_round_Mb"].sum()),
        float(slm_qlora["h5"]["rs_test_acc"][-1]),
    ]
    s_vals = [
        float(slm_qsora["wall_seconds"]) / 60.0,
        _gpu0_energy_wh(Path(slm_qsora["telemetry_dir"]) / "gpu_metrics.csv"),
        float(slm_qsora["h5"]["total_transmitted_per_round_Mb"].sum()),
        float(slm_qsora["h5"]["rs_test_acc"][-1]),
    ]
    xb = np.arange(len(metrics_lbls))
    wb = 0.35
    # Normaliza cada métrica pelo maior valor do par para exibir na mesma escala
    q_norm = [q / max(q, s, 1e-6) * 100.0 for q, s in zip(q_vals, s_vals, strict=False)]
    s_norm = [s / max(q, s, 1e-6) * 100.0 for q, s in zip(q_vals, s_vals, strict=False)]
    ax_bar.bar(xb - wb / 2, q_norm, wb, label="SLM QLoRA", color="#ff7f0e")
    ax_bar.bar(xb + wb / 2, s_norm, wb, label="SLM QSoRA", color="#1f77b4")
    for i, (qn, sn, qv, sv) in enumerate(zip(q_norm, s_norm, q_vals, s_vals, strict=False)):
        ax_bar.text(i - wb / 2, qn + 2, f"{qv:.1f}", ha="center", va="bottom", fontsize=8.5)
        ax_bar.text(i + wb / 2, sn + 2, f"{sv:.1f}", ha="center", va="bottom", fontsize=8.5)
    ax_bar.set_ylim(0, 140)
    ax_bar.set_xticks(xb)
    ax_bar.set_xticklabels(metrics_lbls, fontsize=9)
    ax_bar.set_ylabel("Escala Relativa (% do maior)")
    ax_bar.set_title("Resumo de Custo FL (Valores Absolutos no Topo)")
    ax_bar.legend(loc="upper right")

    fig.suptitle(
        "Custo Real em Aprendizado Federado (OxfordPets, 5 rodadas, 2 clientes): QLoRA vs QSoRA",
        y=1.01,
    )
    _save_fig(fig, out_dir, "07_qlora_vs_qsora_custo_fl")


def _plot_qlora_vs_qsora_aux_accuracy(results_root: Path, out_dir: Path) -> None:
    """Compara a acurácia auxiliar média (rs_aux_simple_avg_acc) dos modelos da fig. 06."""
    specs = [
        (
            "SLM (OxfordPets)",
            [
                (
                    "BF16 LoRA (4c, 15r)",
                    "qlora_validation_suite/runs/val_slm_lora_bf16",
                    "#d62728",
                    "o",
                    "-",
                ),
                (
                    "4-bit QLoRA (4c, 15r)",
                    "qlora_validation_suite/runs/val_slm_qlora_nf4",
                    "#ff7f0e",
                    "s",
                    "-",
                ),
                (
                    "4-bit QSoRA (2c, 15r)",
                    "qsora_validation_suite/runs/val_slm_qsora_15r",
                    "#1f77b4",
                    "^",
                    "-",
                ),
            ],
        ),
        (
            "d1-2B (Banking77)",
            [
                (
                    "BF16 LoRA (4c, 5r)",
                    "qlora_validation_suite/runs/val_d1_lora_bf16",
                    "#d62728",
                    "o",
                    "-",
                ),
                (
                    "4-bit QLoRA (4c, 5r)",
                    "qlora_validation_suite/runs/val_d1_qlora_nf4",
                    "#ff7f0e",
                    "s",
                    "-",
                ),
                (
                    "4-bit QSoRA (4c, 6r)",
                    "qsora_validation_suite/runs/val_d1_qsora_robust_4c6r",
                    "#1f77b4",
                    "^",
                    "-",
                ),
            ],
        ),
        (
            "LFM3-VL-2B (Banking77)",
            [
                (
                    "BF16 LoRA (4c, 3r, PaCA=8*)",
                    "legados/testes_exploratorios/fl_puma_Banking77_lfm_lora_prune1_ala1_adaptpaca_adaptrank",
                    "#d62728",
                    "o",
                    "--",
                ),
                (
                    "4-bit QLoRA (4c, 3r*)",
                    "legados/testes_exploratorios/fl_lfm_qlora_test",
                    "#ff7f0e",
                    "s",
                    "--",
                ),
                (
                    "4-bit QSoRA (4c, 6r)",
                    "qsora_validation_suite/runs/val_lfm_qsora_robust_4c6r",
                    "#1f77b4",
                    "^",
                    "-",
                ),
            ],
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.0))
    curve_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
    ax_bar = axes[1, 1]

    variant_labels = ["BF16 LoRA", "4-bit QLoRA", "4-bit QSoRA"]
    variant_colors = ["#d62728", "#ff7f0e", "#1f77b4"]
    final_matrix: list[list[float]] = []
    round_tags: list[list[str]] = []

    for ax, (group_title, items) in zip(curve_axes, specs, strict=False):
        group_vals: list[float] = []
        group_tags: list[str] = []
        for label, rel_dir, color, marker, ls in items:
            h5 = _load_h5_bundle(results_root / rel_dir)
            aux = h5.get("rs_aux_simple_avg_acc")
            if aux is None or len(aux) == 0:
                group_vals.append(0.0)
                group_tags.append("N/A")
                continue
            rounds = np.arange(1, len(aux) + 1)
            ax.plot(
                rounds,
                aux,
                label=f"{label}: {aux[-1]:.2f}%",
                color=color,
                marker=marker,
                linestyle=ls,
                linewidth=2.0,
                markersize=5.5,
            )
            group_vals.append(float(aux[-1]))
            group_tags.append(f"{len(aux)}r")
        final_matrix.append(group_vals)
        round_tags.append(group_tags)

        ax.set_title(f"{group_title} — Aux Accuracy por Rodada")
        ax.set_xlabel("Rodada")
        ax.set_ylabel("Aux Accuracy Média (%)")
        ax.set_ylim(-2, 106)
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    x = np.arange(len(specs))
    w = 0.24
    for v_idx, (v_label, color) in enumerate(zip(variant_labels, variant_colors, strict=False)):
        vals = [row[v_idx] for row in final_matrix]
        tags = [row[v_idx] for row in round_tags]
        bars = ax_bar.bar(x + (v_idx - 1) * w, vals, w, label=v_label, color=color)
        for idx, (bar, val, rtag) in enumerate(zip(bars, vals, tags, strict=False)):
            suffix = "*" if (idx == 2 and v_idx < 2) else ""
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                val + 1.5,
                f"{val:.1f}%\n({rtag}{suffix})",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax_bar.set_title("Aux Accuracy Final por Estratégia")
    ax_bar.set_ylabel("Aux Accuracy Média (%)")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([s[0].replace(" (", "\n(") for s in specs])
    ax_bar.set_ylim(0, 122)
    ax_bar.legend(loc="upper right", ncol=3, fontsize=8.5)

    fig.suptitle(
        "Acurácia Auxiliar Média (rs_aux_simple_avg_acc): BF16 LoRA vs QLoRA vs QSoRA\n"
        "(* LFM BF16 LoRA e QLoRA possuem apenas 3 rodadas nos logs exploratórios)",
        y=1.01,
    )
    _save_fig(fig, out_dir, "08_qlora_vs_qsora_aux_accuracy")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera gráficos da suíte de validação QSoRA por dataset."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("src/results/qsora_validation_suite/validation_manifest.json"),
    )
    parser.add_argument(
        "--long-manifest",
        type=Path,
        default=Path("src/results/qsora_validation_suite/long_pruning_manifest.json"),
    )
    parser.add_argument(
        "--robust-manifest",
        type=Path,
        default=Path("src/results/qsora_validation_suite/robust_lfm_d1_manifest.json"),
    )
    parser.add_argument(
        "--bench-json",
        type=Path,
        default=Path("src/results/qsora_validation_suite/qlora_vs_qsora_gpu_benchmark.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("src/results/qsora_validation_suite/plots"),
    )
    args = parser.parse_args()

    by_id: dict[str, dict[str, Any]] = {}
    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        for item in manifest["experiments"]:
            h5 = _load_h5_bundle(Path(item["results_dir"]))
            gpu = _load_gpu_csv(Path(item["telemetry_dir"]) / "gpu_metrics.csv")
            by_id[item["id"]] = {**item, "h5": h5, "gpu": gpu}

        pets_records = [by_id[k] for k in ("slm_qlora", "slm_qsora") if k in by_id]
        bank_records = [by_id[k] for k in ("d1_qsora_banking77", "lfm_qsora") if k in by_id]

        if pets_records:
            _plot_suite(
                pets_records,
                "OxfordPets (37 classes, 5 rodadas) — SLM QLoRA vs QSoRA",
                "01_oxfordpets_slm_qlora_vs_qsora",
                args.out_dir,
            )
        if bank_records:
            _plot_suite(
                bank_records,
                "Banking77 (77 classes, 3 rodadas) — d1-2B QSoRA vs LFM3-VL-2B QSoRA",
                "02_banking77_d1_vs_lfm_qsora",
                args.out_dir,
            )
        if pets_records or bank_records:
            _plot_efficiency_bars(pets_records, bank_records, args.out_dir)
            _write_summary(pets_records, bank_records, args.out_dir.parent)
        if "slm_qlora" in by_id and "slm_qsora" in by_id:
            _plot_qlora_vs_qsora_fl_cost(by_id["slm_qlora"], by_id["slm_qsora"], args.out_dir)

    post_prune_mb: dict[str, float] = {}
    if args.long_manifest.exists():
        long_entry = json.loads(args.long_manifest.read_text(encoding="utf-8"))
        _plot_long_pruning(long_entry, args.out_dir)
        long_h5 = _load_h5_bundle(Path(long_entry["results_dir"]))
        if "Model_size_per_round_Mb" in long_h5 and len(long_h5["Model_size_per_round_Mb"]) > 0:
            post_prune_mb["SLM (OxfordPets)"] = float(long_h5["Model_size_per_round_Mb"][-1])

    if args.robust_manifest.exists():
        robust = json.loads(args.robust_manifest.read_text(encoding="utf-8"))
        robust_records = []
        for item in robust.get("experiments", []):
            h5 = _load_h5_bundle(Path(item["results_dir"]))
            gpu = _load_gpu_csv(Path(item["telemetry_dir"]) / "gpu_metrics.csv")
            robust_records.append({**item, "h5": h5, "gpu": gpu})
            if "Model_size_per_round_Mb" in h5 and len(h5["Model_size_per_round_Mb"]) > 0:
                if item["id"] == "d1_qsora_robust":
                    post_prune_mb["d1-2B (Banking77)"] = float(h5["Model_size_per_round_Mb"][-1])
                elif item["id"] == "lfm_qsora_robust":
                    post_prune_mb["LFM3-VL-2B (Banking77)"] = float(
                        h5["Model_size_per_round_Mb"][-1]
                    )
        if robust_records:
            _plot_suite(
                robust_records,
                "Banking77 (77 classes, 4 clientes, 6 rodadas) — d1-2B vs LFM3-VL-2B QSoRA",
                "05_banking77_robust_4c_6r",
                args.out_dir,
            )

    if args.bench_json.exists():
        bench_items = json.loads(args.bench_json.read_text(encoding="utf-8"))
        _plot_qlora_vs_qsora_vram(bench_items, post_prune_mb, args.out_dir)

    results_root = args.out_dir.parent.parent
    if results_root.exists():
        _plot_qlora_vs_qsora_aux_accuracy(results_root, args.out_dir)


if __name__ == "__main__":
    main()
