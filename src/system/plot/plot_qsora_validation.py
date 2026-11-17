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
        "--out-dir",
        type=Path,
        default=Path("src/results/qsora_validation_suite/plots"),
    )
    args = parser.parse_args()

    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        by_id: dict[str, dict[str, Any]] = {}
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

    if args.long_manifest.exists():
        long_entry = json.loads(args.long_manifest.read_text(encoding="utf-8"))
        _plot_long_pruning(long_entry, args.out_dir)

    if args.robust_manifest.exists():
        robust = json.loads(args.robust_manifest.read_text(encoding="utf-8"))
        robust_records = []
        for item in robust.get("experiments", []):
            h5 = _load_h5_bundle(Path(item["results_dir"]))
            gpu = _load_gpu_csv(Path(item["telemetry_dir"]) / "gpu_metrics.csv")
            robust_records.append({**item, "h5": h5, "gpu": gpu})
        if robust_records:
            _plot_suite(
                robust_records,
                "Banking77 (77 classes, 4 clientes, 6 rodadas) — d1-2B vs LFM3-VL-2B QSoRA",
                "05_banking77_robust_4c_6r",
                args.out_dir,
            )


if __name__ == "__main__":
    main()
