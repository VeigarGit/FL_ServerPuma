#!/usr/bin/env python3
"""Gera acurácia, dados e tempo de troca por encontro para execuções V2V/V2I.

Na raiz, para reproduzir os três PDFs da comparação:
    uv run python src/v2v_fl/plot/plot_exchange_time_.py \
        --run-dir src/results/runs/v2v/validacao-01 \
        --run-dir src/results/runs/v2i/validacao-ofc

--accuracy weighted usa média ponderada por amostras em vez da média dos
participantes. Os rótulos n= indicam participantes, que podem mudar a cada encontro.
Os dados são upload + download dos modelos em MiB, sem duplicar remetente/receptor
nem incluir mensagens de controle ou retransmissões TCP. O tempo padrão é
protocol_time (até aplicação/confirmação, incluindo agregação e espera).
--time-metric total_time inclui também avaliação e persistência.

As legendas identificam execução/seed. Fontes e seus hashes ficam nos metadados
dos PDFs. Sem --run-dir, preserva o gráfico legado ETC vs exchange_time.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plot_v2v_accuracy import (
    _save_and_close,
    _set_title,
    load_encounter_runs,
    plot_encounter_metric,
)


def plot_etc_vs_exchange(log_file: Path, output_dir: Path) -> None:
    """Comparação legada; exchange_time inclui todo o encontro nos logs novos."""
    events = []
    for line in log_file.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("type") != "encounter_formed":
            continue
        if event.get("status", "completed") != "completed":
            continue
        etc, exchange = event.get("etc_seconds"), event.get("exchange_time")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in (etc, exchange)):
            continue
        if all(np.isfinite(v) and v >= 0 for v in (etc, exchange)):
            events.append(event)
    if not events:
        raise ValueError(f"Nenhum encontro contém ETC e exchange_time válidos: {log_file}")
    events.sort(key=lambda event: event["encounter_id"])
    ids = [event["encounter_id"] for event in events]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(
        ids,
        [event["etc_seconds"] for event in events],
        color="#2ECC71",
        marker="o",
        linewidth=2,
        label="ETC previsto",
    )
    ax.plot(
        ids,
        [event["exchange_time"] for event in events],
        color="#3498DB",
        marker="s",
        linewidth=2,
        label="Duração do encontro",
    )
    ax.set_xlabel("Encounter ID")
    ax.set_ylabel("Time (s)")
    ax.set_xticks(ids)
    ax.set_ylim(bottom=0)
    _set_title(ax, "ETC previsto vs duração do encontro")
    ax.legend(fontsize=18)
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save_and_close(fig, output_dir, "etc_vs_exchange.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        help="Pasta com run.json e eventos JSONL; repita para comparar",
    )
    source.add_argument("--log-file", type=Path, help="JSONL para o gráfico legado ETC vs tempo")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--accuracy", choices=["mean", "weighted"], default="mean")
    parser.add_argument(
        "--time-metric", choices=["protocol_time", "total_time"], default="protocol_time"
    )
    args = parser.parse_args()
    try:
        if args.run_dir:
            runs = load_encounter_runs(args.run_dir)
            for metric in (f"accuracy_{args.accuracy}", "transfer_bytes", args.time_metric):
                plot_encounter_metric(runs, args.output_dir, metric)
        else:
            project_root = Path(__file__).resolve().parents[3]
            log_file = args.log_file or project_root / "src/results/v2v_events.jsonl"
            plot_etc_vs_exchange(log_file, args.output_dir)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Erro ao gerar gráficos: {exc}\n")


if __name__ == "__main__":
    main()
