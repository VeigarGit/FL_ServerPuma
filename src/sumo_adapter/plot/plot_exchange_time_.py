#!/usr/bin/env python3
"""
plot_exchange_time_.py
=====================
Gera um grafico de linhas comparando o ETC (Estimated Contact Time)
previsto pela cinematica do SUMO vs o tempo real que a troca P2P
levou para cada encontro da simulacao V2V.

Requer que o campo 'exchange_time' esteja presente no v2v_events.jsonl
(adicionado ao sumo_docker_orchestrator.py).

Uso (de dentro de src/sumo_adapter/):
  uv run python plot/plot_exchange_time_.py
"""

import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    # Procurar o arquivo de log
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "results" / "v2v_events.jsonl",
        Path("../results/v2v_events.jsonl").resolve(),
        Path("../../results/v2v_events.jsonl").resolve(),
    ]

    log_file = None
    for c in candidates:
        if c.exists():
            log_file = c
            break

    if log_file is None:
        print("Erro: Arquivo v2v_events.jsonl nao encontrado.", file=sys.stderr)
        print("Execute uma simulacao primeiro com o sumo_docker_orchestrator.py")
        sys.exit(1)

    print(f"Lendo: {log_file}")

    # Extrair dados dos encontros formados
    etcs = []
    exchanges = []
    encounter_ids = []
    num_clients = []

    with open(log_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if data.get("type") != "encounter_formed":
                continue

            etc = data.get("etc_seconds")
            exchange = data.get("exchange_time")

            if exchange is None:
                continue
            if isinstance(etc, str):
                continue

            etcs.append(float(etc))
            exchanges.append(float(exchange))
            encounter_ids.append(f"Enc {data.get('encounter_id', len(etcs))}")
            num_clients.append(len(data.get("clients", [])))

    if not etcs:
        print(
            "Nenhum dado de 'exchange_time' encontrado no log.",
            file=sys.stderr,
        )
        print(
            "Voce precisa rodar uma NOVA simulacao (com o orquestrador atualizado) "
            "para que os logs incluam esse campo."
        )
        sys.exit(1)

    print(f"  {len(etcs)} encontros com dados de ETC e exchange_time")

    # ── Plotar ────────────────────────────────────────────────────────────
    x = list(range(len(etcs)))

    fig, ax = plt.subplots(figsize=(max(10, len(etcs) * 1.0), 6))

    ax.plot(
        x, etcs,
        color="#2ECC71", marker="o", linewidth=2, markersize=6,
        label="ETC Previsto (s)",
    )
    ax.plot(
        x, exchanges,
        color="#3498DB", marker="s", linewidth=2, markersize=6,
        label="Tempo Real P2P (s)",
    )

    # Anotacoes de valor em cada ponto
    for i in range(len(etcs)):
        ax.annotate(
            f"{etcs[i]:.1f}", xy=(i, etcs[i]),
            xytext=(0, 8), textcoords="offset points",
            ha="center", fontsize=7, color="#27AE60",
        )
        ax.annotate(
            f"{exchanges[i]:.1f}", xy=(i, exchanges[i]),
            xytext=(0, -14), textcoords="offset points",
            ha="center", fontsize=7, color="#2980B9",
        )

    ax.set_ylabel("Segundos", fontsize=12)
    ax.set_xlabel("Encontro", fontsize=12)
    ax.set_title(
        "ETC Previsto vs Tempo Real de Troca P2P por Encontro",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(encounter_ids, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Anotacao com numero de clientes abaixo do eixo
    for i, nc in enumerate(num_clients):
        ax.annotate(
            f"{nc}c",
            xy=(i, 0),
            xytext=(0, -22),
            textcoords="offset points",
            ha="center", va="top", fontsize=7, color="gray",
        )

    fig.tight_layout()

    output_path = "plot/etc_vs_exchange.pdf"
    plt.savefig(output_path, dpi=300)
    print(f"\nGrafico salvo em: {output_path}")


if __name__ == "__main__":
    main()
