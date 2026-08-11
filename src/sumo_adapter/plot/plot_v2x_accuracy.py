#!/usr/bin/env python3
"""
plot_v2x_accuracy.py
=====================
Gera dois graficos a partir dos resultados HDF5 salvos pelos clientes:

1. Acuracia individual de cada cliente ao longo das epocas (V2X)
2. Comparacao da media de acuracia entre cenario V2X e cenario centralizado

Uso:
    cd FL_ServerPuma/
    uv run python3 src/sumo_adapter/plot/plot_v2x_accuracy.py

    # Especificando pastas:
    uv run python3 src/sumo_adapter/plot/plot_v2x_accuracy.py \
        --v2x-dir src/results/default_exp \
        --centralized-dir src/results/centralized_exp

    # Salvar PNGs sem abrir janela:
    uv run python3 src/sumo_adapter/plot/plot_v2x_accuracy.py --save
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


# ── Paleta de cores premium ──────────────────────────────────────────────────
# Cores distintas e vibrantes para ate 10 clientes
COLORS = [
    "#2196F3",  # Azul
    "#FF5722",  # Laranja avermelhado
    "#4CAF50",  # Verde
    "#9C27B0",  # Roxo
    "#FF9800",  # Laranja
    "#00BCD4",  # Ciano
    "#E91E63",  # Rosa
    "#795548",  # Marrom
    "#607D8B",  # Cinza azulado
    "#CDDC39",  # Lima
]


def load_results_from_dir(results_dir: Path) -> dict[int, dict]:
    """Carrega todos os arquivos .h5 de um diretorio de resultados.

    Retorna um dicionario {client_idx: {"local_acc": [...], "train_acc": [...]}}.
    Os client_idx sao extraidos do nome do arquivo (ex: client_2_MNIST_...).
    """
    clients = {}

    if not results_dir.exists():
        return clients

    for h5_file in sorted(results_dir.glob("client_*.h5")):
        # Extrair indice do cliente a partir do nome do arquivo
        # Nome esperado: client_<idx>_<dataset>_...h5
        try:
            parts = h5_file.stem.split("_")
            client_idx = int(parts[1])
        except (IndexError, ValueError):
            print(f"  [WARN] Nao foi possivel extrair client_idx de: {h5_file.name}")
            continue

        with h5py.File(h5_file, "r") as hf:
            local_acc = hf["rs_local_acc"][:] if "rs_local_acc" in hf else np.array([])
            train_acc = hf["rs_train_acc"][:] if "rs_train_acc" in hf else np.array([])

        # So incluir clientes com pelo menos 1 epoca de dados
        if len(local_acc) > 0:
            clients[client_idx] = {
                "local_acc": local_acc,
                "train_acc": train_acc,
                "filename": h5_file.name,
            }

    return clients


def plot_individual_accuracy(v2x_clients: dict[int, dict], output_dir: Path, save: bool):
    """Grafico 1: Acuracia de teste de cada cliente V2X ao longo das epocas."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Estilo do fundo
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    for i, (client_idx, data) in enumerate(sorted(v2x_clients.items())):
        epochs = np.arange(1, len(data["local_acc"]) + 1)
        color = COLORS[i % len(COLORS)]

        ax.plot(
            epochs,
            data["local_acc"],
            marker="o",
            markersize=6,
            linewidth=2.5,
            color=color,
            label=f"Cliente {client_idx}",
            alpha=0.9,
        )

    # Estilizacao
    ax.set_xlabel("Época", fontsize=14, color="white", fontweight="bold")
    ax.set_ylabel("Acurácia (%)", fontsize=14, color="white", fontweight="bold")
    ax.set_title(
        "Acurácia por Cliente — Cenário V2X Descentralizado",
        fontsize=16,
        color="white",
        fontweight="bold",
        pad=20,
    )
    ax.legend(
        fontsize=11,
        loc="lower right",
        facecolor="#0f3460",
        edgecolor="#e94560",
        labelcolor="white",
    )
    ax.tick_params(colors="white", labelsize=11)
    ax.grid(True, alpha=0.2, color="white")
    ax.spines["bottom"].set_color("#e94560")
    ax.spines["left"].set_color("#e94560")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Eixo Y: sempre de 0 a 100
    ax.set_ylim(0, 105)

    plt.tight_layout()

    if save:
        out_path = output_dir / "v2x_accuracy_per_client.pdf"
        fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"  Salvo: {out_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_comparison(
    v2x_clients: dict[int, dict],
    centralized_clients: dict[int, dict],
    output_dir: Path,
    save: bool,
):
    """Grafico 2: Comparacao da media de acuracia V2X vs Centralizado."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Estilo do fundo
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    # ── Calcular media V2X ────────────────────────────────────────────────
    # Cada cliente pode ter numeros diferentes de epocas.
    # Usar o maximo de epocas e preencher com NaN para clientes mais curtos.
    if v2x_clients:
        max_epochs_v2x = max(len(d["local_acc"]) for d in v2x_clients.values())
        v2x_matrix = np.full((len(v2x_clients), max_epochs_v2x), np.nan)
        for i, (_, data) in enumerate(sorted(v2x_clients.items())):
            n = len(data["local_acc"])
            v2x_matrix[i, :n] = data["local_acc"]

        v2x_mean = np.nanmean(v2x_matrix, axis=0)
        v2x_std = np.nanstd(v2x_matrix, axis=0)
        epochs_v2x = np.arange(1, max_epochs_v2x + 1)

        ax.plot(
            epochs_v2x,
            v2x_mean,
            marker="s",
            markersize=7,
            linewidth=3,
            color="#e94560",
            label="V2X Descentralizado (D-PSGD)",
        )
        ax.fill_between(
            epochs_v2x,
            v2x_mean - v2x_std,
            v2x_mean + v2x_std,
            alpha=0.15,
            color="#e94560",
        )

    # ── Calcular media Centralizado ───────────────────────────────────────
    if centralized_clients:
        max_epochs_cent = max(len(d["local_acc"]) for d in centralized_clients.values())
        cent_matrix = np.full((len(centralized_clients), max_epochs_cent), np.nan)
        for i, (_, data) in enumerate(sorted(centralized_clients.items())):
            n = len(data["local_acc"])
            cent_matrix[i, :n] = data["local_acc"]

        cent_mean = np.nanmean(cent_matrix, axis=0)
        cent_std = np.nanstd(cent_matrix, axis=0)
        epochs_cent = np.arange(1, max_epochs_cent + 1)

        ax.plot(
            epochs_cent,
            cent_mean,
            marker="D",
            markersize=7,
            linewidth=3,
            color="#0f3460",
            label="Centralizado (FedAvg)",
        )
        ax.fill_between(
            epochs_cent,
            cent_mean - cent_std,
            cent_mean + cent_std,
            alpha=0.15,
            color="#0f3460",
        )
    else:
        # Se nao ha dados centralizados, exibir aviso no grafico
        ax.text(
            0.5,
            0.3,
            "Dados centralizados não disponíveis.\n"
            "Execute o modo centralizado e passe --centralized-dir.",
            transform=ax.transAxes,
            fontsize=13,
            color="#aaa",
            ha="center",
            va="center",
            style="italic",
        )

    # Estilizacao
    ax.set_xlabel("Época / Rodada", fontsize=14, color="white", fontweight="bold")
    ax.set_ylabel("Acurácia Média de Teste (%)", fontsize=14, color="white", fontweight="bold")
    ax.set_title(
        "V2X Descentralizado vs Centralizado — Média de Acurácia",
        fontsize=16,
        color="white",
        fontweight="bold",
        pad=20,
    )
    ax.legend(
        fontsize=12,
        loc="lower right",
        facecolor="#0f3460",
        edgecolor="#e94560",
        labelcolor="white",
    )
    ax.tick_params(colors="white", labelsize=11)
    ax.grid(True, alpha=0.2, color="white")
    ax.spines["bottom"].set_color("#e94560")
    ax.spines["left"].set_color("#e94560")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 105)

    plt.tight_layout()

    if save:
        out_path = output_dir / "v2x_vs_centralized.pdf"
        fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"  Salvo: {out_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plotar acuracia dos clientes V2X e comparar com centralizado",
    )
    parser.add_argument(
        "--v2x-dir",
        type=str,
        default="src/results/default_exp",
        help="Diretorio com os .h5 dos clientes V2X (default: src/results/default_exp)",
    )
    parser.add_argument(
        "--centralized-dir",
        type=str,
        default="src/results/centralized_exp",
        help="Diretorio com os .h5 dos clientes centralizados (default: src/results/centralized_exp)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Salvar graficos como PDF em vez de abrir janela interativa",
    )
    args = parser.parse_args()

    # Resolver caminhos relativos a partir da raiz do projeto
    project_root = Path(__file__).resolve().parents[3]
    v2x_dir = project_root / args.v2x_dir
    centralized_dir = project_root / args.centralized_dir
    output_dir = Path(__file__).resolve().parent

    print(f"Diretorio V2X:          {v2x_dir}")
    print(f"Diretorio Centralizado: {centralized_dir}")
    print()

    # ── Carregar dados ────────────────────────────────────────────────────
    print("Carregando resultados V2X...")
    v2x_clients = load_results_from_dir(v2x_dir)
    if not v2x_clients:
        print("  [ERRO] Nenhum arquivo .h5 encontrado em:", v2x_dir)
        sys.exit(1)
    for idx, data in sorted(v2x_clients.items()):
        print(f"  Cliente {idx}: {len(data['local_acc'])} epocas, "
              f"ultima acc={data['local_acc'][-1]:.2f}%")

    print()
    print("Carregando resultados centralizados...")
    centralized_clients = load_results_from_dir(centralized_dir)
    if not centralized_clients:
        print("  [INFO] Nenhum arquivo .h5 centralizado encontrado.")
        print("         O grafico comparativo mostrara apenas V2X.")
    else:
        for idx, data in sorted(centralized_clients.items()):
            print(f"  Cliente {idx}: {len(data['local_acc'])} rodadas, "
                  f"ultima acc={data['local_acc'][-1]:.2f}%")

    # ── Gerar graficos ────────────────────────────────────────────────────
    print()
    print("Gerando graficos...")
    plot_individual_accuracy(v2x_clients, output_dir, args.save)
    plot_comparison(v2x_clients, centralized_clients, output_dir, args.save)
    print("Concluido!")


if __name__ == "__main__":
    main()
