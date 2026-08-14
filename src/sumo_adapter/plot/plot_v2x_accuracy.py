#!/usr/bin/env python3
"""
plot_v2x_accuracy.py
=====================
Gera graficos a partir dos resultados HDF5 salvos pelos clientes:

1. Acuracia individual de cada cliente ao longo das epocas (V2X)
2. Comparacao da media de acuracia entre cenario V2X e cenario centralizado

Uso:
    cd FL_ServerPuma/
    uv run python3 src/sumo_adapter/plot/plot_v2x_accuracy.py

    # Especificando pastas:
    uv run python3 src/sumo_adapter/plot/plot_v2x_accuracy.py \
        --v2x-dir src/results/default_exp \
        --centralized-dir src/results/centralized_exp

Os graficos sao salvos como PDF no diretorio de resultados do experimento V2X.
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ==============================================================================
# Configurações globais
# ==============================================================================

# Flag para ativar/desativar títulos explicativos nos gráficos.
# True  = títulos aparecem (útil para apresentações e revisões)
# False = títulos ocultos (útil para artigos/papers onde a legenda da figura é suficiente)
SHOW_TITLES = False

plt.rcParams.update({
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
})

# ==============================================================================
# Paleta de cores e estilos para comparação
# ==============================================================================

STYLES = [
    {'color': '#1f77b4', 'marker': 'o',  'ls': '-'},
    {'color': '#d62728', 'marker': 's',  'ls': '--'},
    {'color': '#2ca02c', 'marker': 'D',  'ls': '-.'},
    {'color': '#ff7f0e', 'marker': '^',  'ls': ':'},
    {'color': '#9467bd', 'marker': 'v',  'ls': '-'},
    {'color': '#8c564b', 'marker': 'P',  'ls': '--'},
    {'color': '#e377c2', 'marker': 'X',  'ls': '-.'},
    {'color': '#17becf', 'marker': 'h',  'ls': ':'},
    {'color': '#bcbd22', 'marker': '*',  'ls': '-'},
    {'color': '#7f7f7f', 'marker': 'd',  'ls': '--'},
]


# ==============================================================================
# Funções auxiliares
# ==============================================================================

def _mark_every(n):
    return max(1, n // 10)


def _set_title(ax, title):
    """Aplica título ao eixo apenas se SHOW_TITLES estiver ativado."""
    if SHOW_TITLES:
        ax.set_title(title, fontsize=18, fontweight='bold', pad=12)


def _save_and_close(fig, output_dir, filename):
    """Salva a figura como PDF e fecha."""
    import os
    path = os.path.join(str(output_dir), filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  📊 {path}")


# ==============================================================================
# Funções de carregamento
# ==============================================================================

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
            post_p2p_acc = hf["rs_post_p2p_acc"][:] if "rs_post_p2p_acc" in hf else np.array([])
            post_p2p_enc_ids = hf["rs_post_p2p_encounter_ids"][:] if "rs_post_p2p_encounter_ids" in hf else np.array([])

        # So incluir clientes com pelo menos 1 epoca de dados
        if len(local_acc) > 0:
            clients[client_idx] = {
                "local_acc": local_acc,
                "train_acc": train_acc,
                "post_p2p_acc": post_p2p_acc,
                "post_p2p_enc_ids": post_p2p_enc_ids,
                "filename": h5_file.name,
            }

    return clients


# ==============================================================================
# 01 - Acurácia Individual por Cliente (V2X)
# ==============================================================================

def plot_individual_accuracy(v2x_clients: dict[int, dict], output_dir: Path):
    """Acuracia de teste de cada cliente V2X ao longo das epocas."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for i, (client_idx, data) in enumerate(sorted(v2x_clients.items())):
        s = STYLES[i % len(STYLES)]
        epochs = np.arange(1, len(data["local_acc"]) + 1)

        ax.plot(
            epochs,
            data["local_acc"],
            color=s['color'],
            marker=s['marker'],
            markevery=_mark_every(len(data["local_acc"])),
            linestyle=s['ls'],
            label=f"Client {client_idx}",
        )

    _set_title(ax, "Per-Client Accuracy — V2X Decentralized Scenario")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    _save_and_close(fig, output_dir, "v2x_accuracy_per_client.pdf")



# ==============================================================================
# 02 - Acurácia Pós-Agregação P2P por Cliente
# ==============================================================================

def plot_post_p2p_accuracy(v2x_clients: dict[int, dict], output_dir: Path):
    """Acuracia de teste de cada cliente imediatamente apos cada agregacao P2P."""
    # Filtrar clientes que possuem dados de agregacao P2P
    clients_with_p2p = {
        idx: data for idx, data in v2x_clients.items()
        if len(data.get("post_p2p_acc", [])) > 0
    }

    if not clients_with_p2p:
        print("  [INFO] Nenhum dado de acuracia pos-P2P encontrado. Pulando grafico.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, (client_idx, data) in enumerate(sorted(clients_with_p2p.items())):
        s = STYLES[i % len(STYLES)]
        enc_ids = data["post_p2p_enc_ids"].astype(int)
        acc = data["post_p2p_acc"]

        ax.plot(
            enc_ids,
            acc,
            color=s['color'],
            marker=s['marker'],
            markevery=_mark_every(len(acc)),
            linestyle=s['ls'],
            label=f"Client {client_idx}",
        )

    _set_title(ax, "Post-P2P Aggregation Accuracy — V2X Decentralized")
    ax.set_xlabel("Encounter ID")
    ax.set_ylabel("Accuracy (%)")
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    _save_and_close(fig, output_dir, "v2x_post_p2p_accuracy.pdf")


# ==============================================================================
# 03 - Comparação V2X vs Centralizado (Média de Acurácia)
# ==============================================================================

def plot_comparison(
    v2x_clients: dict[int, dict],
    centralized_clients: dict[int, dict],
    output_dir: Path,
):
    """Comparacao da media de acuracia V2X vs Centralizado."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # ── Calcular media V2X ────────────────────────────────────────────────
    if v2x_clients:
        max_epochs_v2x = max(len(d["local_acc"]) for d in v2x_clients.values())
        v2x_matrix = np.full((len(v2x_clients), max_epochs_v2x), np.nan)
        for i, (_, data) in enumerate(sorted(v2x_clients.items())):
            n = len(data["local_acc"])
            v2x_matrix[i, :n] = data["local_acc"]

        v2x_mean = np.nanmean(v2x_matrix, axis=0)
        v2x_std = np.nanstd(v2x_matrix, axis=0)
        epochs_v2x = np.arange(1, max_epochs_v2x + 1)

        s = STYLES[0]
        ax.plot(
            epochs_v2x,
            v2x_mean,
            color=s['color'],
            marker=s['marker'],
            markevery=_mark_every(len(v2x_mean)),
            linestyle=s['ls'],
            label="V2X Decentralized",
        )
        ax.fill_between(
            epochs_v2x,
            v2x_mean - v2x_std,
            v2x_mean + v2x_std,
            alpha=0.1,
            color=s['color'],
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

        s = STYLES[1]
        ax.plot(
            epochs_cent,
            cent_mean,
            color=s['color'],
            marker=s['marker'],
            markevery=_mark_every(len(cent_mean)),
            linestyle=s['ls'],
            label="Centralized",
        )
        ax.fill_between(
            epochs_cent,
            cent_mean - cent_std,
            cent_mean + cent_std,
            alpha=0.1,
            color=s['color'],
        )
    else:
        ax.text(
            0.5, 0.3,
            "Centralized data not available.\n"
            "Run centralized mode and pass --centralized-dir.",
            transform=ax.transAxes, fontsize=13,
            color="#999", ha="center", va="center", style="italic",
        )

    _set_title(ax, "V2X Decentralized vs Centralized")
    ax.set_xlabel("Epoch / Round")
    ax.set_ylabel("Average Test Accuracy (%)")
    ax.set_ylim(0, 100)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    _save_and_close(fig, output_dir, "v2x_vs_centralized.pdf")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plotar acuracia dos clientes V2X e comparar com centralizado",
    )
    parser.add_argument(
        "--v2x-dir",
        type=str,
        default="src/results/default_exp_v2x",
        help="Diretorio com os .h5 dos clientes V2X (default: src/results/default_exp_v2x)",
    )
    parser.add_argument(
        "--centralized-dir",
        type=str,
        default="src/results/default_exp",
        help="Diretorio com os .h5 dos clientes centralizados (default: src/results/centralized_exp)",
    )
    args = parser.parse_args()

    # Resolver caminhos relativos a partir da raiz do projeto
    project_root = Path(__file__).resolve().parents[3]
    v2x_dir = project_root / args.v2x_dir
    centralized_dir = project_root / args.centralized_dir

    # Salvar graficos na pasta de resultados do experimento V2X
    output_dir = Path(__file__).resolve().parent

    print(f"Diretorio V2X:          {v2x_dir}")
    print(f"Diretorio Centralizado: {centralized_dir}")
    print(f"Saida:                  {output_dir}")
    print()

    # ── Carregar dados ────────────────────────────────────────────────────
    print("Carregando resultados V2X...")
    v2x_clients = load_results_from_dir(v2x_dir)
    if not v2x_clients:
        print("  [ERRO] Nenhum arquivo .h5 encontrado em:", v2x_dir)
        sys.exit(1)
    for idx, data in sorted(v2x_clients.items()):
        p2p_info = f", {len(data['post_p2p_acc'])} encontros P2P" if len(data.get('post_p2p_acc', [])) > 0 else ""
        print(f"  Cliente {idx}: {len(data['local_acc'])} epocas{p2p_info}, "
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
    plot_individual_accuracy(v2x_clients, output_dir)
    plot_post_p2p_accuracy(v2x_clients, output_dir)
    plot_comparison(v2x_clients, centralized_clients, output_dir)
    print("\nConcluido!")


if __name__ == "__main__":
    main()
