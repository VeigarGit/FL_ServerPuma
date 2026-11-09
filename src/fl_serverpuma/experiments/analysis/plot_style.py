# Responsabilidade: Centraliza estilos, opções de apresentação, seleção de experimentos e
# salvamento das figuras.
#
# Principais funções: _mark_every, _set_title, _save_and_close.

"""Experiment plotting — plot_style."""

from fl_serverpuma.paths import SOURCE_ROOT
import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

SHOW_TITLES = False

SHOW_RANK_IN_LEGEND = False

plt.rcParams.update({
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
})

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

SCRIPT_DIR = str(SOURCE_ROOT / "system" / "plot")

DEFAULT_RESULTS_BASE = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', 'results'))

DATASET_EXPERIMENTS = {
    "OxfordPets": [
        "fl_puma_clip_lora_prune1_ala1_paca12",
        "fl_puma_OxfordPets_clip_sora_with_schedule_prune1_ala1_paca12",
        "fl_puma_OxfordPets_clip_sora_with_schedule_prune1_ala1_adaptpaca",
    ],
    "Flowers102": [
        "fl_puma_Flowers102_clip_lora_prune1_ala1_paca12",
        "fl_puma_Flowers102_clip_sora_with_schedule_prune1_ala1_paca12",
        "fl_puma_Flowers102_clip_sora_with_schedule_prune1_ala1_adaptpaca",
    ],
    "DTD": [
        "fl_puma_DTD_clip_lora_prune1_ala1_paca12",
        "fl_puma_DTD_clip_sora_with_schedule_prune1_ala1_paca12",
        "fl_puma_DTD_clip_sora_with_schedule_prune1_ala1_adaptpaca",
    ],
    "FGVCAircraft": [
        "fl_puma_FGVCAircraft_clip_lora_prune1_ala1_paca12",
        "fl_puma_FGVCAircraft_clip_sora_with_schedule_prune1_ala1_paca12",
        "fl_puma_FGVCAircraft_clip_sora_with_schedule_prune1_ala1_adaptpaca",
    ],
}

EXPERIMENTOS_PARA_PLOTAR = DATASET_EXPERIMENTS["OxfordPets"]


def _mark_every(n):
    return max(1, n // 10)


def _set_title(ax, title):
    """Aplica título ao eixo apenas se SHOW_TITLES estiver ativado."""
    if SHOW_TITLES:
        ax.set_title(title, fontsize=18, fontweight='bold', pad=12)


def _save_and_close(fig, output_dir, filename):
    """Salva a figura como PDF e fecha."""
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  📊 {path}")
