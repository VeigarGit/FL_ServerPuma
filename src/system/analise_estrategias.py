#!/usr/bin/env python3
"""
Análise Comparativa de Estratégias de Federated Learning
=========================================================
Compara 3 estratégias (SoRA Estático, LoRA Padrão, PUMA-GT/Adapt PaCa)
em 4 datasets (OxfordPets, DTD, FGVCAircraft, Flowers102).

Métricas coletadas:
  1. Rodada de convergência (janela deslizante com desvio padrão)
  2. Total de dados transmitidos ao final do treinamento (MB)
  3. Acurácia final (%)
  4. Tempo médio por rodada (s) — média ao longo de todo o treinamento
  5. Média de dados transmitidos por rodada (MB)

Resultados: tabela formatada no terminal + CSV.
"""

import os
import sys
import argparse
import csv
import glob
from pathlib import Path

import numpy as np
import h5py


# ======================================================================
# CONFIGURAÇÃO: Mapeamento das pastas de resultados
# ======================================================================
# Cada entrada: (dataset_label, estrategia_label, pasta_relativa, padrão_glob_h5)
#
# O padrão glob para os H5 do servidor usa "*run*.h5" — isso casa com
# server_<dataset>_<strategy>_rank8_paca*_prune_freq*_FedAVG_run<N>.h5
#
# Para adicionar novos experimentos, basta acrescentar entradas aqui.
# ======================================================================

EXPERIMENTS = [
    # --- OxfordPets ---
    ("OxfordPets", "LoRA Padrão",
     "1_lora_padrao_clip_lora_prune1_ala1_paca12_20260801_180443",
     "server_OxfordPets_*_run*.h5"),

    ("OxfordPets", "SoRA Estático",
     "2_sora_estatico_clip_sora_with_schedule_prune1_ala1_paca12_20260802_011456",
     "server_OxfordPets_*_run*.h5"),

    ("OxfordPets", "PUMA-GT",
     "3_pumagt_clip_sora_with_schedule_prune1_ala1_adaptpaca_20260802_095514",
     "server_OxfordPets_*_run*.h5"),

    # --- DTD ---
    ("DTD", "LoRA Padrão",
     "DTD_run_lora_padrao_clip_lora_prune1_ala1_paca12_20260811_174358",
     "server_DTD_*_run*.h5"),

    ("DTD", "SoRA Estático",
     "DTD_run_sora_estatico_clip_sora_with_schedule_prune1_ala1_paca12_20260811_074803",
     "server_DTD_*_run*.h5"),

    ("DTD", "PUMA-GT",
     "DTD_run_sora_adapt_paca_clip_sora_with_schedule_prune1_ala1_adaptpaca_20260810_191511",
     "server_DTD_*_run*.h5"),

    # --- FGVCAircraft ---
    ("FGVCAircraft", "LoRA Padrão",
     "FGVCAircraft_run_lora_padrao_clip_lora_prune1_ala1_paca12_20260812_073136",
     "server_FGVCAircraft_*_run*.h5"),

    ("FGVCAircraft", "SoRA Estático",
     "FGVCAircraft_run_sora_estatico_clip_sora_with_schedule_prune1_ala1_paca12_20260809_123022",
     "server_FGVCAircraft_*_run*.h5"),

    ("FGVCAircraft", "PUMA-GT",
     "FGVCAircraft_run_sora_adapt_paca_clip_sora_with_schedule_prune1_ala1_adaptpaca_20260810_091733",
     "server_FGVCAircraft_*_run*.h5"),

    # --- Flowers102 ---
    ("Flowers102", "LoRA Padrão",
     "Flowers102_run_lora_padrao_clip_lora_prune1_ala1_paca12_20260808_080534",
     "server_Flowers102_*_run*.h5"),

    ("Flowers102", "SoRA Estático",
     "Flowers102_run_sora_estatico_clip_sora_with_schedule_prune1_ala1_paca12_20260807_160845",
     "server_Flowers102_*_run*.h5"),

    ("Flowers102", "PUMA-GT",
     "Flowers102_run_pumagt_clip_sora_with_schedule_prune1_ala1_adaptpaca_20260806_192626",
     "server_Flowers102_*_run*.h5"),
]


# ======================================================================
# PARÂMETROS DA DETECÇÃO DE CONVERGÊNCIA
# ======================================================================
CONVERGENCE_WINDOW = 20     # Tamanho da janela deslizante (em rodadas)
CONVERGENCE_THRESHOLD = 0.5 # Desvio padrão da acurácia dentro da janela (%)


def detect_convergence_round(accuracy_curve: np.ndarray,
                              window: int = CONVERGENCE_WINDOW,
                              threshold: float = CONVERGENCE_THRESHOLD) -> int:
    """
    Detecta a rodada de convergência usando janela deslizante.

    Percorre a curva de acurácia e, para cada posição i (a partir de i = window),
    calcula o desvio padrão da acurácia nas últimas `window` rodadas.
    A rodada de convergência é a primeira onde o desvio padrão cai abaixo
    do `threshold`.

    Se nunca convergir, retorna o número total de rodadas (não convergiu).

    Retorna:
        int: Número da rodada onde a convergência foi detectada (1-indexed).
    """
    n = len(accuracy_curve)
    if n < window:
        return n  # Dados insuficientes para a janela

    for i in range(window, n + 1):
        window_data = accuracy_curve[i - window:i]
        std = np.std(window_data)
        if std < threshold:
            return i  # 1-indexed (rodada i = posição i no array)

    return n  # Nunca convergiu dentro das rodadas disponíveis


def load_single_h5(filepath: str) -> dict:
    """
    Carrega os datasets relevantes de um único arquivo H5 do servidor.

    Retorna um dicionário com:
        - acc_curve: np.ndarray com a acurácia por rodada
        - round_time: np.ndarray com o tempo de cada rodada (s)
        - total_transmitted_per_round: np.ndarray com os dados transmitidos por rodada (MB)
    """
    with h5py.File(filepath, 'r') as f:
        acc_curve = f['rs_test_acc'][:]
        round_time = f['Round_time'][:]
        total_per_round = f['total_transmitted_per_round_Mb'][:]

    return {
        'acc_curve': acc_curve,
        'round_time': round_time,
        'total_transmitted_per_round': total_per_round,
    }


def analyze_single_run(data: dict) -> dict:
    """
    Calcula as métricas para um único run.

    Retorna:
        dict com:
            - convergence_round: int
            - total_data_transmitted_mb: float
            - final_accuracy: float
            - avg_round_time_s: float
            - avg_data_per_round_mb: float
    """
    acc = data['acc_curve']
    rtime = data['round_time']
    tpr = data['total_transmitted_per_round']

    conv_round = detect_convergence_round(acc)
    total_data = float(np.sum(tpr))
    final_acc = float(acc[-1])
    avg_time = float(np.mean(rtime))
    avg_data_round = float(np.mean(tpr))

    return {
        'convergence_round': conv_round,
        'total_data_transmitted_mb': total_data,
        'final_accuracy': final_acc,
        'avg_round_time_s': avg_time,
        'avg_data_per_round_mb': avg_data_round,
    }


def analyze_experiment(results_dir: str, folder: str, h5_glob: str) -> dict:
    """
    Analisa todos os runs de um experimento (pasta).

    Retorna dict com média ± desvio padrão de cada métrica,
    e o número de runs encontrados.
    """
    exp_path = os.path.join(results_dir, folder)
    if not os.path.isdir(exp_path):
        return None

    h5_files = sorted(glob.glob(os.path.join(exp_path, h5_glob)))
    if not h5_files:
        return None

    # Acumular métricas de cada run
    metrics_keys = [
        'convergence_round', 'total_data_transmitted_mb',
        'final_accuracy', 'avg_round_time_s', 'avg_data_per_round_mb'
    ]
    all_metrics = {k: [] for k in metrics_keys}

    for h5f in h5_files:
        try:
            data = load_single_h5(h5f)
            run_metrics = analyze_single_run(data)
            for k in metrics_keys:
                all_metrics[k].append(run_metrics[k])
        except Exception as e:
            print(f"  [WARN] Erro ao processar {os.path.basename(h5f)}: {e}")

    n_runs = len(all_metrics['convergence_round'])
    if n_runs == 0:
        return None

    # Calcular média ± desvio padrão
    result = {'n_runs': n_runs}
    for k in metrics_keys:
        arr = np.array(all_metrics[k])
        result[f'{k}_mean'] = float(np.mean(arr))
        result[f'{k}_std'] = float(np.std(arr))

    return result


def format_metric(mean: float, std: float, fmt: str = '.2f', n_runs: int = 10) -> str:
    """Formata métrica como 'mean ± std'. Se n_runs==1, mostra só a média."""
    if n_runs == 1:
        return f"{mean:{fmt}}"
    return f"{mean:{fmt}} ± {std:{fmt}}"


def compute_improvement(puma: dict, lora: dict) -> dict:
    """
    Calcula a melhoria percentual do PUMA-GT em relação ao LoRA Padrão.

    Para acurácia (higher is better):  ((PUMA - LoRA) / LoRA) * 100
    Para as demais (lower is better):  ((LoRA - PUMA) / LoRA) * 100

    Valores positivos significam que o PUMA-GT é melhor.
    """
    imp = {}

    # Acurácia: maior é melhor
    lora_acc = lora['final_accuracy_mean']
    puma_acc = puma['final_accuracy_mean']
    imp['accuracy'] = ((puma_acc - lora_acc) / lora_acc) * 100 if lora_acc != 0 else 0.0

    # Convergência: menor é melhor
    lora_conv = lora['convergence_round_mean']
    puma_conv = puma['convergence_round_mean']
    imp['convergence'] = ((lora_conv - puma_conv) / lora_conv) * 100 if lora_conv != 0 else 0.0

    # Dados totais: menor é melhor
    lora_data = lora['total_data_transmitted_mb_mean']
    puma_data = puma['total_data_transmitted_mb_mean']
    imp['total_data'] = ((lora_data - puma_data) / lora_data) * 100 if lora_data != 0 else 0.0

    # Tempo médio por rodada: menor é melhor
    lora_time = lora['avg_round_time_s_mean']
    puma_time = puma['avg_round_time_s_mean']
    imp['avg_time'] = ((lora_time - puma_time) / lora_time) * 100 if lora_time != 0 else 0.0

    # Dados médios por rodada: menor é melhor
    lora_dpr = lora['avg_data_per_round_mb_mean']
    puma_dpr = puma['avg_data_per_round_mb_mean']
    imp['avg_data_round'] = ((lora_dpr - puma_dpr) / lora_dpr) * 100 if lora_dpr != 0 else 0.0

    return imp


def format_improvement(value: float) -> str:
    """Formata improvement com sinal e seta direcional."""
    arrow = "▲" if value > 0 else "▼" if value < 0 else "─"
    return f"{arrow} {value:+.2f}%"


def main():
    # ======================================================================
    # ARGUMENTOS DE LINHA DE COMANDO
    # ======================================================================
    parser = argparse.ArgumentParser(
        description='Análise comparativa de estratégias de Federated Learning'
    )
    parser.add_argument(
        '--improvement', action='store_true',
        help='Adiciona linha de improvement PUMA-GT vs LoRA Padrão (em %%) para cada dataset'
    )
    args = parser.parse_args()

    # Caminho base dos resultados
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, '..', 'results')
    results_dir = os.path.normpath(results_dir)

    if not os.path.exists(results_dir):
        print(f"[ERRO] Diretório de resultados não encontrado: {results_dir}")
        sys.exit(1)

    print("=" * 120)
    print("ANÁLISE COMPARATIVA DE ESTRATÉGIAS — Federated Learning")
    print("=" * 120)
    print(f"Diretório de resultados: {results_dir}")
    print(f"Convergência: janela={CONVERGENCE_WINDOW} rodadas, threshold={CONVERGENCE_THRESHOLD}% (desvio padrão)")
    print()

    # Coletar resultados
    all_results = []

    for dataset, strategy, folder, h5_glob in EXPERIMENTS:
        print(f"Processando: {dataset} / {strategy} ...", end=" ")
        result = analyze_experiment(results_dir, folder, h5_glob)

        if result is None:
            print("⚠️  NÃO ENCONTRADO ou SEM DADOS")
            all_results.append({
                'dataset': dataset,
                'strategy': strategy,
                'n_runs': 0,
                'status': 'MISSING',
            })
            continue

        n = result['n_runs']
        print(f"✅ {n} run(s)")
        all_results.append({
            'dataset': dataset,
            'strategy': strategy,
            'status': 'OK',
            **result,
        })

    # ======================================================================
    # EXIBIÇÃO: Tabela formatada no terminal
    # ======================================================================
    print()
    print("=" * 160)
    header = (
        f"{'Dataset':<15} | {'Estratégia':<16} | {'Runs':>4} | "
        f"{'Acurácia Final (%)':>22} | {'Rodada Converg.':>20} | "
        f"{'Dados Total (MB)':>22} | {'Tempo Médio/Rod (s)':>22} | "
        f"{'Dados Médios/Rod (MB)':>24}"
    )
    print(header)
    print("-" * 160)

    for r in all_results:
        if r['status'] == 'MISSING':
            print(
                f"{r['dataset']:<15} | {r['strategy']:<16} | {'---':>4} | "
                f"{'--- SEM DADOS ---':>22} | {'---':>20} | "
                f"{'---':>22} | {'---':>22} | {'---':>24}"
            )
            continue

        n = r['n_runs']
        acc = format_metric(r['final_accuracy_mean'], r['final_accuracy_std'], '.2f', n)
        conv = format_metric(r['convergence_round_mean'], r['convergence_round_std'], '.1f', n)
        total_data = format_metric(r['total_data_transmitted_mb_mean'], r['total_data_transmitted_mb_std'], '.2f', n)
        avg_time = format_metric(r['avg_round_time_s_mean'], r['avg_round_time_s_std'], '.2f', n)
        avg_data = format_metric(r['avg_data_per_round_mb_mean'], r['avg_data_per_round_mb_std'], '.2f', n)

        print(
            f"{r['dataset']:<15} | {r['strategy']:<16} | {n:>4} | "
            f"{acc:>22} | {conv:>20} | "
            f"{total_data:>22} | {avg_time:>22} | "
            f"{avg_data:>24}"
        )

    # ======================================================================
    # EXIBIÇÃO: Linhas de Improvement (PUMA-GT vs LoRA Padrão)
    # ======================================================================
    improvement_data = {}  # dataset -> dict de improvements
    if args.improvement:
        print("-" * 160)
        datasets_list = ["OxfordPets", "DTD", "FGVCAircraft", "Flowers102"]
        for ds in datasets_list:
            puma = next((r for r in all_results if r['dataset'] == ds and r['strategy'] == 'PUMA-GT' and r['status'] == 'OK'), None)
            lora = next((r for r in all_results if r['dataset'] == ds and r['strategy'] == 'LoRA Padrão' and r['status'] == 'OK'), None)

            if puma is None or lora is None:
                print(
                    f"{ds:<15} | {'⚡ Improvement':<16} | {'---':>4} | "
                    f"{'--- DADOS INSUF. ---':>22} | {'---':>20} | "
                    f"{'---':>22} | {'---':>22} | {'---':>24}"
                )
                continue

            imp = compute_improvement(puma, lora)
            improvement_data[ds] = imp

            acc_imp = format_improvement(imp['accuracy'])
            conv_imp = format_improvement(imp['convergence'])
            data_imp = format_improvement(imp['total_data'])
            time_imp = format_improvement(imp['avg_time'])
            dpr_imp = format_improvement(imp['avg_data_round'])

            print(
                f"{ds:<15} | {'⚡ Improvement':<16} | {'':>4} | "
                f"{acc_imp:>22} | {conv_imp:>20} | "
                f"{data_imp:>22} | {time_imp:>22} | "
                f"{dpr_imp:>24}"
            )

    print("=" * 160)

    # ======================================================================
    # EXIBIÇÃO: Análise comparativa por dataset
    # ======================================================================
    print("\n")
    datasets = ["OxfordPets", "DTD", "FGVCAircraft", "Flowers102"]
    strategies = ["LoRA Padrão", "SoRA Estático", "PUMA-GT"]

    for ds in datasets:
        ds_results = [r for r in all_results if r['dataset'] == ds and r['status'] == 'OK']
        if len(ds_results) < 2:
            continue

        print(f"📊 COMPARATIVO — {ds}")
        print("-" * 80)

        # Encontrar melhor acurácia
        best_acc = max(ds_results, key=lambda x: x['final_accuracy_mean'])
        # Encontrar menor total de dados
        least_data = min(ds_results, key=lambda x: x['total_data_transmitted_mb_mean'])
        # Encontrar convergência mais rápida
        fastest_conv = min(ds_results, key=lambda x: x['convergence_round_mean'])
        # Encontrar menor tempo por rodada
        fastest_round = min(ds_results, key=lambda x: x['avg_round_time_s_mean'])

        print(f"  🏆 Melhor acurácia:         {best_acc['strategy']} ({best_acc['final_accuracy_mean']:.2f}%)")
        print(f"  📉 Menos dados transmitidos: {least_data['strategy']} ({least_data['total_data_transmitted_mb_mean']:.2f} MB)")
        print(f"  ⚡ Convergência mais rápida: {fastest_conv['strategy']} (rodada {fastest_conv['convergence_round_mean']:.1f})")
        print(f"  🚀 Rodada mais rápida:       {fastest_round['strategy']} ({fastest_round['avg_round_time_s_mean']:.2f}s)")
        print()

    # ======================================================================
    # SALVAR CSV
    # ======================================================================
    csv_path = os.path.join(results_dir, 'analise_comparativa_estrategias.csv')
    csv_columns = [
        'Dataset', 'Estrategia', 'N_Runs',
        'Acuracia_Final_Mean', 'Acuracia_Final_Std',
        'Rodada_Convergencia_Mean', 'Rodada_Convergencia_Std',
        'Dados_Total_MB_Mean', 'Dados_Total_MB_Std',
        'Tempo_Medio_Rodada_s_Mean', 'Tempo_Medio_Rodada_s_Std',
        'Dados_Medios_Rodada_MB_Mean', 'Dados_Medios_Rodada_MB_Std',
    ]

    # Se --improvement ativo, adicionar colunas de improvement ao CSV
    if args.improvement:
        csv_columns.extend([
            'Improvement_Acuracia_%',
            'Improvement_Convergencia_%',
            'Improvement_Dados_Total_%',
            'Improvement_Tempo_Medio_%',
            'Improvement_Dados_Rodada_%',
        ])

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()

        for r in all_results:
            if r['status'] == 'MISSING':
                writer.writerow({
                    'Dataset': r['dataset'],
                    'Estrategia': r['strategy'],
                    'N_Runs': 0,
                })
                continue

            writer.writerow({
                'Dataset': r['dataset'],
                'Estrategia': r['strategy'],
                'N_Runs': r['n_runs'],
                'Acuracia_Final_Mean': f"{r['final_accuracy_mean']:.4f}",
                'Acuracia_Final_Std': f"{r['final_accuracy_std']:.4f}",
                'Rodada_Convergencia_Mean': f"{r['convergence_round_mean']:.2f}",
                'Rodada_Convergencia_Std': f"{r['convergence_round_std']:.2f}",
                'Dados_Total_MB_Mean': f"{r['total_data_transmitted_mb_mean']:.4f}",
                'Dados_Total_MB_Std': f"{r['total_data_transmitted_mb_std']:.4f}",
                'Tempo_Medio_Rodada_s_Mean': f"{r['avg_round_time_s_mean']:.4f}",
                'Tempo_Medio_Rodada_s_Std': f"{r['avg_round_time_s_std']:.4f}",
                'Dados_Medios_Rodada_MB_Mean': f"{r['avg_data_per_round_mb_mean']:.4f}",
                'Dados_Medios_Rodada_MB_Std': f"{r['avg_data_per_round_mb_std']:.4f}",
            })

        # Linhas de improvement no CSV
        if args.improvement:
            for ds, imp in improvement_data.items():
                writer.writerow({
                    'Dataset': ds,
                    'Estrategia': 'IMPROVEMENT (PUMA-GT vs LoRA)',
                    'N_Runs': '',
                    'Improvement_Acuracia_%': f"{imp['accuracy']:.4f}",
                    'Improvement_Convergencia_%': f"{imp['convergence']:.4f}",
                    'Improvement_Dados_Total_%': f"{imp['total_data']:.4f}",
                    'Improvement_Tempo_Medio_%': f"{imp['avg_time']:.4f}",
                    'Improvement_Dados_Rodada_%': f"{imp['avg_data_round']:.4f}",
                })

    print(f"\n📁 Resultados salvos em: {csv_path}")
    print()


if __name__ == '__main__':
    main()
