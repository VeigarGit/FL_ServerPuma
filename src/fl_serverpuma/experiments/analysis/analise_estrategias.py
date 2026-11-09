# Responsabilidade: CLI e apresentação dos resumos comparativos entre estratégias.
#
# Principais funções: main.

"""Strategy comparison — analise_estrategias."""

from fl_serverpuma.paths import SOURCE_ROOT
import os
import sys
import argparse
import csv
from fl_serverpuma.experiments.analysis.strategy_metrics import analyze_experiment
from fl_serverpuma.experiments.analysis.strategy_metrics import format_metric
from fl_serverpuma.experiments.analysis.strategy_metrics import compute_improvement
from fl_serverpuma.experiments.analysis.strategy_metrics import format_improvement
from fl_serverpuma.experiments.analysis.strategy_settings import CONVERGENCE_THRESHOLD
from fl_serverpuma.experiments.analysis.strategy_settings import CONVERGENCE_WINDOW
from fl_serverpuma.experiments.analysis.strategy_settings import EXPERIMENTS

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
    base_dir = str(SOURCE_ROOT / "system" / "plot")
    candidates = [
        os.path.normpath(os.path.join(base_dir, '..', '..', 'results')),
        os.path.normpath(os.path.join(base_dir, '..', 'results')),
    ]
    results_dir = next((c for c in candidates if os.path.exists(c)), candidates[0])

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
        f"{'Dataset':<15} | {'Estratégia':<20} | {'Runs':>4} | "
        f"{'Acurácia Final (%)':>22} | {'Rodada Converg.':>20} | "
        f"{'Dados Total (MB)':>22} | {'Tempo Médio/Rod (s)':>22} | "
        f"{'Dados Médios/Rod (MB)':>24}"
    )
    print(header)
    print("-" * 168)

    for r in all_results:
        if r['status'] == 'MISSING':
            print(
                f"{r['dataset']:<15} | {r['strategy']:<20} | {'---':>4} | "
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
            f"{r['dataset']:<15} | {r['strategy']:<20} | {n:>4} | "
            f"{acc:>22} | {conv:>20} | "
            f"{total_data:>22} | {avg_time:>22} | "
            f"{avg_data:>24}"
        )

    # ======================================================================
    # EXIBIÇÃO: Linhas de Improvement (PUMA-GT Rank Adap vs LoRA Padrão)
    # ======================================================================
    improvement_data = {}  # dataset -> dict de improvements
    if args.improvement:
        print("-" * 168)
        datasets_list = ["OxfordPets", "DTD", "FGVCAircraft", "Flowers102"]
        for ds in datasets_list:
            puma = next((r for r in all_results if r['dataset'] == ds and r['strategy'] == 'PUMA-GT Rank Adap' and r['status'] == 'OK'), None)
            if puma is None:
                puma = next((r for r in all_results if r['dataset'] == ds and r['strategy'] == 'PUMA-GT' and r['status'] == 'OK'), None)
            lora = next((r for r in all_results if r['dataset'] == ds and r['strategy'] == 'LoRA Padrão' and r['status'] == 'OK'), None)

            if puma is None or lora is None:
                print(
                    f"{ds:<15} | {'⚡ Improvement':<20} | {'---':>4} | "
                    f"{'--- DADOS INSUF. ---':>22} | {'---':>20} | "
                    f"{'---':>22} | {'---':>22} | {'---':>24}"
                )
                continue

            imp = compute_improvement(puma, lora)
            improvement_data[ds] = (imp, puma['strategy'])

            acc_imp = format_improvement(imp['accuracy'])
            conv_imp = format_improvement(imp['convergence'])
            data_imp = format_improvement(imp['total_data'])
            time_imp = format_improvement(imp['avg_time'])
            dpr_imp = format_improvement(imp['avg_data_round'])

            print(
                f"{ds:<15} | {'⚡ Improvement':<20} | {'':>4} | "
                f"{acc_imp:>22} | {conv_imp:>20} | "
                f"{data_imp:>22} | {time_imp:>22} | "
                f"{dpr_imp:>24}"
            )

    print("=" * 168)

    # ======================================================================
    # EXIBIÇÃO: Eleição do melhor por atributo (por dataset)
    # ======================================================================
    print("\n")
    datasets = ["OxfordPets", "DTD", "FGVCAircraft", "Flowers102"]

    # Definição dos atributos com: (label, chave_mean, higher_is_better)
    attributes = [
        ("Acurácia Final",       'final_accuracy_mean',              True),
        ("Rodada Converg.",      'convergence_round_mean',           False),
        ("Dados Total (MB)",     'total_data_transmitted_mb_mean',   False),
        ("Tempo Médio/Rod (s)",  'avg_round_time_s_mean',            False),
        ("Dados Médios/Rod (MB)",'avg_data_per_round_mb_mean',       False),
    ]

    # Contagem global de vitórias por estratégia
    global_wins = {}  # strategy -> count

    for ds in datasets:
        ds_results = [r for r in all_results if r['dataset'] == ds and r['status'] == 'OK']
        if len(ds_results) < 2:
            continue

        print(f"🏅 ELEIÇÃO DO MELHOR — {ds}")
        print("-" * 90)
        print(f"  {'Atributo':<25} | {'Melhor Estratégia':<22} | {'Valor':>15}")
        print(f"  {'-'*25}-+-{'-'*22}-+-{'-'*15}")

        for label, key, higher_is_better in attributes:
            if higher_is_better:
                best = max(ds_results, key=lambda x: x[key])
            else:
                best = min(ds_results, key=lambda x: x[key])

            val = best[key]
            strat = best['strategy']

            # Contabilizar vitória global
            global_wins[strat] = global_wins.get(strat, 0) + 1

            print(f"  {label:<25} | {strat:<22} | {val:>15.2f}")

        print()

    # ======================================================================
    # EXIBIÇÃO: Resumo global de vitórias
    # ======================================================================
    if global_wins:
        print("🏆 RESUMO GLOBAL — Vitórias por estratégia (total de 'melhor' em todos os datasets)")
        print("=" * 60)
        sorted_wins = sorted(global_wins.items(), key=lambda x: x[1], reverse=True)
        for rank, (strat, count) in enumerate(sorted_wins, 1):
            medal = ["🥇", "🥈", "🥉", "  "][min(rank - 1, 3)]
            print(f"  {medal} {strat:<22} : {count} vitória(s)")
        print("=" * 60)
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
            for ds, (imp, strat_name) in improvement_data.items():
                writer.writerow({
                    'Dataset': ds,
                    'Estrategia': f'IMPROVEMENT ({strat_name} vs LoRA)',
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
