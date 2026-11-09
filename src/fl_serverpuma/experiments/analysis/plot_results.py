# Responsabilidade: Seleciona experimentos, carrega resultados e coordena a geração dos
# gráficos.
#
# Principais funções: main.

"""Experiment plotting — plot_results."""

import os
import sys
from fl_serverpuma.experiments.analysis.result_loading import parse_experiment
from fl_serverpuma.experiments.analysis.accuracy_plots import plot_01_acuracia_global
from fl_serverpuma.experiments.analysis.accuracy_plots import plot_02_train_loss
from fl_serverpuma.experiments.analysis.accuracy_plots import plot_03_local_vs_treino
from fl_serverpuma.experiments.analysis.communication_plots import plot_04_banda_acumulada
from fl_serverpuma.experiments.analysis.communication_plots import plot_05_banda_por_rodada
from fl_serverpuma.experiments.analysis.communication_plots import plot_15_mb_vs_tempo
from fl_serverpuma.experiments.analysis.model_plots import plot_06_tamanho_modelo
from fl_serverpuma.experiments.analysis.model_plots import plot_07_parametros_treinaveis
from fl_serverpuma.experiments.analysis.model_plots import plot_08_paca_evolucao
from fl_serverpuma.experiments.analysis.timing_plots import plot_09_tempo_por_rodada
from fl_serverpuma.experiments.analysis.timing_plots import plot_10_tempo_treinamento_clientes
from fl_serverpuma.experiments.analysis.timing_plots import plot_11_tempo_comunicacao_clientes
from fl_serverpuma.experiments.analysis.timing_plots import plot_12_decomposicao_temporal
from fl_serverpuma.experiments.analysis.efficiency_plots import plot_13_acuracia_vs_tempo
from fl_serverpuma.experiments.analysis.efficiency_plots import plot_14_resumo_comparativo
from fl_serverpuma.experiments.analysis.plot_style import SHOW_RANK_IN_LEGEND
from fl_serverpuma.experiments.analysis.plot_style import SHOW_TITLES
from fl_serverpuma.experiments.analysis.plot_style import EXPERIMENTOS_PARA_PLOTAR
from fl_serverpuma.experiments.analysis.plot_style import DEFAULT_RESULTS_BASE

def main():
    exp_names = EXPERIMENTOS_PARA_PLOTAR

    if not exp_names:
        print("❌ Nenhum experimento definido na lista EXPERIMENTOS_PARA_PLOTAR.")
        sys.exit(1)


    print(f"{'='*60}")
    print(f"📊 Carregando {len(exp_names)} experimento(s)...")
    print(f"{'='*60}")

    experiments = []
    for name in exp_names:
        print(f"\n→ {name}")
        exp = parse_experiment(name)
        if exp and exp['server']:
            experiments.append(exp)
            sv = exp['server']
            print(f"  ✅ Servidor: {sv['num_runs']} runs, {len(sv['acc_mean'])} rounds")
            if exp['client']:
                print(f"  ✅ Clientes: {exp['client']['num_clients']} clientes")

    if not experiments:
        print("\n❌ Nenhum experimento válido encontrado.")
        sys.exit(1)

    # Diretório de saída
    if len(experiments) == 1:
        output_dir = os.path.join(experiments[0]['results_dir'], "graficos")
    else:
        output_dir = os.path.join(DEFAULT_RESULTS_BASE, "graficos_comparativos")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"📂 Salvando gráficos em: {output_dir}")
    print(f"{'='*60}")
    print(f"📝 Títulos nos gráficos: {'ATIVADOS' if SHOW_TITLES else 'DESATIVADOS'}")
    print(f"🏷️ Legenda com Rank: {'ATIVADA' if SHOW_RANK_IN_LEGEND else 'DESATIVADA'}\n")

    # Gerar todos os gráficos (em ordem sequencial 01-15)
    # --- Desempenho do Modelo ---
    plot_01_acuracia_global(experiments, output_dir)
    plot_02_train_loss(experiments, output_dir)
    plot_03_local_vs_treino(experiments, output_dir)

    # --- Custos de Comunicação ---
    plot_04_banda_acumulada(experiments, output_dir)
    plot_05_banda_por_rodada(experiments, output_dir)

    # --- Estrutura do Modelo ---
    plot_06_tamanho_modelo(experiments, output_dir)
    plot_07_parametros_treinaveis(experiments, output_dir)
    plot_08_paca_evolucao(experiments, output_dir)

    # --- Análise Temporal ---
    plot_09_tempo_por_rodada(experiments, output_dir)
    plot_10_tempo_treinamento_clientes(experiments, output_dir)
    plot_11_tempo_comunicacao_clientes(experiments, output_dir)
    plot_12_decomposicao_temporal(experiments, output_dir)

    # --- Eficiência (Acurácia vs Tempo) ---
    plot_13_acuracia_vs_tempo(experiments, output_dir)

    # --- Resumo ---
    plot_14_resumo_comparativo(experiments, output_dir)

    # --- MB vs Tempo ---
    plot_15_mb_vs_tempo(experiments, output_dir)

    print(f"\n✅ {len(experiments)} experimento(s) plotados com sucesso em: {output_dir}")

if __name__ == '__main__':
    main()
