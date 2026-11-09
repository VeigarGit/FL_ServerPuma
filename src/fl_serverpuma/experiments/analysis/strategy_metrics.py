# Responsabilidade: Carrega resultados por execução, detecta convergência e calcula
# estatísticas e melhorias entre estratégias.
#
# Principais funções: detect_convergence_round, analyze_single_run, analyze_experiment,
# compute_improvement.

"""Strategy comparison — strategy_metrics."""

import os
import glob
import numpy as np
import h5py
from fl_serverpuma.experiments.analysis.strategy_settings import CONVERGENCE_THRESHOLD
from fl_serverpuma.experiments.analysis.strategy_settings import CONVERGENCE_WINDOW

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
