# Responsabilidade: Interpreta nomes de experimentos, lê HDF5 de servidor/clientes e prepara
# séries de comunicação e tempos.
#
# Principais funções: parse_experiment.

"""Experiment plotting — result_loading."""

import os
import glob
import numpy as np
import h5py
import re
from fl_serverpuma.experiments.analysis.plot_style import SHOW_RANK_IN_LEGEND
from fl_serverpuma.experiments.analysis.plot_style import DEFAULT_RESULTS_BASE

def parse_experiment(exp_name, results_base=None):
    """Analisa um diretório de experimento e retorna seus metadados + dados."""
    if results_base is None:
        results_base = DEFAULT_RESULTS_BASE
    results_dir = os.path.join(results_base, exp_name)
    if not os.path.isdir(results_dir):
        print(f"⚠️ Diretório não encontrado: {results_dir}")
        return None

    server_files = sorted(glob.glob(os.path.join(results_dir, "server_*.h5")))
    if not server_files:
        print(f"⚠️ Nenhum arquivo de servidor em: {results_dir}")
        return None

    # Extrair metadados do nome do arquivo
    sample = os.path.basename(server_files[0])

    # Exemplo: server_OxfordPets_sora_with_schedule_rank8_paca12_prune_freq1_FedAVG_run1.h5
    # Ou: server_OxfordPets_lora_rank8_paca12_withou_Prune_freq1_FedAVG_run1_20260817_142300.h5
    pattern = r"server_(?P<dataset>[^_]+)_(?P<strategy>.+)_rank(?P<rank>\d+)_paca(?P<paca>\d+)_(?P<prune>.*?)_freq(?P<freq>\d+)_(?P<algo>[^_]+)_run\d+(?:_\d{8}_\d{6})?\.h5"

    match = re.match(pattern, sample)
    if not match:
        print(f"⚠️ Não foi possível extrair metadados do arquivo (padrão regex falhou): {sample}")
        return None

    dataset   = match.group('dataset')
    strategy  = match.group('strategy')
    rank      = int(match.group('rank'))
    paca      = int(match.group('paca'))
    prune_str = match.group('prune')
    freq      = int(match.group('freq'))
    algo      = match.group('algo')

    # Label legível para os gráficos
    if SHOW_RANK_IN_LEGEND:
        is_adap_rank = any(term in exp_name.lower() for term in ["adap_rank", "adaptrank", "adaptive_rank"])
        rank_str = f" r={rank} (AdaRank)" if is_adap_rank else f" r={rank}"
    else:
        rank_str = ""

    if "adalora" in exp_name.lower():
        label = f"AdaLoRA{rank_str}"
    elif "sora" in exp_name.lower():
        if "adaptpaca" in exp_name.lower():
            label = f"PUMA-GT{rank_str}"
        else:
            label = f"Static SoRA{rank_str}"
    elif "lora" in exp_name.lower():
        label = f"LoRA{rank_str}"
    else:
        label = f"{strategy.upper()}{rank_str} p={paca} ({algo})"
        if "withou" not in prune_str.lower():
            label += " +prune"

    # Carregar dados do servidor
    server_data = _load_server_h5(server_files)

    # Carregar dados dos clientes
    client_pattern = os.path.join(results_dir, f"client_*_{dataset}_{strategy}_rank{rank}_paca*_{algo}_run*.h5")
    client_files = sorted(glob.glob(client_pattern))
    client_data = _load_client_h5(client_files) if client_files else None

    return {
        'exp_name':    exp_name,
        'label':       label,
        'dataset':     dataset,
        'strategy':    strategy,
        'rank':        rank,
        'paca':        paca,
        'prune':       prune_str,
        'freq':        freq,
        'algo':        algo,
        'server':      server_data,
        'client':      client_data,
        'results_dir': results_dir,
    }


def _load_server_h5(files):
    acc, loss, mb, mb_bruto, time_, params, size_, paca = [], [], [], [], [], [], [], []
    client_training_time = []
    client_comm_time = []
    server_pruning_time = []
    client_eval_time = []
    server_processing_time = []
    for f in files:
        with h5py.File(f, 'r') as hf:
            acc.append(np.array(hf['rs_test_acc']))
            loss_arr = np.array(hf['rs_train_loss']) if 'rs_train_loss' in hf else (
                np.array(hf['rs_test_loss']) if 'rs_test_loss' in hf else np.zeros_like(np.array(hf['rs_test_acc']))
            )
            loss.append(loss_arr)
            mb.append(np.array(hf['sended_model_Mb']))
            mb_bruto.append(np.array(hf['Sended_without_quant']))
            time_.append(np.array(hf['Round_time']))
            params.append(np.array(hf.get('Trainable_params', np.zeros_like(np.array(hf['Round_time'])))))
            size_.append(np.array(hf.get('Model_size_per_round_Mb', np.zeros_like(np.array(hf['Round_time'])))))

            if 'rs_client_paca' in hf:
                paca.append(np.array(hf['rs_client_paca']))
            else:
                paca.append(None)

            if 'rs_client_training_time' in hf:
                client_training_time.append(np.array(hf['rs_client_training_time']))
            else:
                client_training_time.append(None)

            if 'rs_client_comm_time' in hf:
                client_comm_time.append(np.array(hf['rs_client_comm_time']))
            else:
                client_comm_time.append(None)

            if 'server_pruning_time' in hf:
                server_pruning_time.append(np.array(hf['server_pruning_time']))
            else:
                server_pruning_time.append(None)

            if 'rs_client_eval_time' in hf:
                client_eval_time.append(np.array(hf['rs_client_eval_time']))
            else:
                client_eval_time.append(None)

            if 'rs_server_processing_time' in hf:
                server_processing_time.append(np.array(hf['rs_server_processing_time']))
            else:
                server_processing_time.append(None)

    paca_mean = None
    if all(p is not None for p in paca):
        paca_mean = np.mean(paca, axis=0)

    valid_client_training_time = [t for t in client_training_time if t is not None]
    client_training_time_mean = None
    if valid_client_training_time:
        min_len = min(len(t) for t in valid_client_training_time)
        valid_client_training_time = [t[:min_len] for t in valid_client_training_time]
        client_training_time_mean = np.mean(valid_client_training_time, axis=0)

    valid_client_comm_time = [t for t in client_comm_time if t is not None]
    client_comm_time_mean = None
    if valid_client_comm_time:
        min_len = min(len(t) for t in valid_client_comm_time)
        valid_client_comm_time = [t[:min_len] for t in valid_client_comm_time]
        client_comm_time_mean = np.mean(valid_client_comm_time, axis=0)

    valid_server_pruning_time = [t for t in server_pruning_time if t is not None]
    server_pruning_time_mean = None
    if valid_server_pruning_time:
        min_len = min(len(t) for t in valid_server_pruning_time)
        valid_server_pruning_time = [t[:min_len] for t in valid_server_pruning_time]
        server_pruning_time_mean = np.mean(valid_server_pruning_time, axis=0)

    valid_client_eval_time = [t for t in client_eval_time if t is not None]
    client_eval_time_mean = None
    if valid_client_eval_time:
        min_len = min(len(t) for t in valid_client_eval_time)
        valid_client_eval_time = [t[:min_len] for t in valid_client_eval_time]
        client_eval_time_mean = np.mean(valid_client_eval_time, axis=0)

    valid_server_processing_time = [t for t in server_processing_time if t is not None]
    server_processing_time_mean = None
    if valid_server_processing_time:
        min_len = min(len(t) for t in valid_server_processing_time)
        valid_server_processing_time = [t[:min_len] for t in valid_server_processing_time]
        server_processing_time_mean = np.mean(valid_server_processing_time, axis=0)

    return {
        'acc_mean': np.mean(acc, axis=0), 'acc_std': np.std(acc, axis=0),
        'loss_mean': np.mean(loss, axis=0), 'loss_std': np.std(loss, axis=0),
        'mb_mean': np.mean(mb, axis=0), 'mb_std': np.std(mb, axis=0),
        'mb_bruto_mean': np.mean(mb_bruto, axis=0),
        'time_mean': np.mean(time_, axis=0), 'time_std': np.std(time_, axis=0),
        'params_mean': np.mean(params, axis=0),
        'size_mb_mean': np.mean(size_, axis=0), 'size_mb_std': np.std(size_, axis=0),
        'paca_mean': paca_mean,
        'client_training_time_mean': client_training_time_mean,
        'client_comm_time_mean': client_comm_time_mean,
        'server_pruning_time_mean': server_pruning_time_mean,
        'client_eval_time_mean': client_eval_time_mean,
        'server_processing_time_mean': server_processing_time_mean,
        'num_runs': len(files),
    }


def _load_client_h5(files):
    g_acc, g_loss, l_acc, t_acc = [], [], [], []
    clients_set = set()
    for f in files:
        clients_set.add(os.path.basename(f).split('_')[1])
        with h5py.File(f, 'r') as hf:
            g_acc.append(np.array(hf['rs_global_acc']))
            g_loss.append(np.array(hf['rs_global_loss']))
            l_acc.append(np.array(hf['rs_local_acc']))
            t_acc.append(np.array(hf['rs_train_acc']))

    return {
        'global_acc_mean': np.mean(g_acc, axis=0), 'global_acc_std': np.std(g_acc, axis=0),
        'global_loss_mean': np.mean(g_loss, axis=0), 'global_loss_std': np.std(g_loss, axis=0),
        'local_acc_mean': np.mean(l_acc, axis=0), 'local_acc_std': np.std(l_acc, axis=0),
        'train_acc_mean': np.mean(t_acc, axis=0), 'train_acc_std': np.std(t_acc, axis=0),
        'num_clients': len(clients_set),
    }


def _compute_mb_per_round(d):
    """Calcula MB trafegados por rodada individual (ida + volta) a partir do acumulado."""
    num_rounds = len(d['time_mean'])
    total_envios = len(d['mb_mean'])
    if num_rounds == 0 or total_envios == 0:
        return None
    clients_per_round = max(1, total_envios // num_rounds)
    indices = [min(total_envios - 1, r * clients_per_round) for r in range(1, num_rounds + 1)]
    mb_accumulated = [d['mb_mean'][idx] for idx in indices]
    mb_discrete = []
    for r in range(len(mb_accumulated)):
        if r == 0:
            mb_round = mb_accumulated[r]
        else:
            mb_round = max(0, mb_accumulated[r] - mb_accumulated[r-1])
        # Multiplicamos por 2: o .h5 salva apenas envio servidor→cliente,
        # mas o cliente devolve os mesmos tensores (upload ≈ download).
        mb_discrete.append(mb_round * 2.0)
    return mb_discrete, mb_accumulated, indices


def _get_train_time_per_round(d):
    """Extrai o tempo médio de treinamento por rodada."""
    t_train = d.get('client_training_time_mean')
    if t_train is not None:
        if t_train.ndim > 1:
            return np.mean(t_train, axis=1)
        return t_train
    # Fallback: estima como 50% do tempo total da rodada
    return d['time_mean'] * 0.5
