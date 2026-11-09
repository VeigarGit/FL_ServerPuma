# Responsabilidade: Grava métricas e metadados do servidor em HDF5 com os nomes e caminhos
# originais.
#
# Principais funções: save_results.

"""Server server results."""

from fl_serverpuma.paths import SOURCE_ROOT
import logging
import h5py

logger = logging.getLogger(__name__)

def save_results(state, i: int):
    i_str = str(i)
    a = "prune" if state.args.prune == 1 else "withou_Prune"
    b = "FedALA" if getattr(state, 'argalgo', 0) == 0 else "FedAVG"

    paca_val = state.args.paca if (isinstance(state.args.paca, (int, float)) and not isinstance(state.args.paca, bool) and state.args.paca > 0) else 0
    rank_val = state.args.rank if (isinstance(state.args.rank, (int, float)) and not isinstance(state.args.rank, bool) and state.args.rank > 0) else 8
    ts_suffix = f"_{state.args.timestamp}" if hasattr(state.args, 'timestamp') and state.args.timestamp else ""
    algo = f"{state.args.dataset}_{state.args.strategy}_rank{rank_val}_paca{paca_val}_{a}_freq{state.args.prune_freq}_{b}_run{state.args.run_id}{ts_suffix}"

    current_dir = (SOURCE_ROOT / "system")
    result_path = current_dir.parent / "results" / state.args.exp_name
    result_path.mkdir(parents=True, exist_ok=True)

    # Padronizando com o prefixo 'server_' para não misturar com o client
    file_path = result_path / f"server_{algo}.h5"

    with h5py.File(file_path, 'w') as hf:
        hf.create_dataset('rs_test_acc', data=state.rs_test_acc)
        hf.create_dataset('rs_test_loss', data=state.rs_test_loss)
        hf.create_dataset('sended_model_Mb', data=state.sended_ammount)
        hf.create_dataset('Sended_without_quant', data=state.sended_withouquant)
        hf.create_dataset('Aggregated_clients', data=state.aggregated_clients)
        hf.create_dataset('Round_time', data=state.round_time)
        hf.create_dataset('Model_size_per_round_Mb', data=state.model_size_per_round)
        hf.create_dataset('Trainable_params', data=state.trainable_params_per_round)
        hf.create_dataset('rs_client_paca', data=state.rs_client_paca)
        hf.create_dataset('rs_client_training_time', data=state.rs_client_training_time)
        hf.create_dataset('rs_client_comm_time', data=state.rs_client_comm_time)
        hf.create_dataset('rs_client_eval_time', data=state.rs_client_eval_time)
        hf.create_dataset('rs_server_processing_time', data=state.rs_server_processing_time)
        hf.create_dataset('server_pruning_time', data=state.server_pruning_time)
        hf.create_dataset('received_from_clients_Mb', data=state.received_ammount)
        hf.create_dataset('total_transmitted_per_round_Mb', data=state.total_transmitted_per_round)
