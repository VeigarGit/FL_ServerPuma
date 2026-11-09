# Responsabilidade: Grava métricas locais/globais e opcionais de P2P em HDF5, preservando
# nomes de arquivos e atributos.
#
# Principais funções: save_results.

"""Client results."""

from fl_serverpuma.paths import SOURCE_ROOT
import logging
import h5py

logger = logging.getLogger(__name__)

def save_results(args, rs_global_acc, rs_global_loss, rs_ala_acc, rs_local_acc, rs_train_acc, idx=0, argalgo=0, rs_post_p2p_acc=None, rs_post_p2p_encounter_ids=None, rs_p2p_data_sent_mb=None, rs_p2p_data_sent_enc_ids=None):
    b = "FedALA" if argalgo == 0 else "FedAVG"

    paca_val = args.paca if (isinstance(args.paca, (int, float)) and not isinstance(args.paca, bool) and args.paca > 0) else 0
    rank_val = args.rank if (isinstance(args.rank, (int, float)) and not isinstance(args.rank, bool) and args.rank > 0) else 8
    ts_suffix = f"_{args.timestamp}" if hasattr(args, 'timestamp') and args.timestamp else ""
    algo = f"client_{idx}_{args.dataset}_{args.strategy}_rank{rank_val}_paca{paca_val}_{b}_run{args.run_id}{ts_suffix}"

    current_dir = (SOURCE_ROOT / "system")
    result_path = current_dir.parent / "results" / args.exp_name
    result_path.mkdir(parents=True, exist_ok=True)

    file_path = result_path / f"{algo}.h5"
    logger.info(f"Salvando resultados em: {file_path}")

    with h5py.File(file_path, 'w') as hf:
        hf.create_dataset('rs_global_acc', data=rs_global_acc)
        hf.create_dataset('rs_global_loss', data=rs_global_loss)
        if rs_ala_acc is not None and len(rs_ala_acc) > 0:
            hf.create_dataset('rs_ala_acc', data=rs_ala_acc)
        hf.create_dataset('rs_local_acc', data=rs_local_acc)
        hf.create_dataset('rs_train_acc', data=rs_train_acc)
        if rs_post_p2p_acc is not None and len(rs_post_p2p_acc) > 0:
            hf.create_dataset('rs_post_p2p_acc', data=rs_post_p2p_acc)
        if rs_post_p2p_encounter_ids is not None and len(rs_post_p2p_encounter_ids) > 0:
            hf.create_dataset('rs_post_p2p_encounter_ids', data=rs_post_p2p_encounter_ids)
        if rs_p2p_data_sent_mb is not None and len(rs_p2p_data_sent_mb) > 0:
            hf.create_dataset('rs_p2p_data_sent_mb', data=rs_p2p_data_sent_mb)
        if rs_p2p_data_sent_enc_ids is not None and len(rs_p2p_data_sent_enc_ids) > 0:
            hf.create_dataset('rs_p2p_data_sent_enc_ids', data=rs_p2p_data_sent_enc_ids)
        hf.attrs['paca_used'] = paca_val
