"""Client result lists and HDF5 persistence."""
import logging
from dataclasses import dataclass, field
from pathlib import Path

import h5py

logger = logging.getLogger(__name__)

@dataclass
class ClientMetrics:
    rs_global_acc: list = field(default_factory=list)
    rs_global_loss: list = field(default_factory=list)
    rs_ala_acc: list = field(default_factory=list)
    rs_local_acc: list = field(default_factory=list)
    rs_train_acc: list = field(default_factory=list)
    rs_post_p2p_acc: list = field(default_factory=list)
    rs_post_p2p_encounter_ids: list = field(default_factory=list)
    rs_p2p_data_sent_mb: list = field(default_factory=list)
    rs_p2p_data_sent_enc_ids: list = field(default_factory=list)

def save_results(args, rs_global_acc, rs_global_loss, rs_ala_acc, rs_local_acc, rs_train_acc, idx=0, argalgo=0, rs_post_p2p_acc=None, rs_post_p2p_encounter_ids=None, rs_p2p_data_sent_mb=None, rs_p2p_data_sent_enc_ids=None):
    b = "FedALA" if argalgo == 0 else "FedAVG"

    paca_val = args.paca if (isinstance(args.paca, (int, float)) and not isinstance(args.paca, bool) and args.paca > 0) else 0
    rank_val = args.rank if (isinstance(args.rank, (int, float)) and not isinstance(args.rank, bool) and args.rank > 0) else 8
    ts_suffix = f"_{args.timestamp}" if hasattr(args, 'timestamp') and args.timestamp else ""
    algo = f"client_{idx}_{args.dataset}_{args.strategy}_rank{rank_val}_paca{paca_val}_{b}_run{args.run_id}{ts_suffix}"

    current_dir = Path(__file__).resolve().parent
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
