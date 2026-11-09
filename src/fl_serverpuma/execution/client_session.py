# Responsabilidade: Agrupa modelo, dados, configuração e listas de métricas de um cliente, sem
# copiar seus objetos.
#
# Principais funções/classes: ClientSession.

from dataclasses import dataclass, field
from typing import Any

@dataclass
class ClientSession:
    """Objects and metric lists shared by one client execution."""
    args: Any
    device: Any
    model: Any
    train_loader: Any
    test_loader: Any
    ala: Any
    run_config: Any
    rs_global_acc: list = field(default_factory=list)
    rs_global_loss: list = field(default_factory=list)
    rs_ala_acc: list = field(default_factory=list)
    rs_local_acc: list = field(default_factory=list)
    rs_train_acc: list = field(default_factory=list)
    rs_post_p2p_acc: list = field(default_factory=list)
    rs_post_p2p_encounter_ids: list = field(default_factory=list)
    rs_p2p_data_sent_mb: list = field(default_factory=list)
    rs_p2p_data_sent_enc_ids: list = field(default_factory=list)
