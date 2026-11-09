# Responsabilidade: Inicializa o modelo global e guarda configuração, conexões, locks, dados e
# históricos de uma execução em ServerState.
#
# Principais funções/classes: ServerState.

"""Model, synchronization primitives and mutable history for one server run."""

import threading
import logging
import torch
from fl_serverpuma.models.clip.config import load_config
from fl_serverpuma.models.clip.config import resolve_run_modes
from fl_serverpuma.models.clip.config import build_run_config
from fl_serverpuma.models.clip.model import build_model
from fl_serverpuma.algorithms.fl_math import resize_model_to_pruned
from fl_serverpuma.models.slm.config import load_config as load_config_slm
from fl_serverpuma.models.slm.config import resolve_run_modes as resolve_run_modes_slm
from fl_serverpuma.models.slm.config import build_run_config as build_run_config_slm
from fl_serverpuma.models.slm.model import build_model as build_model_slm
from fl_serverpuma.models.cnn import SimpleModel
from fl_serverpuma.models.complexity import LayerComplexityCalculator
from fl_serverpuma.data.server_data import load_test_data

logger = logging.getLogger(__name__)

class ServerState:
    def __init__(self, args):
        self.args = args
        self.config = None
        self.slm_collator = None
        self.test_loaders_cache = {}

        # Uso do Structural Pattern Matching (Substituindo a cadeia de IFs)
        match args.model:
            case 'cnn':
                match args.dataset:
                    case 'MNIST':
                        self.global_model = SimpleModel(in_features=1, num_classes=10, dim=1024)
                        self.input_size = (1, 1, 28, 28)
                    case 'Cifar10':
                        self.global_model = SimpleModel(in_features=args.in_features, num_classes=10, dim=args.dim)
                        self.input_size = (1, 3, 32, 32)
                    case 'Cifar100':
                        self.global_model = SimpleModel(in_features=args.in_features, num_classes=args.num_classes, dim=args.dim)
                        self.input_size = (1, 3, 32, 32)
                    case _:
                        logger.error(f"Dataset não reconhecido: {args.dataset}")
                        raise ValueError(f"Dataset inválido: {args.dataset}")
            case 'clip' | 'slm':
                if args.model == 'clip':
                    config = load_config(args.config)
                else:
                    config = load_config_slm(args.config)
                self.config = config

                # --- INÍCIO DA SOBRESCRITA DA ESTRATÉGIA ---
                # Pega a flag do terminal (--strategy) e força o modo no dicionário do YAML
                if args.strategy == 'lora':
                    print("Linha 82 =========================================")
                    config["model"]["lora"]["mode"] = "with_lora"
                elif args.strategy == 'sora_no_schedule':
                    print("Linha 85 =========================================")
                    config["model"]["lora"]["mode"] = "with_sora_no_schedule" 
                elif args.strategy == 'sora_with_schedule':
                    print("Linha 88 =========================================")
                    config["model"]["lora"]["mode"] = "with_sora_schedule" 
                elif args.strategy == 'full_ft':
                    print("Linha 91 =========================================")
                    config["model"]["lora"]["mode"] = "without_lora"
                elif args.strategy == 'adalora':
                    print("Linha AdaLoRA =========================================")
                    config["model"]["lora"]["mode"] = "with_adalora"

                if "paca" not in config["model"]:
                    config["model"]["paca"] = {}

                if "lora" in config["model"]:
                    config["model"]["lora"]["r"] = args.rank

                if args.paca is not None and args.paca > 0:
                    config["model"]["paca"]["enabled"] = True
                    config["model"]["paca"]["upper_layers"] = args.paca
                else:
                    config["model"]["paca"]["enabled"] = False
                    config["model"]["paca"]["upper_layers"] = None

                if args.model == 'clip':
                    run_mode = resolve_run_modes(config)[0]
                    run_config = build_run_config(config, run_mode=run_mode)

                    self.global_model = build_model(
                        config=run_config, 
                        num_classes=args.num_classes, 
                        device=args.device
                    )
                    self.input_size = (1, 3, 224, 224)
                else:
                    run_mode = resolve_run_modes_slm(config)[0]
                    run_config = build_run_config_slm(config, run_mode=run_mode)

                    self.global_model = build_model_slm(
                        config=run_config, 
                        num_classes=args.num_classes, 
                        device=args.device
                    )
                    self.input_size = None # SLM usa o processor

                    try:
                        from fl_serverpuma.models.slm.data import CustomCollator
                        from transformers import AutoProcessor
                        model_name = self.config.get("model", {}).get("name", "Qwen/Qwen3-VL-2B-Instruct") if self.config else "Qwen/Qwen3-VL-2B-Instruct"
                        processor = AutoProcessor.from_pretrained(model_name)
                        label_to_idx = {i: i for i in range(1000)}
                        label_to_idx.update({str(i): i for i in range(1000)})
                        self.slm_collator = CustomCollator(processor, label_to_idx)
                    except Exception as e:
                        logger.warning(f"Não foi possível pré-carregar CustomCollator para SLM na inicialização: {e}")


        if args.load_model:
            logger.info(f"Carregando pesos salvos de: {args.load_model}")
            state_dict = torch.load(args.load_model, map_location=args.device)

            # 1. Redimensiona a arquitetura do modelo em memória para o tamanho podado
            self.global_model = resize_model_to_pruned(self.global_model, state_dict)

            # 2. Agora os formatos coincidem, podemos carregar os pesos com segurança
            self.global_model.load_state_dict(state_dict, strict=False)
            self.global_state = self.global_model.state_dict()


        self.global_model = self.global_model.to(args.device)
        self.rs_test_acc = []
        self.rs_test_loss = []
        self.global_state = self.global_model.state_dict()
        self.lock = threading.Lock()
        self.client_data = {}
        self.client_connections = []
        self.client_addresses = []
        self.size_fc = 25
        self.client_idx = []
        self.clients_info = {}
        self.prune = args.prune
        self.sended_ammount = [0]
        self.sended_withouquant = [0]
        self.received_ammount = [0]
        self.total_transmitted_per_round = []
        self.rs_client_paca = []
        self.rs_client_training_time = []
        self.rs_client_comm_time = []
        self.rs_client_eval_time = []
        self.rs_server_processing_time = []
        self.server_pruning_time = []
        self.aggregated_clients = []
        self.round_time = []
        self.model_size_per_round = []
        self.trainable_params_per_round = []
        self.bit = []
        self.complexity_calculated = False

        # Performance tracking variables
        self.time_threshold = 0.0
        self.mean_training_time = 0.0

        # LHDQ: cache do estado anterior por cliente (para delta coding)
        self.previous_sent_state = {}   # client_id → state_dict enviado na rodada anterior
        self.previous_recv_state = {}   # client_id → state_dict recebido na rodada anterior

        if self.args.model == 'slm':
            alpha_list, beta_list = [0], [0]
            logger.info("Cálculo de complexidade ignorado para SLM (forward multimodal complexo).")
        else:
            calculator = LayerComplexityCalculator(self.global_model, self.input_size)
            alpha_list, beta_list = calculator.calculate_alpha_beta()

        self.alpha_list = alpha_list
        self.beta_list = beta_list
        self.complexity_calculated = True

        logger.info(f"Calculado: {len(alpha_list)} camadas")
        logger.info(f"Total α: {sum(alpha_list):,} parâmetros")
        logger.info(f"Total β: {sum(beta_list):,} FLOPs")

        # Detectar total de encoder layers dinamicamente (para PaCA)
        if hasattr(self.global_model, 'vision_model') and hasattr(self.global_model.vision_model, 'encoder'):
            self.total_encoder_layers = len(self.global_model.vision_model.encoder.layers)
        elif self.args.model == 'slm':
            from fl_serverpuma.models.slm.adaptation import _get_encoder_layers
            layers = _get_encoder_layers(self.global_model)
            self.total_encoder_layers = len(layers) if layers else self.args.paca
        else:
            self.total_encoder_layers = self.args.paca  # fallback
        logger.info(f"Total encoder layers detectado: {self.total_encoder_layers}")

        # Dicionário para armazenar máscaras por cliente
        self.client_masks = {}

        self.test_loader = load_test_data(self, args.dataset, args.test_client_idx, args.batch_size)
        if self.test_loader is not None:
            self.test_loaders_cache[args.test_client_idx if args.test_client_idx is not None else 0] = self.test_loader
        else:
            logger.warning("Could not load test data. Evaluation will be skipped.")
