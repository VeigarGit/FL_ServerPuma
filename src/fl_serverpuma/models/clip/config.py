# Responsabilidade: Lê YAML/CLI, resolve modos de execução e monta configurações e caminhos de
# saída. Família CLIP.
#
# Principais funções: get_device, parse_args, load_config, resolve_run_modes,
# build_run_config, build_output_path.

"""CLIP model support — config."""

import argparse
from pathlib import Path
import torch
import yaml

DEFAULT_CONFIG_PATH = Path("train_config.yml")

VALID_LORA_MODES = {
    "with_lora",
    "with_adalora",
    "without_lora",
    "both",
    "with_sora_no_schedule",
    "with_sora_schedule",
}

SORA_MODES = {"with_sora_no_schedule", "with_sora_schedule"}


def get_device():
    """Detecção de Hardware. Identifica se o treino ocorrerá em GPU (CUDA) ou CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    """Captura o caminho do arquivo de configuração YAML via linha de comando."""
    parser = argparse.ArgumentParser(description="Fine-tune CLIP vision encoder for classification.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def load_config(config_path):
    """Gestão de Configurações. Carrega, valida chaves obrigatórias e normaliza modos de LoRA."""
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("The YAML config must define a mapping at the top level.")

    required_keys = {"dataset", "model", "training", "optimizer", "scheduler", "output"}
    missing_keys = required_keys - config.keys()
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise KeyError(f"Missing required config sections: {missing}")

    lora_config = config["model"].get("lora", {})
    lora_mode = lora_config.get("mode")
    if lora_mode is None:
        if "enabled" in lora_config:
            lora_mode = "with_lora" if lora_config["enabled"] else "without_lora"
        else:
            lora_mode = "with_lora"
        lora_config["mode"] = lora_mode

    if lora_mode not in VALID_LORA_MODES:
        expected = ", ".join(sorted(VALID_LORA_MODES))
        raise ValueError(f"Invalid model.lora.mode: {lora_mode!r}. Expected one of: {expected}")

    return config


def resolve_run_modes(config):
    """Lógica de Experimentos. Determina se haverá treinos comparativos (com vs sem LoRA)."""
    lora_mode = config["model"]["lora"]["mode"]
    if lora_mode == "both":
        return ["with_lora", "without_lora"]
    return [lora_mode]


def build_run_config(config, run_mode):
    """Gera um clone da configuração global ajustado para um modo de execução específico."""
    run_config = yaml.safe_load(yaml.safe_dump(config))
    run_config["model"]["lora"]["mode"] = run_mode
    return run_config


def build_output_path(base_output_path, run_mode, multiple_runs):
    """Define o caminho final do arquivo de pesos baseado no experimento atual."""
    output_path = Path(base_output_path)
    if not multiple_runs:
        return output_path

    return output_path.with_name(f"{output_path.stem}_{run_mode}{output_path.suffix}")
