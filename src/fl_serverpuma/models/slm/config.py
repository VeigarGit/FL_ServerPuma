# Responsabilidade: Lê YAML/CLI, resolve modos de execução e monta configurações e caminhos de
# saída. Família SLM.
#
# Principais funções: get_device, parse_args, load_config, resolve_run_modes,
# build_run_config, build_output_path.

"""SLM model support — config."""

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
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SLM for classification.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to YAML config.")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    lora_config = config["model"].get("lora", {})
    lora_mode = lora_config.get("mode", "with_lora")
    lora_config["mode"] = lora_mode
    return config


def resolve_run_modes(config):
    lora_mode = config["model"]["lora"]["mode"]
    return ["with_lora", "without_lora"] if lora_mode == "both" else [lora_mode]


def build_run_config(config, run_mode):
    run_config = yaml.safe_load(yaml.safe_dump(config))
    run_config["model"]["lora"]["mode"] = run_mode
    return run_config


def build_output_path(base_output_path, run_mode, multiple_runs):
    output_path = Path(base_output_path)
    if not multiple_runs: return output_path
    return output_path.with_name(f"{output_path.stem}_{run_mode}{output_path.suffix}")
