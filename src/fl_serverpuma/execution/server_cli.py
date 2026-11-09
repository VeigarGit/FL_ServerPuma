# Responsabilidade: Define os argumentos e valores padrão da CLI do servidor.
#
# Principais funções: parse_args.

"""Server command-line arguments."""

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning Server')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--clients-per-round', type=int, default=8)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='Cifar10')
    parser.add_argument('--test-client-idx', type=int, default=0)
    parser.add_argument('--in-features', type=int, default=3)
    parser.add_argument('--num-classes', type=int, default=100)
    parser.add_argument('--dim', type=int, default=1600)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-clients', type=int, default=10)
    parser.add_argument('--prune', type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument('--pm', type=str, default='OPALA', choices=['OPALA', 'SNIP', 'NISP'])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('--experiments', type=int, default=1)
    parser.add_argument('--model', type=str, default="cnn", choices=["cnn", "clip", "slm"])
    parser.add_argument('--config', type=str, default="lora_clip/train_config.yml")
    parser.add_argument('--prune-freq', type=int, default=0)

    parser.add_argument('--save-model', type=str, default=None, help='Caminho para salvar os pesos treinados (.pt)')
    parser.add_argument('--load-model', type=str, default=None, help='Caminho para carregar pesos pré-treinados (.pt)')

    parser.add_argument('--run-id', type=int, default=1, help='ID da simulação atual')
    parser.add_argument('--exp-name', type=str, default='default_exp', help='Nome da sessão')
    parser.add_argument('--timestamp', type=str, default='', help='Timestamp para diferenciar rodadas de simulação')
    parser.add_argument('--strategy', type=str, default='lora', choices=['lora', 'adalora', 'sora_with_schedule', 'sora_no_schedule'])
    parser.add_argument('--rank', type=int, default=8, help='Rank para o SoRA/LoRA')
    parser.add_argument('--paca', type=int, default=12, help='Número de camadas do modelo base para injetar adaptadores (PaCA)')
    parser.add_argument('--adaptive-paca', action='store_true', help='Ativa PaCA dinâmico: servidor ajusta automaticamente o número de camadas por cliente baseado em latência')
    parser.add_argument('--adaptive-paca-min', type=int, default=2, help='Número mínimo de camadas que o PaCA adaptativo pode alcançar')
    parser.add_argument('--adaptive-rank', action='store_true', help='Ativa rank adaptativo: servidor ajusta automaticamente o rank por cliente baseado em latência')
    parser.add_argument('--adaptive-rank-min', type=int, default=2, help='Rank mínimo para o rank adaptativo')
    parser.add_argument('--adaptive-rank-max', type=int, default=8, help='Rank máximo para o rank adaptativo')
    parser.add_argument('--allow-paca-upscale', action='store_true', help='Permite a subida de PaCA (por padrão apenas desce)')
    parser.add_argument('--allow-rank-upscale', action='store_true', help='Permite a subida de Rank (por padrão apenas desce)')
    parser.add_argument('--min-rank', type=int, default=2, help='Rank mínimo permitido por módulo SoRA durante poda iterativa')
    parser.add_argument('--sora-prune', action='store_true', help='Ativa poda iterativa de rank do SoRA (independente do --prune CNN)')
    parser.add_argument('--delta-coding', action='store_true', help='Ativa LHDQ (Low Huffman-coded Delta Quantization) em vez de int8')
    return parser.parse_args()
