# Responsabilidade: Define os argumentos e valores padrão da CLI do cliente.
#
# Principais funções: parse_args.

"""Client cli."""

import argparse
import logging

logger = logging.getLogger(__name__)

def parse_args():
    # Colocar um argumento -t para servir como um for 
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='Cifar100')
    parser.add_argument('--client-idx', type=int, default=0)
    parser.add_argument('--in-features', type=int, default=3)
    parser.add_argument('--num-classes', type=int, default=100)
    parser.add_argument('--dim', type=int, default=1600)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--random-client', action='store_true')
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument("--ala", type=int, default=0)
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument('--config', type=str, default="lora_clip/train_config.yml")
    parser.add_argument('--experiments', type=int, default=1)
    parser.add_argument('--run-id', type=int, default=1)
    parser.add_argument('--exp-name', type=str, default='default_exp', help='Nome da sessão')
    parser.add_argument('--timestamp', type=str, default='', help='Timestamp para diferenciar rodadas de simulação')
    parser.add_argument('--strategy', type=str, default='lora', choices=['lora', 'adalora', 'sora_with_schedule', 'sora_no_schedule'])
    parser.add_argument('--rank', type=int, default=8, help='Rank para o SoRA/LoRA') 
    parser.add_argument('--paca', type=int, default=12, help='Número de camadas para a estratégia PaCA')

    # --- Parâmetros de Sparsidade (SoRA) ---
    parser.add_argument('--sparse-lr', type=float, default=None, help='Taxa de aprendizado para a poda (Softshrink)')
    parser.add_argument('--max-lambda', type=float, default=None, help='Força máxima da penalidade L1 (max_lambda)')

    # --- PaCA Heterogêneo ---
    parser.add_argument('--random-paca', action='store_true', help='Sorteia um valor de PaCA aleatório para este cliente')
    parser.add_argument('--paca-min', type=int, default=1, help='Valor mínimo de PaCA no sorteio aleatório')
    parser.add_argument('--paca-max', type=int, default=12, help='Valor máximo de PaCA no sorteio aleatório')
    parser.add_argument('--paca-list', type=str, default=None, help='Lista de PaCAs pré-definidos por cliente (ex: "4,6,8,12"). O cliente usa o valor na posição client_idx %% len(lista)')
    parser.add_argument('--prune', type=int, default=1, choices=[0, 1], help='Habilitar pruning adaptativo (0=sim, 1=não, default: 1)')
    parser.add_argument('--mode', type=str, default='centralized', choices=['centralized', 'decentralized'])
    parser.add_argument('--max-epochs', type=int, default=0, help='Limite de epocas no modo descentralizado (0 = infinito)')
    parser.add_argument('--delta-coding', action='store_true', help='Ativa LHDQ (Low Huffman-coded Delta Quantization) em vez de int8')

    # --- Otimização de Avaliação ---
    parser.add_argument('--skip-post-eval', action='store_true', help='Pula todas as avaliações pós-treino no cliente (local_test_acc e train_acc) para máxima velocidade, mantendo apenas Global Model Test Acc')
    parser.add_argument('--skip-train-eval', action='store_true', help='Pula apenas a avaliação demorada no conjunto de treino pós-treino')

    return parser.parse_args()
