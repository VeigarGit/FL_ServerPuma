import os
import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt

# ==========================================
# Configurações das Simulações
# ==========================================
DATASET = "OxfordPets"
RESULTS_BASE = "../../results"
OUTPUT_DIR = "../../results/plots_comparisons"

# Cria a pasta de saída para os gráficos organizados
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Definição das pastas geradas nos seus experimentos
runs = [
    {
        "name": "LoRA (PaCA=12, Sem Pruning)",
        "folder": "lora_paca12_sempruning_semALA_20260710_124548",
        "color": "#1f77b4", # Azul
        "marker": "o",
        "ls": "--"
    },
    {
        "name": "SoRA (PaCA=12, Com Pruning)",
        "folder": "sora_paca12_compruning_semALA_20260710_125527",
        "color": "#ff7f0e", # Laranja
        "marker": "s",
        "ls": "-."
    },
    {
        "name": "SoRA (PaCA Adaptativo, Com Pruning)",
        "folder": "sora_paca_adaptativo_compruning_semALA_20260710_175212",
        "color": "#2ca02c", # Verde
        "marker": "*",
        "ls": "-"
    },
    {
        "name": "SoRA (PaCA Adaptativo, Sem Pruning)",
        "folder": "sora_paca_adaptativo_sempruning_semALA_20260710_133049",
        "color": "#d62728", # Vermelho
        "marker": "^",
        "ls": ":"
    }
]

# ==========================================
# Carregamento dos Dados
# ==========================================
def load_results(folder_name):
    # Procura o arquivo .h5 gerado pelo servidor dentro da respectiva pasta
    pattern = os.path.join(RESULTS_BASE, folder_name, f"server_{DATASET}_*.h5")
    files = glob.glob(pattern)
    
    if not files:
        print(f"⚠️ Nenhum arquivo encontrado em {pattern}")
        return None
    
    f = files[0]
    print(f"✅ Lendo: {f}")
    
    with h5py.File(f, 'r') as hf:
        acc = np.array(hf['rs_test_acc'])
        
        # O sended_model_Mb salva o total acumulado a CADA CLIENTE. Para plotar por rodada,
        # pegamos o último valor de banda registrado em cada rodada.
        mb_raw = np.array(hf['sended_model_Mb'])
        if len(mb_raw) > 0 and mb_raw[0] == 0:
            mb_raw = mb_raw[1:]
        clientes_por_rodada = len(mb_raw) // len(acc) if len(acc) > 0 else 1
        mb_acumulada = mb_raw[clientes_por_rodada - 1::clientes_por_rodada] if len(mb_raw) > 0 else mb_raw
        
        mb_bruto_raw = np.array(hf['Sended_without_quant'])
        if len(mb_bruto_raw) > 0 and mb_bruto_raw[0] == 0:
            mb_bruto_raw = mb_bruto_raw[1:]
        mb_bruto_acumulada = mb_bruto_raw[clientes_por_rodada - 1::clientes_por_rodada] if len(mb_bruto_raw) > 0 else mb_bruto_raw
        
        data = {
            'acc': acc,
            'loss': np.array(hf['rs_train_loss']),
            'mb': mb_acumulada,
            'mb_bruto': mb_bruto_acumulada,
            'time': np.array(hf['Round_time'])
        }
        data['time_acum'] = np.cumsum(data['time'])
        
        if 'Trainable_params' in hf:
            data['params'] = np.array(hf['Trainable_params'])
        if 'Model_size_per_round_Mb' in hf:
            data['size_mb'] = np.array(hf['Model_size_per_round_Mb'])
            
        if 'rs_client_paca' in hf:
            data['paca'] = np.array(hf['rs_client_paca'])
            
    return data

# Extrai os dados de cada configuração
for run in runs:
    run['data'] = load_results(run['folder'])

# ==========================================
# Funções de Plotagem Automática
# ==========================================
def plot_metric(metric_key, title, ylabel, filename, plot_bruto=False):
    plt.figure(figsize=(10, 6))
    plotou_algo = False
    
    for run in runs:
        if run['data'] is not None and metric_key in run['data']:
            d = run['data'][metric_key]
            eixo = range(1, len(d) + 1)
            
            # Se for gráfico de rede, plota a versão sem quantização clarinha ao fundo
            if plot_bruto and 'mb_bruto' in run['data']:
                plt.plot(eixo, run['data']['mb_bruto'], color=run['color'], linestyle=':', alpha=0.3)
                
            plt.plot(eixo, d, label=run['name'], color=run['color'], linestyle=run['ls'], marker=run['marker'], markersize=7)
            plotou_algo = True

    if plotou_algo:
        plt.title(f"{title} - {DATASET}", fontsize=14, pad=15)
        plt.xlabel("Rodadas (Rounds)", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Otimiza o layout e salva sem cortar bordas
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📸 Gráfico salvo em: {save_path}")
    plt.close()

def plot_x_vs_y(x_key, y_key, title, xlabel, ylabel, filename):
    plt.figure(figsize=(10, 6))
    plotou_algo = False
    
    for run in runs:
        if run['data'] is not None and x_key in run['data'] and y_key in run['data']:
            x_data = run['data'][x_key]
            y_data = run['data'][y_key]
            
            # Garantir que tenham o mesmo tamanho
            min_len = min(len(x_data), len(y_data))
            x_data = x_data[:min_len]
            y_data = y_data[:min_len]
            
            plt.plot(x_data, y_data, label=run['name'], color=run['color'], linestyle=run['ls'], marker=run['marker'], markersize=7)
            plotou_algo = True

    if plotou_algo:
        plt.title(f"{title} - {DATASET}", fontsize=14, pad=15)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📸 Gráfico salvo em: {save_path}")
    plt.close()

print("\nGerando Gráficos...")
# Plot 1: Acurácia
plot_metric('acc', "Acurácia Global de Teste", "Acurácia (%)", "01_acuracia_global.png")

# Plot 2: Custo de Comunicação
plot_metric('mb', "Custo de Comunicação Acumulado", "Megabytes Acumulados (MB)", "02_custo_rede.png", plot_bruto=True)

# Plot 3: Latência por Rodada
plot_metric('time', "Latência (Tempo de Treinamento por Rodada)", "Tempo (Segundos)", "03_latencia_rodada.png")

# Plot 4: Perda de Treinamento
plot_metric('loss', "Perda (Loss) Global de Treinamento", "Loss", "04_perda_treinamento.png")

# Plot 5: Banda / Tamanho do Modelo por Rodada
plot_metric('size_mb', "Tamanho Efetivo dos Adaptadores por Rodada", "Tamanho do Modelo (MB)", "05_tamanho_modelo.png")

# Plot 6: Acurácia vs Banda Acumulada (MB)
plot_x_vs_y('mb', 'acc', "Acurácia Global vs Banda Acumulada", "Banda Acumulada Trafegada (MB)", "Acurácia (%)", "06_acuracia_vs_banda.png")

# Plot 7: Evolução do PaCA por Cliente
def plot_paca_clients():
    plt.figure(figsize=(12, 6))
    plotou_algo = False
    
    for run in runs:
        if run['data'] is not None and 'paca' in run['data']:
            paca_matrix = run['data']['paca'] # shape: (rounds, clients)
            rounds = range(1, len(paca_matrix) + 1)
            
            paca_mean = np.mean(paca_matrix, axis=1)
            plt.plot(rounds, paca_mean, label=f"{run['name']} (Média)", color=run['color'], linestyle=run['ls'], linewidth=2)
            
            if "Adaptativo" in run['name']:
                for c in range(paca_matrix.shape[1]):
                    plt.plot(rounds, paca_matrix[:, c], color=run['color'], alpha=0.15, linewidth=1)
            plotou_algo = True

    if plotou_algo:
        plt.title(f"Evolução do PaCA (Média e por Cliente) - {DATASET}", fontsize=14, pad=15)
        plt.xlabel("Rodadas", fontsize=12)
        plt.ylabel("Camadas (PaCA)", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, "07_evolucao_paca.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📸 Gráfico salvo em: {save_path}")
    plt.close()

plot_paca_clients()

# Plot 7: Banda Acumulada vs Tempo Acumulado
plot_x_vs_y('time_acum', 'mb', "Banda Acumulada vs Tempo Acumulado", "Tempo Acumulado de Treinamento (Segundos)", "Banda Acumulada Trafegada (MB)", "07_banda_vs_tempo.png")

# Plot 8: Acurácia vs Tempo Acumulado
plot_x_vs_y('time_acum', 'acc', "Acurácia vs Tempo Acumulado", "Tempo Acumulado de Treinamento (Segundos)", "Acurácia (%)", "08_acuracia_vs_tempo.png")

print(f"\n✨ Sucesso! Todos os gráficos foram salvos na pasta: {OUTPUT_DIR}")
