import os
import glob
import re

def main():
    # Define o caminho base da pasta results (considerando que este script está em src/system/)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, '..', 'results')
    
    if not os.path.exists(results_dir):
        print(f"Diretório não encontrado: {results_dir}")
        return

    # Busca as pastas dos experimentos, especificamente as iniciadas em 1_, 2_, 3_ e 4_
    experiment_dirs = [d for d in os.listdir(results_dir) 
                       if os.path.isdir(os.path.join(results_dir, d)) and re.match(r'^[1-4]_', d)]
    experiment_dirs.sort()

    results = {}

    print("Processando logs, aguarde...\n")
    for d in experiment_dirs:
        dir_path = os.path.join(results_dir, d)
        
        # Encontra o arquivo server.log dentro de logs/sim_*
        sim_logs = glob.glob(os.path.join(dir_path, 'logs', 'sim_*', 'server.log'))
        
        if not sim_logs:
            continue
        
        sim_durations = []
        for log_file in sim_logs:
            sim_time = 0.0
            try:
                with open(log_file, 'r', errors='ignore') as f:
                    for line in f:
                        # Procura a linha: "Round 150 duration: 21.30 seconds"
                        match = re.search(r'duration:\s*([\d\.]+)\s*seconds', line, re.IGNORECASE)
                        if match:
                            sim_time += float(match.group(1))
            except Exception as e:
                print(f"Erro ao ler {log_file}: {e}")
            
            # Só armazena se encontrou tempos
            if sim_time > 0:
                sim_durations.append(sim_time)
                
        if sim_durations:
            avg_duration = sum(sim_durations) / len(sim_durations)
            results[d] = {
                'avg_time_sec': avg_duration,
                'avg_time_min': avg_duration / 60,
                'num_sims': len(sim_durations)
            }

    if not results:
        print("Nenhum log encontrado ou nenhum tempo computado.")
        return

    # Tentativa de mapear os experimentos automaticamente
    exp_keys = {
        'lora': next((k for k in results if k.startswith('1_')), None),
        'sora': next((k for k in results if k.startswith('2_')), None),
        'prop1': next((k for k in results if k.startswith('3_')), None),
        'prop2': next((k for k in results if k.startswith('4_')), None),
    }

    # Print Tabela de Tempos Médios
    print("=" * 110)
    print(f"{'Experimento':<60} | {'Simulações':<10} | {'Tempo (s)':<12} | {'Tempo (min)':<12}")
    print("-" * 110)
    for k, v in results.items():
        print(f"{k:<60} | {v['num_sims']:<10} | {v['avg_time_sec']:<12.2f} | {v['avg_time_min']:<12.2f}")
    print("=" * 110)
    print()

    # Função auxiliar para realizar a comparação
    def print_comparacao(alvo_key, baseline_key, nome_baseline):
        if not alvo_key or not baseline_key:
            return
        
        t_alvo = results[alvo_key]['avg_time_sec']
        t_base = results[baseline_key]['avg_time_sec']
        
        reducao_tempo = ((t_base - t_alvo) / t_base) * 100
        ganho_vel = ((t_base / t_alvo) - 1) * 100
        
        print(f"👉 Comparando [{alvo_key}] com o baseline [{nome_baseline}]:")
        if reducao_tempo > 0:
            print(f"   📉 Redução de Tempo : -{reducao_tempo:.2f}% (terminou mais cedo)")
            print(f"   🚀 Ganho Velocidade : +{ganho_vel:.2f}% (throughput superior)")
        else:
            print(f"   ⚠️ Aumento de Tempo : +{abs(reducao_tempo):.2f}% (ficou mais lento)")
        print()

    print("--- ANÁLISE DE PROPOSTA FINAL (Exp 4) ---")
    if exp_keys['prop2'] and exp_keys['lora']:
        print_comparacao(exp_keys['prop2'], exp_keys['lora'], exp_keys['lora'][:15]+"...")
    if exp_keys['prop2'] and exp_keys['sora']:
        print_comparacao(exp_keys['prop2'], exp_keys['sora'], exp_keys['sora'][:15]+"...")
        
    print("--- ANÁLISE DE PROPOSTA INTERMEDIÁRIA (Exp 3) ---")
    if exp_keys['prop1'] and exp_keys['lora']:
        print_comparacao(exp_keys['prop1'], exp_keys['lora'], exp_keys['lora'][:15]+"...")
    if exp_keys['prop1'] and exp_keys['sora']:
        print_comparacao(exp_keys['prop1'], exp_keys['sora'], exp_keys['sora'][:15]+"...")

if __name__ == '__main__':
    main()
