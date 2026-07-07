import json

with open('plot_results.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            # Replace old directory constants
            if 'SERVER_RESULTS_DIR = "../results/"' in line:
                new_source.append('EXP_NAME = "fl_session_TODO" # INSIRA O NOME DA PASTA DO SEU EXPERIMENTO AQUI (ex: fl_session_20260706_120400)\n')
                new_source.append('RESULTS_DIR = f"../results/{EXP_NAME}/"\n')
                continue
            if 'CLIENT_RESULTS_DIR = "./dados_compartilhados/"' in line:
                continue # We'll just use RESULTS_DIR for both now
            
            # Replace SERVER patterns
            if 'f"{dataset}_{strategy}_rank{rank}_paca{paca}_{prune}_{algo}_run*.h5"' in line:
                line = line.replace('f"{dataset}', 'f"server_{dataset}')
            if 'f"{dataset}_{strategy}_paca{paca}_{prune}_{algo}_run*.h5"' in line:
                line = line.replace('f"{dataset}', 'f"server_{dataset}')
            if 'SERVER_RESULTS_DIR' in line:
                line = line.replace('SERVER_RESULTS_DIR', 'RESULTS_DIR')
                
            # Replace CLIENT patterns
            if 'f"{dataset}_{strategy}_rank{rank}_paca*_{algo}_client_*_run*.h5"' in line:
                line = line.replace('f"{dataset}_{strategy}_rank{rank}_paca*_{algo}_client_*_run*.h5"', 'f"client_*_{dataset}_{strategy}_rank{rank}_paca*_{algo}_run*.h5"')
            if 'CLIENT_RESULTS_DIR' in line:
                line = line.replace('CLIENT_RESULTS_DIR', 'RESULTS_DIR')
                
            # Replace hardcoded directory paths in other cells
            if 'results_dir = "../results"' in line:
                line = line.replace('results_dir = "../results"', 'results_dir = f"../results/{EXP_NAME}"')
            if 'RESULTS_DIR = "../results/*.h5"' in line:
                line = line.replace('RESULTS_DIR = "../results/*.h5"', 'RESULTS_DIR = f"../results/{EXP_NAME}/server_*.h5"')
            if 'CLIENT_RESULTS_DIR = "./dados_compartilhados/*.h5"' in line:
                line = line.replace('CLIENT_RESULTS_DIR = "./dados_compartilhados/*.h5"', 'CLIENT_RESULTS_DIR = f"../results/{EXP_NAME}/client_*.h5"')

            new_source.append(line)
        cell['source'] = new_source

with open('plot_results.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
