#!/bin/bash

# Mudar para diretório do dataset
cd ../dataset

# Gerar dataset Cifar100
python generate_Cifar100.py noniid - dir

# Voltar para system (ou diretório principal)
cd ../system
sleep 15
# Create tmux session
tmux new-session -d -s myapp 'python server.py'

# Create panes for clients
tmux split-window -h 'python client.py --client-idx 0 --host 'localhost''
tmux split-window -v 'python client.py --client-idx 1 --host 'localhost''  # Corrigido: idx 1

# Attach to session
tmux attach-session -t myapp

# or tmux a