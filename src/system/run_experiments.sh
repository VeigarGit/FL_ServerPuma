#!/bin/bash

# Loop de 3 até 4
for current_rank in {6..7}
do
    echo "========================================================="
    echo "🤖 INICIANDO EXPERIMENTO COM RANK = $current_rank"
    echo "========================================================="
    
    # Execução do script (sem comentários no meio dos argumentos)
    ./run.sh \
        --clients 5 \
        --model clip \
        --dataset OxfordPets \
        --rounds 30 \
        --num-classes 37 \
        --ala 1 \
        --simulations 1 \
        --strategy lora \
        --paca 12 \
        --prune 1 \
        --rank $current_rank
        
    echo "✅ Experimento com RANK = $current_rank finalizado."
    echo "Aguardando 5 segundos para esfriar antes do próximo..."
    sleep 5
done

echo "🎉 Todos os experimentos de Rank 3 e 4 foram concluídos!"