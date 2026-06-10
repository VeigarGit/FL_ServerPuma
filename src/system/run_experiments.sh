#!/bin/bash

# Loop de 2 até 8
for current_rank in {2..4}
do
    echo "========================================================="
    echo "🤖 INICIANDO EXPERIMENTO COM RANK = $current_rank"
    echo "========================================================="
    
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
        # --prune-freq 1 \
        --rank $current_rank
        
    echo "✅ Experimento com RANK = $current_rank finalizado."
    echo "Aguardando 5 segundos para esfriar antes do próximo..."
    sleep 5
done

echo "🎉 Todos os experimentos de Rank 2 a 8 foram concluídos!"