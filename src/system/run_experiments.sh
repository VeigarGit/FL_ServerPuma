#!/bin/bash

# Vamos iterar a flag --ala entre 1 (FedAVG padrão/Desligado) e 0 (FedALA/Ligado)
for current_ala in 1 0
do
    if [ $current_ala -eq 0 ]; then
        ala_name="FedALA (LIGADO)"
    else
        ala_name="FedAVG (DESLIGADO)"
    fi
    
    echo "========================================================="
    echo "🤖 INICIANDO EXPERIMENTO SORA RANK 8 | MODO: $ala_name"
    echo "========================================================="
    
    # Execução do script
    ./run.sh \
        --clients 5 \
        --model clip \
        --dataset OxfordPets \
        --rounds 30 \
        --num-classes 37 \
        --ala $current_ala \
        --simulations 1 \
        --strategy sora_no_schedule \
        --paca 0 \
        --prune 1 \
        --rank 8
        
    echo "✅ Experimento com MODO = $ala_name finalizado."
    echo "Aguardando 5 segundos para limpar buffers da GPU antes do próximo..."
    sleep 5
done

echo "🎉 Ambos os experimentos (Com e Sem FedALA) foram concluídos!"