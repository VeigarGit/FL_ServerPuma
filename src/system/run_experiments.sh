#!/bin/bash

# Vamos iterar sobre as estratégias e sobre a flag --ala (1 = Sem FedALA, 0 = Com FedALA)
for strategy in lora sora_with_schedule
do
    for current_ala in 1 0
    do
        if [ $current_ala -eq 0 ]; then
            ala_name="FedALA (LIGADO)"
        else
            ala_name="FedAVG (DESLIGADO)"
        fi
        
        echo "========================================================="
        echo "🤖 INICIANDO EXPERIMENTO $strategy RANK 8 PACA 12 | MODO: $ala_name"
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
            --strategy $strategy \
            --paca 12 \
            --prune 0 \
            --rank 8
            
        echo "✅ Experimento $strategy com MODO = $ala_name finalizado."
        echo "Aguardando 5 segundos para limpar buffers da GPU antes do próximo..."
        sleep 5
    done
done

echo "🎉 Todas as 4 simulações foram concluídas!"