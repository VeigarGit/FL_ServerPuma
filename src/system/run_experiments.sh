#!/bin/bash

echo "========================================================="
echo "🚀 INICIANDO BATERIA DE EXPERIMENTOS (50 RODADAS | 3 SIMULAÇÕES)"
echo "========================================================="

# Configurações globais
CLIENTS=5
MODEL="clip"
DATASET="OxfordPets"
ROUNDS=50
NUM_CLASSES=37
ALA=1 # 1 = FedAVG (sem FedALA)
SIMULATIONS=3
PACA=12
RANK=8

# ==============================================================================
# CENÁRIO 1: LoRA (Sem Pruning, Sem FedALA, PaCA Fixo)
# ==============================================================================
echo "▶️ [1/4] Executando: LoRA (PaCA 12, Sem Pruning)"
./run.sh \
    -s lora_paca12_sempruning_semALA \
    -p 9501 \
    --clients $CLIENTS \
    --model $MODEL \
    --dataset $DATASET \
    --rounds $ROUNDS \
    --num-classes $NUM_CLASSES \
    --ala $ALA \
    --simulations $SIMULATIONS \
    --strategy lora \
    --paca $PACA \
    --prune 1 \
    --prune-freq 1 \
    --rank $RANK

echo "✅ Cenário 1 concluído. Limpando memória (10s)..."
sleep 10

# ==============================================================================
# CENÁRIO 2: SoRA (Com Pruning, Sem FedALA, PaCA Fixo)
# ==============================================================================
echo "▶️ [2/4] Executando: SoRA (PaCA 12, Com Pruning)"
./run.sh \
    -s sora_paca12_compruning_semALA \
    -p 9502 \
    --clients $CLIENTS \
    --model $MODEL \
    --dataset $DATASET \
    --rounds $ROUNDS \
    --num-classes $NUM_CLASSES \
    --ala $ALA \
    --simulations $SIMULATIONS \
    --strategy sora_with_schedule \
    --paca $PACA \
    --prune 0 \
    --prune-freq 1 \
    --rank $RANK

echo "✅ Cenário 2 concluído. Limpando memória (10s)..."
sleep 10

# ==============================================================================
# CENÁRIO 3: SoRA Adaptativo (Com Pruning, Sem FedALA, PaCA Adaptativo)
# ==============================================================================
echo "▶️ [3/4] Executando: SoRA (PaCA Adaptativo, Com Pruning)"
./run.sh \
    -s sora_paca_adaptativo_compruning_semALA \
    -p 9503 \
    --clients $CLIENTS \
    --model $MODEL \
    --dataset $DATASET \
    --rounds $ROUNDS \
    --num-classes $NUM_CLASSES \
    --ala $ALA \
    --simulations $SIMULATIONS \
    --strategy sora_with_schedule \
    --paca $PACA \
    --adaptive-paca \
    --prune 0 \
    --prune-freq 1 \
    --rank $RANK

echo "✅ Cenário 3 concluído. Limpando memória (10s)..."
sleep 10

# ==============================================================================
# CENÁRIO 4: SoRA Adaptativo (Sem Pruning, Sem FedALA, PaCA Adaptativo)
# ==============================================================================
echo "▶️ [4/4] Executando: SoRA (PaCA Adaptativo, Sem Pruning)"
./run.sh \
    -s sora_paca_adaptativo_sempruning_semALA \
    -p 9504 \
    --clients $CLIENTS \
    --model $MODEL \
    --dataset $DATASET \
    --rounds $ROUNDS \
    --num-classes $NUM_CLASSES \
    --ala $ALA \
    --simulations $SIMULATIONS \
    --strategy sora_with_schedule \
    --paca $PACA \
    --adaptive-paca \
    --prune 1 \
    --prune-freq 1 \
    --rank $RANK

echo "✅ Cenário 4 concluído."

echo "========================================================="
echo "🎉 TODOS OS EXPERIMENTOS FORAM CONCLUÍDOS COM SUCESSO!"
echo "========================================================="