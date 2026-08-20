#!/usr/bin/env python3
"""Detailed trajectory analysis: Sora vs Lora divergence point."""
import h5py
import numpy as np
import glob
import os

results_base = '../results'
lora_dir = os.path.join(results_base, 'david_clip_lora_prune1_ala1_20260716_173830')
sora_dir = os.path.join(results_base, 'sora_rankminimo_v2')
sora_v2_dir = os.path.join(results_base, 'david_v2_clip_sora_with_schedule_prune1_ala1_adaptpaca_20260718_155430')

def load_acc_loss_params(exp_dir):
    files = sorted(glob.glob(os.path.join(exp_dir, 'server_*.h5')))
    acc, loss, params, size_ = [], [], [], []
    for f in files:
        with h5py.File(f, 'r') as hf:
            acc.append(np.array(hf['rs_test_acc']))
            loss.append(np.array(hf['rs_train_loss']))
            if 'Trainable_params' in hf:
                params.append(np.array(hf['Trainable_params']))
            if 'Model_size_per_round_Mb' in hf:
                size_.append(np.array(hf['Model_size_per_round_Mb']))
    return {
        'acc': np.mean(acc, axis=0),
        'loss': np.mean(loss, axis=0),
        'params': np.mean(params, axis=0) if params else None,
        'size': np.mean(size_, axis=0) if size_ else None,
    }

lora = load_acc_loss_params(lora_dir)
sora = load_acc_loss_params(sora_dir)
sora_v2 = load_acc_loss_params(sora_v2_dir)

print("=" * 80)
print("TRAJETÓRIA DE ACURÁCIA E LOSS - COMPARATIVO DETALHADO")
print("=" * 80)

# Show every 25 rounds
checkpoints = [1, 10, 25, 50, 75, 100, 125, 134, 150, 175, 200, 225, 250, 275, 300]

print(f"\n{'Round':>6} | {'LoRA Acc':>10} | {'SoRA Acc':>10} | {'SoRA_v2 Acc':>12} | {'LoRA Loss':>10} | {'SoRA Loss':>10} | {'SoRA_v2 Loss':>12}")
print("-" * 90)

for r in checkpoints:
    idx = r - 1
    if idx < len(lora['acc']) and idx < len(sora['acc']) and idx < len(sora_v2['acc']):
        print(f"{r:>6} | {lora['acc'][idx]:>9.2f}% | {sora['acc'][idx]:>9.2f}% | {sora_v2['acc'][idx]:>11.2f}% | {lora['loss'][idx]:>10.4f} | {sora['loss'][idx]:>10.4f} | {sora_v2['loss'][idx]:>12.4f}")

print("\n" + "=" * 80)
print("EVOLUÇÃO DOS PARÂMETROS TREINÁVEIS")
print("=" * 80)

print(f"\n{'Round':>6} | {'LoRA Params':>14} | {'SoRA Params':>14} | {'SoRA_v2 Params':>16} | {'SoRA Red%':>10} | {'SoRA_v2 Red%':>12}")
print("-" * 85)

for r in checkpoints:
    idx = r - 1
    lp = lora['params'][idx] if lora['params'] is not None else 0
    sp = sora['params'][idx] if sora['params'] is not None else 0
    sp2 = sora_v2['params'][idx] if sora_v2['params'] is not None else 0
    sr = (1 - sp/max(sora['params'][0], 1)) * 100 if sora['params'] is not None else 0
    sr2 = (1 - sp2/max(sora_v2['params'][0], 1)) * 100 if sora_v2['params'] is not None else 0
    print(f"{r:>6} | {lp:>14,.0f} | {sp:>14,.0f} | {sp2:>16,.0f} | {sr:>9.1f}% | {sr2:>11.1f}%")

# Key analysis: where does the accuracy gap open?
print("\n" + "=" * 80)
print("ANÁLISE DO GAP DE ACURÁCIA LORA vs SORA")
print("=" * 80)

gap = lora['acc'] - sora['acc']
gap_v2 = lora['acc'] - sora_v2['acc']

# Find when gap exceeds 5%
for i in range(len(gap)):
    if gap[i] > 5:
        print(f"\n⚠️  GAP LoRA-SoRA > 5% primeiro em rodada {i+1}: LoRA={lora['acc'][i]:.2f}% vs SoRA={sora['acc'][i]:.2f}%")
        if sora['params'] is not None:
            print(f"   Nesse ponto SoRA tinha {sora['params'][i]:,.0f} params ({(1-sora['params'][i]/sora['params'][0])*100:.1f}% reduzido)")
        break

for i in range(len(gap)):
    if gap[i] > 10:
        print(f"\n⚠️  GAP LoRA-SoRA > 10% primeiro em rodada {i+1}: LoRA={lora['acc'][i]:.2f}% vs SoRA={sora['acc'][i]:.2f}%")
        if sora['params'] is not None:
            print(f"   Nesse ponto SoRA tinha {sora['params'][i]:,.0f} params ({(1-sora['params'][i]/sora['params'][0])*100:.1f}% reduzido)")
        break

# Identify peak for SoRA and what happened after
peak_sora = np.argmax(sora['acc'])
print(f"\n📌 SoRA peak: rodada {peak_sora+1} com {sora['acc'][peak_sora]:.2f}%")
print(f"   Loss nessa rodada: {sora['loss'][peak_sora]:.4f}")
if sora['params'] is not None:
    print(f"   Params nessa rodada: {sora['params'][peak_sora]:,.0f} ({(1-sora['params'][peak_sora]/sora['params'][0])*100:.1f}% reduzido)")

# After peak, how much does it drop?
post_peak_acc = sora['acc'][peak_sora:]
print(f"   Depois do peak: min={np.min(post_peak_acc):.2f}%, média={np.mean(post_peak_acc):.2f}%")
print(f"   Queda do peak até o final: {sora['acc'][peak_sora] - sora['acc'][-1]:.2f}%")

# Compare loss trajectories
print(f"\n📈 DIVERGÊNCIA LOSS-ACCURACY (OVERFITTING CHECK):")
for name, data in [("SoRA", sora), ("SoRA_v2", sora_v2)]:
    peak = np.argmax(data['acc'])
    loss_at_peak = data['loss'][peak]
    loss_final = data['loss'][-1]
    acc_at_peak = data['acc'][peak]
    acc_final = data['acc'][-1]
    print(f"\n   {name}:")
    print(f"      Peak acc rodada {peak+1}: acc={acc_at_peak:.2f}%, loss={loss_at_peak:.4f}")
    print(f"      Final rodada {len(data['acc'])}: acc={acc_final:.2f}%, loss={loss_final:.4f}")
    print(f"      Loss mudou: {loss_at_peak:.4f} → {loss_final:.4f} ({loss_final-loss_at_peak:+.4f})")
    print(f"      Acc mudou: {acc_at_peak:.2f}% → {acc_final:.2f}% ({acc_final-acc_at_peak:+.2f}%)")
    if loss_final > loss_at_peak and acc_final < acc_at_peak:
        print(f"      🔴 OVERFITTING CLARO: loss subiu E accuracy caiu após o peak!")
    elif acc_final < acc_at_peak:
        print(f"      🟡 ACC DEGRADOU mas loss nem tanto, pode ser instabilidade ou pruning destrutivo")
