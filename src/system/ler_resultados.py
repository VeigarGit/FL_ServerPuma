import h5py
import matplotlib.pyplot as plt
import os

# Os três arquivos gerados pela sua bateria de testes
ARQUIVOS = {
    "Baseline (FedAVG sem Poda)": "../results/Cifar100_withou_Prune_FedAVG.h5",
    "FedALA Puro (Sem Poda)": "../results/Cifar100_withou_Prune_FedALA.h5",
    "Sistema Completo (FedALA + Poda)": "../results/Cifar100_prune_FedALA.h5"
}

plt.figure(figsize=(14, 6))

# Gráfico 1: Acurácia
plt.subplot(1, 2, 1)
for label, path in ARQUIVOS.items():
    if os.path.exists(path):
        with h5py.File(path, 'r') as hf:
            acc = hf['rs_test_acc'][:]
            plt.plot(range(1, len(acc) + 1), acc, label=label, linewidth=2)
plt.title('Comparação de Acurácia (CIFAR-100)')
plt.xlabel('Rodadas')
plt.ylabel('Acurácia (%)')
plt.grid(True)
plt.legend()

# Gráfico 2: Loss
plt.subplot(1, 2, 2)
for label, path in ARQUIVOS.items():
    if os.path.exists(path):
        with h5py.File(path, 'r') as hf:
            loss = hf['rs_train_loss'][:] # ou rs_test_loss dependendo de como salvou
            plt.plot(range(1, len(loss) + 1), loss, label=label, linewidth=2)
plt.title('Comparação de Perda / Loss')
plt.xlabel('Rodadas')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('comparacao_final.png')
print("Gráfico de comparação salvo como 'comparacao_final.png'!")