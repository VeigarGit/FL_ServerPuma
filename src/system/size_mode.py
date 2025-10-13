def get_model_size(model):
    total_params = 0
    for param in model.parameters():
        total_params += param.numel() * param.element_size()  # número de elementos * tamanho de cada elemento em bytes
    return total_params / (1024 ** 2)  # converte para megabytes (MB)

# Exemplo de uso
# model = ...  # seu modelo PyTorch
# print(f"Tamanho do modelo: {get_model_size(model):.2f} MB")
