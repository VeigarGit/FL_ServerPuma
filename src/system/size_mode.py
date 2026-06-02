def get_model_size(model):
    total_params = 0
    for param in model.parameters():
        total_params += param.numel() * param.element_size()  # número de elementos * tamanho de cada elemento em bytes
    return total_params / (1024 ** 2)  # converte para megabytes (MB)

def get_trainable_size_and_params(model):
    """Retorna o tamanho (em MB) e a quantidade apenas dos parâmetros treináveis."""
    total_size = 0
    total_params = 0
    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()
            total_size += param.numel() * param.element_size()
    return total_size / (1024 ** 2), total_params

# Exemplo de uso
# model = ...  # seu modelo PyTorch
# print(f"Tamanho do modelo: {get_model_size(model):.2f} MB")
