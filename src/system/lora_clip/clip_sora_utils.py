def apply_sora_hot_swap(model, global_state):
    """
    HOT-SWAP: Analisa os pesos recebidos do servidor e substitui os adaptadores
    SoRA locais se o servidor tiver reduzido o Rank deles.
    """
    from lora_clip.sora import SoRALinear, SoRAWrappedLinear
    
    swapped = 0
    for name, module in model.named_modules():
        if isinstance(module, SoRAWrappedLinear) and module.sora is not None:
            # A matriz A tem o formato (r, in_features). Vamos descobrir o 'r' que o servidor enviou!
            lora_A_key = f"{name}.sora.lora_A"
            if lora_A_key in global_state:
                new_rank = global_state[lora_A_key].shape[0] 
                current_rank = module.sora.r
                
                if new_rank != current_rank:
                    logger.debug(f"Hot-Swap na {name}: Rank {current_rank} -> {new_rank}")
                    
                    # Cria a nova peça de hardware (adaptador menor) com as configs originais
                    new_sora = SoRALinear(
                        in_features=module.original.in_features,
                        out_features=module.original.out_features,
                        r=new_rank,
                        lora_alpha=module.sora.lora_alpha,
                        lora_dropout=module.sora.lora_dropout.p if isinstance(module.sora.lora_dropout, nn.Dropout) else 0.0
                    ).to(next(model.parameters()).device)
                    
                    # Desparafusa a antiga e coloca a nova
                    module.sora = new_sora
                    swapped += 1
                    
    if swapped > 0:
        logger.info(f"🔧 Hot-Swap Concluído: {swapped} adaptadores substituídos para o novo tamanho.")
    return model