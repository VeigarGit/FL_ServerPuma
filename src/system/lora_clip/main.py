import os
import torch
import pickle
import sys
from transformers import CLIPProcessor

# Importa as funções auxiliares e de configuração do clip_setup
from clip_setup import (
    parse_args, load_config, get_device, resolve_run_modes, 
    build_run_config, build_output_path, build_dataloaders,
    build_model, build_optimizer, build_scheduler, 
    benchmark_attention, quantize_weights
)
from trainer import ModelTrainer 
from sora import pre_prune_whole_model, re_freeze_vision_model

def main():
    """
    Pipeline completo de execução.
    Esta função orquestra todo o ciclo de vida do projeto:
    1. Inicialização: Configura hardware (GPU/CPU) e carrega parâmetros do YAML.
    2. Dados: Prepara os iteradores de imagens (DataLoaders) e mapeia classes.
    3. Modelo: Constrói o CLIP com suporte a SDPA e injeta adaptadores (LoRA/SoRA).
    4. Otimização Prévia: Opcionalmente reduz o backbone original via pre-pruning.
    5. Treinamento: Executa o loop via ModelTrainer, monitorando acurácia e esparsidade.
    6. Refinamento: Se SoRA for usado, realiza a poda estrutural para gerar um modelo compacto.
    7. Retreino: Depois da poda estrutural, o modelo compacto é submetido a uma nova avaliação
    8. Compressão: Aplica quantização INT8 para reduzir o peso final em até 4x.
    9. Exportação: Salva os pesos otimizados para uso futuro.
    """
    args = parse_args()
    config = load_config(args.config)
    device = get_device()

    # Inicializa o processador e os dataloaders
    processor = CLIPProcessor.from_pretrained(config["model"]["name"])
    train_loader, eval_loader, class_names = build_dataloaders(config, processor, device)
    
    run_modes = resolve_run_modes(config)
    multiple_runs = len(run_modes) > 1

    for run_mode in run_modes:
        run_config = build_run_config(config, run_mode=run_mode)

        # 1. Constrói o modelo (já com SDPA)
        model = build_model(run_config, num_classes=len(class_names), device=device)
        
        # 2. Pré-Poda opcional do backbone (se configurado)
        sora_config = run_config["model"].get("sora", {})
        # if sora_config.get("pre_prune_ratio"):
        #     ratio = sora_config["pre_prune_ratio"]
        #     model = pre_prune_whole_model(model, prune_ratio=ratio, device=device)
        #     model = re_freeze_vision_model(model)

        # 3. Benchmark de performance da atenção
        benchmark_attention(model, eval_loader, device)

        # 4. Inicializa otimizadores e schedulers
        optimizer, sparse_optimizer = build_optimizer(model, run_config)
        scheduler = build_scheduler(optimizer, run_config)
        sparse_scheduler = build_scheduler(sparse_optimizer, run_config) if sparse_optimizer else None
        
        is_sora = run_mode in run_config["model"].get("sora_modes", ["with_sora_no_schedule", "with_sora_schedule"])

        # 5. Configura o Trainer e executa o treino
        trainer = ModelTrainer(
            model=model, 
            train_loader=train_loader, 
            eval_loader=eval_loader, 
            optimizer=optimizer, 
            sparse_optimizer=sparse_optimizer, 
            scheduler=scheduler, 
            sparse_scheduler=sparse_scheduler, 
            config=run_config, 
            is_sora=is_sora
        )
        
        total_epochs = run_config["training"]["epochs"]
        print(f"\nStarting run: {run_mode}")
        trainer.execute_epochs(total_epochs)

        # BenchMark de Inferência
        trainer.benchmark_inference(eval_loader, device)

        # 6. Finalização: Poda estrutural do SoRA, extração de pesos treinados e do modelo compacto
        trainable_state_dict, post_prunning_model = trainer.finalize()

        final_trainer = ModelTrainer(
            model=post_prunning_model, 
            train_loader=train_loader, 
            eval_loader=eval_loader, 
            optimizer=optimizer, 
            sparse_optimizer=sparse_optimizer, 
            scheduler=scheduler, 
            sparse_scheduler=sparse_scheduler, 
            config=run_config, 
            is_sora=False
        )

        # 7. Avaliação do modelo compacto
        print("Iniciando a avaliação do modelo compacto:")
        final_trainer.execute_epochs(total_epochs)
        
        final_trainer.benchmark_inference(eval_loader, device)

        # 8. Quantização INT8 para redução drástica de tamanho
        print(f"Tamanho antes da quantização: {sys.getsizeof(pickle.dumps(trainable_state_dict)) / 1024**2:.2f} MB")
        final_state_dict = quantize_weights(trainable_state_dict)
        print(f"Tamanho após quantização: {sys.getsizeof(pickle.dumps(final_state_dict)) / 1024**2:.2f} MB")

        # 9. Salvamento dos pesos
        output_path = build_output_path(
            config["output"]["weights_path"],
            run_mode=run_mode,
            multiple_runs=multiple_runs,
        )
        
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        torch.save(final_state_dict, output_path)
        print(f"Saved optimized weights to {output_path}")

if __name__ == "__main__":
    main()