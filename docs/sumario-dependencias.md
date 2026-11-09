# Sumário hierárquico de dependências

Levantamento do código presente no workspace em 08/11/2026. Este documento organiza os arquivos como capítulos de um livro: o arquivo importado aparece indentado abaixo do arquivo que o importa.

## Como ler

- **1** identifica o arquivo que inicia um capítulo; **1.1** é uma dependência direta dele; **1.1.1** é uma dependência do arquivo **1.1**.
- A indentação representa uma relação de importação, não a estrutura de pastas nem a ordem de execução.
- Cada arquivo é expandido na primeira ocorrência. Quando reaparece em outro ramo, o link numérico leva à sua seção completa; a relação com o novo consumidor continua visível.
- **[import local/condicional]** indica que todos os imports dessa dependência no arquivo pai estão dentro de funções, classes ou blocos condicionais. Isso não significa que todos esses ramos serão executados.
- **[sem imports internos]** indica ausência de imports para outros arquivos do projeto; o arquivo ainda pode usar bibliotecas externas, dados ou configurações.
- Os links com nomes de arquivos abrem o código; os links numéricos navegam dentro deste sumário.
- Os nomes dos módulos são relativos a `src/fl_serverpuma/`; caminhos do notebook e dos scripts aparecem a partir da raiz do projeto.

**Cobertura:** 120 módulos Python, 185 relações distintas entre módulos e 7 dependências internas do notebook. Imports repetidos do mesmo arquivo são agrupados em uma entrada. Os marcadores `__init__.py` estão no apêndice.

A árvore considera imports explícitos, inclusive dentro de funções, sem executar o projeto. Não representa chamadas de funções, leitura de YAML/datasets, imports dinâmicos ou dependências externas como PyTorch. Os comandos de inicialização estão separados ao final. Esta é uma fotografia do código: mudanças nos imports exigem atualizar o documento.

Não foram encontrados ciclos entre os módulos nos imports analisados.

Cada nova função ou método deve ser registrado em **Funções/métodos**, sob a primeira ocorrência expandida do arquivo responsável. Use o nome qualificado para métodos e funções aninhadas; atualize o registro também ao renomear, mover ou excluir. Essas listas identificam definições, não chamadas entre funções. Regras completas em [AGENTS.md](../AGENTS.md).

## Capítulos

- [1. execution/server.py](#sec-1)
- [2. execution/client.py](#sec-2)
- [3. models/clip/main.py](#sec-3)
- [4. models/slm/main.py](#sec-4)
- [5. simulation/sumo/orchestrator.py](#sec-5)
- [6. simulation/sumo/create_trips.py](#sec-6)
- [7. simulation/docker/compose.py](#sec-7)
- [8. experiments/analysis/plot_results.py](#sec-8)
- [9. experiments/analysis/analise_estrategias.py](#sec-9)
- [10. experiments/analysis/v2v_accuracy.py](#sec-10)
- [11. experiments/analysis/exchange_time.py](#sec-11)
- [12. experiments/analysis/analyze_sora_vs_lora.py](#sec-12)
- [13. experiments/analysis/analise_tempo.py](#sec-13)
- [14. experiments/analysis/result_utils.py](#sec-14)
- [15. experiments/analysis/get_mean_std.py](#sec-15)
- [16. experiments/analysis/update_notebook.py](#sec-16)
- [17. src/system/plot/flopsxpercentage.ipynb](#sec-17)
- [18. data/generators/generate_AGNews.py](#sec-18)
- [19. data/generators/generate_AmazonReview.py](#sec-19)
- [20. data/generators/generate_Cifar10.py](#sec-20)
- [21. data/generators/generate_Cifar100.py](#sec-21)
- [22. data/generators/generate_Country211.py](#sec-22)
- [23. data/generators/generate_DTD.py](#sec-23)
- [24. data/generators/generate_Digit5.py](#sec-24)
- [25. data/generators/generate_DomainNet.py](#sec-25)
- [26. data/generators/generate_ECG.py](#sec-26)
- [27. data/generators/generate_EMNIST.py](#sec-27)
- [28. data/generators/generate_FGVCAircraft.py](#sec-28)
- [29. data/generators/generate_FashionMNIST.py](#sec-29)
- [30. data/generators/generate_Flowers102.py](#sec-30)
- [31. data/generators/generate_GTSRB.py](#sec-31)
- [32. data/generators/generate_HAR.py](#sec-32)
- [33. data/generators/generate_MNIST.py](#sec-33)
- [34. data/generators/generate_Omniglot.py](#sec-34)
- [35. data/generators/generate_OxfordPets.py](#sec-35)
- [36. data/generators/generate_PAMAP2.py](#sec-36)
- [37. data/generators/generate_Shakespeare.py](#sec-37)
- [38. data/generators/generate_SogouNews.py](#sec-38)
- [39. data/generators/generate_StanfordCars.py](#sec-39)
- [40. data/generators/generate_TinyImagenet.py](#sec-40)
- [41. models/flops.py](#sec-41)

## Árvore de dependências

<a id="sec-1"></a>

## 1. [execution/server.py](../src/fl_serverpuma/execution/server.py)

**Funções/métodos:** `main`.

Ponto de entrada do servidor: lê a CLI, seleciona dispositivo e inicia os experimentos.

- <a id="sec-1-1"></a>**1.1** [execution/server_cli.py](../src/fl_serverpuma/execution/server_cli.py) — Define os argumentos e valores padrão da CLI do servidor. **[sem imports internos]**

    **Funções/métodos:** `parse_args`.

- <a id="sec-1-2"></a>**1.2** [execution/round_coordinator.py](../src/fl_serverpuma/execution/round_coordinator.py) — Coordena conexões e rodadas, dispara trocas por cliente, chama agregação, avaliação e persistência e mantém a sequência do experimento.

    **Funções/métodos:** `run_server`.

    - <a id="sec-1-2-1"></a>**1.2.1** [models/clip/sora_pruning.py](../src/fl_serverpuma/models/clip/sora_pruning.py) — Compacta o rank SoRA, converte para LoRA, ajusta congelamento e extrai parâmetros treináveis. Família CLIP.

        **Funções/métodos:** `prune_sora_to_lora_and_report`, `prune_sora_to_lora_and_report.PrunedLoRA.__init__`, `prune_sora_to_lora_and_report.PrunedLoRA.forward`, `prune_sora_iterative`, `pre_prune_whole_model`, `re_freeze_vision_model`, `get_trainable_state_dict`.

        - <a id="sec-1-2-1-1"></a>**1.2.1.1** [models/clip/sora_layers.py](../src/fl_serverpuma/models/clip/sora_layers.py) — Define as camadas SoRALinear e SoRAWrappedLinear e sua propagação direta. Família CLIP. **[sem imports internos]**

            **Funções/métodos:** `SoRALinear.__init__`, `SoRALinear.forward`, `SoRAWrappedLinear.__init__`, `SoRAWrappedLinear.forward`.

    - <a id="sec-1-2-2"></a>**1.2.2** [algorithms/fl_math.py](../src/fl_serverpuma/algorithms/fl_math.py) — Reúne quantização e desquantização, avaliação de modelos, cópia de parâmetros e ajuste das dimensões aos estados podados. **[sem imports internos]**

        **Funções/métodos:** `dequantization`, `quantization`, `evaluate_model`, `set_parameters`, `resize_model_to_pruned`.

    - <a id="sec-1-2-3"></a>**1.2.3** [models/clip/clip_sora_utils.py](../src/fl_serverpuma/models/clip/clip_sora_utils.py) — Reduz o rank de matrizes SoRA no state_dict de CLIP, preservando a lógica específica dessa família. **[sem imports internos]**

        **Funções/métodos:** `reduce_sora_state_dict_rank`.

    - <a id="sec-1-2-4"></a>**1.2.4** [models/slm/sora_pruning.py](../src/fl_serverpuma/models/slm/sora_pruning.py) — Compacta o rank SoRA, converte para LoRA, ajusta congelamento e extrai parâmetros treináveis. Família SLM.

        **Funções/métodos:** `prune_sora_to_lora_and_report`, `prune_sora_to_lora_and_report.PrunedLoRA.__init__`, `prune_sora_to_lora_and_report.PrunedLoRA.forward`, `prune_sora_iterative`, `pre_prune_whole_model`, `re_freeze_vision_model`, `get_trainable_state_dict`.

        - <a id="sec-1-2-4-1"></a>**1.2.4.1** [models/slm/sora_layers.py](../src/fl_serverpuma/models/slm/sora_layers.py) — Define as camadas SoRALinear e SoRAWrappedLinear e sua propagação direta. Família SLM. **[sem imports internos]**

            **Funções/métodos:** `SoRALinear.__init__`, `SoRALinear.forward`, `SoRAWrappedLinear.__init__`, `SoRAWrappedLinear.forward`.

    - <a id="sec-1-2-5"></a>**1.2.5** [models/size.py](../src/fl_serverpuma/models/size.py) — Mede tamanho total/treinável, quantidade de parâmetros e tamanho de tensores de um state_dict. **[sem imports internos]**

        **Funções/métodos:** `get_model_size`, `get_trainable_size_and_params`, `get_state_dict_size_mb`.

    - <a id="sec-1-2-6"></a>**1.2.6** [communication/tcp.py](../src/fl_serverpuma/communication/tcp.py) — Envia e recebe mensagens serializadas com pickle, com cabeçalho de tamanho, leitura completa, timeout e medição de bytes/taxa. **[sem imports internos]**

        **Funções/métodos:** `send_data`, `recvall`, `recv_data`.

    - <a id="sec-1-2-7"></a>**1.2.7** [algorithms/aggregation.py](../src/fl_serverpuma/algorithms/aggregation.py) — Agrega estados parciais com pesos por cliente; recebe dispositivo e máscaras explicitamente, sem depender do estado completo do servidor.

        **Funções/métodos:** `aggregate_models`.

        - <a id="sec-1-2-7-1"></a>**1.2.7.1** [algorithms/alignment.py](../src/fl_serverpuma/algorithms/alignment.py) — Alinha tensores locais/podados às dimensões globais e constrói máscaras de cobertura para agregação. **[sem imports internos]**

            **Funções/métodos:** `_are_compatible_with_pruning`, `_align_to_global_structure`, `_align_with_mask`, `_get_mask`, `_align_conv_weights`, `_align_linear_weights`, `_align_bias`.

    - <a id="sec-1-2-8"></a>**1.2.8** [algorithms/adaptation/latency.py](../src/fl_serverpuma/algorithms/adaptation/latency.py) — Calcula a meta de latência e o timeout amortecido a partir das medições dos clientes. **[sem imports internos]**

        **Funções/métodos:** `set_threshold`.

    - <a id="sec-1-2-9"></a>**1.2.9** [communication/server_exchange.py](../src/fl_serverpuma/communication/server_exchange.py) — Executa a troca de uma rodada com um cliente: prepara o estado a enviar, aplica as políticas e recebe atualização e métricas sob os locks existentes.

        **Funções/métodos:** `handle_client`.

        - <a id="sec-1-2-9-1"></a>**1.2.9.1** [models/clip/sora_pruning.py](../src/fl_serverpuma/models/clip/sora_pruning.py) — dependência já detalhada em [1.2.1](#sec-1-2-1).

        - <a id="sec-1-2-9-2"></a>**1.2.9.2** [models/slm/sora_pruning.py](../src/fl_serverpuma/models/slm/sora_pruning.py) — dependência já detalhada em [1.2.4](#sec-1-2-4).

        - <a id="sec-1-2-9-3"></a>**1.2.9.3** [algorithms/pruning/structural.py](../src/fl_serverpuma/algorithms/pruning/structural.py) — Aplica poda e reestruturação do modelo; mantém funções de restauração de tamanho por máscaras. **[sem imports internos]**

            **Funções/métodos:** `prune_and_restructure`, `restore_to_original_size`.

        - <a id="sec-1-2-9-4"></a>**1.2.9.4** [algorithms/pruning/nisp.py](../src/fl_serverpuma/algorithms/pruning/nisp.py) — Calcula importância de neurônios a partir de ativações e aplica a poda da camada totalmente conectada. **[sem imports internos]**

            **Funções/métodos:** `get_frl_output`, `get_frl_output.hook`, `calculate_neuron_importance`, `propagate_importance_fc`, `prune_fc1`.

        - <a id="sec-1-2-9-5"></a>**1.2.9.5** [algorithms/pruning/snip.py](../src/fl_serverpuma/algorithms/pruning/snip.py) — Calcula máscaras de poda SNIP com gradientes e aplica as máscaras ao modelo. **[sem imports internos]**

            **Funções/métodos:** `snip_pruning`, `apply_mask`.

        - <a id="sec-1-2-9-6"></a>**1.2.9.6** [communication/tcp.py](../src/fl_serverpuma/communication/tcp.py) — dependência já detalhada em [1.2.6](#sec-1-2-6).

        - <a id="sec-1-2-9-7"></a>**1.2.9.7** [algorithms/fl_math.py](../src/fl_serverpuma/algorithms/fl_math.py) — dependência já detalhada em [1.2.2](#sec-1-2-2).

        - <a id="sec-1-2-9-8"></a>**1.2.9.8** [communication/lhdq.py](../src/fl_serverpuma/communication/lhdq.py) — Comprime e reconstrói deltas entre estados do modelo usando quantização em três níveis e codificação Huffman; identifica mensagens nesse formato. **[sem imports internos]**

            **Funções/métodos:** `_huffman_pack`, `_huffman_unpack`, `_quantize_delta_3level`, `_compute_p_delta`, `_dequantize_3level`, `lhdq_encode`, `lhdq_decode`, `is_lhdq_encoded`.

        - <a id="sec-1-2-9-9"></a>**1.2.9.9** [algorithms/adaptation/pruning.py](../src/fl_serverpuma/algorithms/adaptation/pruning.py) — Calcula a taxa de poda usando custo computacional, comunicação e histórico de tempo; inclui fallback. **[sem imports internos]**

            **Funções/métodos:** `new_set_amount_prune`, `_fallback_prune_calculation`, `_adjust_pruning_with_history`, `set_amount_prune`.

        - <a id="sec-1-2-9-10"></a>**1.2.9.10** [algorithms/adaptation/paca.py](../src/fl_serverpuma/algorithms/adaptation/paca.py) — Decide PaCA por latência, perda e acurácia; aplica cooldown, limites e reversão, atualizando o histórico da política. **[sem imports internos]**

            **Funções/métodos:** `calculate_adaptive_paca`, `_check_accuracy_signal`, `_is_excluded_by_paca`.

        - <a id="sec-1-2-9-11"></a>**1.2.9.11** [algorithms/adaptation/rank.py](../src/fl_serverpuma/algorithms/adaptation/rank.py) — Decide rank adaptativo por latência, perda e acurácia, preservando limites e histórico. **[sem imports internos]**

            **Funções/métodos:** `calculate_adaptive_rank`, `_check_accuracy_signal_rank`.

        - <a id="sec-1-2-9-12"></a>**1.2.9.12** [data/server_data.py](../src/fl_serverpuma/data/server_data.py) — Carrega e prepara os dados de teste globais, incluindo tokenização para SLM.

            **Funções/métodos:** `_prepare_slm_test_loader`, `load_test_data`.

            - <a id="sec-1-2-9-12-1"></a>**1.2.9.12.1** [data/client_data.py](../src/fl_serverpuma/data/client_data.py) — Lê partições locais de treino/teste dos clientes, incluindo tratamento específico de texto e Shakespeare. **[sem imports internos]**

                **Funções/métodos:** `read_data`, `read_client_data`, `read_client_data_text`, `read_client_data_Shakespeare`.

            - <a id="sec-1-2-9-12-2"></a>**1.2.9.12.2** [models/slm/data.py](../src/fl_serverpuma/models/slm/data.py) **[import local/condicional]** — Define collator, codificação de rótulos e construção dos DataLoaders. Família SLM. **[sem imports internos]**

                **Funções/métodos:** `CustomCollator.__init__`, `CustomCollator.__call__`, `resolve_label_metadata`, `encode_label`, `build_dataloaders`.

        - <a id="sec-1-2-9-13"></a>**1.2.9.13** [models/size.py](../src/fl_serverpuma/models/size.py) — dependência já detalhada em [1.2.5](#sec-1-2-5).

    - <a id="sec-1-2-10"></a>**1.2.10** [data/server_data.py](../src/fl_serverpuma/data/server_data.py) — dependência já detalhada em [1.2.9.12](#sec-1-2-9-12).

    - <a id="sec-1-2-11"></a>**1.2.11** [persistence/server_results.py](../src/fl_serverpuma/persistence/server_results.py) — Grava métricas e metadados do servidor em HDF5 com os nomes e caminhos originais.

        **Funções/métodos:** `save_results`.

        - <a id="sec-1-2-11-1"></a>**1.2.11.1** [paths.py](../src/fl_serverpuma/paths.py) — Define SOURCE_ROOT, usado para localizar datasets, resultados e recursos operacionais a partir do diretório src. **[sem imports internos]**

    - <a id="sec-1-2-12"></a>**1.2.12** [models/slm/slm_sora_utils.py](../src/fl_serverpuma/models/slm/slm_sora_utils.py) **[import local/condicional]** — Reduz o rank de matrizes SoRA no state_dict de SLM, preservando a lógica específica dessa família. **[sem imports internos]**

        **Funções/métodos:** `reduce_sora_state_dict_rank`.

- <a id="sec-1-3"></a>**1.3** [execution/server_state.py](../src/fl_serverpuma/execution/server_state.py) — Inicializa o modelo global e guarda configuração, conexões, locks, dados e históricos de uma execução em ServerState.

    **Funções/métodos:** `ServerState.__init__`.

    - <a id="sec-1-3-1"></a>**1.3.1** [models/clip/config.py](../src/fl_serverpuma/models/clip/config.py) — Lê YAML/CLI, resolve modos de execução e monta configurações e caminhos de saída. Família CLIP. **[sem imports internos]**

        **Funções/métodos:** `get_device`, `parse_args`, `load_config`, `resolve_run_modes`, `build_run_config`, `build_output_path`.

    - <a id="sec-1-3-2"></a>**1.3.2** [models/clip/model.py](../src/fl_serverpuma/models/clip/model.py) — Define o classificador e constrói o modelo com os adaptadores selecionados. Família CLIP.

        **Funções/métodos:** `CLIPForClassification.__init__`, `CLIPForClassification.forward`, `build_model`.

        - <a id="sec-1-3-2-1"></a>**1.3.2.1** [models/clip/adaptation.py](../src/fl_serverpuma/models/clip/adaptation.py) — Aplica SoRA e controla as camadas/gradientes habilitados por PaCA em tempo de execução. Família CLIP.

            **Funções/métodos:** `apply_sora`, `_get_encoder_layers`, `_apply_paca_gradient_boundary`, `_apply_paca_gradient_boundary.detach_hook`, `update_paca_runtime`.

            - <a id="sec-1-3-2-1-1"></a>**1.3.2.1.1** [models/clip/sora_layers.py](../src/fl_serverpuma/models/clip/sora_layers.py) — dependência já detalhada em [1.2.1.1](#sec-1-2-1-1).

        - <a id="sec-1-3-2-2"></a>**1.3.2.2** [models/clip/config.py](../src/fl_serverpuma/models/clip/config.py) — dependência já detalhada em [1.3.1](#sec-1-3-1).

    - <a id="sec-1-3-3"></a>**1.3.3** [algorithms/fl_math.py](../src/fl_serverpuma/algorithms/fl_math.py) — dependência já detalhada em [1.2.2](#sec-1-2-2).

    - <a id="sec-1-3-4"></a>**1.3.4** [models/slm/config.py](../src/fl_serverpuma/models/slm/config.py) — Lê YAML/CLI, resolve modos de execução e monta configurações e caminhos de saída. Família SLM. **[sem imports internos]**

        **Funções/métodos:** `get_device`, `parse_args`, `load_config`, `resolve_run_modes`, `build_run_config`, `build_output_path`.

    - <a id="sec-1-3-5"></a>**1.3.5** [models/slm/model.py](../src/fl_serverpuma/models/slm/model.py) — Define o classificador e constrói o modelo com os adaptadores selecionados. Família SLM.

        **Funções/métodos:** `SLMForClassification.__init__`, `SLMForClassification.forward`, `build_model`.

        - <a id="sec-1-3-5-1"></a>**1.3.5.1** [models/slm/adaptation.py](../src/fl_serverpuma/models/slm/adaptation.py) — Aplica SoRA e controla as camadas/gradientes habilitados por PaCA em tempo de execução. Família SLM.

            **Funções/métodos:** `apply_sora`, `_get_encoder_layers`, `_apply_paca_gradient_boundary`, `_apply_paca_gradient_boundary.detach_hook`, `update_paca_runtime`.

            - <a id="sec-1-3-5-1-1"></a>**1.3.5.1.1** [models/slm/sora_layers.py](../src/fl_serverpuma/models/slm/sora_layers.py) — dependência já detalhada em [1.2.4.1](#sec-1-2-4-1).

        - <a id="sec-1-3-5-2"></a>**1.3.5.2** [models/slm/config.py](../src/fl_serverpuma/models/slm/config.py) — dependência já detalhada em [1.3.4](#sec-1-3-4).

    - <a id="sec-1-3-6"></a>**1.3.6** [models/cnn.py](../src/fl_serverpuma/models/cnn.py) — Define SimpleModel: CNN com duas etapas convolucionais e camadas totalmente conectadas para classificação. **[sem imports internos]**

        **Funções/métodos:** `SimpleModel.__init__`, `SimpleModel.forward`.

    - <a id="sec-1-3-7"></a>**1.3.7** [models/complexity.py](../src/fl_serverpuma/models/complexity.py) — Calcula complexidade por camada e medidas SynExp por meio de LayerComplexityCalculator e SynExpSimple.

        **Funções/métodos:** `LayerComplexityCalculator.__init__`, `LayerComplexityCalculator.calculate_alpha_beta`, `LayerComplexityCalculator._calculate_parameters_per_layer`, `LayerComplexityCalculator._calculate_flops_per_layer`, `SynExpSimple.__init__`, `SynExpSimple.compute_densities`.

        - <a id="sec-1-3-7-1"></a>**1.3.7.1** [models/flops.py](../src/fl_serverpuma/models/flops.py) **[import local/condicional]** — Conta ou estima FLOPs de um modelo para determinada entrada. **[sem imports internos]**

            **Funções/métodos:** `count_flops`, `count_flops.conv2d_flops`, `count_flops.linear_flops`, `count_flops.attention_flops`, `count_flops.maxpool2d_flops`, `count_flops.hook_fn`, `count_flops.hook_fn.hook`, `estimate_flops`.

    - <a id="sec-1-3-8"></a>**1.3.8** [data/server_data.py](../src/fl_serverpuma/data/server_data.py) — dependência já detalhada em [1.2.9.12](#sec-1-2-9-12).

    - <a id="sec-1-3-9"></a>**1.3.9** [models/slm/data.py](../src/fl_serverpuma/models/slm/data.py) **[import local/condicional]** — dependência já detalhada em [1.2.9.12.2](#sec-1-2-9-12-2).

    - <a id="sec-1-3-10"></a>**1.3.10** [models/slm/adaptation.py](../src/fl_serverpuma/models/slm/adaptation.py) **[import local/condicional]** — dependência já detalhada em [1.3.5.1](#sec-1-3-5-1).

<a id="sec-2"></a>

## 2. [execution/client.py](../src/fl_serverpuma/execution/client.py)

**Funções/métodos:** `main`.

Ponto de entrada do cliente: cria a sessão e seleciona o fluxo centralizado ou P2P.

- <a id="sec-2-1"></a>**2.1** [execution/client_cli.py](../src/fl_serverpuma/execution/client_cli.py) — Define os argumentos e valores padrão da CLI do cliente. **[sem imports internos]**

    **Funções/métodos:** `parse_args`.

- <a id="sec-2-2"></a>**2.2** [execution/client_initialization.py](../src/fl_serverpuma/execution/client_initialization.py) — Seleciona dispositivo/PaCA, constrói modelo e dados, configura ALA e entrega a sessão pronta.

    **Funções/métodos:** `initialize_client`.

    - <a id="sec-2-2-1"></a>**2.2.1** [models/clip/config.py](../src/fl_serverpuma/models/clip/config.py) — dependência já detalhada em [1.3.1](#sec-1-3-1).

    - <a id="sec-2-2-2"></a>**2.2.2** [models/clip/model.py](../src/fl_serverpuma/models/clip/model.py) — dependência já detalhada em [1.3.2](#sec-1-3-2).

    - <a id="sec-2-2-3"></a>**2.2.3** [algorithms/personalization/ala.py](../src/fl_serverpuma/algorithms/personalization/ala.py) — Implementa ALA: adaptação local da combinação entre parâmetros globais e locais usando dados do cliente. **[sem imports internos]**

        **Funções/métodos:** `ALA.__init__`, `ALA.adaptive_local_aggregation`.

    - <a id="sec-2-2-4"></a>**2.2.4** [algorithms/personalization/ala_lora.py](../src/fl_serverpuma/algorithms/personalization/ala_lora.py) — Implementa a adaptação local ALA para modelos com adaptadores LoRA. **[sem imports internos]**

        **Funções/métodos:** `ALA_LoRA.__init__`, `ALA_LoRA.adaptive_local_aggregation`.

    - <a id="sec-2-2-5"></a>**2.2.5** [models/cnn.py](../src/fl_serverpuma/models/cnn.py) — dependência já detalhada em [1.3.6](#sec-1-3-6).

    - <a id="sec-2-2-6"></a>**2.2.6** [data/client_loaders.py](../src/fl_serverpuma/data/client_loaders.py) — Constrói os DataLoaders locais conforme dataset e família do modelo.

        **Funções/métodos:** `load_data`.

        - <a id="sec-2-2-6-1"></a>**2.2.6.1** [data/client_data.py](../src/fl_serverpuma/data/client_data.py) — dependência já detalhada em [1.2.9.12.1](#sec-1-2-9-12-1).

        - <a id="sec-2-2-6-2"></a>**2.2.6.2** [models/slm/data.py](../src/fl_serverpuma/models/slm/data.py) **[import local/condicional]** — dependência já detalhada em [1.2.9.12.2](#sec-1-2-9-12-2).

    - <a id="sec-2-2-7"></a>**2.2.7** [execution/client_session.py](../src/fl_serverpuma/execution/client_session.py) — Agrupa modelo, dados, configuração e listas de métricas de um cliente, sem copiar seus objetos. **[sem imports internos]**

    - <a id="sec-2-2-8"></a>**2.2.8** [models/slm/config.py](../src/fl_serverpuma/models/slm/config.py) **[import local/condicional]** — dependência já detalhada em [1.3.4](#sec-1-3-4).

    - <a id="sec-2-2-9"></a>**2.2.9** [models/slm/model.py](../src/fl_serverpuma/models/slm/model.py) **[import local/condicional]** — dependência já detalhada em [1.3.5](#sec-1-3-5).

- <a id="sec-2-3"></a>**2.3** [execution/centralized_client.py](../src/fl_serverpuma/execution/centralized_client.py) — Executa o ciclo de comunicação, treinamento e avaliação do cliente conectado ao servidor.

    **Funções/métodos:** `run_centralized`.

    - <a id="sec-2-3-1"></a>**2.3.1** [communication/tcp.py](../src/fl_serverpuma/communication/tcp.py) — dependência já detalhada em [1.2.6](#sec-1-2-6).

    - <a id="sec-2-3-2"></a>**2.3.2** [algorithms/fl_math.py](../src/fl_serverpuma/algorithms/fl_math.py) — dependência já detalhada em [1.2.2](#sec-1-2-2).

    - <a id="sec-2-3-3"></a>**2.3.3** [communication/lhdq.py](../src/fl_serverpuma/communication/lhdq.py) — dependência já detalhada em [1.2.9.8](#sec-1-2-9-8).

    - <a id="sec-2-3-4"></a>**2.3.4** [execution/local_training.py](../src/fl_serverpuma/execution/local_training.py) — Executa uma etapa local de treino e personalização e seleciona os pesos retornados ao servidor.

        **Funções/métodos:** `local_training`, `local_initialization`.

        - <a id="sec-2-3-4-1"></a>**2.3.4.1** [models/clip/optimization.py](../src/fl_serverpuma/models/clip/optimization.py) — Constrói otimizadores e scheduler conforme o modo de adaptação. Família CLIP.

            **Funções/métodos:** `build_optimizer`, `build_scheduler`.

            - <a id="sec-2-3-4-1-1"></a>**2.3.4.1.1** [models/clip/sora_optimization.py](../src/fl_serverpuma/models/clip/sora_optimization.py) — Implementa SparseAdamW e GateSparsifier para atualizar e esparsificar gates. Família CLIP. **[sem imports internos]**

                **Funções/métodos:** `SparseAdamW.__init__`, `SparseAdamW._build_lambda_list`, `SparseAdamW.step_lambda`, `SparseAdamW.step`, `GateSparsifier.__init__`, `GateSparsifier._build_lambda_list`, `GateSparsifier.step_lambda`, `GateSparsifier.zero_grad`, `GateSparsifier.step`.

            - <a id="sec-2-3-4-1-2"></a>**2.3.4.1.2** [models/clip/config.py](../src/fl_serverpuma/models/clip/config.py) — dependência já detalhada em [1.3.1](#sec-1-3-1).

        - <a id="sec-2-3-4-2"></a>**2.3.4.2** [models/clip/sora_pruning.py](../src/fl_serverpuma/models/clip/sora_pruning.py) — dependência já detalhada em [1.2.1](#sec-1-2-1).

        - <a id="sec-2-3-4-3"></a>**2.3.4.3** [algorithms/fl_math.py](../src/fl_serverpuma/algorithms/fl_math.py) — dependência já detalhada em [1.2.2](#sec-1-2-2).

        - <a id="sec-2-3-4-4"></a>**2.3.4.4** [models/state_mapping.py](../src/fl_serverpuma/models/state_mapping.py) — Mapeia chaves de estados sequenciais para os nomes de parâmetros de SimpleModel. **[sem imports internos]**

            **Funções/métodos:** `map_sequential_to_simplemodel`.

        - <a id="sec-2-3-4-5"></a>**2.3.4.5** [models/clip/training.py](../src/fl_serverpuma/models/clip/training.py) **[import local/condicional]** — Executa uma época de treino, mede esparsidade e avalia o classificador. Família CLIP. **[sem imports internos]**

            **Funções/métodos:** `train_epoch`, `compute_gate_sparsity`, `evaluate`.

        - <a id="sec-2-3-4-6"></a>**2.3.4.6** [models/clip/adaptation.py](../src/fl_serverpuma/models/clip/adaptation.py) **[import local/condicional]** — dependência já detalhada em [1.3.2.1](#sec-1-3-2-1).

        - <a id="sec-2-3-4-7"></a>**2.3.4.7** [models/slm/optimization.py](../src/fl_serverpuma/models/slm/optimization.py) **[import local/condicional]** — Constrói otimizadores e scheduler conforme o modo de adaptação. Família SLM.

            **Funções/métodos:** `build_optimizer`, `build_scheduler`.

            - <a id="sec-2-3-4-7-1"></a>**2.3.4.7.1** [models/slm/sora_optimization.py](../src/fl_serverpuma/models/slm/sora_optimization.py) — Implementa SparseAdamW e GateSparsifier para atualizar e esparsificar gates. Família SLM. **[sem imports internos]**

                **Funções/métodos:** `SparseAdamW.__init__`, `SparseAdamW._build_lambda_list`, `SparseAdamW.step_lambda`, `SparseAdamW.step`, `GateSparsifier.__init__`, `GateSparsifier._build_lambda_list`, `GateSparsifier.step_lambda`, `GateSparsifier.zero_grad`, `GateSparsifier.step`.

            - <a id="sec-2-3-4-7-2"></a>**2.3.4.7.2** [models/slm/config.py](../src/fl_serverpuma/models/slm/config.py) — dependência já detalhada em [1.3.4](#sec-1-3-4).

        - <a id="sec-2-3-4-8"></a>**2.3.4.8** [models/slm/training.py](../src/fl_serverpuma/models/slm/training.py) **[import local/condicional]** — Executa uma época de treino, mede esparsidade e avalia o classificador. Família SLM. **[sem imports internos]**

            **Funções/métodos:** `train_epoch`, `compute_gate_sparsity`, `evaluate`.

        - <a id="sec-2-3-4-9"></a>**2.3.4.9** [models/slm/adaptation.py](../src/fl_serverpuma/models/slm/adaptation.py) **[import local/condicional]** — dependência já detalhada em [1.3.5.1](#sec-1-3-5-1).

        - <a id="sec-2-3-4-10"></a>**2.3.4.10** [models/slm/sora_pruning.py](../src/fl_serverpuma/models/slm/sora_pruning.py) **[import local/condicional]** — dependência já detalhada em [1.2.4](#sec-1-2-4).

    - <a id="sec-2-3-5"></a>**2.3.5** [persistence/client_results.py](../src/fl_serverpuma/persistence/client_results.py) — Grava métricas locais/globais e opcionais de P2P em HDF5, preservando nomes de arquivos e atributos.

        **Funções/métodos:** `save_results`.

        - <a id="sec-2-3-5-1"></a>**2.3.5.1** [paths.py](../src/fl_serverpuma/paths.py) — dependência já detalhada em [1.2.11.1](#sec-1-2-11-1).

- <a id="sec-2-4"></a>**2.4** [execution/peer_client.py](../src/fl_serverpuma/execution/peer_client.py) — Executa treinamento descentralizado, listener P2P e reação aos sinais de encontros, mantendo a sincronização original.

    **Funções/métodos:** `run_decentralized`, `run_decentralized.wait_for_round_signal`, `run_decentralized.wait_for_encounter`, `run_decentralized.p2p_server_thread`.

    - <a id="sec-2-4-1"></a>**2.4.1** [paths.py](../src/fl_serverpuma/paths.py) — dependência já detalhada em [1.2.11.1](#sec-1-2-11-1).

    - <a id="sec-2-4-2"></a>**2.4.2** [models/clip/sora_pruning.py](../src/fl_serverpuma/models/clip/sora_pruning.py) — dependência já detalhada em [1.2.1](#sec-1-2-1).

    - <a id="sec-2-4-3"></a>**2.4.3** [communication/tcp.py](../src/fl_serverpuma/communication/tcp.py) — dependência já detalhada em [1.2.6](#sec-1-2-6).

    - <a id="sec-2-4-4"></a>**2.4.4** [algorithms/fl_math.py](../src/fl_serverpuma/algorithms/fl_math.py) — dependência já detalhada em [1.2.2](#sec-1-2-2).

    - <a id="sec-2-4-5"></a>**2.4.5** [execution/local_training.py](../src/fl_serverpuma/execution/local_training.py) — dependência já detalhada em [2.3.4](#sec-2-3-4).

    - <a id="sec-2-4-6"></a>**2.4.6** [persistence/client_results.py](../src/fl_serverpuma/persistence/client_results.py) — dependência já detalhada em [2.3.5](#sec-2-3-5).

    - <a id="sec-2-4-7"></a>**2.4.7** [models/slm/sora_pruning.py](../src/fl_serverpuma/models/slm/sora_pruning.py) **[import local/condicional]** — dependência já detalhada em [1.2.4](#sec-1-2-4).

<a id="sec-3"></a>

## 3. [models/clip/main.py](../src/fl_serverpuma/models/clip/main.py)

**Funções/métodos:** `main`.

Executa o pipeline independente de CLIP: configuração, dados, construção, treinamento, poda, avaliação e exportação.

- <a id="sec-3-1"></a>**3.1** [models/clip/config.py](../src/fl_serverpuma/models/clip/config.py) — dependência já detalhada em [1.3.1](#sec-1-3-1).

- <a id="sec-3-2"></a>**3.2** [models/clip/data.py](../src/fl_serverpuma/models/clip/data.py) — Define collator, codificação de rótulos e construção dos DataLoaders. Família CLIP. **[sem imports internos]**

    **Funções/métodos:** `CustomCollator.__init__`, `CustomCollator.__call__`, `resolve_label_metadata`, `encode_label`, `build_dataloaders`.

- <a id="sec-3-3"></a>**3.3** [models/clip/model.py](../src/fl_serverpuma/models/clip/model.py) — dependência já detalhada em [1.3.2](#sec-1-3-2).

- <a id="sec-3-4"></a>**3.4** [models/clip/optimization.py](../src/fl_serverpuma/models/clip/optimization.py) — dependência já detalhada em [2.3.4.1](#sec-2-3-4-1).

- <a id="sec-3-5"></a>**3.5** [models/clip/measurement.py](../src/fl_serverpuma/models/clip/measurement.py) — Executa benchmark de inferência e quantiza pesos para exportação. Família CLIP. **[sem imports internos]**

    **Funções/métodos:** `benchmark_attention`, `quantize_weights`.

- <a id="sec-3-6"></a>**3.6** [models/clip/trainer.py](../src/fl_serverpuma/models/clip/trainer.py) — Coordena épocas e fases de treinamento de CLIP, impressão de métricas, benchmark de inferência e finalização.

    **Funções/métodos:** `ModelTrainer.__init__`, `ModelTrainer.print_metrics`, `ModelTrainer.execute_epochs`, `ModelTrainer.benchmark_inference`, `ModelTrainer.finalize`.

    - <a id="sec-3-6-1"></a>**3.6.1** [models/clip/training.py](../src/fl_serverpuma/models/clip/training.py) — dependência já detalhada em [2.3.4.5](#sec-2-3-4-5).

    - <a id="sec-3-6-2"></a>**3.6.2** [models/clip/sora_pruning.py](../src/fl_serverpuma/models/clip/sora_pruning.py) — dependência já detalhada em [1.2.1](#sec-1-2-1).

- <a id="sec-3-7"></a>**3.7** [models/clip/sora_pruning.py](../src/fl_serverpuma/models/clip/sora_pruning.py) — dependência já detalhada em [1.2.1](#sec-1-2-1).

<a id="sec-4"></a>

## 4. [models/slm/main.py](../src/fl_serverpuma/models/slm/main.py)

**Funções/métodos:** `main`.

Executa o pipeline independente de SLM: configuração, dados, construção, treinamento, poda, avaliação e exportação.

- <a id="sec-4-1"></a>**4.1** [models/slm/config.py](../src/fl_serverpuma/models/slm/config.py) — dependência já detalhada em [1.3.4](#sec-1-3-4).

- <a id="sec-4-2"></a>**4.2** [models/slm/data.py](../src/fl_serverpuma/models/slm/data.py) — dependência já detalhada em [1.2.9.12.2](#sec-1-2-9-12-2).

- <a id="sec-4-3"></a>**4.3** [models/slm/model.py](../src/fl_serverpuma/models/slm/model.py) — dependência já detalhada em [1.3.5](#sec-1-3-5).

- <a id="sec-4-4"></a>**4.4** [models/slm/optimization.py](../src/fl_serverpuma/models/slm/optimization.py) — dependência já detalhada em [2.3.4.7](#sec-2-3-4-7).

- <a id="sec-4-5"></a>**4.5** [models/slm/measurement.py](../src/fl_serverpuma/models/slm/measurement.py) — Quantiza pesos para exportação; benchmark_attention permanece sem implementação, como na versão anterior. Família SLM. **[sem imports internos]**

    **Funções/métodos:** `benchmark_attention`, `quantize_weights`.

- <a id="sec-4-6"></a>**4.6** [models/slm/trainer.py](../src/fl_serverpuma/models/slm/trainer.py) — Coordena épocas e fases de treinamento de SLM, impressão de métricas, benchmark de inferência e finalização.

    **Funções/métodos:** `ModelTrainer.__init__`, `ModelTrainer.print_metrics`, `ModelTrainer.execute_epochs`, `ModelTrainer.benchmark_inference`, `ModelTrainer.finalize`.

    - <a id="sec-4-6-1"></a>**4.6.1** [models/slm/training.py](../src/fl_serverpuma/models/slm/training.py) — dependência já detalhada em [2.3.4.8](#sec-2-3-4-8).

    - <a id="sec-4-6-2"></a>**4.6.2** [models/slm/sora_pruning.py](../src/fl_serverpuma/models/slm/sora_pruning.py) — dependência já detalhada em [1.2.4](#sec-1-2-4).

- <a id="sec-4-7"></a>**4.7** [models/slm/sora_pruning.py](../src/fl_serverpuma/models/slm/sora_pruning.py) — dependência já detalhada em [1.2.4](#sec-1-2-4).

<a id="sec-5"></a>

## 5. [simulation/sumo/orchestrator.py](../src/fl_serverpuma/simulation/sumo/orchestrator.py)

**Funções/métodos:** `main`.

CLI do orquestrador SUMO; valida opções e inicia o loop de simulação.

- <a id="sec-5-1"></a>**5.1** [simulation/sumo/simulation_loop.py](../src/fl_serverpuma/simulation/sumo/simulation_loop.py) — Avança a simulação TraCI e coordena encontros, sincronização, contêineres e registros de eventos.

    **Funções/métodos:** `run_orchestrator`.

    - <a id="sec-5-1-1"></a>**5.1.1** [simulation/sumo/compose.py](../src/fl_serverpuma/simulation/sumo/compose.py) — Gera o Docker Compose dos clientes V2V. Quando executado diretamente, apenas orienta usar o orquestrador.

        **Funções/métodos:** `generate`.

        - <a id="sec-5-1-1-1"></a>**5.1.1.1** [paths.py](../src/fl_serverpuma/paths.py) — dependência já detalhada em [1.2.11.1](#sec-1-2-11-1).

    - <a id="sec-5-1-2"></a>**5.1.2** [simulation/sumo/mobility.py](../src/fl_serverpuma/simulation/sumo/mobility.py) — Define VehicleState/V2VEvent e calcula distâncias, tempo de contato e agrupamentos de veículos.

        **Funções/métodos:** `euclidean_distance`, `estimate_contact_time`, `cluster_etc`, `find_clusters`.

        - <a id="sec-5-1-2-1"></a>**5.1.2.1** [simulation/sumo/settings.py](../src/fl_serverpuma/simulation/sumo/settings.py) — Centraliza caminhos, constantes e configuração de log da orquestração V2V.

            - <a id="sec-5-1-2-1-1"></a>**5.1.2.1.1** [paths.py](../src/fl_serverpuma/paths.py) — dependência já detalhada em [1.2.11.1](#sec-1-2-11-1).

    - <a id="sec-5-1-3"></a>**5.1.3** [simulation/sumo/signals.py](../src/fl_serverpuma/simulation/sumo/signals.py) — Escreve sinais atômicos de encontros e aguarda marcadores de conclusão de épocas dos clientes.

        **Funções/métodos:** `write_encounter_signal`, `wait_all_clients_epoch_done`.

        - <a id="sec-5-1-3-1"></a>**5.1.3.1** [simulation/sumo/settings.py](../src/fl_serverpuma/simulation/sumo/settings.py) — dependência já detalhada em [5.1.2.1](#sec-5-1-2-1).

    - <a id="sec-5-1-4"></a>**5.1.4** [simulation/sumo/containers.py](../src/fl_serverpuma/simulation/sumo/containers.py) — Inicia e encerra os contêineres do experimento por Docker Compose.

        **Funções/métodos:** `docker_compose_up`, `docker_compose_down`.

        - <a id="sec-5-1-4-1"></a>**5.1.4.1** [simulation/sumo/settings.py](../src/fl_serverpuma/simulation/sumo/settings.py) — dependência já detalhada em [5.1.2.1](#sec-5-1-2-1).

    - <a id="sec-5-1-5"></a>**5.1.5** [simulation/sumo/settings.py](../src/fl_serverpuma/simulation/sumo/settings.py) — dependência já detalhada em [5.1.2.1](#sec-5-1-2-1).

- <a id="sec-5-2"></a>**5.2** [simulation/sumo/settings.py](../src/fl_serverpuma/simulation/sumo/settings.py) — dependência já detalhada em [5.1.2.1](#sec-5-1-2-1).

<a id="sec-6"></a>

## 6. [simulation/sumo/create_trips.py](../src/fl_serverpuma/simulation/sumo/create_trips.py)

**Funções/métodos:** `main`.

CLI de geração de viagens; seleciona rede/opções e coordena geração e gravação das rotas.

- <a id="sec-6-1"></a>**6.1** [simulation/sumo/network.py](../src/fl_serverpuma/simulation/sumo/network.py) — Lê a rede SUMO e extrai arestas/conexões e seus metadados.

    **Funções/métodos:** `parse_network`.

    - <a id="sec-6-1-1"></a>**6.1.1** [simulation/sumo/route_settings.py](../src/fl_serverpuma/simulation/sumo/route_settings.py) — Centraliza os padrões e caminhos de geração de rotas. **[sem imports internos]**

- <a id="sec-6-2"></a>**6.2** [simulation/sumo/routing.py](../src/fl_serverpuma/simulation/sumo/routing.py) — Gera trajetos válidos e estima os tempos de percurso/simulação.

    **Funções/métodos:** `generate_route`, `_edge_traverse_time`, `estimate_simulation_time`.

    - <a id="sec-6-2-1"></a>**6.2.1** [simulation/sumo/route_settings.py](../src/fl_serverpuma/simulation/sumo/route_settings.py) — dependência já detalhada em [6.1.1](#sec-6-1-1).

- <a id="sec-6-3"></a>**6.3** [simulation/sumo/route_files.py](../src/fl_serverpuma/simulation/sumo/route_files.py) — Grava rotas XML e atualiza o arquivo de configuração SUMO.

    **Funções/métodos:** `generate_routes_xml`, `update_sumocfg`.

    - <a id="sec-6-3-1"></a>**6.3.1** [simulation/sumo/routing.py](../src/fl_serverpuma/simulation/sumo/routing.py) — dependência já detalhada em [6.2](#sec-6-2).

    - <a id="sec-6-3-2"></a>**6.3.2** [simulation/sumo/route_settings.py](../src/fl_serverpuma/simulation/sumo/route_settings.py) — dependência já detalhada em [6.1.1](#sec-6-1-1).

- <a id="sec-6-4"></a>**6.4** [simulation/sumo/route_settings.py](../src/fl_serverpuma/simulation/sumo/route_settings.py) — dependência já detalhada em [6.1.1](#sec-6-1-1).

<a id="sec-7"></a>

## 7. [simulation/docker/compose.py](../src/fl_serverpuma/simulation/docker/compose.py)

**Funções/métodos:** `generate_docker_compose`.

Gera o Docker Compose do experimento centralizado, com servidor, clientes, parâmetros, rede e volumes.

- <a id="sec-7-1"></a>**7.1** [paths.py](../src/fl_serverpuma/paths.py) — dependência já detalhada em [1.2.11.1](#sec-1-2-11-1).

<a id="sec-8"></a>

## 8. [experiments/analysis/plot_results.py](../src/fl_serverpuma/experiments/analysis/plot_results.py)

**Funções/métodos:** `main`.

Seleciona experimentos, carrega resultados e coordena a geração dos gráficos.

- <a id="sec-8-1"></a>**8.1** [experiments/analysis/result_loading.py](../src/fl_serverpuma/experiments/analysis/result_loading.py) — Interpreta nomes de experimentos, lê HDF5 de servidor/clientes e prepara séries de comunicação e tempos.

    **Funções/métodos:** `parse_experiment`, `_load_server_h5`, `_load_client_h5`, `_compute_mb_per_round`, `_get_train_time_per_round`.

    - <a id="sec-8-1-1"></a>**8.1.1** [experiments/analysis/plot_style.py](../src/fl_serverpuma/experiments/analysis/plot_style.py) — Centraliza estilos, opções de apresentação, seleção de experimentos e salvamento das figuras.

        **Funções/métodos:** `_mark_every`, `_set_title`, `_save_and_close`.

        - <a id="sec-8-1-1-1"></a>**8.1.1.1** [paths.py](../src/fl_serverpuma/paths.py) — dependência já detalhada em [1.2.11.1](#sec-1-2-11-1).

- <a id="sec-8-2"></a>**8.2** [experiments/analysis/accuracy_plots.py](../src/fl_serverpuma/experiments/analysis/accuracy_plots.py) — Gera gráficos de acurácia global, perda de treinamento e comparação entre acurácia local de teste e de treino.

    **Funções/métodos:** `plot_01_acuracia_global`, `plot_02_train_loss`, `plot_03_local_vs_treino`.

    - <a id="sec-8-2-1"></a>**8.2.1** [experiments/analysis/plot_style.py](../src/fl_serverpuma/experiments/analysis/plot_style.py) — dependência já detalhada em [8.1.1](#sec-8-1-1).

- <a id="sec-8-3"></a>**8.3** [experiments/analysis/communication_plots.py](../src/fl_serverpuma/experiments/analysis/communication_plots.py) — Gera gráficos do volume de dados transferido por rodada, acumulado e em função do tempo.

    **Funções/métodos:** `plot_04_banda_acumulada`, `plot_05_banda_por_rodada`, `plot_15_mb_vs_tempo`.

    - <a id="sec-8-3-1"></a>**8.3.1** [experiments/analysis/plot_style.py](../src/fl_serverpuma/experiments/analysis/plot_style.py) — dependência já detalhada em [8.1.1](#sec-8-1-1).

    - <a id="sec-8-3-2"></a>**8.3.2** [experiments/analysis/result_loading.py](../src/fl_serverpuma/experiments/analysis/result_loading.py) — dependência já detalhada em [8.1](#sec-8-1).

- <a id="sec-8-4"></a>**8.4** [experiments/analysis/model_plots.py](../src/fl_serverpuma/experiments/analysis/model_plots.py) — Gera gráficos do tamanho dos adaptadores, quantidade de parâmetros treináveis e evolução do PaCA.

    **Funções/métodos:** `plot_06_tamanho_modelo`, `plot_07_parametros_treinaveis`, `plot_08_paca_evolucao`.

    - <a id="sec-8-4-1"></a>**8.4.1** [experiments/analysis/plot_style.py](../src/fl_serverpuma/experiments/analysis/plot_style.py) — dependência já detalhada em [8.1.1](#sec-8-1-1).

- <a id="sec-8-5"></a>**8.5** [experiments/analysis/timing_plots.py](../src/fl_serverpuma/experiments/analysis/timing_plots.py) — Gera gráficos de duração das rodadas, treino, comunicação, processamento e avaliação.

    **Funções/métodos:** `plot_09_tempo_por_rodada`, `plot_10_tempo_treinamento_clientes`, `plot_11_tempo_comunicacao_clientes`, `plot_12_decomposicao_temporal`.

    - <a id="sec-8-5-1"></a>**8.5.1** [experiments/analysis/plot_style.py](../src/fl_serverpuma/experiments/analysis/plot_style.py) — dependência já detalhada em [8.1.1](#sec-8-1-1).

- <a id="sec-8-6"></a>**8.6** [experiments/analysis/efficiency_plots.py](../src/fl_serverpuma/experiments/analysis/efficiency_plots.py) — Relaciona acurácia, comunicação e tempo para visualizar eficiência.

    **Funções/métodos:** `plot_13_acuracia_vs_tempo`, `plot_14_resumo_comparativo`.

    - <a id="sec-8-6-1"></a>**8.6.1** [experiments/analysis/plot_style.py](../src/fl_serverpuma/experiments/analysis/plot_style.py) — dependência já detalhada em [8.1.1](#sec-8-1-1).

    - <a id="sec-8-6-2"></a>**8.6.2** [experiments/analysis/result_loading.py](../src/fl_serverpuma/experiments/analysis/result_loading.py) — dependência já detalhada em [8.1](#sec-8-1).

- <a id="sec-8-7"></a>**8.7** [experiments/analysis/plot_style.py](../src/fl_serverpuma/experiments/analysis/plot_style.py) — dependência já detalhada em [8.1.1](#sec-8-1-1).

<a id="sec-9"></a>

## 9. [experiments/analysis/analise_estrategias.py](../src/fl_serverpuma/experiments/analysis/analise_estrategias.py)

**Funções/métodos:** `main`.

CLI e apresentação dos resumos comparativos entre estratégias.

- <a id="sec-9-1"></a>**9.1** [paths.py](../src/fl_serverpuma/paths.py) — dependência já detalhada em [1.2.11.1](#sec-1-2-11-1).

- <a id="sec-9-2"></a>**9.2** [experiments/analysis/strategy_metrics.py](../src/fl_serverpuma/experiments/analysis/strategy_metrics.py) — Carrega resultados por execução, detecta convergência e calcula estatísticas e melhorias entre estratégias.

    **Funções/métodos:** `detect_convergence_round`, `load_single_h5`, `analyze_single_run`, `analyze_experiment`, `format_metric`, `compute_improvement`, `format_improvement`.

    - <a id="sec-9-2-1"></a>**9.2.1** [experiments/analysis/strategy_settings.py](../src/fl_serverpuma/experiments/analysis/strategy_settings.py) — Define experimentos, janela de convergência e limiares da comparação. **[sem imports internos]**

- <a id="sec-9-3"></a>**9.3** [experiments/analysis/strategy_settings.py](../src/fl_serverpuma/experiments/analysis/strategy_settings.py) — dependência já detalhada em [9.2.1](#sec-9-2-1).

<a id="sec-10"></a>

## 10. [experiments/analysis/v2v_accuracy.py](../src/fl_serverpuma/experiments/analysis/v2v_accuracy.py)

**Funções/métodos:** `_mark_every`, `_set_title`, `_save_and_close`, `load_results_from_dir`, `plot_individual_accuracy`, `plot_post_p2p_accuracy`, `plot_comparison`, `main`.

Carrega resultados dos clientes V2V e produz gráficos de acurácia individual, pós-P2P e comparações.

- <a id="sec-10-1"></a>**10.1** [paths.py](../src/fl_serverpuma/paths.py) — dependência já detalhada em [1.2.11.1](#sec-1-2-11-1).

<a id="sec-11"></a>

## 11. [experiments/analysis/exchange_time.py](../src/fl_serverpuma/experiments/analysis/exchange_time.py)

**Funções/métodos:** `main`.

Lê eventos V2V e compara o tempo de contato estimado pelo SUMO com a duração real das trocas P2P em um gráfico.

- <a id="sec-11-1"></a>**11.1** [paths.py](../src/fl_serverpuma/paths.py) — dependência já detalhada em [1.2.11.1](#sec-1-2-11-1).

<a id="sec-12"></a>

## 12. [experiments/analysis/analyze_sora_vs_lora.py](../src/fl_serverpuma/experiments/analysis/analyze_sora_vs_lora.py)

**Funções/métodos:** `parse_args`, `load_acc_loss_params`.

Carrega métricas de acurácia, perda e parâmetros para comparar os resultados de SoRA e LoRA.

Sem imports internos explícitos para outros arquivos do projeto.

<a id="sec-13"></a>

## 13. [experiments/analysis/analise_tempo.py](../src/fl_serverpuma/experiments/analysis/analise_tempo.py)

**Funções/métodos:** `main`, `main.print_comparacao`.

Lê logs dos experimentos, extrai durações de rodadas e compara tempos médios de execução.

- <a id="sec-13-1"></a>**13.1** [paths.py](../src/fl_serverpuma/paths.py) — dependência já detalhada em [1.2.11.1](#sec-1-2-11-1).

<a id="sec-14"></a>

## 14. [experiments/analysis/result_utils.py](../src/fl_serverpuma/experiments/analysis/result_utils.py)

**Funções/métodos:** `average_data`, `get_all_results_for_one_algo`, `read_data_then_delete`.

Lê resultados HDF5 de múltiplas execuções e calcula estatísticas; read_data_then_delete pode excluir o arquivo lido quando solicitado.

Sem imports internos explícitos para outros arquivos do projeto.

<a id="sec-15"></a>

## 15. [experiments/analysis/get_mean_std.py](../src/fl_serverpuma/experiments/analysis/get_mean_std.py)

Lê da entrada padrão o nome de um arquivo .out e calcula média e desvio padrão das acurácias encontradas. Executa esse processamento ao importar ou executar o arquivo.

Sem imports internos explícitos para outros arquivos do projeto.

<a id="sec-16"></a>

## 16. [experiments/analysis/update_notebook.py](../src/fl_serverpuma/experiments/analysis/update_notebook.py)

Atualiza caminhos e padrões de nomes de resultados no notebook plot_results.ipynb do diretório de trabalho. A leitura e a gravação ocorrem ao importar ou executar este arquivo.

Sem imports internos explícitos para outros arquivos do projeto.

<a id="sec-17"></a>

## 17. [src/system/plot/flopsxpercentage.ipynb](../src/system/plot/flopsxpercentage.ipynb)

**Funções/métodos:** `dequantization (célula 3)`, `quantization (célula 3)`, `load_test_data (célula 4)`.

Compara FLOPs entre métodos e percentuais de poda, reunindo preparação, medições e gráficos.

- <a id="sec-17-1"></a>**17.1** [models/cnn.py](../src/fl_serverpuma/models/cnn.py) — dependência já detalhada em [1.3.6](#sec-1-3-6).

- <a id="sec-17-2"></a>**17.2** [data/client_data.py](../src/fl_serverpuma/data/client_data.py) — dependência já detalhada em [1.2.9.12.1](#sec-1-2-9-12-1).

- <a id="sec-17-3"></a>**17.3** [algorithms/pruning/structural.py](../src/fl_serverpuma/algorithms/pruning/structural.py) — dependência já detalhada em [1.2.9.3](#sec-1-2-9-3).

- <a id="sec-17-4"></a>**17.4** [models/size.py](../src/fl_serverpuma/models/size.py) — dependência já detalhada em [1.2.5](#sec-1-2-5).

- <a id="sec-17-5"></a>**17.5** [models/complexity.py](../src/fl_serverpuma/models/complexity.py) — dependência já detalhada em [1.3.7](#sec-1-3-7).

- <a id="sec-17-6"></a>**17.6** [algorithms/pruning/nisp.py](../src/fl_serverpuma/algorithms/pruning/nisp.py) — dependência já detalhada em [1.2.9.4](#sec-1-2-9-4).

- <a id="sec-17-7"></a>**17.7** [algorithms/pruning/snip.py](../src/fl_serverpuma/algorithms/pruning/snip.py) — dependência já detalhada em [1.2.9.5](#sec-1-2-9-5).

<a id="sec-18"></a>

## 18. [data/generators/generate_AGNews.py](../src/fl_serverpuma/data/generators/generate_AGNews.py)

**Funções/métodos:** `generate_dataset`.

Prepara notícias AGNews e gera partições por cliente conforme as opções e regras já existentes no gerador.

- <a id="sec-18-1"></a>**18.1** [data/partitioning.py](../src/fl_serverpuma/data/partitioning.py) — Verifica configurações existentes, distribui dados entre clientes, separa treino/teste e grava partições e metadados; usa Path nos caminhos de persistência. **[sem imports internos]**

    **Funções/métodos:** `check`, `separate_data`, `split_data`, `save_file`.

- <a id="sec-18-2"></a>**18.2** [data/language.py](../src/fl_serverpuma/data/language.py) — Converte caracteres, palavras e textos em representações numéricas; reúne tokenização, vocabulário e utilitários de embeddings. **[sem imports internos]**

    **Funções/métodos:** `letter_to_index`, `_one_hot`, `letter_to_vec`, `word_to_indices`, `split_line`, `_word_to_index`, `line_to_indices`, `bag_of_words`, `get_word_emb_arr`, `val_to_vec`, `tokenizer`.

<a id="sec-19"></a>

## 19. [data/generators/generate_AmazonReview.py](../src/fl_serverpuma/data/generators/generate_AmazonReview.py)

**Funções/métodos:** `load_amazon`, `generate_dataset`.

Prepara avaliações AmazonReview e gera partições por cliente conforme as opções e regras já existentes no gerador.

- <a id="sec-19-1"></a>**19.1** [data/partitioning.py](../src/fl_serverpuma/data/partitioning.py) — dependência já detalhada em [18.1](#sec-18-1).

<a id="sec-20"></a>

## 20. [data/generators/generate_Cifar10.py](../src/fl_serverpuma/data/generators/generate_Cifar10.py)

**Funções/métodos:** `generate_dataset`.

Prepara imagens CIFAR-10 e gera partições por cliente conforme as opções e regras já existentes no gerador.

- <a id="sec-20-1"></a>**20.1** [paths.py](../src/fl_serverpuma/paths.py) — dependência já detalhada em [1.2.11.1](#sec-1-2-11-1).

- <a id="sec-20-2"></a>**20.2** [data/partitioning.py](../src/fl_serverpuma/data/partitioning.py) — dependência já detalhada em [18.1](#sec-18-1).

<a id="sec-21"></a>

## 21. [data/generators/generate_Cifar100.py](../src/fl_serverpuma/data/generators/generate_Cifar100.py)

**Funções/métodos:** `generate_dataset`.

Prepara imagens CIFAR-100 e gera partições por cliente conforme as opções e regras já existentes no gerador.

- <a id="sec-21-1"></a>**21.1** [paths.py](../src/fl_serverpuma/paths.py) — dependência já detalhada em [1.2.11.1](#sec-1-2-11-1).

- <a id="sec-21-2"></a>**21.2** [data/partitioning.py](../src/fl_serverpuma/data/partitioning.py) — dependência já detalhada em [18.1](#sec-18-1).

<a id="sec-22"></a>

## 22. [data/generators/generate_Country211.py](../src/fl_serverpuma/data/generators/generate_Country211.py)

**Funções/métodos:** `generate_dataset`, `generate_dataset.load_data`.

Prepara imagens Country211 e gera partições por cliente conforme as opções e regras já existentes no gerador.

- <a id="sec-22-1"></a>**22.1** [data/partitioning.py](../src/fl_serverpuma/data/partitioning.py) — dependência já detalhada em [18.1](#sec-18-1).

<a id="sec-23"></a>

## 23. [data/generators/generate_DTD.py](../src/fl_serverpuma/data/generators/generate_DTD.py)

**Funções/métodos:** `generate_dataset`, `generate_dataset.load_data`.

Prepara texturas DTD e gera partições por cliente conforme as opções e regras já existentes no gerador.

- <a id="sec-23-1"></a>**23.1** [data/partitioning.py](../src/fl_serverpuma/data/partitioning.py) — dependência já detalhada em [18.1](#sec-18-1).

<a id="sec-24"></a>

## 24. [data/generators/generate_Digit5.py](../src/fl_serverpuma/data/generators/generate_Digit5.py)

**Funções/métodos:** `load_mnist`, `load_mnist_m`, `load_svhn`, `load_syn`, `load_usps`, `Digit5Dataset.__init__`, `Digit5Dataset.__getitem__`, `Digit5Dataset.__len__`, `digit5_dataset_read`, `generate_dataset`.

Prepara domínios de imagens de dígitos Digit5 e gera partições por cliente conforme as opções e regras já existentes no gerador.

- <a id="sec-24-1"></a>**24.1** [data/partitioning.py](../src/fl_serverpuma/data/partitioning.py) — dependência já detalhada em [18.1](#sec-18-1).

<a id="sec-25"></a>

## 25. [data/generators/generate_DomainNet.py](../src/fl_serverpuma/data/generators/generate_DomainNet.py)

**Funções/métodos:** `read_domainnet_data`, `DomainNet.__init__`, `DomainNet.__getitem__`, `DomainNet.__len__`, `get_domainnet_dloader`, `generate_dataset`.

Prepara imagens de múltiplos domínios DomainNet e gera partições por cliente conforme as opções e regras já existentes no gerador.

- <a id="sec-25-1"></a>**25.1** [data/partitioning.py](../src/fl_serverpuma/data/partitioning.py) — dependência já detalhada em [18.1](#sec-18-1).

<a id="sec-26"></a>

## 26. [data/generators/generate_ECG.py](../src/fl_serverpuma/data/generators/generate_ECG.py)

**Funções/métodos:** `load_ecg_datasets`, `generate_ecg_dataset`.

Prepara sinais ECG a partir de arquivos CSV locais e gera partições por cliente conforme as opções e regras já existentes no gerador.

- <a id="sec-26-1"></a>**26.1** [data/partitioning.py](../src/fl_serverpuma/data/partitioning.py) — dependência já detalhada em [18.1](#sec-18-1).

<a id="sec-27"></a>

## 27. [data/generators/generate_EMNIST.py](../src/fl_serverpuma/data/generators/generate_EMNIST.py)

**Funções/métodos:** `generate_dataset`.

Prepara caracteres EMNIST e gera partições por cliente conforme as opções e regras já existentes no gerador.

- <a id="sec-27-1"></a>**27.1** [data/partitioning.py](../src/fl_serverpuma/data/partitioning.py) — dependência já detalhada em [18.1](#sec-18-1).

<a id="sec-28"></a>

## 28. [data/generators/generate_FGVCAircraft.py](../src/fl_serverpuma/data/generators/generate_FGVCAircraft.py)

**Funções/métodos:** `generate_dataset`, `generate_dataset.load_data`.

Prepara imagens de aeronaves FGVC-Aircraft e gera partições por cliente conforme as opções e regras já existentes no gerador.

- <a id="sec-28-1"></a>**28.1** [paths.py](../src/fl_serverpuma/paths.py) — dependência já detalhada em [1.2.11.1](#sec-1-2-11-1).

- <a id="sec-28-2"></a>**28.2** [data/partitioning.py](../src/fl_serverpuma/data/partitioning.py) — dependência já detalhada em [18.1](#sec-18-1).

<a id="sec-29"></a>

## 29. [data/generators/generate_FashionMNIST.py](../src/fl_serverpuma/data/generators/generate_FashionMNIST.py)

**Funções/métodos:** `generate_dataset`.

Prepara imagens Fashion-MNIST e gera partições por cliente conforme as opções e regras já existentes no gerador.

- <a id="sec-29-1"></a>**29.1** [data/partitioning.py](../src/fl_serverpuma/data/partitioning.py) — dependência já detalhada em [18.1](#sec-18-1).

<a id="sec-30"></a>

## 30. [data/generators/generate_Flowers102.py](../src/fl_serverpuma/data/generators/generate_Flowers102.py)

**Funções/métodos:** `generate_dataset`, `generate_dataset.load_data`.

Prepara imagens Flowers102 e gera partições por cliente conforme as opções e regras já existentes no gerador.

- <a id="sec-30-1"></a>**30.1** [paths.py](../src/fl_serverpuma/paths.py) — dependência já detalhada em [1.2.11.1](#sec-1-2-11-1).

- <a id="sec-30-2"></a>**30.2** [data/partitioning.py](../src/fl_serverpuma/data/partitioning.py) — dependência já detalhada em [18.1](#sec-18-1).

<a id="sec-31"></a>

## 31. [data/generators/generate_GTSRB.py](../src/fl_serverpuma/data/generators/generate_GTSRB.py)

**Funções/métodos:** `generate_dataset`, `generate_dataset.load_data`.

Prepara imagens de sinais de trânsito GTSRB e gera partições por cliente conforme as opções e regras já existentes no gerador.

- <a id="sec-31-1"></a>**31.1** [data/partitioning.py](../src/fl_serverpuma/data/partitioning.py) — dependência já detalhada em [18.1](#sec-18-1).

<a id="sec-32"></a>

## 32. [data/generators/generate_HAR.py](../src/fl_serverpuma/data/generators/generate_HAR.py)

**Funções/métodos:** `generate_dataset`, `load_data_HAR`.

Prepara dados de reconhecimento de atividades humanas HAR e gera partições por cliente conforme as opções e regras já existentes no gerador.

- <a id="sec-32-1"></a>**32.1** [data/har.py](../src/fl_serverpuma/data/har.py) — Lê e formata dados de sensores/atividades, separa treino/teste e salva partições para os geradores HAR/PAMAP2. **[sem imports internos]**

    **Funções/métodos:** `format_data_x`, `format_data_y`, `read_ids`, `split_data`, `save_file`.

<a id="sec-33"></a>

## 33. [data/generators/generate_MNIST.py](../src/fl_serverpuma/data/generators/generate_MNIST.py)

**Funções/métodos:** `generate_dataset`.

Prepara dígitos MNIST e gera partições por cliente conforme as opções e regras já existentes no gerador.

- <a id="sec-33-1"></a>**33.1** [paths.py](../src/fl_serverpuma/paths.py) — dependência já detalhada em [1.2.11.1](#sec-1-2-11-1).

- <a id="sec-33-2"></a>**33.2** [data/partitioning.py](../src/fl_serverpuma/data/partitioning.py) — dependência já detalhada em [18.1](#sec-18-1).

<a id="sec-34"></a>

## 34. [data/generators/generate_Omniglot.py](../src/fl_serverpuma/data/generators/generate_Omniglot.py)

**Funções/métodos:** `generate_dataset`.

Prepara caracteres Omniglot e gera partições por cliente conforme as opções e regras já existentes no gerador.

- <a id="sec-34-1"></a>**34.1** [data/partitioning.py](../src/fl_serverpuma/data/partitioning.py) — dependência já detalhada em [18.1](#sec-18-1).

<a id="sec-35"></a>

## 35. [data/generators/generate_OxfordPets.py](../src/fl_serverpuma/data/generators/generate_OxfordPets.py)

**Funções/métodos:** `generate_dataset`.

Prepara imagens OxfordPets e gera partições por cliente conforme as opções e regras já existentes no gerador.

- <a id="sec-35-1"></a>**35.1** [paths.py](../src/fl_serverpuma/paths.py) — dependência já detalhada em [1.2.11.1](#sec-1-2-11-1).

- <a id="sec-35-2"></a>**35.2** [data/partitioning.py](../src/fl_serverpuma/data/partitioning.py) — dependência já detalhada em [18.1](#sec-18-1).

<a id="sec-36"></a>

## 36. [data/generators/generate_PAMAP2.py](../src/fl_serverpuma/data/generators/generate_PAMAP2.py)

**Funções/métodos:** `generate_dataset`, `load_data_PAMAP2`, `del_labels`, `adjust_idx_labels`, `complete_HR`.

Prepara dados de atividades PAMAP2 e gera partições por cliente conforme as opções e regras já existentes no gerador.

- <a id="sec-36-1"></a>**36.1** [data/har.py](../src/fl_serverpuma/data/har.py) — dependência já detalhada em [32.1](#sec-32-1).

<a id="sec-37"></a>

## 37. [data/generators/generate_Shakespeare.py](../src/fl_serverpuma/data/generators/generate_Shakespeare.py)

**Funções/métodos:** `process_x`, `process_y`, `generate_dataset`.

Prepara texto Shakespeare e gera partições por cliente conforme as opções e regras já existentes no gerador.

- <a id="sec-37-1"></a>**37.1** [data/language.py](../src/fl_serverpuma/data/language.py) — dependência já detalhada em [18.2](#sec-18-2).

<a id="sec-38"></a>

## 38. [data/generators/generate_SogouNews.py](../src/fl_serverpuma/data/generators/generate_SogouNews.py)

**Funções/métodos:** `generate_dataset`.

Prepara notícias SogouNews e gera partições por cliente conforme as opções e regras já existentes no gerador.

- <a id="sec-38-1"></a>**38.1** [data/partitioning.py](../src/fl_serverpuma/data/partitioning.py) — dependência já detalhada em [18.1](#sec-18-1).

- <a id="sec-38-2"></a>**38.2** [data/language.py](../src/fl_serverpuma/data/language.py) — dependência já detalhada em [18.2](#sec-18-2).

<a id="sec-39"></a>

## 39. [data/generators/generate_StanfordCars.py](../src/fl_serverpuma/data/generators/generate_StanfordCars.py)

**Funções/métodos:** `generate_dataset`, `generate_dataset.load_data`.

Prepara imagens StanfordCars e gera partições por cliente conforme as opções e regras já existentes no gerador.

- <a id="sec-39-1"></a>**39.1** [data/partitioning.py](../src/fl_serverpuma/data/partitioning.py) — dependência já detalhada em [18.1](#sec-18-1).

<a id="sec-40"></a>

## 40. [data/generators/generate_TinyImagenet.py](../src/fl_serverpuma/data/generators/generate_TinyImagenet.py)

**Funções/métodos:** `ImageFolder_custom.__init__`, `ImageFolder_custom.__getitem__`, `ImageFolder_custom.__len__`, `generate_dataset`.

Prepara imagens Tiny ImageNet e gera partições por cliente conforme as opções e regras já existentes no gerador.

- <a id="sec-40-1"></a>**40.1** [data/partitioning.py](../src/fl_serverpuma/data/partitioning.py) — dependência já detalhada em [18.1](#sec-18-1).

<a id="sec-41"></a>

## 41. [models/flops.py](../src/fl_serverpuma/models/flops.py)

Dependência já detalhada em [1.3.7.1](#sec-1-3-7-1).

## Apêndice A — inicialização por scripts e Docker

Estas relações são comandos de execução, não imports Python. O encadeamento abaixo usa referências aos capítulos já detalhados.

- **A.1** [src/system/run_all_sora.sh](../src/system/run_all_sora.sh)

    - **A.1.1** Executa [src/system/run.sh](../src/system/run.sh).

- **A.2** [src/system/run.sh](../src/system/run.sh)

    - **A.2.1** Executa [execution/server.py](../src/fl_serverpuma/execution/server.py) — ver [1](#sec-1).

    - **A.2.2** Executa [execution/client.py](../src/fl_serverpuma/execution/client.py) — ver [2](#sec-2).

    O gerador de dataset também é executado, mas seu nome é selecionado em tempo de execução a partir de `DATASET`: `fl_serverpuma.data.generators.generate_<dataset>`. Os geradores aparecem nos capítulos acima.

- **A.3** [docker/dockerfile.server](../docker/dockerfile.server)

    - **A.3.1** Executa [execution/server.py](../src/fl_serverpuma/execution/server.py) — ver [1](#sec-1).

- **A.4** [docker/dockerfile.client](../docker/dockerfile.client)

    - **A.4.1** Executa [execution/client.py](../src/fl_serverpuma/execution/client.py) — ver [2](#sec-2).

- **A.5** [docker/build-optimized.sh](../docker/build-optimized.sh)

    - **A.5.1** Executa [simulation/docker/compose.py](../src/fl_serverpuma/simulation/docker/compose.py) — ver [7](#sec-7).

`src/system/run_all_pumagt.sh` é um link simbólico para `run_all_sora.sh`. `docker/runclient.sh` inicia serviços via Compose; essa chamada não é tratada como import de um módulo Python.

## Apêndice B — marcadores de pacote

Os arquivos abaixo não têm imports explícitos nem inicialização operacional. Foram separados da árvore porque sua presença identifica pacotes Python, e não uma dependência explícita entre módulos. O Python pode carregá-los implicitamente ao importar o pacote.

- [__init__.py](../src/fl_serverpuma/__init__.py)
- [algorithms/__init__.py](../src/fl_serverpuma/algorithms/__init__.py)
- [algorithms/adaptation/__init__.py](../src/fl_serverpuma/algorithms/adaptation/__init__.py)
- [algorithms/personalization/__init__.py](../src/fl_serverpuma/algorithms/personalization/__init__.py)
- [algorithms/pruning/__init__.py](../src/fl_serverpuma/algorithms/pruning/__init__.py)
- [communication/__init__.py](../src/fl_serverpuma/communication/__init__.py)
- [data/__init__.py](../src/fl_serverpuma/data/__init__.py)
- [data/generators/__init__.py](../src/fl_serverpuma/data/generators/__init__.py)
- [execution/__init__.py](../src/fl_serverpuma/execution/__init__.py)
- [experiments/__init__.py](../src/fl_serverpuma/experiments/__init__.py)
- [experiments/analysis/__init__.py](../src/fl_serverpuma/experiments/analysis/__init__.py)
- [models/__init__.py](../src/fl_serverpuma/models/__init__.py)
- [models/clip/__init__.py](../src/fl_serverpuma/models/clip/__init__.py)
- [models/slm/__init__.py](../src/fl_serverpuma/models/slm/__init__.py)
- [persistence/__init__.py](../src/fl_serverpuma/persistence/__init__.py)
- [simulation/__init__.py](../src/fl_serverpuma/simulation/__init__.py)
- [simulation/docker/__init__.py](../src/fl_serverpuma/simulation/docker/__init__.py)
- [simulation/sumo/__init__.py](../src/fl_serverpuma/simulation/sumo/__init__.py)

## Documentação relacionada

- [Guia de desenvolvimento, arquitetura e responsabilidades](../AGENTS.md).
