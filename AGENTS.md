# AGENTS.md — Guia de Engenharia e Arquitetura (`FL_ServerPuma`)

Este documento é a referência técnica oficial para agentes de IA e desenvolvedores que atuam no repositório **FL_ServerPuma** (**PUMA** / **PUMA-GT Plus**). Ele consolida a arquitetura, fluxos de dados, protocolos de rede, invariantes matemáticos, convenções de código e armadilhas críticas do sistema.

---

## 1. Visão Geral do Projeto

O **FL_ServerPuma** é um framework de pesquisa e experimentação para **Aprendizado Federado (Federated Learning — FL)** focado em dispositivos de borda restritos, modelos de fundação multimodais e redes veiculares (V2X). Seus pilares científicos são:

1. **Personalização Adaptativa (FedALA)**:
   - Camada de Agregação Local Adaptativa para CNNs (`src/system/ALA.py`) e para adaptadores PEFT/LoRA (`src/system/ALA_LoRA.py`).
2. **Eficiência Computacional e Estrutural**:
   - **CNNs**: Poda estrutural de filtros/canais com reestruturação física de camadas via **OPALA** (`src/system/prunning.py`), **NISP** (`src/system/prunning_nisp.py`) e **SNIP** (`src/system/prunning_snip.py`).
   - **Foundation Models (`CLIP` e `SLM` / `Qwen3-VL-2B-Instruct`)**: Fine-tuning eficiente via **LoRA**, **AdaLoRA**, **SoRA** (*Sparse Low-Rank Adaptation* com poda iterativa de rank no servidor), **QLoRA** (quantização 4-bit NF4 + DoubleQuant) e **PaCA** (*Partial Calibration* / seleção dinâmica das $k$ camadas superiores).
   - **PUMA-GT Plus**: Motor analítico no servidor que calcula o Índice de Capacidade $\gamma^c$ por cliente a cada rodada (combinando latência sistêmica $\eta^c$, dificuldade estatística Non-IID $\psi^c$ e decaimento por platô $\phi^c_{\text{plateau}}$) para alocar dinamicamente **PaCA** (`--adaptive-paca`) e **Rank** (`--adaptive-rank`).
3. **Compressão de Comunicação**:
   - **Quantização Simétrica INT8** padrão (`src/utils/fl_math.py`) ou **LHDQ** (*Low Huffman-coded Delta Quantization*, ~1.67 bits/parâmetro em `src/utils/lhdq.py` via `--delta-coding`).
4. **Aprendizado Federado Veicular (V2V e V2I Over-the-Air)**:
   - Simulação de mobilidade urbana via **SUMO/TraCI** (mapa Marambaia) acoplada a contêineres Docker com agregação analógica **Over-the-Air (OTA)** sob path loss COST-231 Hata, desvanecimento Rayleigh com erro de CSI e beamforming MIMO/MRC (`src/v2v_fl/`).

---

## 2. Mapa de Arquitetura e Diretórios

```text
FL_ServerPuma/
├── pyproject.toml                  # Manifesto do projeto (Python 3.11.*, uv, hatchling, ruff)
├── requirements.txt                # Dependências congeladas (espelho para Docker)
├── docker/
│   ├── dockerfile.server           # Imagem CPU para servidor centralizado e RSU V2I
│   ├── dockerfile.client           # Imagem CUDA 11.8 / PyTorch para clientes FL
│   ├── generate_compose.py         # Gerador de docker-compose.generated.yml (centralizado + docker-tc)
│   ├── generate_v2v_compose.py     # Gerador de compose para clientes V2V descentralizados
│   └── build-optimized.sh          # Script de build com BuildKit das imagens Docker
└── src/
    ├── system/                     # Motor principal de FL (Server, Client, Modelos, Scripts)
    │   ├── server.py               # Servidor centralizado (agregação, γ^c, poda SoRA, avaliação, HDF5)
    │   ├── client.py               # Cliente unificado (modos: centralized, decentralized V2V/V2I)
    │   ├── run.sh                  # Orquestrador local via tmux (gerencia dataset, GPUs, server e clientes)
    │   ├── run_all_*.sh            # Suites de benchmark automatizado (PUMA-GT Plus, LoRA/SLM)
    │   ├── model.py                # SimpleModel (CNN 2xConv + 2xFC para MNIST/CIFAR10/CIFAR100)
    │   ├── ALA.py                  # FedALA tradicional para modelos densos (CNN)
    │   ├── ALA_LoRA.py             # FedALA adaptado para parâmetros PEFT (LoRA/AdaLoRA/SoRA) e head
    │   ├── prunning.py             # Poda estrutural OPALA + reestruturação física (prune_and_restructure)
    │   ├── prunning_nisp.py        # Poda estrutural baseada em NISP (propagação de importância)
    │   ├── prunning_snip.py        # Poda estrutural baseada em SNIP (sensibilidade de conexão)
    │   ├── flops.py                # Estimativa analítica de FLOPs/MACs por camada para CNN
    │   ├── result_utils.py         # Exportação de resultados e matrizes de confusão em HDF5/TXT
    │   ├── lora_clip/              # Subsistema CLIP (openai/clip-vit-base-patch32)
    │   │   ├── clip_setup.py       # CLIPForClassification, PaCA, update_paca_runtime, hooks de gradiente
    │   │   ├── sora.py             # SoRALinear, SoRAWrappedLinear e GateSparsifier (proximal L1)
    │   │   ├── clip_sora_utils.py  # Poda iterativa de rank no servidor (reduce_sora_state_dict_rank)
    │   │   ├── trainer.py          # Loop de treino/avaliação CLIP (AdamW unificado + GateSparsifier)
    │   │   └── train_config.yml    # Hiperparâmetros base das estratégias PEFT para CLIP
    │   ├── lora_slm/               # Subsistema VLM/SLM (Qwen/Qwen3-VL-2B-Instruct)
    │   │   ├── slm_setup.py        # SLMForClassification (bfloat16, 28 camadas, CustomCollator, PaCA)
    │   │   ├── qlora.py            # QLoRA 4-bit NF4 + DoubleQuant, PagedAdamW8bit, validação de payload
    │   │   ├── sora.py             # Implementação SoRA para camadas lineares da SLM
    │   │   ├── slm_sora_utils.py   # Poda iterativa de rank para SLM (preserva ordenação ordinal de keep_idx)
    │   │   ├── trainer.py          # Loop de treino SLM (gradient accumulation, AMP bfloat16, PagedAdamW8bit)
    │   │   └── train_config.yml    # Hiperparâmetros base (batch_size=2, grad_accum=8, r=16)
    │   └── plot/                   # Scripts de pós-processamento e geração de gráficos científicos
    ├── utils/                      # Utilitários compartilhados de rede, matemática e dados
    │   ├── network_utils.py        # Framing TCP (4 bytes !I + pickle) e validação do protocolo V2I (v1)
    │   ├── fl_math.py              # Quantização INT8, agregação multi-rank/PaCA, resize_model_to_pruned
    │   ├── lhdq.py                 # Compressão diferencial LHDQ (quantis ternários + Huffman ~1.67 bpp)
    │   ├── data_utils.py           # Leitura de partições .npz (read_client_data, read_server_Test_data)
    │   └── model_utils.py          # Escrita atômica (write_json_atomic, save_state_dict_atomic), seed
    ├── dataset/                    # Geradores de dados particionados (IID / Non-IID Patológico / Dirichlet)
    │   ├── utils/dataset_utils.py  # Funções canônicas: separate_data, split_data, save_file
    │   └── generate_*.py           # Scripts para MNIST, Cifar10, Cifar100, EuroSAT, Flowers102, GTSRB, OxfordPets, etc.
    └── v2v_fl/                     # Subsistema de FL Veicular Descentralizado (V2V e V2I)
        ├── v2v_docker_orchestrator.py # Orquestrador SUMO + Docker para V2V D-PSGD (eleição por Medoid)
        ├── v2i_docker_orchestrator.py # Orquestrador SUMO + Docker para V2I (handover entre RSUs fixas)
        ├── rsu_server.py           # Servidor TCP multi-RSU stateless em CPU (protocolo V2I v1)
        ├── run_simulations.sh      # Suite automatizada de experimentos comparativos V2V vs V2I
        ├── experiment_report.py    # Consolidador de métricas e relatórios de experimentos V2X
        ├── validate_mobility.py    # Validador de conectividade e mobilidade da malha SUMO
        ├── over_the_air/
        │   ├── ota_channel.py      # Canal físico (COST-231 Hata, Rayleigh, CSI imperfeito, MMSE/MRC SVD)
        │   └── ota_aggregation.py  # Simulação OTA ponderada por amostras de treino + diagnóstico L2
        ├── plot/                   # Scripts de visualização V2X (plot_v2v_accuracy.py, plot_exchange_time_.py)
        └── maps/                   # Topologias SUMO (marambaia e marambaia_recorte) e configs de RSUs
```

---

## 3. Ambiente, Dependências e Comandos Essenciais

### 3.1 Gerenciamento de Pacotes e Python
- **Versão do Python**: `==3.11.*` (estritamente definido em `pyproject.toml`).
- **Gerenciador**: [`uv`](https://docs.astral.sh/uv/) com `hatchling` (instala `src/system`, `src/dataset` e `src/utils` como pacotes editáveis).
- **Variáveis de Ambiente Recomendadas**:
  - `OMP_NUM_THREADS=1` (evita contenção de CPU ao rodar múltiplos clientes em paralelo).
  - `HF_HOME` apontando para diretório persistente ao usar modelos `clip` ou `slm`.

```bash
# Sincronizar o ambiente virtual (.venv)
uv sync

# Executar linter e formatador (Ruff configurado em pyproject.toml)
uvx ruff check src/
uvx ruff format --check src/
```

> **Nota sobre Testes**: O repositório não possui suíte de testes unitários automatizados (`pytest`). Toda validação de mudanças deve ser feita via checagem estática (`uvx ruff check`), verificação de sintaxe (`uv run python -m py_compile <arquivo>`) e execução controlada de rodadas curtas (`--rounds 2`).

### 3.2 Geração de Datasets Particionados
Os datasets ficam em `src/dataset/<NomeDataset>/{train,test}/*.npz` com metadados em `config.json`.

> **Atenção à Interface CLI dos Geradores**:
> - `generate_MNIST.py` usa `argparse` (`--num-clients N`).
> - Os demais geradores (`generate_OxfordPets.py`, `generate_Flowers102.py`, `generate_Cifar10.py`, etc.) leem `sys.argv` posicional: `<iid|noniid> <balance|-> <pat|dir> <num_clients>`.
> - A função `check()` em `src/dataset/utils/dataset_utils.py` **reutiliza** o dataset existente se os parâmetros em `config.json` coincidirem. Se você alterar o número de clientes em um gerador que não passa `num_clients` para `check()`, remova `src/dataset/<Dataset>/config.json` para forçar a regeneração.

```bash
# Exemplo: Gerar OxfordPets para 6 clientes em distribuição Non-IID Dirichlet (α=0.1)
PYTHONPATH=src uv run --no-sync python src/dataset/generate_OxfordPets.py noniid - dir 6

# Exemplo: Gerar MNIST para 10 clientes em distribuição Non-IID Dirichlet
PYTHONPATH=src uv run --no-sync python src/dataset/generate_MNIST.py noniid - dir --num-clients 10
```

### 3.3 Execução Local Centralizada (`run.sh` via `tmux`)
O script `src/system/run.sh` é o ponto de entrada padrão para experimentos centralizados locais. Ele valida/gera o dataset, cria uma sessão `tmux` (`fl_session` ou customizada via `--session-name`), inicia o `server.py` na janela 0 e distribui os processos `client.py` nas GPUs disponíveis (`i % NUM_GPUS`).

```bash
cd src/system

# 1. CNN + MNIST + FedALA (--ala 0) + Poda OPALA (--prune 1)
./run.sh --model cnn --dataset MNIST --clients 10 --rounds 20 --ala 0 --prune 1 --method OPALA

# 2. CLIP + Flowers102 + SoRA + PUMA-GT Plus (PaCA e Rank Adaptativos + LHDQ)
./run.sh --model clip --dataset Flowers102 --clients 6 --rounds 50 \
  --strategy sora_with_schedule --prune-freq 5 --adaptive-paca --adaptive-rank --delta-coding --ala 0

# 3. SLM (Qwen3-VL-2B) + OxfordPets + QLoRA (4-bit NF4)
./run.sh --model slm --dataset OxfordPets --clients 6 --rounds 25 \
  --strategy lora --qlora --ala 1 --epochs 2
```

### 3.4 Execução em Contêineres Docker e V2X (V2V / V2I)
```bash
# Build otimizado das imagens Docker (fl-server:latest e fl-client:latest)
./docker/build-optimized.sh

# Execução Centralizada via Docker Compose (com controle de tráfego docker-tc)
uv run python docker/generate_compose.py --clients 5 --rounds 10 --ala 0 --prune 1 --dataset MNIST --model cnn
docker compose -f docker/docker-compose.generated.yml up

# Execução Descentralizada V2V-OTA com SUMO (a partir da raiz do repositório)
PYTHONPATH=src uv run python src/v2v_fl/v2v_docker_orchestrator.py \
  --clients 10 --rounds 20 --dataset MNIST --model cnn --ala 0 --radius 300 --tx-power 23.0

# Execução Descentralizada V2I-OTA com RSUs fixas stateless (a partir da raiz do repositório)
PYTHONPATH=src uv run python src/v2v_fl/v2i_docker_orchestrator.py \
  --clients 10 --rounds 20 --dataset MNIST --model cnn --ala 0 --rsu-antennas 16
```

---

## 4. Fluxos de Execução e Protocolos de Rede

### 4.1 Protocolo Centralizado (TCP Socket + Pickle)
Definido em `src/utils/network_utils.py`. Toda mensagem usa um cabeçalho de **4 bytes big-endian (`!I`)** indicando o tamanho do payload serializado com `pickle.dumps`.

**Ordem Estrita de Mensagens (NÃO altere um lado sem alterar o outro):**
1. **Handshake Inicial (Conexão)**:
   - Cliente $\rightarrow$ Servidor: `send_data(s, client_idx)` (`int`).
2. **Início da Rodada $t$ (Servidor $\rightarrow$ Cliente)**:
   - *Apenas se `model in ['clip', 'slm']`*:
     - Servidor envia `client_paca` (`int`: número de camadas superiores ativas).
     - Servidor envia `client_rank` (`int`: rank alvo alocada para o cliente).
   - Servidor envia `payload` (`dict`: `state_dict` comprimido via INT8, LHDQ ou `bf16` adapter-only no QLoRA).
   - Servidor envia `prune` (`int`: flag de poda estrutural para CNN).
   - *Apenas se `round_num >= 2 and prune == 1 and model == 'cnn'`*:
     - Servidor envia `max_amount` (`float`: taxa de poda calculada analiticamente por $\gamma^c$).
3. **Fim do Treino Local na Rodada $t$ (Cliente $\rightarrow$ Servidor)**:
   - Cliente envia na sequência exata:
     1. `payload_to_send` (`dict`: pesos locais treinados e comprimidos)
     2. `len(train_loader)` (`int`: número de batches de treino)
     3. `args.ala` (`int`: flag ALA)
     4. `client_training_time` (`float`: segundos gastos em treino + FedALA)
     5. `test_accuracy` (`float`: acurácia local em %)
     6. `test_loss` (`float`: loss local média)
     7. `client_processing_overhead` (`float`: segundos gastos em compressão/descompressão/reestruturação)
4. **Barreira de Fim de Rodada (Servidor $\rightarrow$ Cliente)**:
   - Se `model == 'slm'`: Servidor envia `send_data(conn, "end")` e cliente lê com `recv_data(s)`.
   - Se `model in ['cnn', 'clip']`: Servidor envia bytes crus `conn.send(b"end")` e cliente lê com `s.recv(3)`.

### 4.2 Fluxo Descentralizado V2V-OTA (`v2v_docker_orchestrator.py`)
1. **Sincronização por Arquivos Atômicos** em `src/results/runs/v2v/<run_id>/encounters/`:
   - Na inicialização, os clientes sincronizam o mesmo modelo inicial de referência (`models/initial_global_model.pt`) para garantir ponto de partida idêntico em CNNs.
   - O orquestrador cria `.start_round_<r>`.
   - Clientes executam treino local, salvam `models/client_<id>_local.pt` atomicamente e criam `.epoch_done_client_<id>`.
   - O orquestrador avança o SUMO (`traci.simulationStep()`), detecta cliques de veículos dentro de `--radius` cujo **Tempo de Contato Estimado** (`estimate_contact_time` via cinemática quadrática 2D) seja $\ge$ `--min-contact-time` (padrão 6.0s), e elege o veículo **Medoid** (`find_medoid`) como agregador.
   - O orquestrador grava `encounter_<id>.json` (`type: "v2v"`).
2. **Comunicação P2P + Canal OTA Ponderado por Amostras**:
   - Veículos transmissores conectam via TCP em `fl-client-v2v-<aggregator_id>:9000` e enviam payload validado contendo `(client_idx, enc_id, train_samples, snapshot)`.
   - O agregador aplica o canal sem fio (`src/v2v_fl/over_the_air/ota_aggregation.py` + `ota_channel.py`): path loss COST-231 Hata (2 GHz), fading Rayleigh com erro de CSI ($\epsilon=0.1$), beamforming MIMO receptor MMSE/MRC via SVD ($N=8$ antenas) e ruído térmico AWGN (100 kHz).
   - **Ponderação proporcional a `train_samples`**: Com $M$ participantes e pesos normalizados $q_k = M \cdot \frac{n_k}{\sum_{j} n_j}$, o agregador combina seu modelo local com o sinal OTA recebido:
     $$\mathbf{w}_{\text{agg}} = \frac{q_{\text{local}}}{M}\mathbf{w}_{\text{local}} + \left(\frac{\sum_{k \in \text{tx}} q_k}{M}\right)\hat{\mathbf{w}}_{\text{OTA}}$$
     Além disso, registra diagnóstico detalhado de distorção (`relative_l2_error`, `noise_std`, `effective_model_weight`).
   - O agregador devolve $\mathbf{w}_{\text{agg}}$ via socket aos transmissores; todos gravam `.applied_enc_<id>_client_<c>.json` e `.done_enc_<id>_client_<c>`.

### 4.3 Fluxo Descentralizado V2I-OTA (`v2i_docker_orchestrator.py` + `rsu_server.py`)
- As **RSUs** (*Roadside Units*, definidas em `src/v2v_fl/maps/rsu_config_marambaia.json` ou `rsu_config_marambaia_recorte.json`) rodam em um único contêiner CPU `fl-rsu-service` (`python -m v2v_fl.rsu_server` na porta 9000).
- **RSU Stateless**: A RSU **não mantém estado de modelo entre encontros**. A cada encontro V2I (`type: "v2i"`), obrigatoriamente com $K \ge 2$ veículos sob cobertura da mesma antena, os veículos fazem upload de seus `state_dict` e `train_samples`, a RSU aplica `aggregate_ota_pure` ($N=16$ antenas por padrão, ponderado por `train_samples`), devolve o modelo agregado e descarta os tensores da memória.
- **Protocolo Validado V2I (`V2I_PROTOCOL_VERSION = 1` em `src/utils/network_utils.py`)**:
  - Healthcheck: `{"protocol_version": 1, "msg_type": "PING"}` $\rightarrow$ `PONG`.
  - Encontro: Cliente envia `UPLOAD` (validado contra `run_id`, `rsu_id`, `encounter_id`, `client_id`, `train_samples` e tensores densos finitos em CPU) $\rightarrow$ RSU responde `RESULT` (ou `ERROR`) $\rightarrow$ Cliente confirma aplicação local enviando `APPLIED` $\rightarrow$ RSU publica `rsu_status_enc_<id>.json` (`status: "COMPLETED"`).
  - **Restrição atual**: O modo V2I suporta exclusivamente `--model cnn` (`client.py` valida e aborta caso contrário).

---

## 5. Modelos, Estratégias PEFT e Motor Matemático (PUMA-GT Plus)

### 5.1 Famílias de Modelos (`--model`)
| Flag `--model` | Classe Principal | Backbone / Arquitetura | Precisão Padrão | Camadas Totais (PaCA Default) |
| :--- | :--- | :--- | :--- | :--- |
| `cnn` | `SimpleModel` (`src/system/model.py`) | 2x `Conv2d` + ReLU + MaxPool $\rightarrow$ 2x `Linear` | `float32` | N/A (usa poda OPALA/NISP/SNIP) |
| `clip` | `CLIPForClassification` (`src/system/lora_clip/clip_setup.py`) | `openai/clip-vit-base-patch32` (`vision_model` + `classifier`) | `float32` (`sdpa`) | 12 camadas (`encoder.layers`) |
| `slm` | `SLMForClassification` (`src/system/lora_slm/slm_setup.py`) | `Qwen/Qwen3-VL-2B-Instruct` (`language_model` + `classifier` no último token não-pad) | `bfloat16` (`sdpa`) | 28 camadas (`language_model.layers`) |

### 5.2 Equação do Índice de Capacidade ($\gamma^c$) no Servidor
Calculado em `calculate_capacity_index` (`src/system/server.py`) a cada rodada para guiar a poda CNN, o **PaCA Adaptativo** (`--adaptive-paca`) e o **Rank Adaptativo** (`--adaptive-rank`):

$$\gamma^c = \text{clamp}\left(\eta^c \times \psi^c \times \phi^c_{\text{plateau}},\; \gamma_{\min}=0.15,\; 1.0\right)$$

Onde:
1. **Fator de Eficiência Sistêmica ($\eta^c$)**:
   $$\eta^c = \min\left(1.0,\; \frac{T_{\text{target}}}{T_{\text{full}}^c}\right), \quad T_{\text{target}} = 1.15 \times \overline{T}, \quad T_{\text{full}}^c = T_{\text{train}}^c + T_{\text{comm}}^c + T_{\text{eval}}^c + T_{\text{proc}}^c$$
   Clientes rápidos operam em capacidade máxima ($\eta^c = 1.0$); clientes lentos (*stragglers*) têm $\eta^c < 1.0$.
2. **Fator de Dificuldade Estatística Non-IID ($\psi^c$)**:
   $$\psi^c = \left(\text{clamp}\left(\frac{\text{Acc}_{\text{avg}}}{\text{Acc}^c},\; 0.80,\; 1.35\right)\right)^\delta \quad (\delta = 0.5 \text{ padrão})$$
   - **Invariante de Justiça Non-IID**: Clientes com acurácia **abaixo** da média global têm $\text{Acc}_{\text{avg}} / \text{Acc}^c > 1.0 \implies \psi^c > 1.0$ (**ganham** mais camadas PaCA e maior Rank para aprender distribuições difíceis). Se `client_loss > 1.2 * global_loss`, força-se $\psi^c \ge 1.15$.
3. **Fator de Platô ($\phi^c_{\text{plateau}}$) e Mecanismos de Segurança**:
   - Após a rodada 12, se a acurácia do cliente estabilizar ($\mu_{\text{acc}} \ge 0.90 \cdot \text{Acc}_{\text{best}}$ e $\sigma_{\text{acc}} < 1.8\%$ em janela de 5 rodadas), $\phi^c_{\text{plateau}} = \max(0.45,\; 1.0 - 0.025 \times \text{plateau\_rounds})$.
   - **Freio de Emergência Global (*Plateau Brake*)**: Se a acurácia global cair $> 3.0\text{ pp}$ em relação ao pico das últimas 10 rodadas, o decaimento por platô é suspenso ($\phi^c_{\text{plateau}} = 1.0$).
   - **Panic Reversal**: Se a acurácia de um cliente cair $> 2.0\text{ pp}$ logo após uma redução de PaCA/Rank, o servidor reverte imediatamente (+2 camadas / +2 ranks) e zera o contador de platô.
   - **Cooldown e Limites de Transição**: PaCA varia no máximo $\pm 2$ camadas por rodada com cooldown de 2 rodadas; Rank varia no máximo $-1$ (redução) ou $+2$ (aumento) por rodada com cooldown de 3 rodadas.

### 5.3 SoRA, Poda Iterativa de Rank e Agregação Heterogênea
- **SoRA (`sora.py`)**: Envolve camadas `nn.Linear` com matrizes `lora_A` $(r \times d_{\text{in}})$, `lora_B` $(d_{\text{out}} \times r)$ e um vetor `gate` treinável $(1 \times r)$ inicializado em $0.1$. Otimizado com `AdamW` unificado (2 grupos de parâmetros) + `GateSparsifier` que aplica *soft-thresholding* proximal $L_1$: $g \leftarrow \text{softshrink}(g,\; \lambda_{\text{sparse}} \cdot \text{lr}_{\text{gate}})$.
- **Poda Iterativa no Servidor (`--prune-freq N`)**: A cada $N$ rodadas, `reduce_sora_state_dict_rank` (`clip_sora_utils.py` / `slm_sora_utils.py`) remove dimensões onde $|g_i| \le \tau_{\text{adapt}}$ ($\tau_{\text{adapt}}$ cresce linearmente de $10^{-3}$ a $10^{-2}$ ao longo das rodadas), preservando no mínimo `min_rank = 2` dimensões. Em seguida, `resize_model_to_pruned` (`src/utils/fl_math.py`) substitui os `nn.Parameter` do modelo sem alterar `sora.scaling = lora_alpha / base_r`.
- **Agregação Heterogênea (`aggregate_models` em `src/utils/fl_math.py`)**:
  - Suporta clientes com diferentes ranks (truncamento/alinhamento até o rank global) e diferentes janelas PaCA (camadas congeladas no cliente não enviam atualizações e o servidor preserva os pesos globais naquelas fatias fazendo média apenas sobre os clientes que treinaram a camada).

### 5.4 QLoRA (`--qlora` em `src/system/lora_slm/qlora.py`)
- Disponível para `--model slm --strategy lora` no modo centralizado.
- Carrega o backbone `Qwen3-VL-2B-Instruct` em **4-bit NF4** com **Double Quantization** (`bnb_4bit_compute_dtype=torch.bfloat16`), congela 100% do backbone (`verify_qlora_freezing` garante que nenhum `Params4bit` tenha `requires_grad=True`) e usa o otimizador `bnb.optim.PagedAdamW8bit`.
- **Invariante de Comunicação QLoRA**: Apenas os adaptadores LoRA (`slm_model.<peft_canonical_key>`) e a cabeça `classifier.*` trafegam pela rede em `bfloat16` (validado por `assert_adapter_only_payload`). A quantização INT8 de transporte é desativada no modo QLoRA, e o servidor acumula os deltas em `float32` antes de converter de volta para `bfloat16`.

---

## 6. Esquema de Resultados e Métricas (HDF5 e JSONL)

- **Resultados Centralizados (`src/results/`)**:
  - Salvos como arquivos `.h5` (HDF5 via `h5py`) nomeados com o resumo dos hiperparâmetros:
    - Servidor: `server_<model>_<strategy>_<dataset>_r<rounds>_c<clients>_..._.h5`
    - Clientes: `client_<id>_<model>_<strategy>_<dataset>_..._.h5`
  - **Datasets internos do `.h5` do Servidor**:
    - `accuracy`, `loss`, `train_loss`, `round_time`, `Server_Evaluation_Time`
    - `bytes_sent`, `bytes_received`, `amount_pruned`
    - `avg_paca_layers`, `avg_rank`, `total_active_params` (modelos `clip` e `slm`)
    - `confusion_matrix` (matriz de confusão final no conjunto de teste do servidor)
- **Resultados V2V / V2I (`src/results/runs/<v2v|v2i>/<run_id>/`)**:
  - `run.json`: Manifesto completo da simulação (argumentos, topologia, resumo final).
  - `events/v2v_events.jsonl` ou `events/v2i_events.jsonl`: Log estruturado em tempo real de todos os encontros, distâncias, SINR/SNR de canal OTA, perdas e acurácias pós-agregação.
  - `metrics/`: Arquivos `.h5` por veículo compatíveis com os scripts de plotagem em `src/system/plot/`.

---

## 7. Convenções de Código e Engenharia de Software

1. **Estilo e Formatação (`pyproject.toml`)**:
   - Python `3.11`, indentação de **4 espaços**, limite de linha de **99 caracteres**.
   - Regras Ruff ativas: `E`, `F`, `I` (isort), `UP` (pyupgrade), `B` (flake8-bugbear), `SIM` (flake8-simplify), com `E501` ignorado.
2. **Idioma**:
   - Comentários, docstrings, mensagens de log e mensagens de commit são escritos em **Português (PT-BR)**.
   - Mensagens de commit seguem **Conventional Commits** em português: `feat(escopo): descrição`, `fix(escopo): ...`, `refactor(ota): ...`, `docs(readme): ...`.
3. **Operações de Arquivo Compartilhadas (Host $\leftrightarrow$ Docker)**:
   - Sempre utilize `write_json_atomic` e `save_state_dict_atomic` (`src/utils/model_utils.py`) ao gravar arquivos em `src/results/runs/`. Elas criam arquivos temporários no mesmo filesystem (`mkstemp`), aplicam permissão `0o644` (permitindo leitura/remoção tanto pelo usuário do host quanto pelo `root` do contêiner), executam `os.fsync` e fazem substituição atômica via `os.replace`.

---

## 8. Armadilhas Críticas e Invariantes (ATENÇÃO MÁXIMA)

Antes de modificar argumentos CLI, loops de treinamento ou funções de rede, verifique estas particularidades históricas do código:

### ⚠️ 1. Semântica Invertida da Flag `--ala`
Em **todo o projeto** (`server.py`, `client.py`, `run.sh`, `generate_compose.py`, `v2v_docker_orchestrator.py`, `v2i_docker_orchestrator.py`):
- **`--ala 0`** significa **FedALA ATIVADO** (`if args.ala == 0:` executa `ALA` / `ALA_LoRA`).
- **`--ala 1`** significa **FedALA DESATIVADO** (usa substituição direta / FedAvg puro).
- **Nunca** inverta essa lógica para "0=falso, 1=verdadeiro", pois todos os scripts de benchmark (`run_all_*.sh`) e geradores de gráficos dependem de `ala0` = Com FedALA e `ala1` = Sem FedALA.

### ⚠️ 2. Divergência de Documentação da Flag `--prune`
- No código real de treinamento (`client.py:local_training` e `server.py`), a poda estrutural de CNN é executada quando **`prune == 1`** (`if round_num >= 2 and prune == 1:`).
- Contudo, as strings `help=` de `argparse` em `client.py`, `v2v_docker_orchestrator.py` e `v2i_docker_orchestrator.py` dizem erroneamente `"0=sim, 1=não"`, enquanto `server.py` e `run.sh` documentam e usam corretamente **`1` = Poda Ativada** e **`0` = Poda Desativada**. Ao acionar poda estrutural de CNN, passe sempre `--prune 1`.

### ⚠️ 3. Dependência de Diretório de Trabalho (`CWD = src/system/`) e `--dataset-dir`
- Por padrão (quando `--dataset-dir` não é informado), `src/utils/data_utils.py` (`read_client_data` e `read_server_Test_data`) carrega dados usando o caminho relativo **`../dataset/<dataset>/train/<idx>.npz`**.
- Portanto, `server.py`, `client.py` (em modo centralizado sem `--dataset-dir`) e `run.sh` **devem ser executados com diretório de trabalho em `src/system/`** (como configurado em `run.sh` e no `WORKDIR /app/src/system` dos Dockerfiles).
- Nos cenários V2V/V2I, é possível selecionar diretórios customizados de partição via `--dataset-dir` (ex.: `Cifar10_alpha1`). Já os orquestradores `src/v2v_fl/v2v_docker_orchestrator.py`, `src/v2v_fl/v2i_docker_orchestrator.py` e `rsu_server.py` devem ser executados a partir da **raiz do repositório** com `PYTHONPATH=src`.

### ⚠️ 4. Desnormalização de Imagens entre Dataset `.npz`, `CLIP` e `SLM`
- O gerador `src/dataset/generate_OxfordPets.py` salva os tensores `.npz` **já normalizados com a média e desvio padrão do CLIP** (gerando valores negativos até $\approx -1.8$), enquanto outros geradores (`MNIST`, `Cifar10`) salvam em `[-1, 1]` ou `[0, 1]`.
- O modelo `slm` (`Qwen3-VL-2B-Instruct`) utiliza o `AutoProcessor` da HuggingFace, que espera imagens PIL RGB em `[0, 255]` (`uint8`).
- Por isso, tanto `client.py` quanto `server.py` contêm uma rotina explícita que detecta `if x_tensor.min() < -0.1:` (caso `OxfordPets`, `Flowers102`, `EuroSAT`, `GTSRB`), **reverte a normalização do CLIP** (`x * std + mean`) e converte para `uint8` antes de alimentar o `CustomCollator` da SLM. Não remova essa conversão ao adicionar novos datasets para a SLM.

### ⚠️ 5. Reestruturação Física de CNN (`map_sequential_to_simplemodel`)
- Quando a poda estrutural de CNN (`prune_and_restructure` em `src/system/prunning.py`) reduz o número de filtros de uma camada convolucional, ela retorna um `nn.Sequential` cujas chaves de `state_dict` são índices numéricos (`"0.weight"`, `"3.weight"`, `"7.weight"`, `"9.weight"`).
- O `SimpleModel` espera chaves nomeadas (`"conv1.0.weight"`, `"conv2.0.weight"`, `"fc1.0.weight"`, `"fc.weight"`). A função `map_sequential_to_simplemodel()` em `src/system/client.py` realiza essa tradução e reconstrói as camadas internas do `SimpleModel` com as novas dimensões antes do envio ao servidor.

### ⚠️ 6. Sincronia de Estado de Referência no LHDQ (`--delta-coding`)
- Na compressão diferencial LHDQ (`src/utils/lhdq.py`), o transmissor calcula $\Delta P = P_t - P_{\text{ref}}$.
- Para evitar *drift* acumulativo de quantização entre transmissor e receptor, o transmissor **deve** executar imediatamente `lhdq_decode(encoded_payload, previous_sent_state)` para atualizar seu próprio `previous_sent_state` (como feito em `server.py` e `client.py`). Além disso, quando o tamanho de um tensor muda (ex.: poda de rank SoRA ou poda estrutural CNN), o LHDQ faz fallback automático para envio do tensor completo naquela rodada e reinicia a referência.

