# FL_ServerPuma — YOLO federada por câmera

Esta é a versão da branch **`yolo_review`**: YOLO11n da Ultralytics, uma câmera por cliente e **FedAvg ponderado pelo número de imagens de treino**. Um servidor e clientes locais trocam pesos por TCP em `127.0.0.1`. Os clientes treinam **em sequência**, usando uma única GPU; as imagens permanecem no dataset. O README descreve apenas essa versão.

```bash
git clone --branch yolo_review --single-branch \
  https://github.com/VeigarGit/FL_ServerPuma.git
cd FL_ServerPuma
uv sync --project yolo --locked
```

## Requisitos e instalação

Use Linux com GPU NVIDIA para o treinamento completo, Python 3.11 e [uv](https://docs.astral.sh/uv/getting-started/installation/). O ambiente isolado `yolo/.venv` fixa Ultralytics 8.3.0, PyTorch 2.0.1, torchvision 0.15.2 e NumPy 1.26.4; `yolo/uv.lock` registra as dependências. Não é necessário instalar o ambiente principal do repositório, Docker, tmux ou bibliotecas CLIP/SLM.

Verifique a GPU após `uv sync`:

```bash
nvidia-smi
uv run --project yolo --locked python -c \
  "import torch; print('torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU disponível:', torch.cuda.is_available())"
```

O wheel Linux de PyTorch 2.0.1 do ambiente usa CUDA 11.7. É necessário um driver NVIDIA compatível; o toolkit CUDA instalado no sistema não substitui o runtime do wheel. Consulte a [matriz oficial do PyTorch 2.0.1](https://pytorch.org/get-started/previous-versions/#v201). Para usar deliberadamente CUDA 11.8, mantenha as versões e instale os wheels cu118 no ambiente:

```bash
uv pip install --python yolo/.venv/bin/python \
  torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu118
```

Nesse caso, execute com `yolo/.venv/bin/python` diretamente: `uv run`/`uv sync` pode restaurar os wheels do lock padrão. Registre a alteração no relatório do experimento. CPU é suportada para testes pequenos; esta versão não oferece MPS nem execução distribuída entre máquinas.

Os pesos `yolo11n.pt` são baixados no primeiro treino. Para uma máquina sem internet, baixe antecipadamente e passe `--weights /caminho/yolo11n.pt`.

## Dataset já dividido por câmera

```text
dataset/
├── camera_01/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
└── camera_02/
    ├── images/{train,val,test}/
    └── labels/{train,val,test}/
```

Cada imagem deve ter um `.txt` de mesmo nome e caminho relativo em `labels`. As linhas usam `classe x_centro y_centro largura altura`, com coordenadas normalizadas. Um `.txt` vazio indica uma imagem negativa. Os IDs do YAML fornecido são:

| ID | Classe |
|---|---|
| 0 | person |
| 1 | bicycle |
| 2 | car |
| 3 | motorcycle |
| 4 | airplane |
| 5 | bus |
| 6 | train |
| 7 | truck |

`configs/yolo.yaml` preserva essas classes e os splits `images/train`, `images/val` e `images/test`. O `path` absoluto do YAML original é substituído por `--dataset-root`. As câmeras são descobertas nas subpastas, ordenadas por nome e associadas a IDs de cliente no arquivo `data/clients.yaml` de cada execução.

Nenhuma imagem é copiada ou reparticionada. O preparador gera YAMLs por câmera e um YAML global com listas de diretórios. Todos os três splits devem ter imagens, e cada câmera precisa de pelo menos um objeto anotado no treino. Classes, caixas, imagens corrompidas, labels órfãos e conteúdo duplicado entre splits são verificados antes do treinamento. A Ultralytics poderá criar caches de labels ao lado dos dados.

A comparação de conteúdo não comprova separação temporal: confira na origem que frames vizinhos do mesmo vídeo não vazaram entre treino, validação e teste.

Valide primeiro, sem carregar pesos ou usar GPU:

```bash
uv run --project yolo --locked python src/system/yolo/run.py \
  --config configs/yolo.yaml \
  --dataset-root /caminho/dataset \
  --validate-only
```

Labels ausentes causam erro por padrão. Se tiver confirmado que representam imagens negativas, use `--allow-missing-labels`; os arquivos ausentes continuam listados em `data/validation_report.yaml`.

## Executar

Execute os comandos a partir da raiz do repositório.

Teste inicial com duas câmeras e duas rodadas:

```bash
uv run --project yolo python src/system/yolo/run.py \
  --config configs/yolo.yaml \
  --dataset-root /caminho/dataset \
  --cameras camera_01 camera_02 \
  --rounds 2
```

Treinamento completo, com todas as câmeras descobertas:

```bash
uv run --project yolo python src/system/yolo/run.py \
  --config configs/yolo.yaml \
  --dataset-root /caminho/dataset
```

Teste reduzido em CPU:

```bash
uv run --project yolo --locked python src/system/yolo/run.py \
  --config configs/yolo.yaml \
  --dataset-root /caminho/dataset \
  --cameras camera_01 camera_02 \
  --rounds 2 --device cpu --imgsz 64 --batch 2
```

O servidor também aceita o mesmo fluxo pelo ponto de entrada existente:

```bash
uv run --project yolo --locked python src/system/server.py \
  --model yolo --config configs/yolo.yaml --dataset-root /caminho/dataset
```

O lançador inicia e encerra `client.py --model yolo` automaticamente, um processo por câmera e rodada; não é necessário abrir terminais adicionais. Não inicie clientes extras manualmente na porta interna do experimento.

### Parâmetros

| Opção | Padrão | Uso |
|---|---|---|
| `--cameras` | Todas | Nomes exatos das subpastas, separados por espaços |
| `--rounds` | 20 | Rodadas globais |
| `--local-epochs` | 1 | Épocas de cada cliente em cada rodada |
| `--imgsz` | 640 | Resolução, múltiplo de 32 |
| `--batch` | 4 | Imagens por batch; reduza em caso de falta de memória |
| `--device` | `0` | Índice de uma GPU CUDA ou `cpu` |
| `--weights` | `yolo11n.pt` | Checkpoint inicial; arquitetura de detecção YOLO11n |
| `--output` | `src/results/yolo` | Pasta base; cada execução cria uma subpasta única |
| `--timeout` | 3600 | Segundos máximos de espera pela resposta de um cliente |

AdamW, `lr0=0.001`, seed 42, AMP desativado e `workers=0` estão em `configs/yolo.yaml`. Opções de linha de comando prevalecem sobre a configuração. Cada rodada recria o otimizador, desativa warmup e parada antecipada local, usa `nbs=batch` e `close_mosaic=0`. As demais opções de aumento de dados seguem a Ultralytics 8.3.0.

A cabeça com oito classes é inicializada uma vez, carregando os pesos compatíveis do modelo pré-treinado. Todos os clientes começam cada rodada com exatamente os mesmos pesos globais. O servidor agrega o modelo bruto da última época local, antes da seleção de checkpoint/EMA da Ultralytics. Tensores flutuantes, incluindo estatísticas de BatchNorm, são agregados em FP32; buffers inteiros permanecem iguais aos globais. A versão não aplica poda, FedALA, LoRA ou compressão.

## Avaliação e resultados

O caminho de resultados é impresso ao iniciar. Cada execução produz:

```text
src/results/yolo/<execucao>/
├── config.yaml
├── data/                  # YAMLs, clientes e relatório do dataset
├── baseline.json          # Avaliação inicial antes do ajuste federado
├── metrics.csv
├── metrics.json           # Global, câmeras, classes, tempos e bytes
├── test.json              # Teste final do melhor modelo
├── best_global.pt
├── last_global.pt
├── checkpoints/round_001.pt
├── clients/               # Artefatos do treinamento local
├── evaluation/            # Avaliações globais e por câmera
└── logs/                  # Um log por câmera e rodada
```

A referência inicial é o modelo de oito classes recém-inicializado a partir dos pesos pré-treinados, antes de ajustar ao dataset. Ela não representa o detector COCO original de 80 classes.

O modelo global é avaliado após cada rodada sobre a **união das imagens de validação**. O mAP global é calculado nessa união, não pela média dos mAPs das câmeras. São registrados mAP50, mAP50–95, precisão e recall, além de resultados por câmera e por classe com anotações no split. Classes ausentes não recebem uma métrica fictícia.

O melhor modelo é selecionado por mAP50–95 de validação; em empate, permanece a primeira rodada. `test` é usado somente ao concluir, para avaliar esse modelo globalmente e por câmera. Os tempos locais incluem treino e validação interna da Ultralytics; `exchange_seconds` inclui o tempo esperando o cliente. Os bytes incluem o cabeçalho de transporte.

Se um cliente falhar, nenhum update daquela rodada será agregado nem substituirá o último checkpoint global concluído. A execução encerra com erro e indica o log. Não há retomada automática de rodada nesta versão.

### Inferência

```bash
uv run --project yolo --locked python - <<'PY'
from ultralytics import YOLO
model = YOLO('src/results/yolo/<execucao>/best_global.pt')
model.predict(source='/caminho/imagem.jpg', device=0, save=True)
PY
```

Para CPU, troque `device=0` por `device='cpu'`.

## Testes e solução de erros

```bash
uv run --project yolo --locked pytest tests/yolo -q
```

Teste completo sintético, duas câmeras e duas rodadas em CPU (baixa os pesos se necessário):

```bash
PUMA_YOLO_SMOKE=1 uv run --project yolo --locked \
  pytest tests/yolo/test_smoke.py -q -s
```

| Problema | Ação |
|---|---|
| CUDA indisponível | Verifique driver e wheel PyTorch; use `--device cpu` para teste pequeno |
| Falta de memória | Reduza `--batch 2` ou `--batch 1`, e depois a resolução; reinicie o experimento |
| Dataset inválido | Leia `data/validation_report.yaml` na pasta de resultados |
| Label ausente | Crie `.txt` vazio se a imagem for negativa, ou autorize explicitamente os negativos sem label |
| Conteúdo duplicado entre splits | Corrija os splits na origem e valide novamente |
| Cliente falhou ou excedeu tempo | Consulte `logs/`; aumente `--timeout` se o treino local precisa de mais tempo |
| Download de pesos falhou | Passe um checkpoint local com `--weights` |

Os testes sintéticos verificam funcionamento e contratos de agregação; não estimam qualidade no dataset real.
