# spec.md — Módulo de IA para Detecção Automática de Doenças Foliares em Alface

> **Documento de especificação para agente de geração de código.**
> Escopo: **exclusivamente** o módulo de Inteligência Artificial (visão computacional) em Python.
> Fora de escopo: firmware ESP32, broker MQTT, banco de dados, dashboard React, app mobile.
> Idioma do código: identificadores e docstrings em português; mensagens de log em português.

---

## 1. Contexto e objetivo

O TCC *"Sistema integrado de hardware e software para detecção precoce de doenças foliares em alface (Lactuca sativa) em estufas de agricultura familiar"* (UNIP, 2026) propõe um ecossistema IoT cujo núcleo é um modelo de visão computacional baseado em **Ultralytics YOLO11**.

Este módulo deve entregar:

1. Um **modelo treinado** capaz de classificar imagens de folhas de alface em três classes: **saudável**, **bacteriano** (*Xanthomonas campestris*) e **fúngico** (*Bremia lactucae*).
2. Um **pipeline reproduzível** de preparação de dados, treinamento, avaliação e exportação.
3. Uma **camada de inferência** (`predizer(imagem) -> resultado`) com contrato estável, pronta para ser consumida pela API FastAPI do back-end.

Requisitos do TCC rastreados por este módulo:

| ID | Requisito | Meta |
| --- | --- | --- |
| RF02 | Detectar e classificar patógenos foliares (bacteriano, fúngico ou saudável) | Funcional |
| RNF01 | Tempo de inferência por imagem | **< 100 ms** |
| RNF02 | Acurácia na classificação das três classes | **> 90%** |
| RNF07 | Código documentado e versionado em Git | Funcional |
| H1 | Acurácia > 90% na distinção bacteriano × fúngico em estágios iniciais | Hipótese |
| H2 | Inferência em tempo real < 100 ms | Hipótese |
| CT01 | Detecção de *Bremia lactucae* | Classe `fungico`, confiança ≥ 0,85 |
| CT02 | Detecção de *Xanthomonas campestris* | Classe `bacteriano`, confiança ≥ 0,85 |
| CT03 | Classificação de folha saudável | Classe `saudavel`, recall ≥ 90% |
| CT04 | Tempo de inferência no servidor | < 100 ms |

---

## 2. Divergência crítica: classificação × detecção

**O agente NÃO deve tentar treinar um modelo de detecção com o dataset atual.**

O texto do TCC (seções 5.3.4, 6.1.1, 6.2.6.3, tabela `deteccoes`) descreve consistentemente **detecção de objetos** com *bounding boxes*, anotações em formato YOLO `.txt` e pré-processamento em 640×640. O dataset disponível em `desenvolvimento/dataset/` é, de fato, um **dataset de classificação de imagem inteira**: diretórios por classe, **zero arquivos de anotação**, nenhum `data.yaml`.

Decisão a implementar:

- **Tarefa:** `classify` (Ultralytics YOLO11-cls).
- **Modelo base:** `yolo11n-cls.pt` (principal) e `yolo11s-cls.pt` (comparativo). São as variantes indicadas na Tabela 3 do TCC para dispositivos embarcados.
- O agente deve registrar essa divergência em `docs/DIVERGENCIAS.md`, com o texto de justificativa abaixo, para que o autor a incorpore ao TCC:

> O dataset público adotado fornece rótulos em nível de imagem, sem anotações de caixas delimitadoras. Optou-se, portanto, pela tarefa de classificação de imagens (YOLO11-cls), que atende integralmente ao requisito RF02 — identificar se a folha está saudável, com infecção bacteriana ou fúngica — e aos critérios de aceitação CT01–CT03. A localização por caixas delimitadoras exigiria anotação manual do conjunto, atividade não prevista no cronograma da Seção 6.2.6.6.

Consequências que o agente deve tratar:

- O contrato de saída da inferência mantém os campos `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h`, porém **sempre `None`**. As colunas correspondentes na tabela `deteccoes` precisam ser `NULLABLE` — anotar isso em `docs/DIVERGENCIAS.md`.
- Como substituto de baixo custo para a visualização "onde está a lesão" exigida pelo dashboard (Seção 7.4), implementar **mapa de calor Grad-CAM** (item 8, opcional-recomendado).
- `imgsz` de treino será **224** (padrão de classificação), não 640. Justificativa: a mediana de resolução do dataset é 474×355; 640 apenas interpola sem agregar informação, e triplica o custo de CPU. Registrar em `docs/DIVERGENCIAS.md`.

---

## 3. Ambiente de execução (verificado nesta máquina)

```
Python      3.13.5
ultralytics 8.4.95
torch       2.13.0+cpu     # CUDA NÃO disponível
Plataforma  Windows 11 (win32)
```

Regras:

- O código **deve funcionar em CPU**. Detectar o dispositivo com `torch.cuda.is_available()` e selecionar `device="cuda"` ou `device="cpu"` automaticamente; nunca hard-codar.
- Em CPU, desabilitar AMP (`amp=False`) e usar `workers=0` no Windows para evitar erros de multiprocessing.
- Treinar `yolo11n-cls` em 224 px sobre ~2,4 mil imagens em CPU é viável (estimativa: 30–90 s/época). Se o usuário quiser acelerar, o agente deve gerar também `desenvolvimento/notebooks/treino_colab.ipynb` para execução com GPU no Google Colab — **sem** tornar o Colab obrigatório.
- Fixar sementes (`seed=42`, `deterministic=True`) e registrar as versões das bibliotecas em `docs/AMBIENTE.md`.

Dependências a declarar em `requirements.txt`:

```
ultralytics>=8.4.95
torch>=2.13.0
pillow>=11.0
numpy>=2.0
pandas>=2.2
matplotlib>=3.9
scikit-learn>=1.5
onnx>=1.17
onnxruntime>=1.20
```

---

## 4. Dataset: estado real medido

Raiz: `desenvolvimento/dataset/` — **2.813 arquivos**, todos legíveis, 100% RGB (1 arquivo RGBA a converter).

### 4.1 Distribuição atual

| Split | Bacterial | fungal | healthy | Total | % |
| --- | ---: | ---: | ---: | ---: | ---: |
| `train` | 732 | 581 | 1.049 | 2.362 | 84,0% |
| `valid` | 75 | 75 | 75 | 225 | 8,0% |
| `test`  | 75 | 75 | 76 | 226 | 8,0% |
| **Total** | **882** | **731** | **1.200** | **2.813** | 100% |

Formatos: 1.825 `.png`, 981 `.jpg`, 7 `.jpeg`.
Resolução: largura 129–2048 px (mediana 474), altura 129–2048 px (mediana 355). Tamanhos mais frequentes: 225×225 (161), 275×183 (101), 474×355 (100). Apenas 354 imagens são quadradas.

### 4.2 Problemas identificados — tratamento obrigatório

**(P1) Vazamento entre splits (duplicatas exatas).**
Dos 2.813 arquivos, apenas **2.216 têm hash MD5 distinto** — 597 arquivos são cópias byte-idênticas de outros. Existem **373 grupos de duplicatas**, dos quais **104 cruzam a fronteira dos splits**. Exemplos verificados:

```
train/Bacterial/ba14_4.png   ≡  test/Bacterial/ba14_0.png
train/Bacterial/ba13_4.png   ≡  valid/Bacterial/ba13_1.png
train/Bacterial/ba1_4.png    ≡  valid/Bacterial/ba1_0.png  ≡  test/Bacterial/ba1_1.png
```

Métricas medidas sobre o `test/` atual são **cientificamente inválidas** — o modelo estaria sendo avaliado em imagens que memorizou. Isso invalidaria a verificação de RNF02/H1.

**(P2) Vazamento de grupo (variantes aumentadas da mesma folha).**
O padrão de nome `<origem>_<n>` (ex.: `ba12_0`, `ba12_2`, `ba12_3`) indica variantes aumentadas de uma mesma folha original. **115 grupos de origem** aparecem em mais de um split. Mesmo quando não são byte-idênticas, são quase-duplicatas.

**(P3) Diversidade real muito baixa na classe `Bacterial`.**
Contagem de origens distintas (prefixo antes de `_`):

| Classe | Imagens | Origens distintas |
| --- | ---: | ---: |
| healthy | 1.200 | 601 |
| fungal | 731 | 328 |
| **Bacterial** | **882** | **≈ 54** |

As 282 imagens `ba*` da classe `Bacterial` derivam de aproximadamente **53 folhas originais**. É um teto real de generalização e deve constar da Seção 8 (Limitações) do TCC.

**(P4) Splits dominados por dados sintéticos.**
Há 800 arquivos nomeados `augmented_*.jpg`, sem origem rastreável. Eles compõem **2/3 de `valid/` (150/225) e 2/3 de `test/` (150/226)**. Em `train/fungal` não existe nenhum `augmented_*`, mas em `valid/fungal` e `test/fungal` existem 50 cada — procedência inconsistente.

**(P5) Proporção divergente do TCC.**
A Seção 6.1.1 especifica divisão estratificada 70/15/15. A divisão atual é 84/8/8.

**(P6) Desbalanceamento.**
Em `train`: healthy 44,4%, Bacterial 31,0%, fungal 24,6%.

### 4.3 Etapa obrigatória: `desenvolvimento/scripts/01_preparar_dataset.py`

O agente **deve** gerar este script e executá-lo antes de qualquer treino. Ele **não modifica** `desenvolvimento/dataset/` (que permanece intacto como fonte imutável) e produz `desenvolvimento/dataset_clean/`.

Comportamento exigido:

1. Varrer recursivamente `desenvolvimento/dataset/` e catalogar todos os arquivos em um DataFrame com: caminho, classe, split original, hash MD5, largura, altura, formato, modo de cor.
2. **Deduplicar por hash MD5**, mantendo uma única cópia por hash (a de menor caminho lexicográfico, para determinismo).
3. **Definir a chave de agrupamento** `grupo_origem` = nome do arquivo sem extensão, removendo sufixos ` - Copy`, ` - Copy (2)` e o sufixo `_<dígitos>` final. Arquivos `augmented_*` recebem `grupo_origem = "augmented_<classe>_<indice>"` (grupo unitário, pois a origem é irrecuperável).
4. **Re-dividir com `StratifiedGroupKFold`** (scikit-learn) ou equivalente, em **70% treino / 15% validação / 15% teste**, garantindo que:
   - nenhum `grupo_origem` apareça em mais de um split;
   - a proporção entre as três classes seja preservada em cada split;
   - `random_state=42`.
5. Normalizar as imagens de saída: converter para RGB (tratar o arquivo RGBA), reescrever como `.jpg` com qualidade 95, preservando a proporção original **sem** redimensionar (o redimensionamento fica a cargo do `imgsz` do Ultralytics).
6. Escrever a árvore no layout que o Ultralytics `classify` espera — atenção: o diretório precisa chamar-se **`val`**, não `valid`:

```
desenvolvimento/dataset_clean/
├── train/
│   ├── saudavel/
│   ├── bacteriano/
│   └── fungico/
├── val/
│   ├── saudavel/ ...
└── test/
    ├── saudavel/ ...
```

7. Emitir `desenvolvimento/reports/dataset_auditoria.md` e `desenvolvimento/reports/dataset_inventario.csv` contendo: contagens antes/depois, nº de duplicatas removidas, nº de grupos por classe, distribuição final por split, histograma de resoluções, e a **lista completa dos 104 grupos de duplicatas cruzadas** que foram resolvidos. Este relatório é material direto para as Seções 6.1.1 e 8 do TCC.
8. Falhar com erro explícito (`sys.exit(1)`) se, após a re-divisão, qualquer `grupo_origem` ou hash aparecer em mais de um split — verificação de integridade obrigatória.

### 4.4 Mapeamento de classes — canônico e imutável

| Pasta no dataset original | Índice | Rótulo canônico (API/banco) | Rótulo para exibição (dashboard) |
| --- | :---: | --- | --- |
| `Bacterial` | 0 | `bacteriano` | Bacteriano (*Xanthomonas campestris*) |
| `fungal` | 1 | `fungico` | Fúngico (*Bremia lactucae*) |
| `healthy` | 2 | `saudavel` | Saudável |

Os rótulos canônicos são exatamente os valores do `ENUM` da coluna `classe` na tabela `deteccoes` do TCC (`'saudavel','bacteriano','fungico'`) — sem acentos, minúsculos. Definir esse mapeamento em **um único módulo** (`src/ia_alface/classes.py`) e importá-lo em todos os demais; nunca redeclarar strings de classe.

> A ordem de índices do Ultralytics é a **ordem alfabética dos diretórios**. Como as pastas de saída serão `bacteriano`, `fungico`, `saudavel`, a ordem alfabética coincide com a tabela acima. O código de inferência deve, ainda assim, ler `model.names` em tempo de execução em vez de assumir a ordem.

---

## 5. Estrutura de arquivos a gerar

> **Nota (atualizada após a implementação).** A árvore abaixo reflete a
> organização final adotada: a raiz contém apenas o que vai para produção, e
> todo o ciclo de pesquisa — dataset, scripts, testes, relatórios — foi agrupado
> sob `desenvolvimento/`. Ver `desenvolvimento/README.md`.

```
TCC/
├── README.md                        # índice do projeto
├── spec.md                          # este documento
├── requirements.txt                 # dependências de PRODUÇÃO
├── .gitignore
│
├── api/                             # PRODUÇÃO — serviço HTTP de inferência
│   ├── main.py                      # rotas, ciclo de vida, tratamento de erros
│   ├── esquemas.py                  # contrato de requisição/resposta
│   ├── configuracao.py              # configuração por variáveis de ambiente
│   ├── cliente.py                   # cliente de linha de comando
│   ├── cliente_web.html             # página de teste interativo
│   └── README.md
│
├── src/ia_alface/                   # PRODUÇÃO — biblioteca
│   ├── __init__.py
│   ├── classes.py                   # mapeamento canônico de classes
│   ├── config.py                    # caminhos + hiperparâmetros
│   ├── dados.py                     # inventário, hashing, agrupamento, split
│   ├── treino.py                    # rotina de treinamento
│   ├── avaliacao.py                 # métricas, matriz de confusão, gráficos
│   ├── inferencia.py                # classe DetectorDoencasAlface
│   ├── explicabilidade.py           # Grad-CAM
│   └── exportacao.py                # exportação ONNX/TorchScript
│
├── modelos/                         # PRODUÇÃO — pesos (.pt, .onnx); não versionados
├── docs/                            # documentação
│   ├── README.md
│   ├── AMBIENTE.md
│   ├── DIVERGENCIAS.md
│   └── DOCUMENTACAO_TECNICA_ABNT.md
│
└── desenvolvimento/                 # NÃO VAI PARA PRODUÇÃO
    ├── README.md
    ├── requirements-dev.txt         # dependências extras de treino/teste
    ├── scripts/                     # pipeline 01–05 + utilitários
    ├── tests/                       # testes automatizados
    ├── notebooks/                   # treino em GPU no Colab
    ├── reports/                     # saídas para o TCC
    ├── tcc/                         # texto da monografia e material de apoio
    ├── dataset/                     # FONTE IMUTÁVEL — não versionado
    ├── dataset_clean/               # gerado por 01_preparar_dataset.py
    ├── runs/                        # execuções do Ultralytics
    └── pesos_base/                  # pesos pré-treinados baixados
```

Regras de código:

- Type hints em todas as funções públicas; docstrings estilo Google em português.
- Nenhum caminho absoluto no código — resolver tudo relativo à raiz do projeto via `pathlib.Path(__file__).resolve().parents[N]`.
- Toda configuração em `config.py` como `@dataclass`, sobrescrevível por argumentos de linha de comando (`argparse`).
- Logging via `logging` (nível `INFO`), nunca `print` em módulos de biblioteca.
- `.gitignore` deve excluir `desenvolvimento/dataset_clean/`, `desenvolvimento/runs/`, `modelos/*.pt`, `modelos/*.onnx`, `__pycache__/`.

---

## 6. Treinamento — `desenvolvimento/scripts/02_treinar.py`

Configuração base (registrar todos os valores em `desenvolvimento/reports/treino_config.json`):

```python
model = YOLO("yolo11n-cls.pt")          # transfer learning a partir de pesos ImageNet
model.train(
    data="dataset_clean",
    epochs=100,
    imgsz=224,
    batch=32,                            # reduzir para 16 se houver falta de memória
    patience=20,                          # early stopping
    optimizer="AdamW",
    lr0=0.001,
    cos_lr=True,
    dropout=0.1,
    label_smoothing=0.05,
    seed=42,
    deterministic=True,
    device=dispositivo,                   # "cuda" se disponível, senão "cpu"
    amp=(dispositivo == "cuda"),
    workers=0,                            # obrigatório no Windows
    project="desenvolvimento/runs/classify",
    name="yolo11n_cls_alface",
    plots=True,
    # Data augmentation em tempo de treino (Seção 6.1.1 do TCC):
    fliplr=0.5,
    flipud=0.3,
    degrees=20.0,
    scale=0.4,
    translate=0.1,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.4,                            # variação de brilho — mitiga a limitação de
                                          # iluminação apontada na Seção 8 do TCC
    erasing=0.2,
)
```

Observações de implementação:

- O aumento de dados deve ser feito **exclusivamente em tempo de treino pelo Ultralytics**, nunca gravado em disco. O dataset original já foi inflado por augmentation offline (P4), o que foi a causa raiz do vazamento; repetir esse padrão reintroduziria o problema.
- Treinar também `yolo11s-cls.pt` com a mesma configuração e produzir `desenvolvimento/reports/comparativo_variantes.md` (parâmetros, acurácia top-1, latência CPU, tamanho do arquivo). Isso preenche a Tabela 3 do TCC com dados próprios.
- Não usar `class_weights` como primeira tentativa. Se a matriz de confusão revelar que `fungico` (classe minoritária, 24,6%) apresenta recall abaixo de 0,90, então aplicar sobre-amostragem da classe minoritária ou perda ponderada, e documentar a intervenção.
- Copiar `desenvolvimento/runs/classify/<name>/weights/best.pt` para `modelos/yolo11n_cls_alface.pt` ao final.

---

## 7. Avaliação — `desenvolvimento/scripts/03_avaliar.py`

Avaliar **sempre no split `test` do `dataset_clean`**, jamais no `desenvolvimento/dataset/test` original.

Saídas obrigatórias em `desenvolvimento/reports/`:

1. `metricas_test.json` — acurácia global (top-1), e por classe: precisão, recall, F1-score, suporte.
2. `matriz_confusao.png` e `matriz_confusao.csv` — 3×3, valores absolutos e normalizados por linha.
3. `curva_treino.png` — perda de treino/validação e acurácia por época.
4. `erros_classificacao/` — as 20 imagens com maior erro de confiança, com rótulo verdadeiro e predito no nome do arquivo. Insumo qualitativo para a discussão do TCC.
5. `avaliacao_test.md` — relatório em português consolidando os itens acima, com uma tabela explícita de verificação:

| Critério | Meta | Medido | Status |
| --- | --- | --- | --- |
| RNF02 / H1 — acurácia global | > 90% | *(preencher)* | *(APROVADO/REPROVADO)* |
| CT01 — confiança média em `fungico` | ≥ 0,85 | *(preencher)* | |
| CT02 — confiança média em `bacteriano` | ≥ 0,85 | *(preencher)* | |
| CT03 — recall de `saudavel` | ≥ 90% | *(preencher)* | |
| CT04 / RNF01 — latência mediana por imagem | < 100 ms | *(preencher)* | |

**O agente não deve ajustar hiperparâmetros observando o split `test`.** Todo ajuste usa exclusivamente o split `val`. O `test` é avaliado uma única vez, no final.

**Reportar o resultado real, mesmo se abaixo da meta.** Se a acurácia ficar abaixo de 90%, o relatório deve declará-lo explicitamente e listar hipóteses diagnósticas (diversidade insuficiente da classe `Bacterial` — ver P3; confusão entre lesão bacteriana e fúngica em estágio inicial; variação de iluminação). Não inflar métricas, não reintroduzir dados vazados para atingir o número.

---

## 8. Explicabilidade (opcional, recomendado)

`src/ia_alface/explicabilidade.py`: gerar mapa de calor Grad-CAM sobre a última camada convolucional do backbone, retornando um PNG com a sobreposição.

Justificativa: o dashboard especificado na Seção 7.4 do TCC prevê "bounding boxes sobrepostos às imagens capturadas". Como o modelo é de classificação, o Grad-CAM entrega a informação de localização aproximada da lesão sem exigir anotação manual do dataset — atendendo à intenção do requisito. Se implementado, adicionar ao contrato de saída o campo `heatmap_url` e documentar em `docs/DIVERGENCIAS.md`.

---

## 9. Contrato de inferência para integração IoT

`src/ia_alface/inferencia.py` deve expor uma classe com carregamento único do modelo (o back-end FastAPI a instanciará uma vez, no *startup*, e reutilizará entre requisições — carregar o modelo por requisição inviabilizaria RNF01).

```python
class DetectorDoencasAlface:
    def __init__(
        self,
        caminho_modelo: str | Path = "modelos/yolo11n_cls_alface.pt",
        dispositivo: str | None = None,       # None => autodetecta
        limiar_confianca: float = 0.85,       # CT01/CT02
    ) -> None: ...

    def predizer(self, imagem: str | Path | bytes | np.ndarray) -> ResultadoDeteccao: ...

    def predizer_lote(self, imagens: list) -> list[ResultadoDeteccao]: ...
```

`ResultadoDeteccao` como `@dataclass` serializável (`to_dict()`), com campos alinhados 1:1 à tabela `deteccoes` do TCC:

```python
@dataclass
class ResultadoDeteccao:
    classe: str                     # 'saudavel' | 'bacteriano' | 'fungico'
    confianca: float                # [0.0, 1.0], arredondado a 3 decimais
    probabilidades: dict[str, float]  # score de cada uma das 3 classes
    patogeno_detectado: bool        # True se classe != 'saudavel'
    alerta: bool                    # True se patogeno_detectado E confianca >= limiar
    tempo_inferencia_ms: float      # medido; alimenta a verificação de RNF01
    bbox_x: float | None = None     # sempre None — ver Seção 2
    bbox_y: float | None = None
    bbox_w: float | None = None
    bbox_h: float | None = None
    modelo_versao: str = ""         # nome do arquivo de pesos, para rastreabilidade
    heatmap_url: str | None = None  # ver Seção 8
```

Requisitos comportamentais:

- Aceitar entrada como caminho, `bytes` (o ESP32-CAM envia JPEG via HTTP POST, conforme Seção 6.2.6.3), ou array NumPy.
- Validar a entrada e lançar `ValueError` com mensagem em português para imagem corrompida, vazia ou de formato não suportado. Nunca retornar um resultado silenciosamente inválido — um falso "saudável" causado por imagem corrompida é o pior modo de falha possível neste sistema.
- Ser thread-safe para leitura (o worker MQTT e a rota HTTP são concorrentes).
- `alerta=True` é o gatilho que o back-end usará para acionar a notificação push (RF07/CT07).
- **Não** implementar aqui rotas HTTP, cliente MQTT ou acesso a banco. Este módulo é uma biblioteca pura. Fornecer apenas, em `docs/README.md`, um exemplo de 15 linhas de como integrá-lo a um endpoint FastAPI `/analyze`.

---

## 10. Exportação e benchmark de latência

`desenvolvimento/scripts/04_exportar.py`:
- Exportar para **ONNX** (`opset=17`, `simplify=True`, `imgsz=224`, `dynamic=False`) em `modelos/yolo11n_cls_alface.onnx`.
- Exportar também para **TorchScript**, como alternativa de implantação.
- Validar que a saída ONNX (via `onnxruntime`) coincide com a do modelo PyTorch, com tolerância `atol=1e-3`, em pelo menos 20 imagens do split `test`. Falhar se divergir.

`desenvolvimento/scripts/05_benchmark_latencia.py`:
- Medir latência de `predizer()` sobre 100 imagens do split `test`, após 10 iterações de aquecimento.
- Reportar em `desenvolvimento/reports/benchmark_latencia.md`: mediana, média, p95, p99, mínimo, máximo — separadamente para PyTorch/CPU e ONNX/CPU, e para GPU se disponível.
- Declarar explicitamente se a mediana atende a RNF01 (< 100 ms) e em qual hardware, informando CPU e SO. A medição vale para o servidor de inferência; a Tabela 16 do TCC compara com dispositivos de borda — não confundir os dois contextos no relatório.

---

## 11. Testes — `desenvolvimento/tests/`

Usar `pytest`. Cobertura mínima exigida:

- `test_classes.py`: o mapeamento pasta → índice → rótulo canônico é bijetivo; os três rótulos canônicos correspondem exatamente ao `ENUM` do banco.
- `test_dados.py`: a função de `grupo_origem` normaliza corretamente `ba12_3.png`, `ba1_4 - Copy.png`, `augmented_0.jpg`, `h100_2.png`; a re-divisão não produz interseção de grupos nem de hashes entre splits (usar um conjunto sintético pequeno, não o dataset real).
- `test_inferencia.py`: `predizer()` sobre uma imagem sintética retorna um `ResultadoDeteccao` válido; `classe` pertence ao conjunto canônico; `sum(probabilidades.values()) ≈ 1.0`; entrada corrompida levanta `ValueError`; `alerta` respeita o limiar.

Os testes não devem depender de pesos treinados existirem — usar `yolo11n-cls.pt` pré-treinado ou `monkeypatch` quando `modelos/` estiver vazio.

---

## 12. Ordem de execução

```bash
pip install -r requirements.txt
python desenvolvimento/scripts/01_preparar_dataset.py     # gera desenvolvimento/dataset_clean/ + auditoria — OBRIGATÓRIO
python desenvolvimento/scripts/02_treinar.py --modelo yolo11n-cls.pt
python desenvolvimento/scripts/02_treinar.py --modelo yolo11s-cls.pt
python desenvolvimento/scripts/03_avaliar.py --pesos modelos/yolo11n_cls_alface.pt
python desenvolvimento/scripts/04_exportar.py
python desenvolvimento/scripts/05_benchmark_latencia.py
pytest -q
```

---

## 13. Critérios de aceitação do módulo

O módulo está concluído quando:

1. `python desenvolvimento/scripts/01_preparar_dataset.py` executa sem erro e a verificação de integridade confirma **zero** hashes e **zero** grupos de origem compartilhados entre `train`, `val` e `test`.
2. `desenvolvimento/dataset_clean/` está dividido em 70/15/15 com estratificação por classe.
3. `modelos/yolo11n_cls_alface.pt` existe e carrega.
4. `desenvolvimento/reports/avaliacao_test.md` existe, com a tabela de verificação preenchida com valores **reais medidos** e status honesto por critério.
5. `desenvolvimento/reports/benchmark_latencia.md` reporta a latência mediana medida e conclui sobre RNF01.
6. `modelos/yolo11n_cls_alface.onnx` existe e sua saída foi validada contra a do PyTorch.
7. `DetectorDoencasAlface.predizer()` funciona a partir de `bytes` de JPEG e retorna o contrato da Seção 9.
8. `pytest` passa integralmente.
9. `docs/DIVERGENCIAS.md` documenta: classificação × detecção, `bbox` nulo, `imgsz` 224 × 640, e — se implementado — o Grad-CAM.
10. `desenvolvimento/reports/dataset_auditoria.md` documenta os problemas P1–P6, prontos para as Seções 6.1.1 e 8 do TCC.

## 14. Restrições

- **Não escrever em `desenvolvimento/dataset/`.** É a fonte imutável.
- **Não modificar** `ENTREGA FINAL corrigido 13_11_2026_1.md`, `documentacao_yolov11.md` nem este `spec.md`.
- **Não implementar** firmware, MQTT, PostgreSQL, React ou React Native.
- **Não baixar** datasets externos nem pesos que não sejam os oficiais `yolo11*-cls.pt` do Ultralytics.
- **Não reportar métrica que não tenha sido medida.** Onde um número for desconhecido, escrever "não medido" — nunca estimar e apresentar como resultado.
- Ultralytics YOLO11 é licenciado sob **AGPL-3.0**. Registrar em `docs/README.md` que o uso é acadêmico, conforme os termos da licença.
