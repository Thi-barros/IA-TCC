# Módulo de IA — Detecção de doenças foliares em alface

Classificador de visão computacional baseado em **Ultralytics YOLO11-cls** que
identifica folhas de alface (*Lactuca sativa*) como **saudáveis**, com infecção
**bacteriana** (*Xanthomonas campestris*) ou **fúngica** (*Bremia lactucae*).

Este pacote é o núcleo de IA do TCC *"Sistema integrado de hardware e software
para detecção precoce de doenças foliares em alface em estufas de agricultura
familiar"* (UNIP, 2026). Ele é uma **biblioteca pura**: não expõe rotas HTTP, não
fala MQTT e não acessa banco de dados. A integração fica a cargo do back-end.

Especificação completa: [`spec.md`](../spec.md) na raiz do projeto.

---

## Obtendo o dataset

O diretório `desenvolvimento/dataset/` **não é versionado** (~966 MB, 2.813 arquivos). Datasets
públicos se referenciam pela origem em vez de serem copiados para o repositório.

Baixe o conjunto de folhas de alface do Kaggle e extraia-o na raiz do projeto, de
modo a obter esta estrutura:

```
desenvolvimento/dataset/
├── train/   Bacterial/  fungal/  healthy/
├── valid/   Bacterial/  fungal/  healthy/
└── test/    Bacterial/  fungal/  healthy/
```

> **Preencher:** insira aqui a URL exata do dataset no Kaggle utilizado no
> trabalho, para permitir a reprodução por terceiros e para citação na
> Seção 6.1.1 do TCC.

A divisão original do download contém vazamento entre os splits; o
`desenvolvimento/scripts/01_preparar_dataset.py` corrige isso e gera o `desenvolvimento/dataset_clean/`. Ver
[`desenvolvimento/reports/dataset_auditoria.md`](../desenvolvimento/reports/dataset_auditoria.md).

## Instalação

```bash
pip install -r requirements.txt
```

O pacote roda em CPU e em GPU; o dispositivo é detectado automaticamente. Não é
necessário instalar o pacote — os scripts adicionam `src/` ao `sys.path`.

## Ordem de execução

```bash
python desenvolvimento/scripts/01_preparar_dataset.py
```

```bash
python desenvolvimento/scripts/02_treinar.py --modelo yolo11n-cls.pt
```

```bash
python desenvolvimento/scripts/02_treinar.py --modelo yolo11s-cls.pt
```

```bash
python desenvolvimento/scripts/03_avaliar.py --pesos modelos/yolo11n_cls_alface.pt
```

```bash
python desenvolvimento/scripts/04_exportar.py
```

```bash
python desenvolvimento/scripts/05_benchmark_latencia.py
```

```bash
python -m pytest -q
```

A etapa 1 é **obrigatória** antes de treinar: ela remove o vazamento entre splits
presente no dataset original. Detalhes em
[`desenvolvimento/reports/dataset_auditoria.md`](../desenvolvimento/reports/dataset_auditoria.md).

> Em CPU, cada época leva alguns minutos. Para acelerar, use
> `desenvolvimento/notebooks/treino_colab.ipynb` (local, não versionado) com GPU e
> traga o arquivo `best.pt` de volta para `modelos/`.

### Utilitários auxiliares

```bash
python desenvolvimento/scripts/testar_imagem.py foto.jpg
```

Classifica imagens avulsas ou uma pasta inteira, mostrando probabilidades, tempo
de inferência e se o alerta dispararia. Opções: `--heatmap` (gera Grad-CAM),
`--como-bytes` (simula o envio do ESP32-CAM), `--json`, `--pesos`, `--limiar`.

```bash
python desenvolvimento/scripts/preparar_zip_colab.py
```

Empacota `desenvolvimento/dataset_clean/` em `dataset_clean.zip` para envio ao Colab, reduzindo o
menor lado das imagens para 320 px (de ~220 MB para ~86 MB). Como o modelo treina
em 224 px e a augmentation recorta no mínimo 60% da área, nenhum pixel útil é
perdido. Use `--sem-reducao` para uma cópia fiel.

## Estrutura

| Caminho | Conteúdo |
| --- | --- |
| `src/ia_alface/classes.py` | Mapeamento canônico de classes — **única** fonte dos rótulos |
| `src/ia_alface/config.py` | Caminhos do projeto e hiperparâmetros |
| `src/ia_alface/dados.py` | Inventário, deduplicação e re-divisão do dataset |
| `src/ia_alface/treino.py` | Rotina de treinamento |
| `src/ia_alface/avaliacao.py` | Métricas, matriz de confusão, latência, relatórios |
| `src/ia_alface/inferencia.py` | `DetectorDoencasAlface` — contrato de integração |
| `src/ia_alface/explicabilidade.py` | Mapas de calor Grad-CAM |
| `src/ia_alface/exportacao.py` | ONNX/TorchScript e validação numérica |
| `desenvolvimento/dataset/` | Fonte imutável — **nunca** escrever aqui |
| `desenvolvimento/dataset_clean/` | Dataset re-dividido (gerado; não versionado) |
| `modelos/` | Pesos treinados e modelos exportados |
| `desenvolvimento/reports/` | Saídas destinadas ao texto do TCC |
| `Dockerfile` | Imagem da API para implantação — ver [`IMPLANTACAO.md`](IMPLANTACAO.md) |
| `desenvolvimento/scripts/verificar_blob.py` | Verificação do acesso ao Azure Blob Storage |

## Uso da biblioteca

```python
from ia_alface.inferencia import DetectorDoencasAlface

detector = DetectorDoencasAlface("modelos/yolo11n_cls_alface.pt")

resultado = detector.predizer("caminho/para/folha.jpg")
print(resultado.classe)        # 'saudavel' | 'bacteriano' | 'fungico'
print(resultado.confianca)     # 0.0 .. 1.0
print(resultado.alerta)        # True -> disparar notificação push
print(resultado.to_dict())     # pronto para JSON / persistência
```

## Serviço de inferência HTTP

Uma API REST pronta para uso está implementada em `api/`. Ela expõe o modelo por
HTTP para que o back-end, o dashboard e o ESP32-CAM o consumam sem instalar
PyTorch:

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Documentação interativa em `/docs`; contrato completo, exemplos de consumo em
Python, JavaScript e Arduino, e variáveis de configuração em
[`../api/README.md`](../api/README.md).

Sem subir o serviço, consulte a cópia estática do Swagger em
[`swagger.html`](swagger.html) ou o contrato em [`openapi.json`](openapi.json).
Ambos são regerados por `python desenvolvimento/scripts/gerar_openapi.py`.

Para executar o serviço em container, localmente ou em Azure Container Apps, ver
[`IMPLANTACAO.md`](IMPLANTACAO.md). O `Dockerfile` empacota apenas `api/`,
`src/ia_alface/` e os pesos — `desenvolvimento/` fica de fora.

## Integração manual com FastAPI

O detector deve ser instanciado **uma vez** no *startup* e reutilizado entre
requisições — carregar o modelo a cada requisição inviabiliza o requisito RNF01
(inferência em menos de 100 ms).

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, UploadFile
from ia_alface.inferencia import DetectorDoencasAlface

detector: DetectorDoencasAlface | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    detector = DetectorDoencasAlface("modelos/yolo11n_cls_alface.pt")
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/analyze")
async def analisar(imagem: UploadFile = File(...)):
    try:
        resultado = detector.predizer(await imagem.read())
    except ValueError as erro:
        raise HTTPException(status_code=400, detail=str(erro)) from erro
    # resultado.alerta == True -> persistir em 'deteccoes' e acionar o push (RF07)
    return resultado.to_dict()
```

O `ResultadoDeteccao` foi desenhado para mapear 1:1 a tabela `deteccoes`
especificada na Seção 6.2.6.8 do TCC. **Atenção:** os campos `bbox_*` são sempre
`None` — ver [`DIVERGENCIAS.md`](DIVERGENCIAS.md).

## Mapa de calor (opcional)

```python
from ia_alface.explicabilidade import GradCAM

explicador = GradCAM("modelos/yolo11n_cls_alface.pt")
explicador.salvar_sobreposicao("folha.jpg", "static/heatmaps/folha.png")
```

Reutilize a instância de `GradCAM` entre chamadas: a construção carrega o modelo.

## Relatórios gerados

| Arquivo | Seção do TCC que alimenta |
| --- | --- |
| `desenvolvimento/reports/dataset_auditoria.md` | 6.1.1 (Coleta de dados) e 8 (Limitações) |
| `desenvolvimento/reports/avaliacao_test.md` | 7.1 (Desempenho do modelo) |
| `desenvolvimento/reports/matriz_confusao.png` | 7.1 |
| `desenvolvimento/reports/curva_treino.png` | 7.1 |
| `desenvolvimento/reports/comparativo_variantes.md` | Tabela 3 (variantes YOLOv11) |
| `desenvolvimento/reports/benchmark_latencia.md` | 7.2 (Pipeline de inferência) e RNF01 |
| `desenvolvimento/reports/erros_classificacao/` | Discussão qualitativa |

## Licença

O Ultralytics YOLO11 é distribuído sob **AGPL-3.0**. O uso neste projeto é
**estritamente acadêmico**, em conformidade com os termos dessa licença, que
permite uso educacional e exige que trabalhos derivados sejam publicados sob a
mesma licença. Uso comercial requer a Enterprise License da Ultralytics
(<https://www.ultralytics.com/license>).
