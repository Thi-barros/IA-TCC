# Detecção Automática de Doenças Foliares em Alface

Módulo de inteligência artificial do TCC *Sistema integrado de hardware e
software para detecção precoce de doenças foliares em alface (Lactuca sativa) em
estufas de agricultura familiar* — UNIP, 2026.

Classificador de visão computacional baseado em **Ultralytics YOLO11-cls** que
identifica folhas de alface como **saudáveis**, com infecção **bacteriana**
(*Xanthomonas campestris*) ou **fúngica** (*Bremia lactucae*), exposto por uma
API REST para consumo pelo restante do sistema.

| Métrica | Resultado |
| --- | ---: |
| Acurácia no conjunto de teste | **95,06%** |
| F1-score macro | 0,9415 |
| Latência mediana (ONNX/CPU) | 12,73 ms |
| Latência ponta a ponta via HTTP | 53,3 ms |
| Testes automatizados | 93 |

Detalhamento completo em
[`docs/DOCUMENTACAO_TECNICA_ABNT.md`](docs/DOCUMENTACAO_TECNICA_ABNT.md).

---

## Organização do repositório

A raiz contém **apenas o que vai para produção**. Todo o ciclo de pesquisa —
dataset, treinamento, avaliação, relatórios — está agrupado em
`desenvolvimento/` e não é implantado.

```
TCC/
├── api/                 serviço HTTP de inferência        PRODUÇÃO
├── src/ia_alface/       biblioteca de IA                  PRODUÇÃO
├── modelos/             pesos treinados (.pt, .onnx)      PRODUÇÃO
├── docs/                documentação
├── requirements.txt     dependências de produção
├── spec.md              especificação do módulo
└── desenvolvimento/     tudo que NÃO vai para produção
```

| Caminho | Conteúdo | Vai para produção |
| --- | --- | :---: |
| `api/` | Rotas HTTP, cliente CLI, página de teste | sim |
| `src/ia_alface/` | Biblioteca: classes, inferência, Grad-CAM | sim |
| `modelos/` | Pesos treinados (não versionados) | sim |
| `docs/` | Documentação técnica e do projeto | sim |
| `desenvolvimento/scripts/` | Pipeline de treino e avaliação | não |
| `desenvolvimento/tests/` | 93 testes automatizados | não |
| `desenvolvimento/notebooks/` | Treinamento em GPU no Colab | não |
| `desenvolvimento/reports/` | Métricas e figuras para o TCC | não |
| `desenvolvimento/dataset*/` | Conjuntos de imagens | não |
| `desenvolvimento/runs/` | Execuções do Ultralytics | não |
| `desenvolvimento/tcc/` | Texto da monografia | não |

Ver [`desenvolvimento/README.md`](desenvolvimento/README.md) para o detalhamento
do que há lá e por quê.

---

## Uso em produção

Requer apenas `api/`, `src/`, `modelos/` e `requirements.txt`.

```bash
pip install -r requirements.txt
```

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Pré-requisito: `modelos/yolo11n_cls_alface.pt` deve existir. Sem ele o serviço
sobe, mas responde `503` nas rotas de inferência e informa a causa em `/health`.

| Rota | Finalidade |
| --- | --- |
| `GET /teste` | Página para arrastar imagens e testar |
| `GET /health` | Verificação de disponibilidade |
| `GET /info` | Metadados do modelo e classes |
| `POST /analyze` | Classificar imagem (`multipart/form-data`) |
| `POST /analyze/raw` | Classificar imagem (corpo bruto — ESP32-CAM) |
| `GET /docs` | Documentação interativa |

Contrato completo, exemplos em Python, JavaScript e Arduino, e variáveis de
configuração em [`api/README.md`](api/README.md).

---

## Uso em desenvolvimento

```bash
pip install -r desenvolvimento/requirements-dev.txt
```

```bash
python desenvolvimento/scripts/01_preparar_dataset.py
```

```bash
python desenvolvimento/scripts/02_treinar.py
```

```bash
python desenvolvimento/scripts/03_avaliar.py --pesos modelos/yolo11n_cls_alface.pt
```

```bash
python -m pytest -q
```

O passo 1 é obrigatório antes de treinar: o dataset original contém vazamento
entre os subconjuntos, e a rotina o elimina. Detalhes em
[`desenvolvimento/reports/dataset_auditoria.md`](desenvolvimento/reports/dataset_auditoria.md).

Guia completo em [`docs/README.md`](docs/README.md).

---

## Documentação

| Arquivo | Conteúdo |
| --- | --- |
| [`spec.md`](spec.md) | Especificação que originou a implementação |
| [`docs/README.md`](docs/README.md) | Guia de uso do módulo |
| [`docs/DOCUMENTACAO_TECNICA_ABNT.md`](docs/DOCUMENTACAO_TECNICA_ABNT.md) | Documentação técnica em norma ABNT, para o TCC |
| [`docs/DIVERGENCIAS.md`](docs/DIVERGENCIAS.md) | Divergências entre o projeto e a implementação |
| [`docs/AMBIENTE.md`](docs/AMBIENTE.md) | Ambiente, versões e desempenho medido |
| [`api/README.md`](api/README.md) | Contrato da API e exemplos de consumo |
| [`desenvolvimento/README.md`](desenvolvimento/README.md) | O que não vai para produção |

---

## Licença

O Ultralytics YOLO11 é distribuído sob **AGPL-3.0**. O uso neste projeto é
estritamente acadêmico, em conformidade com os termos dessa licença. Uso
comercial requer a Enterprise License da Ultralytics.
