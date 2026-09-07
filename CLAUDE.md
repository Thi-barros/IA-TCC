# Instruções para assistentes de IA

Orientação para trabalhar **neste repositório**. O que construir está em
[`spec.md`](spec.md); a visão geral do projeto está em [`README.md`](README.md).

Projeto: classificador de doenças foliares em alface (YOLO11-cls) + API REST.
Módulo de IA de um TCC de graduação (UNIP, 2026). Código em **português**.

---

## Regras invioláveis

Violar qualquer uma delas produz dano difícil de perceber. Em caso de dúvida,
pergunte antes de contrariar.

1. **`desenvolvimento/dataset/` é imutável.** É a fonte de dados original.
   Nenhum código escreve nela. Saídas vão para `desenvolvimento/dataset_clean/`.

2. **Rótulos de classe vêm só de `src/ia_alface/classes.py`.** Nunca escreva
   `"fungico"`, `"bacteriano"` ou `"saudavel"` como literal em outro arquivo —
   importe. Os valores correspondem ao `ENUM` do PostgreSQL do projeto; uma
   divergência de grafia grava dado errado no banco sem gerar erro.

3. **O split `test` é medido uma única vez.** Todo ajuste de hiperparâmetro usa
   `--split val`. Avaliar repetidamente no teste o transforma em segundo
   conjunto de validação e invalida a métrica final.

4. **Nunca reporte métrica que não foi medida.** Se um número é desconhecido,
   escreva "não medido". Não estime e apresente como resultado. Se a acurácia
   ficar abaixo da meta, reporte assim mesmo.

5. **`src/ia_alface/` é biblioteca pura.** Sem HTTP, MQTT ou banco de dados. A
   camada HTTP vive em `api/` e importa a biblioteca — nunca o contrário.

6. **Aumento de dados só em tempo de treino.** Nunca grave imagens
   transformadas em disco. Foi exatamente isso que causou o vazamento no
   dataset original (104 grupos de duplicatas cruzando splits).

7. **`01_preparar_dataset.py` roda antes de qualquer treino.** O dataset
   original tem vazamento entre splits; a rotina o elimina e falha com exit 1
   se algum sobrar.

---

## Estrutura

A raiz contém **apenas produção**. Todo o ciclo de pesquisa está em
`desenvolvimento/` e não é implantado.

```
api/              serviço HTTP           PRODUÇÃO
src/ia_alface/    biblioteca             PRODUÇÃO
modelos/          pesos .pt/.onnx        PRODUÇÃO
Dockerfile        imagem da API          PRODUÇÃO
docs/             documentação
desenvolvimento/  scripts, testes, dataset, reports, notebooks, tcc
```

A implantação é em **Azure Container Apps** com escala a zero; as imagens
capturadas ficam em **Azure Blob Storage**. Nada disso entra em `src/` nem em
`api/` — a biblioteca e a API continuam sem dependência de nuvem. Ver
`docs/IMPLANTACAO.md`.

**Todos os comandos rodam a partir da raiz do projeto**, nunca de dentro de
subpastas.

### Profundidade de `parents[]` ao resolver a raiz

Difere conforme a localização do arquivo. Errar aqui quebra os imports:

| Arquivo | Expressão |
| --- | --- |
| `api/*.py` | `parents[1]` |
| `src/ia_alface/*.py` | `parents[2]` |
| `desenvolvimento/scripts/*.py` | `parents[2]` |
| `desenvolvimento/tests/conftest.py` | `parents[2]` |

Nunca use caminho absoluto. Caminhos canônicos ficam em
`src/ia_alface/config.py`, classe `Caminhos`.

---

## Comandos

```bash
python -m pytest -q                                    # 101 testes, ~26 s
python -m uvicorn api.main:app --host 0.0.0.0          # API
python desenvolvimento/scripts/01_preparar_dataset.py  # obrigatório antes de treinar
python desenvolvimento/scripts/02_treinar.py
python desenvolvimento/scripts/03_avaliar.py --pesos modelos/yolo11n_cls_alface.pt
python desenvolvimento/scripts/04_exportar.py
python desenvolvimento/scripts/05_benchmark_latencia.py
python desenvolvimento/scripts/testar_imagem.py foto.jpg --heatmap
python api/cliente.py minhas_fotos/                    # via HTTP
python desenvolvimento/scripts/gerar_openapi.py        # docs/openapi.json + docs/swagger.html
python desenvolvimento/scripts/verificar_blob.py       # acesso ao Azure Blob Storage
docker build -t ia-alface:local .                      # imagem da API
docker run --rm -p 8000:8000 ia-alface:local           # container local
```

Rode `pytest` após qualquer alteração em `src/` ou `api/`. Alterou rota ou
esquema? Rode também `gerar_openapi.py`, senão a cópia estática do contrato em
`docs/` passa a mentir sobre o serviço.

---

## Armadilhas conhecidas

Todas já custaram tempo. Não as redescubra.

| Situação | O que fazer |
| --- | --- |
| **Windows** | `workers=0` no Ultralytics é obrigatório; o multiprocessing do DataLoader falha |
| **`label_smoothing`** | Removido no Ultralytics 8.4 (`removed_keys`). Aceito sem erro e ignorado. Não passe |
| **Diretório de validação** | O Ultralytics `classify` exige `val/`, não `valid/` |
| **Carregar `.onnx`/`.torchscript`** | Passe `task="classify"` explicitamente; a dedução só funciona com `.pt` |
| **Inferência no FastAPI** | Sempre via `run_in_threadpool`. Chamada direta em handler `async` trava o event loop por ~40 ms |
| **Grad-CAM** | Exige pesos `.pt`. Não funciona com modelo exportado |
| **Torch em CPU** | ~3,8 min/época; 100 épocas ≈ 6 h. Prefira `desenvolvimento/notebooks/treino_colab.ipynb` |
| **Imagem inválida** | Deve levantar `ValueError` / HTTP 400. **Nunca** retornar resultado — um falso `saudavel` suprime o alerta silenciosamente |
| **PyTorch na imagem Docker** | O pacote do PyPI embute CUDA (~2 GB inúteis em CPU). Instale do índice `download.pytorch.org/whl/cpu` **antes** do `requirements.txt` |
| **OpenCV no container** | O Ultralytics exige `libgl1` e `libglib2.0-0` do sistema. Faltando, a build passa e o container quebra ao subir com `libGL.so.1` |
| **`IA_ALFACE_CORS_ORIGENS=""`** | Variável vazia cai no padrão `["*"]` — mais permissiva que não defini-la. Para restringir, passe um valor real |
| **Escala a zero no Container Apps** | Cobra por tempo acordado, não por requisição. `setInterval` no front-end consultando `/health` impede o container de dormir e esgota a cota |
| **Tag da imagem** | Republicar a mesma tag pode servir a imagem antiga em cache. Incremente sempre (`v1`, `v2`, …) |

---

## Convenções de código

- Identificadores, docstrings e mensagens de log em **português**
- Docstrings estilo Google; *type hints* em toda função pública
- `logging` nos módulos de biblioteca; `print` só em scripts
- Configuração via `@dataclass` em `config.py`, sobrescrevível por `argparse`
- Testes usam duplo de teste para o modelo Ultralytics — a suíte **não** depende
  de pesos treinados

---

## Estado atual

Modelo treinado e avaliado. Resultados medidos (não estimados):

| Métrica | Valor |
| --- | ---: |
| Acurácia no teste | 95,06% |
| F1-score macro | 0,9415 |
| Revocação `fungico` | 0,8462 ← ponto fraco |
| Revocação `saudavel` | 0,9929 |
| Latência mediana ONNX/CPU | 12,73 ms |
| Latência ponta a ponta HTTP | 53,3 ms |
| Testes | 101 |

Dataset após preparação: 2.216 imagens (1.566 / 326 / 324), sem vazamento.

Ao citar números, leia dos arquivos em `desenvolvimento/reports/` — não confie
nesta tabela se houver retreino.

---

## Pendências

- `yolo11s-cls` não foi treinado → `reports/comparativo_variantes.md` não existe
- `docs/DOCUMENTACAO_TECNICA_ABNT.md` tem 5 marcadores `[PREENCHER]`: referência
  do dataset no Kaggle, modelo da GPU do Colab, resultado da inspeção dos erros
- Erros com confiança 1,000 em `reports/erros_classificacao/` não foram
  inspecionados visualmente — podem ser rótulos errados no dataset de origem
- Ao implantar, **defina `IA_ALFACE_CHAVE_API`**. A autenticação existe mas é
  inerte sem a variável — sem defini-la, o serviço com ingress externo aceita
  chamada de qualquer origem. A chave é para consumidores servidor a servidor;
  nunca a coloque num front-end
- Azure Function com gatilho de blob não implementada. Quando for: usar a
  variante **Event Grid** (a padrão faz *polling* e atrasa até 10 min), tratar
  400 como falha definitiva sem retry, e usar o nome do blob como chave natural
  em `deteccoes` — o Event Grid entrega ao menos uma vez
- Fases seguintes do TCC (fora deste módulo): firmware ESP32, MQTT, PostgreSQL,
  dashboard React

---

## Contexto do TCC

Este módulo atende RF02, RNF01, RNF02, RNF07, casos de teste CT01–CT04 e as
hipóteses H1 e H2. Ao alterar comportamento observável, verifique o impacto
nesses requisitos e atualize `docs/DIVERGENCIAS.md` se criar nova divergência
em relação ao projeto original.

Divergências já registradas (D1–D7): classificação em vez de detecção, `bbox`
nulo, `imgsz` 224, augmentation em memória, divisão por grupo, Grad-CAM no lugar
de caixas, `label_smoothing` inativo.

---

## Git

Não versionados: `desenvolvimento/dataset*/`, `runs/`, `pesos_base/`, `*.zip`,
`modelos/*.pt|onnx|torchscript`, `desenvolvimento/notebooks/`,
`desenvolvimento/tcc/`.

Os dois últimos são **não regeneráveis** e existem só no disco do autor.
