# Instruções para assistentes de IA

Orientação para trabalhar **neste repositório**. O que construir está em
[`spec.md`](spec.md); a visão geral do projeto está em [`README.md`](README.md).

Projeto: classificador de doenças foliares em alface (YOLO11-cls) + API REST.
Módulo de IA de um TCC de graduação (UNIP, 2026). Código em **português**.

---

## Regras invioláveis

Violar qualquer uma delas produz dano difícil de perceber. Em caso de dúvida,
pergunte antes de contrariar.

1. **`desenvolvimento/dataset/` e `desenvolvimento/dataset_dominio_bruto/` são
   imutáveis.** São as fontes de dados originais. Nenhum código escreve nelas.
   Saídas vão para `desenvolvimento/dataset_clean/` e
   `desenvolvimento/dataset_dominio/`. No `dataset_dominio_bruto/`, só o autor
   move fotos, e as da estufa **não são regeneráveis**.

2. **Rótulos de classe vêm só de `src/ia_alface/classes.py`.** Nunca escreva
   `"fungico"`, `"bacteriano"` ou `"saudavel"` como literal em outro arquivo —
   importe. Os valores correspondem ao `ENUM` do PostgreSQL do projeto; uma
   divergência de grafia grava dado errado no banco sem gerar erro. O mesmo
   vale para os rótulos do verificador, `alface` e `nao_alface`
   (`CLASSE_ALFACE`, `CLASSE_NAO_ALFACE`). Eles **não** são classes de doença,
   e `nao_alface` ainda não está no `ENUM`.

3. **O split `test` é medido uma única vez por versão do modelo.** Todo ajuste
   de hiperparâmetro ou de limiar usa `--split val`. Avaliar repetidamente no
   teste o transforma em segundo conjunto de validação e invalida a métrica
   final. Os testes do classificador e das versões v1 e v2 do verificador já
   foram consumidos. O script 08 recusa medir de novo uma versão existente,
   e nenhum número desses testes pode servir para ajustar algo.

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
   se algum sobrar. Para o verificador, `06_preparar_dataset_dominio.py` roda
   antes do treino e falha com exit 1 se houver hash ou grupo em mais de um
   split.

---

## Estrutura

A raiz contém **apenas produção**. Todo o ciclo de pesquisa está em
`desenvolvimento/` e não é implantado.

```
api/              serviço HTTP           PRODUÇÃO
src/ia_alface/    biblioteca             PRODUÇÃO
modelos/          pesos .pt/.onnx        PRODUÇÃO (classificador e verificador v2)
Dockerfile        imagem da API          PRODUÇÃO
docs/             documentação
desenvolvimento/  scripts, testes, dataset, reports, notebooks, tcc
```

A implantação é em **Azure Container Apps** com escala a zero; as imagens
capturadas ficam em **Azure Blob Storage**. Nada disso entra em `src/` nem em
`api/` — a biblioteca e a API continuam sem dependência de nuvem.

Comandos de implantação em `docs/IMPLANTACAO.md`; o raciocínio por trás das
escolhas, as alternativas descartadas e o índice de armadilhas em
`docs/DECISOES_INFRAESTRUTURA.md`.

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
python -m pytest -q                                    # 221 testes, ~100 s
python -m uvicorn api.main:app --host 0.0.0.0          # API
python desenvolvimento/scripts/01_preparar_dataset.py  # obrigatório antes de treinar
python desenvolvimento/scripts/02_treinar.py
python desenvolvimento/scripts/03_avaliar.py --pesos modelos/yolo11n_cls_alface.pt
python desenvolvimento/scripts/04_exportar.py
python desenvolvimento/scripts/05_benchmark_latencia.py
python desenvolvimento/scripts/06_preparar_dataset_dominio.py   # verificador: obrigatório antes de treinar
python desenvolvimento/scripts/preparar_zip_colab.py --origem desenvolvimento/dataset_dominio
python desenvolvimento/scripts/07_treinar_verificador.py        # prefira o caderno do Colab
python desenvolvimento/scripts/08_avaliar_verificador.py --split val --rotulo v2
python desenvolvimento/scripts/testar_imagem.py foto.jpg --heatmap
python desenvolvimento/scripts/gerar_documento_tcc.py  # desenvolvimento/tcc/Modulo_IA_TCC.docx
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
| **Grad-CAM em foto não quadrada** | O mapa 7 × 7 vale só para o recorte central de 224 × 224. Projete-o com `regiao_analisada` / `GradCAM.sobreposicao` (`explicabilidade.py`). Esticá-lo sobre a imagem inteira desloca o mapa em até ~280 px numa foto 1280 × 720 da C270 (corrigido em 26/10/2026) |
| **Torch em CPU** | ~3,8 min/época; 100 épocas ≈ 6 h. Prefira `desenvolvimento/notebooks/treino_colab.ipynb` |
| **Imagem inválida** | Deve levantar `ValueError` / HTTP 400. **Nunca** retornar resultado — um falso `saudavel` suprime o alerta silenciosamente |
| **PyTorch na imagem Docker** | O pacote do PyPI embute CUDA (~2 GB inúteis em CPU). Instale do índice `download.pytorch.org/whl/cpu` **antes** do `requirements.txt` |
| **OpenCV no container** | O Ultralytics exige `libgl1` e `libglib2.0-0` do sistema. Faltando, a build passa e o container quebra ao subir com `libGL.so.1` |
| **`IA_ALFACE_CORS_ORIGENS=""`** | Variável vazia cai no padrão `["*"]` — mais permissiva que não defini-la. Para restringir, passe um valor real |
| **Escala a zero no Container Apps** | Cobra por tempo acordado, não por requisição. `setInterval` no front-end consultando `/health` impede o container de dormir e esgota a cota |
| **Tag da imagem** | Republicar a mesma tag pode servir a imagem antiga em cache. Incremente sempre (`v1`, `v2`, …). A `v2` está no ar desde 26/10/2026; a próxima é `v3` |
| **`az` sem MFA** | `az containerapp update` falha com `RequestDisallowedByAzure` se a sessão não tiver MFA; consultas funcionam mesmo assim. `az logout` + `az login`, confirmando o segundo fator |
| **`--query` do `az` no PowerShell** | Parênteses chegam truncados (`length(@)` vira `length(@`), porque o `az` é um `.cmd`. Use consultas sem parênteses, como `--query "[].name" -o tsv` |
| **GHCR: `denied` no `manifest inspect`** | Pode ser credencial vencida ou tag inexistente. Com login válido, tag inexistente responde `manifest unknown`. O token precisa ser clássico, com `write:packages` |
| **Nomes das fotos da C270** | O app Câmera do Windows grava `WIN_AAAAMMDD_HH_MM_SS_Pro.jpg`; o sufixo ` (2)` indica duas fotos no mesmo segundo. A data e a hora vêm do nome, e só sem elas vale a data da pasta |
| **Estrutura do `dataset_dominio_bruto`** | `estufa/<AAAAMMDD>_<projeto\|celular>/<alface\|nao_alface\|duvidosas>/` e `nao_alface/<plantvillage\|plantdoc\|deepweeds\|coco>/`. Pasta da estufa fora do padrão interrompe o 06. `duvidosas/` fica fora do dataset; o celular, só no treino |
| **Sessões da C270** | A sessão termina numa pausa de **mais de 2 h** entre fotos e pode atravessar a meia-noite. Cada sessão fica inteira num split, em ordem cronológica (as mais recentes no teste). Uma única sessão nova vai inteira para o **teste** |
| **Data de corte (`20261024`)** | Sessões iniciadas até 24/10/2026 vão só para o treino (`ConfigDatasetDominio.data_corte_treino`, `--data-corte`). Validação e teste de uma versão nova precisam de sessões novas |
| **Caminho duplicado no Colab** | O Ultralytics 8.4.18x grava em `runs/classify/runs/classify/<nome>`. Leia a pasta real em `modelo.trainer.save_dir` |
| **Artefatos do Colab** | Somem quando o ambiente é encerrado. Baixe o zip de artefatos (`results.csv`, `args.yaml`) antes de fechar. Os da v2 do verificador se perderam, e o registro dela é o log |
| **Latência local** | Meça com os outros programas fechados. Com o PC carregado, a mediana passou de 42,70 para 120,84 ms |
| **PowerShell** | Colar o mesmo comando duas vezes na mesma linha estraga o comando (`pytest -qpython -m pytest -q` não roda teste nenhum) |
| **`reports/predicoes.csv`** | Guarda caminhos anteriores à pasta `desenvolvimento/` (`TCC\dataset_clean\...`). Não "corrija": o 08 cruza por nome de arquivo e classe |

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
| Latência ponta a ponta HTTP (local) | 53,3 ms |
| Latência das duas etapas na nuvem (`v2`, mediana) | 22,62 ms |
| Inicialização a frio na nuvem | 11,27 s (uma medida) |
| Testes | 221 |

Dataset após preparação: 2.216 imagens (1.566 / 326 / 324), sem vazamento.

**Verificador de domínio v2** (`modelos/yolo11n_cls_dominio_v2.pt`, limiar
0,50): o teste foi medido uma vez, em 25/10/2026
(`reports/verificador_v2_test.md`).

| Critério (mínimo fixado antes do teste) | Medido | Situação |
| --- | ---: | --- |
| Folhas doentes aceitas (183/184) | 178/184 | **não atingida** |
| Não alface recusada (337/354) | 340/354 | atingida |
| Latência mediana das duas etapas (< 100 ms) | 51,85 ms | atingida |
| Alface aceita na C270 (72/75) | 75/75 | atingida |
| Não alface recusada na C270 (46/48) | 42/48 | **não atingida** |

O critério de implantação **não foi atingido**. A implantação é decisão do
autor, tomada depois do teste e registrada como exceção. Não a descreva como
meta atingida. No sistema completo, a revocação `fungico` é de 80,8% (63/78),
e a acurácia, de 94,1%. Registro completo: `docs/VERIFICADOR_DOMINIO.md`.

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
  400 como falha definitiva sem retry, tratar `classe = "nao_alface"` como
  resultado final sem retry (sem gravar diagnóstico), decidir pelo `alerta`
  e nunca por "classe ≠ `saudavel`", e usar o nome do blob como chave natural
  em `deteccoes` — o Event Grid entrega ao menos uma vez
- O `ENUM` da coluna `classe` não aceita `nao_alface`: combinar com o grupo do
  banco (D8)
- Inicialização a frio na nuvem com uma única medida (11,27 s): repetir
  para ter uma faixa. Verificador v2 implantado em 26/10/2026 (imagem `v2`)
- ESP32-CAM ainda citado em docstrings da API, no contrato OpenAPI, em READMEs,
  no `spec.md` e em testes. A câmera do projeto é a **Logitech C270**
  (`docs/CONTEXTO_IA_CONFIAVEL.md`, Seção 12, divergência 4)
- Fases seguintes do TCC (fora deste módulo): firmware ESP32, MQTT, PostgreSQL,
  dashboard React

---

## Contexto do TCC

Este módulo atende RF02, RNF01, RNF02, RNF07, casos de teste CT01–CT04 e as
hipóteses H1 e H2. Ao alterar comportamento observável, verifique o impacto
nesses requisitos e atualize `docs/DIVERGENCIAS.md` se criar nova divergência
em relação ao projeto original.

Divergências já registradas (D1–D8): classificação em vez de detecção, `bbox`
nulo, `imgsz` 224, augmentation em memória, divisão por grupo, Grad-CAM no lugar
de caixas, `label_smoothing` inativo, e a cascata com verificador de domínio,
com resposta 200 `nao_alface` e a câmera C270 no lugar do ESP32-CAM.

---

## Git

Não versionados (o `.gitignore` lista as pastas pelo nome, sem curinga
`dataset*/`):
- `desenvolvimento/dataset/`, `desenvolvimento/dataset_clean/`,
  `desenvolvimento/dataset_dominio_bruto/` e `desenvolvimento/dataset_dominio/`;
- `desenvolvimento/runs/` e `desenvolvimento/pesos_base/`;
- `*.zip` e `modelos/*.pt|onnx|torchscript`;
- `desenvolvimento/notebooks/` e `desenvolvimento/tcc/`.

Uma pasta nova de dados precisa de linha própria no `.gitignore`.

**Não regeneráveis**, existem só no disco do autor: `desenvolvimento/notebooks/`,
`desenvolvimento/tcc/` e as fotos da estufa em
`desenvolvimento/dataset_dominio_bruto/estufa/`.
