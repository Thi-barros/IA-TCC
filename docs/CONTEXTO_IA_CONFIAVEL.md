# Contexto confiável — Módulo de IA (classificador de doenças foliares em alface)

> **Para que serve este arquivo.** O `CONTEXTO_TCC_Estufa.md` é o contexto
> **geral** do TCC, consolidado a partir do `.docx` em revisão. Este aqui cobre
> **só o módulo de IA** e foi montado **a partir dos arquivos do repositório**
> (relatórios gerados por script, logs do treino, código-fonte). Cada número
> indica de onde veio. Quando o contexto geral e o repositório discordam, vale o
> repositório, e a divergência fica registrada na [Seção 12](#12-divergências-entre-o-contexto-geral-e-o-repositório).
>
> Consolidado em 14/10/2026. Nenhum script foi reexecutado: os valores são os
> que estão gravados nos arquivos citados.
>
> **Atualizado em 25/10/2026** com o verificador de domínio (Seção 5.7), o
> CT01 do sistema completo (Seção 6), a resposta `nao_alface` da API (Seção 8),
> a contagem de testes (Seção 10), as limitações (Seção 11) e a câmera C270
> (Seção 12, divergência 4). Os números novos vêm de
> `desenvolvimento/reports/verificador_*.md`; decisões e procedência, em
> `docs/VERIFICADOR_DOMINIO.md`.

---

## 0. Como ler (níveis de confiança)

| Selo | Significado | Exemplo de fonte |
| --- | --- | --- |
| **[M]** Medido | Valor gravado por script em arquivo de saída bruto (`.json`, `.csv`, `.yaml`) ou relatório gerado automaticamente | `desenvolvimento/reports/metricas_test.json` |
| **[D]** Documentado | Afirmado em texto do repositório (`spec.md`, `docs/*.md`, `README.md`), **sem** arquivo bruto que o reproduza | `spec.md`, `docs/DOCUMENTACAO_IA_TREINAMENTO_RESULTADOS.md` |
| **[C]** Código | Comportamento lido no código-fonte, não medido | `api/main.py` |
| **[N]** Não medido | Ninguém mediu. **Não estime nem preencha** | — |

**Regras para qualquer IA que use este contexto**

1. Não invente número. O que está como **[N]** deve aparecer como "não medido".
2. Não arredonde de outro jeito nem troque a fonte. Ao citar, use o valor e o selo daqui.
3. Os rótulos de classe são exatamente `bacteriano`, `fungico`, `saudavel`: sem acento, minúsculos. O verificador de domínio usa `alface` / `nao_alface`, que **não** são classes de doença e não estão no `ENUM` do banco.
4. O split `test` foi medido **uma única vez**. Não proponha "reavaliar no teste" para ajustar algo; ajuste é com `val`.
5. Se houver retreino, **todos** os números deste arquivo ficam obsoletos. Releia `desenvolvimento/reports/`.

---

## 1. Escopo do módulo

| Item | Valor | Selo / fonte |
| --- | --- | --- |
| Tarefa | **Classificação** de imagem inteira (não detecção). Sem *bounding boxes* | [C] `src/ia_alface/inferencia.py`; [D] `docs/DIVERGENCIAS.md` D1 |
| Classes (índice → rótulo) | 0 `bacteriano` · 1 `fungico` · 2 `saudavel` | [C] `src/ia_alface/classes.py` |
| Pastas originais do dataset | `Bacterial` → `bacteriano`, `fungal` → `fungico`, `healthy` → `saudavel` | [M] `dataset_auditoria.md` §9 |
| Rótulos de exibição | "Bacteriano (Xanthomonas campestris)", "Fúngico (Bremia lactucae)", "Saudável" | [C] `classes.py` |
| Modelo | YOLO11n-cls (Ultralytics), pré-treinado no ImageNet | [M] `runs/.../args.yaml` (`model: yolo11n-cls.pt`) |
| Resolução de entrada | 224 × 224 | [M] `args.yaml`, `validacao_onnx.json` |
| Localização da lesão | Só aproximada, via **Grad-CAM** (exige pesos `.pt`) | [C] `src/ia_alface/explicabilidade.py`, `api/configuracao.py` |
| Fronteira do módulo | Biblioteca pura (`src/ia_alface/`) + API HTTP (`api/`). **Não** grava em banco, **não** usa MQTT, **não** envia push | [C] docstring de `api/main.py` |
| Verificador de domínio | Segundo YOLO11n-cls, binário (`alface` / `nao_alface`), **antes** do classificador (cascata). Só atua com `IA_ALFACE_MODELO_DOMINIO` definida. O classificador não muda | [C] `src/ia_alface/verificacao.py`, `cascata.py`, `api/configuracao.py`; resultados na Seção 5.7 |
| Câmera do projeto | Webcam **Logitech C270** (não ESP32-CAM) | [D] `docs/VERIFICADOR_DOMINIO.md`, Seção 1 (15/10); `docs/DIVERGENCIAS.md` D8 |

Nota taxonômica [D]: *Bremia lactucae* é oomiceto, não fungo verdadeiro; o
nome "fúngico" foi mantido por consistência com a especificação.

---

## 2. Dataset

### 2.1 Origem e auditoria — [M] `desenvolvimento/reports/dataset_auditoria.md` (gerado em 10/08/2026)

| Item | Valor |
| --- | --- |
| Imagens no dataset original | 2.813 (987 JPEG, 1.826 PNG; 2.812 RGB, 1 RGBA) |
| Resolução | mín. 129 px, mediana 474 × 355 px, máx. 2.048 px; 354 quadradas |
| Divisão original | train 2.362 / valid 225 / test 226 (≈ 84/8/8) |
| Hashes MD5 distintos | 2.216 |
| Cópias byte-idênticas descartadas | 597 |
| Grupos de duplicatas que cruzam splits | **104** |
| Arquivos `augmented_*` (sintéticos) | 800 (train: 500 bacteriano; valid e test: 50 por classe) |
| Grupos de origem após fusão por conteúdo | bacteriano 649 · fungico 426 · saudavel 695 |
| Verificação final | **APROVADO**: zero hashes e zero grupos compartilhados entre splits |

Valores da análise inicial em `spec.md`, **sem** linha correspondente no relatório gerado [D]:

- **115** grupos de origem (padrão `<origem>_<n>`) aparecem em mais de um split.
- As **282** imagens `ba*` da classe bacteriana derivam de **≈ 53 folhas**.
  Atenção: isso conta só os arquivos `ba*`. Não contradiz os 649 grupos
  bacterianos do relatório, que incluem arquivos `th*` e os sintéticos
  `augmented_*` (cada um tratado como grupo unitário).

Referência bibliográfica do dataset no Kaggle: **pendente** (marcador
`[PREENCHER]` em `docs/DOCUMENTACAO_IA_TREINAMENTO_RESULTADOS.md`). O contexto
geral cita "SHAHA, 2024", mas o repositório não traz essa referência.

### 2.2 Dataset preparado — [M] `dataset_auditoria.md` §7

Divisão 70/15/15 estratificada por classe e **disjunta por grupo de folha**
(semente 42), aplicada sobre grupos e não sobre arquivos.

| Split | bacteriano | fungico | saudavel | Total | % |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 510 | 390 | 666 | 1.566 | 70,7 |
| val | 114 | 75 | 137 | 326 | 14,7 |
| test | 106 | 78 | 140 | 324 | 14,6 |
| **Total** | **730** | **543** | **943** | **2.216** | 100 |

Procedimento [D] (`DOCUMENTACAO_IA_...md` §2.2.3): `StratifiedGroupKFold`,
gravação em JPEG qualidade 95, sem redimensionar. O dataset original
(`desenvolvimento/dataset/`) nunca é escrito.

---

## 3. Pré-processamento e aumento de dados

### 3.1 Inferência (avaliação e produção) — [C] `inferencia.py` + Ultralytics

1. Validação: vazia, truncada, corrompida, formato não suportado ou dimensão < 2 px → `ValueError` (HTTP 400). **Nunca** devolve classe.
2. Conversão para RGB.
3. Lado menor → 224 px, recorte central 224 × 224, pixels em [0, 1] [D].
4. Softmax → classe = argmax; `confianca` arredondada a 3 casas; probabilidades a 6 casas [C].

### 3.2 Aumento de dados (somente no treino, somente em memória)

Os valores passados ao Ultralytics [M] `args.yaml`:

```
auto_augment: randaugment   fliplr: 0.5   flipud: 0.3   erasing: 0.2
scale: 0.4   degrees: 20.0   translate: 0.1   hsv_h: 0.015   hsv_s: 0.5   hsv_v: 0.4
```

O que **efetivamente** atua na tarefa `classify` do Ultralytics 8.4.95, segundo
[D] `DOCUMENTACAO_IA_...md` §2.4 (comportamento da biblioteca, não medido):

| Transformação | Parâmetro efetivo |
| --- | --- |
| Recorte aleatório redimensionado | área 60–100% (`1 − scale`), proporção 3:4 a 4:3 |
| Espelhamento horizontal / vertical | p = 0,5 / p = 0,3 |
| RandAugment | 2 operações por imagem, magnitude 9 |
| Random Erasing | p = 0,2 |

`degrees` e `translate` não são usados em `classify`, e `hsv_*` é desativado
quando o RandAugment está ligado. **Cuidado:** `src/ia_alface/config.py` e
`docs/DIVERGENCIAS.md` (D4) ainda listam `degrees`, `translate` e `hsv_*` como
se estivessem ativos. Ao redigir, use a tabela acima.

`label_smoothing` **não** foi aplicado: o Ultralytics 8.4 removeu o parâmetro,
que é aceito e ignorado [D] `DIVERGENCIAS.md` D7. Não escreva que foi usado.

---

## 4. Treinamento

### 4.1 Configuração — [M] `desenvolvimento/runs/classify/yolo11n_cls_alface/args.yaml`

| Hiperparâmetro | Valor |
| --- | --- |
| Épocas máx. / paciência | 100 / 20 |
| Batch / `nbs` | 32 / 64 (acumula 2 lotes por atualização → lote efetivo 64) |
| Otimizador | AdamW, `momentum` 0,937 |
| `lr0` / `lrf` | 0,001 / 0,01 (taxa final 0,00001) |
| Agenda | cossenoidal (`cos_lr: true`), `warmup_epochs` 3 |
| `weight_decay` / `dropout` | 0,0005 / 0,1 |
| `seed` / `deterministic` | 42 / true |
| `device` / `amp` / `workers` | `'0'` (GPU) / true / 2 |
| Caminho dos dados | `/content/dataset_clean` (Google Colab) |

Modelo da GPU: **não registrado** [N]. O `args.yaml` só diz `device: '0'`, e
o documento técnico tem `[PREENCHER]`. O contexto geral cita "Tesla T4", mas
nenhum arquivo do repositório confirma.

Versões exatas de Ultralytics e PyTorch **no Colab**: não registradas [N].
As versões locais estão na Seção 7.

### 4.2 Execução — [M] `runs/.../results.csv`

| Item | Valor |
| --- | --- |
| Épocas executadas | 33 (parada antecipada) |
| Melhor época | **13**: acurácia de validação 0,97546 (318/326), `val/loss` 0,08133 |
| Época 30 | também 0,97546, mas não substitui (exige melhora estrita) |
| Tempo total | 440,559 s (≈ 7,3 min); ≈ 13,4 s/época |
| Evolução | 70,86% (ép. 1) → 88,34% (ép. 2) → 81,60% (ép. 4, fim do warmup) → 91,10% (ép. 6) → 97,55% (ép. 13) |
| LR na última época | 0,000770 |
| Menor `val/loss` | 0,04557 (época 30) |

Leitura de sobreajuste [D]: a perda de validação fica na mesma ordem da de
treino, sem tendência de alta. É interpretação, não teste estatístico.

Treino local em CPU: ≈ 3,8 min/época [D] (`CLAUDE.md`, documento técnico). Foi
o motivo para treinar no Colab.

---

## 5. Resultados

### 5.1 Validação (326 imagens): estimativa otimista, não é resultado oficial — [M] `metricas_test_val.json`

| Classe | Precisão | Revocação | F1 | Suporte |
| --- | ---: | ---: | ---: | ---: |
| bacteriano | 0,9825 | 0,9825 | 0,9825 | 114 |
| fungico | 0,9595 | 0,9467 | 0,9530 | 75 |
| saudavel | 0,9783 | 0,9854 | 0,9818 | 137 |

Acurácia 0,9755 · F1 macro 0,9724.

### 5.2 Teste (324 imagens): **resultado oficial, medido uma vez** — [M] `metricas_test.json`, `avaliacao_test.md` (31/07/2026, CPU, pesos `.pt`)

| Classe | Precisão | Revocação | F1 | Suporte | Confiança média nos acertos | % acertos ≥ 0,85 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bacteriano | 0,9364 | 0,9717 | 0,9537 | 106 | 0,9834 | 98,1% |
| fungico | 0,9429 | **0,8462** | 0,8919 | 78 | 0,9561 | 89,4% |
| saudavel | 0,9653 | 0,9929 | 0,9789 | 140 | 0,9884 | 97,1% |

- Acurácia global **0,9506** (308/324) · F1 macro **0,9415**.
- Precisão macro 0,9482 e revocação macro 0,9369 são médias simples das linhas acima (calculadas, não gravadas no JSON).
- Diferença validação − teste: 2,49 p.p.

### 5.3 Matriz de confusão do teste — [M] `matriz_confusao.csv` / `avaliacao_test.md` §3

| real \ predito | bacteriano | fungico | saudavel | Total |
| --- | ---: | ---: | ---: | ---: |
| bacteriano | **103** | 3 | 0 | 106 |
| fungico | 7 | **66** | 5 | 78 |
| saudavel | 0 | 1 | **139** | 140 |
| Total predito | 110 | 70 | 144 | 324 |

16 erros, por consequência operacional:

| Erro | n | Efeito |
| --- | ---: | --- |
| fungico → saudavel | 5 | **Nenhum alerta** (o pior caso) |
| fungico → bacteriano | 7 | Alerta emitido, tipo de doença errado |
| bacteriano → fungico | 3 | Alerta emitido, tipo de doença errado |
| saudavel → fungico | 1 | Alerta indevido |

Nenhuma folha bacteriana virou saudável e nenhuma saudável virou bacteriana.

### 5.4 Erros de alta confiança — [M] `avaliacao_test.md` §5, `reports/erros_classificacao/`

Três erros com confiança exibida como **1,000** (arredondada a 3 casas):
1 fungico → saudavel e 2 fungico → bacteriano. Não são corrigíveis por limiar.

**Inspeção visual dessas imagens: não feita** [N]. Pode haver rótulo errado no
dataset de origem, mas isso é hipótese, não constatação.

### 5.5 Latência

**Benchmark dedicado**: [M] `benchmark_latencia.md` (02/08/2026). 100 imagens
do teste, 10 de aquecimento descartadas, 1 imagem por chamada via
`DetectorDoencasAlface.predizer`, CPU AMD (Ryzen 5 3500U [D]), Windows 11, sem CUDA.

| Configuração | Mediana | Média | p95 | p99 | Mín. | Máx. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch `.pt` / CPU | 20,10 | 23,23 | 38,27 | 81,34 | 17,07 | 91,20 |
| ONNX / CPU | **12,73** | 16,15 | 32,69 | 78,98 | 9,08 | 85,37 |

ONNX reduz a mediana em 36,7% (calculado).

**Medida durante a avaliação** [M] `metricas_test.json` (`.pt`, sem aquecimento,
324 imagens): mediana 61,81 ms, média 75,76, p95 160,5, máx. 714,46 ms. É outra
medição, não contradiz o benchmark. Não misture as duas.

**Ponta a ponta HTTP local**: 53,3 ms (40,6 de inferência + 12,7 de protocolo)
[D]. Aparece em `README.md` e no documento técnico, **sem arquivo de relatório
que o reproduza**. Cite com essa ressalva.

**Na nuvem, imagem `v2`** — [M] em 26/10/2026, a partir da saída colada pelo
autor (`VERIFICADOR_DOMINIO.md`). Foi um loop no PowerShell, então **não há
relatório em `reports/`**.
- **Protocolo:** rota `/analyze/raw`; 100 imagens sorteadas do `val`
  (semente 42); 10 chamadas de aquecimento descartadas.
- **Duas etapas:** soma dos tempos de inferência do verificador e do
  classificador, medidos pelo servidor. Réplica de 1 vCPU e 2 GiB, pesos
  `.pt`.

| Medida | Mediana | p95 | p99 | Máx. |
| --- | ---: | ---: | ---: | ---: |
| Duas etapas (servidor) | **22,62** | 45,15 | 74,22 | 99,06 |
| Só verificador (servidor) | 11,27 | 23,46 | 31,19 | 36,67 |
| HTTP ponta a ponta, do computador do autor | 104,85 | 505,87 | 738,33 | 808,80 |

- **Inicialização a frio:** **11,27 s** na primeira chamada ao `/health`,
  com zero réplicas. É **uma única medida**.
- **HTTP:** inclui a rede doméstica do autor e não entra no RNF01.

**Não medidos** [N]:
- latência do pipeline completo (upload no Blob → Event Grid → Function → API → banco);
- variação da inicialização a frio (só há uma medida);
- latência só do classificador na nuvem, sem o verificador.

### 5.6 Equivalência do ONNX — [M] `validacao_onnx.json`

20 imagens · divergência máxima 4,172 × 10⁻⁷ · média 3,709 × 10⁻⁸ · tolerância
10⁻³ · **0 classes divergentes** · opset 17 · imgsz 224 → aprovado.

Tamanhos dos arquivos (`.pt` 3,19 MB, `.onnx` 6,17 MB, `.torchscript` 6,33 MB)
e 1.534.947 parâmetros treináveis: [D], só no documento técnico.

### 5.7 Verificador de domínio (cascata) — [M] `verificador_val.md`, `verificador_test.md`, `verificador_v2_val.md`, `verificador_v2_test.md`

Registro completo, decisões e procedência: `docs/VERIFICADOR_DOMINIO.md`.
Texto do TCC: Seções 2.9 e 3.8 de `DOCUMENTACAO_IA_TREINAMENTO_RESULTADOS.md`.

Limiar 0,50 nas duas versões, escolhido no `val` e fixado antes do teste [D]
(`VERIFICADOR_DOMINIO.md`, Seção 1). CPU local, pesos `.pt` nas duas etapas.
O `test` de cada versão foi medido **uma única vez**, e o script 08 recusa
uma segunda medição da mesma versão [C]. Pesos a implantar: v2,
`modelos/yolo11n_cls_dominio_v2.pt`.

**Dados e treino** — [M] `dataset_dominio_auditoria*.md` e log do Colab
colado pelo autor (`VERIFICADOR_DOMINIO.md`, 4.3 e 4.6)

| Item | v1 | v2 |
| --- | --- | --- |
| Imagens (train / val / test) | 3.085 / 622 / 710 | 3.522 / 620 / 753 |
| Fotos da C270 (train / val / test) | 79 / 2 / 80 | 516 / 0 / 123 |
| GPU / Ultralytics | Tesla T4 / 8.4.181 | Tesla T4 / 8.4.187 |
| Épocas executadas / melhor época | 26 / 6 | 56 / 36 |
| Acurácia de validação (**não** é o critério) | 0,99839 | 0,9984 |

Da v2 não existem `results.csv` nem `args.yaml`: os artefatos do Colab não
foram guardados, e o registro do treino é o log.

**Validação** (escolha do limiar; não é resultado oficial)

| Critério | v1 | v2 |
| --- | --- | --- |
| Folhas doentes aceitas | 189/189 | 189/189 |
| Não alface recusada | 293/294 | 294/294 |
| Latência mediana das duas etapas | 49,63 ms | 42,70 ms |
| C270 | 2/2 alface; não alface: não medido | não medido (0 imagens) |

A latência de 42,70 ms da v2 é a segunda medição, com os outros programas
fechados. A primeira, com a máquina carregada, deu 120,84 ms; esse número
está registrado só em `VERIFICADOR_DOMINIO.md`, Seção 4.7 [D].

**Teste — medição única por versão**

| Critério (mínimo da v2) | v1 | v2 | Situação na v2 |
| --- | --- | --- | --- |
| (1) Folhas doentes aceitas (≥ 183/184) | 181/184 (98,4%; IC95 95,3%–99,4%) | 178/184 (96,7%; IC95 93,1%–98,5%) | **não atingida** |
| (2) Não alface recusada (≥ 337/354) | 338/356 (94,9%; IC95 92,2%–96,8%) | 340/354 (96,0%; IC95 93,5%–97,6%) | atingida |
| (3) Latência mediana das duas etapas (< 100 ms) | 53,25 ms | 51,85 ms (só verificador 25,42) | atingida |
| (4) Alface aceita na C270 (≥ 72/75) | 12/30 (40,0%; IC95 24,6%–57,7%) | 75/75 (100,0%; IC95 95,1%–100,0%) | atingida |
| (5) Não alface recusada na C270 (≥ 46/48) | 45/50 (90,0%; IC95 78,6%–95,7%) | 42/48 (87,5%; IC95 75,3%–94,1%) | **não atingida** |

- As fotos da C270 são de sessões diferentes em cada versão (v1: 10–11/10 e
  17/10; v2: 25/10). As outras 630 imagens são as mesmas nas duas.
- Os critérios (4) e (5) só existem para a v2. A v1 não atingiu (1) nem (2)
  e não foi implantada.
- **Critério de implantação da v2: não atingido** (condições 1 e 5). A
  implantação é **decisão do autor, posterior ao teste** [D]
  (`VERIFICADOR_DOMINIO.md`, Seção 1, 25/10; documento técnico, 3.8.5). Não
  escreva que o verificador foi aprovado nem que atingiu as metas.

**Cascata no teste oficial do classificador (324 imagens)**

| Métrica | Só classificador | Cascata v1 | Cascata v2 |
| --- | --- | --- | --- |
| Revocação `fungico` | 66/78 (84,6%) | 64/78 (82,0%) | 63/78 (80,8%) |
| Acurácia | 308/324 (95,1%) | 304/324 (93,8%) | 305/324 (94,1%) |
| Alertas verdadeiros suprimidos | — | 3 | 1 |

O classificador não foi reexecutado: a cascata usa `reports/predicoes.csv`.
A queda de 66 para 63 vem de 3 folhas que o classificador acertava e o
verificador recusou.

**Erros da v2** [D] (inspeção visual, `VERIFICADOR_DOMINIO.md`, 4.8):
- as 6 folhas fúngicas recusadas vêm de 3 imagens de origem, de microscópio
  ou macro;
- as 14 "não alface" aceitas são todas plantas: 6 da C270 (quintal), 7 do
  PlantDoc e 1 do DeepWeeds;
- nenhuma foto de objeto, piso ou lente tapada foi aceita.

**Não medidos** [N]:
- latência na nuvem: **medida** em 26/10 (Seção 5.5): duas etapas com mediana de 22,62 ms; inicialização a frio de 11,27 s numa única medida;
- desempenho com a C270 **fixa** no enquadramento de operação: o teste foi
  uma sessão, com a câmera na mão e as mesmas plantas do treino;
- jardineira com substrato e sem planta, e vaso vazio;
- frequência de outras plantas diante da câmera em operação;
- diagnóstico do classificador em fotos da C270 (não há rótulo de doença).

---

## 6. Requisitos, casos de teste e hipóteses do módulo

### 6.1 Pelo critério do repositório (`spec.md`)

| Item | Critério (`spec.md`) | Medido | Situação |
| --- | --- | --- | --- |
| CT01 | classe `fungico`, **confiança** ≥ 0,85 | confiança média nos acertos 0,9561 | Aprovado [M] |
| CT02 | classe `bacteriano`, confiança ≥ 0,85 | 0,9834 | Aprovado [M] |
| CT03 | revocação `saudavel` ≥ 90% | 99,29% | Aprovado [M] |
| CT04 / RNF01 | inferência < 100 ms no servidor | 12,73 ms (ONNX) / 20,10 ms (`.pt`), benchmark local | Aprovado [M] |
| RNF02 / H1 | acurácia > 90% | 95,06% | Atendido [M] |
| H2 (parte latência) | < 100 ms | idem CT04 | Atendido [M] |
| H2 (parte contenção < 5% da área) | — | — | **Não verificável** neste módulo [N] |
| CT01 no **sistema completo** (verificador v2 + classificador) | sem critério próprio; reportar junto do CT01 | revocação `fungico` 63/78 = 80,8% (classificador sozinho: 66/78 = 84,6%) | [M] `verificador_v2_test.md` |
| RNF01 com o verificador | < 100 ms | 51,85 ms (duas etapas, `.pt`, CPU local); 22,62 ms na nuvem (Seção 5.5) | Atendido [M] `verificador_v2_test.md`; nuvem: saída colada pelo autor |

Os CT01–CT04, RNF02 e H1 acima medem o **classificador**, que não mudou com o
verificador. O sistema completo tem acurácia de 305/324 = 94,1% [M].

**Ressalva obrigatória sobre CT01/CT02:** o critério mede confiança **só nos
acertos**. Ele não enxerga as 12 folhas fúngicas erradas. A revocação fúngica
de **0,8462** tem de aparecer junto com o CT01.

### 6.2 Conflito de critério do CT01 (ver Seção 12)

O contexto geral (Quadro 22 do `.docx`) define o CT01 como **"revocação fúngica
≥ 85%"**. Por esse critério, 84,62% **não passa** (0,38 p.p. abaixo), e o
`.docx` registra "aprovado parcialmente". O repositório define o CT01 por
confiança e o aprova. Os dois critérios precisam ser unificados no texto final.
Nenhum dos dois resultados é falso: eles medem coisas diferentes.

Com o verificador de domínio, o critério do contexto geral aplicado ao
**sistema implantado** dá 80,8% (63/78) [M], mais longe dos 85%. O
classificador sozinho continua em 84,62%. Ao citar, diga de qual dos dois
se trata.

### 6.3 Ação recomendada que não foi executada

O próprio `avaliacao_test.md` indica: revocação fúngica < 0,90 → aplicar
sobreamostragem da classe ou perda ponderada. **Não foi aplicado** [N].
`yolo11s-cls` **não foi treinado**, então não há comparativo de variantes [N].

---

## 7. Ambiente de avaliação e de execução local — [M] `avaliacao_test.md`, `benchmark_latencia.md`, `dataset_auditoria.md`

| Item | Valor |
| --- | --- |
| SO | Windows-11-10.0.26200 |
| CPU | AMD64 Family 23 Model 24 (Ryzen 5 3500U [D]) |
| Python | 3.13.5 |
| torch | 2.13.0+cpu |
| ultralytics | 8.4.95 |
| scikit-learn / onnxruntime | 1.6.1 / 1.28.0 [D] |
| CUDA | indisponível |
| `workers` | 0 (obrigatório no Windows) [C] `config.py` |

---

## 8. API de inferência — [C] `api/main.py`, `api/esquemas.py`, `api/configuracao.py`

| Rota | Método | Autenticação | Função |
| --- | --- | --- | --- |
| `/health` | GET | aberta | modelo carregado? |
| `/info` | GET | aberta | classes, `imgsz`, limiar, suporte a heatmap, limite de MB |
| `/analyze` | POST multipart | `X-API-Key` se configurada | classifica a imagem |
| `/analyze/raw` | POST bytes (`image/jpeg`/`png`) | idem | mesma resposta, corpo cru |
| `/teste` (e `/` redireciona) | GET | aberta | página de teste manual |

`?heatmap=true` acrescenta Grad-CAM em PNG base64. Só funciona com pesos `.pt`;
com `.onnx` a API devolve 400.

**Resposta** (`RespostaDeteccao`): `classe`, `classe_exibicao`, `confianca`,
`probabilidades{bacteriano,fungico,saudavel}`, `patogeno_detectado`, `alerta`,
`tempo_inferencia_ms`, `detectado_em` (UTC ISO 8601), `modelo_versao`,
`bbox_x/y/w/h` (**sempre `null`**), `heatmap_base64`.

**Regra de alerta:** `alerta = patogeno_detectado AND confianca >= 0,85`.
`saudavel` nunca gera alerta.

**Com o verificador ativo** (`IA_ALFACE_MODELO_DOMINIO` definida; desde
25/10/2026) [C] `api/main.py`, `docs/VERIFICADOR_DOMINIO.md` §4.1:

- **Imagem com alface:** o diagnóstico normal, mais o bloco `verificacao`
  (`probabilidade_alface`, `limiar`, `tempo_inferencia_ms`, `modelo_versao`).
  O `tempo_inferencia_ms` principal continua sendo só o do classificador.
- **Imagem sem alface:** **200**, não erro. Os campos ficam assim:
  - `classe = "nao_alface"` e `classe_exibicao = "Alface não encontrada"`;
  - `confianca = 1 − probabilidade_alface` e `probabilidades = {}`;
  - `patogeno_detectado` e `alerta` falsos;
  - `tempo_inferencia_ms = 0` e `modelo_versao` do verificador;
  - `verificacao` preenchido.
- **Consumidor:** decide pelo `alerta`, nunca por "classe ≠ `saudavel`".
- **Banco:** o `ENUM` ainda não aceita `nao_alface` (D8; pendente com o
  grupo do banco).
- **Erros:**
  - o 422 deixou de significar recusa por domínio: agora é só requisição
    malformada (FastAPI);
  - verificador configurado e não carregado dá **503**, e o `/health` mostra
    `verificador_carregado: false`;
  - sem a variável, `verificador_carregado` é `null` e a API responde como
    antes.

**Erros:** 400 (imagem vazia, corrompida ou não suportada), 401 (chave), 413
(> 10 MB por padrão), 422 (requisição malformada), 503 (classificador ou
verificador configurado e não carregado; o serviço sobe mesmo assim), 500
(inesperado).

**Variáveis de ambiente:**

| Variável | Padrão | Observação |
| --- | --- | --- |
| `IA_ALFACE_MODELO` | `modelos/yolo11n_cls_alface.pt` | `.onnx` é mais rápido, mas desliga o Grad-CAM |
| `IA_ALFACE_LIMIAR` | 0,85 | |
| `IA_ALFACE_MAX_MB` | 10 | |
| `IA_ALFACE_DISPOSITIVO` | autodetecta | |
| `IA_ALFACE_CORS_ORIGENS` | `["*"]` | vazio também cai em `["*"]` |
| `IA_ALFACE_CHAVE_API` | vazio = **sem autenticação** | comparação em tempo constante |
| `IA_ALFACE_MODELO_DOMINIO` | ausente = **sem verificador** | caminho dos pesos do verificador; na imagem `v2`, `/app/modelos/yolo11n_cls_dominio_v2.pt` (definido no `Dockerfile`) |
| `IA_ALFACE_LIMIAR_DOMINIO` | 0,5 | probabilidade mínima de `alface`; o valor fixado no `val` é 0,50 |

A inferência roda em `run_in_threadpool`, com *lock* no detector. O modelo é
carregado uma vez na subida.

**Fora do escopo da API:** gravar em `Deteccoes`, enviar push, guardar a imagem
e o heatmap no Blob (`HeatmapBlob`). Isso cabe à Function ou ao back-end.

---

## 9. Implantação — [D] `docs/DECISOES_INFRAESTRUTURA.md` §4, `Dockerfile`

| Item | Valor |
| --- | --- |
| Serviço | Azure Container Apps, ambiente `ia-alface-env` (`ConsumptionOnly`) |
| **Região** | **Brazil South**. Canada Central foi recusada pela assinatura (`RequestDisallowedByAzure`) |
| URL | `https://ia-alface.proudwater-1f30045f.brazilsouth.azurecontainerapps.io` |
| Grupo de recursos | `TCC-imagem` |
| Imagem | `ghcr.io/thi-barros/ia-alface:v2` desde 26/10/2026 (digest `sha256:833cc600…`; 614 MB compactada). Antes, `v1` |
| Réplicas | mín. 0, máx. 2, `cooldownPeriod` 300 s (escala a zero) |
| Recursos | 1 vCPU / 2 GiB |
| Pesos em execução | **`.pt`** (padrão do `Dockerfile`; `IMPLANTACAO.md` confirma). O `.onnx` está na imagem, mas não ativo |
| Chave de API | segredo `chave-api` via `secretref:` |
| CORS | `http://localhost:5173` |
| Verificador de domínio | **Implantado em 26/10/2026** (imagem `v2`). O `/health` na nuvem respondeu `verificador_carregado: true`, com `yolo11n_cls_dominio_v2.pt` [M]. Variáveis do verificador vêm do `ENV` do `Dockerfile` |
| Grad-CAM | A `v2` projeta o mapa só sobre o recorte analisado (correção de 26/10) [C] `explicabilidade.py` |

Consequência: o 12,73 ms (ONNX) **não** é a latência do modelo que está no ar.
O número local equivalente ao que está implantado é o do `.pt`: 20,10 ms de
mediana no benchmark, só o classificador. Na nuvem, as duas etapas juntas
tiveram mediana de 22,62 ms (Seção 5.5).

---

## 10. Testes automatizados

**221 testes aprovados** em 26/10/2026, 101,83 s [M] (saída colada pelo autor;
`VERIFICADOR_DOMINIO.md`). Antes, 214 em 25/10. Os 7 novos são de
`test_explicabilidade.py`, com a projeção do Grad-CAM sobre a região
analisada. Cobrem: correspondência dos rótulos com o
banco, ausência de vazamento, reprodutibilidade da divisão, formato da saída,
rejeição de imagem inválida, a projeção do Grad-CAM e, no verificador, a montagem e a divisão do
dataset de domínio, a avaliação, a cascata e a resposta `nao_alface`. Usam um
duplo de teste do modelo e não dependem de pesos treinados.

Contagens antigas: 101 (antes do verificador, `CLAUDE.md`) e 95
(`DECISOES_INFRAESTRUTURA.md`, antes da autenticação).

---

## 11. Limitações e pendências do módulo

**Limitações** [D]:
1. Domínio: treino e teste do classificador vêm do mesmo dataset público (folhas isoladas). Diagnóstico do classificador em fotos reais da C270: **[N] não medido**, porque não há fotos da C270 com rótulo de doença. As fotos da C270 do verificador só dizem se há alface.
2. Diversidade bacteriana baixa (≈ 53 folhas nos arquivos `ba*`).
3. Conjunto fechado: sozinho, o classificador sempre dá uma das 3 classes. O **verificador de domínio está implementado e medido** (Seção 5.7), mas a v2 **não atingiu** o critério de implantação: 178/184 folhas doentes aceitas e 42/48 "não alface" da C270 recusadas. Sua implantação é uma exceção do autor, decidida depois do teste. Ele ainda aceita outras plantas (14/354 "não alface" aceitas no teste, todas plantas). A evidência na C270 é uma única sessão, com a câmera na mão. A jardineira vazia não foi medida.
4. Recorte central descarta as laterais de imagens retangulares.
5. Grad-CAM 7 × 7 é aproximado e não quantifica área afetada (por isso H2 não fecha).
6. Revocação fúngica 0,8462: é o ponto fraco, e a mitigação não foi aplicada.

**Pendências** (estado em 14/10/2026):
- [ ] Referência do dataset no Kaggle
- [ ] Modelo da GPU e versões no Colab
- [ ] Inspeção visual dos 3 erros com confiança 1,000
- [ ] Referência fitopatológica do Quadro 3
- [ ] Treinar `yolo11s-cls` (`comparativo_variantes.md` não existe)
- [x] Medir latência na nuvem e *cold start* (26/10; Seção 5.5). O *cold start* tem uma única medida
- [ ] Unificar o critério do CT01 entre `.docx` e `spec.md`
- [ ] Corrigir `config.py` / D4, que listam `degrees`, `translate` e `hsv_*` como ativos
- [ ] Azure Function de imagens com gatilho Event Grid (não está neste repositório)
- [ ] `ENUM` da coluna `classe` aceitar `nao_alface`, ou o back-end não gravar essas respostas (D8; grupo do banco)
- [x] Implantar o verificador v2 (passo H) e medir a latência das duas etapas na nuvem (26/10)
- [ ] Verificador: fotos da jardineira com substrato e sem planta e do vaso vazio, e de outras plantas como "não alface" (`VERIFICADOR_DOMINIO.md`, 4.9)
- [ ] Trocar ESP32-CAM por C270 nos arquivos listados na Seção 12, divergência 4. Mexer em `api/` ou `src/` exige rodar `pytest` e `gerar_openapi.py`

---

## 12. Divergências entre o contexto geral e o repositório

| # | Tema | Contexto geral (`CONTEXTO_TCC_Estufa.md`) | Repositório (fonte) | O que usar |
| --- | --- | --- | --- | --- |
| 1 | Região da API | "Container Apps" em **Canada Central** | **Brazil South**; Canada Central recusada (`DECISOES_INFRAESTRUTURA.md`) | Brazil South para a API. IoT Hub e outros recursos podem estar em outra região: confirme com quem os criou |
| 2 | Formato do modelo implantado | "YOLO11n-cls **em ONNX** no Container Apps" | Imagem roda **`.pt`** por padrão (`Dockerfile`, `IMPLANTACAO.md`) | `.pt`. Se trocarem para ONNX via variável, atualizar aqui |
| 3 | Critério do CT01 | Revocação fúngica ≥ 85% → "aprovado parcialmente" | Confiança ≥ 0,85 → "aprovado" (`spec.md`, `avaliacao_test.md`) | Explicitar os dois e unificar no texto (Seção 6.2) |
| 4 | Captura | Webcam Logitech C270 + notebook | Câmera do projeto: **Logitech C270** (informado pelo autor em 15/10/2026; `VERIFICADOR_DOMINIO.md`, Seção 1). As fotos reais do verificador são dela | **Parcialmente resolvida em 25/10/2026.** A C270 já consta no documento técnico (2.8, 2.9, 3.6, 4.1) e na D8 de `DIVERGENCIAS.md`, que lista as seções do texto do TCC a corrigir. O `IMPLANTACAO.md` (topologia) muda no G5. **Ainda citam o ESP32-CAM** (busca de 25/10): docstrings de `api/main.py` e o contrato gerado (`docs/openapi.json`, `docs/swagger.html`), `docs/referencia_api.html`, `api/README.md`, `api/cliente.py`, `README.md`, `docs/AMBIENTE.md`, `docs/DOCUMENTACAO_TECNICA_ABNT.md`, `docs/README.md`, `.env.exemplo`, `spec.md`, `src/ia_alface/inferencia.py`, `desenvolvimento/reports/benchmark_latencia.md`, scripts e testes. Até serem corrigidos, use C270 |
| 5 | GPU do treino | "Tesla T4 (~15 GB)" | Não registrado (`args.yaml` só `device: '0'`; `[PREENCHER]`) | "não registrado", até haver evidência (print do `nvidia-smi` do Colab) |
| 6 | Aumento de dados | RandAugment, recorte, flips, erasing | `config.py` / D4 listam também rotação, translação e HSV | Usar a tabela da Seção 3.2. Corrigir D4 |
| 7 | Banco | Azure SQL (`Deteccoes`, CHECK em `Classe`) | `classes.py` / `CLAUDE.md` falam em `ENUM` do **PostgreSQL** | Os valores coincidem. Atualizar a menção a PostgreSQL |
| 8 | Latência "ponta a ponta" | Não cita 53,3 ms; diz que a ponta a ponta na nuvem não foi medida | 53,3 ms **HTTP local** (`README.md`), sem relatório bruto | 53,3 ms = HTTP local [D]; nuvem = não medido |
| 9 | Dataset no Kaggle | "SHAHA, 2024" | Referência ausente (`[PREENCHER]`) | Confirmar autor, título e URL antes de citar |

---

## 13. Arquivos-fonte (para conferir ou atualizar)

| Arquivo | Conteúdo |
| --- | --- |
| `desenvolvimento/reports/dataset_auditoria.md` | auditoria, dedup, nova divisão |
| `desenvolvimento/runs/classify/yolo11n_cls_alface/args.yaml` | hiperparâmetros efetivos do treino |
| `desenvolvimento/runs/classify/yolo11n_cls_alface/results.csv` | histórico por época |
| `desenvolvimento/reports/metricas_test.json` · `avaliacao_test.md` | resultado oficial (teste) |
| `desenvolvimento/reports/metricas_test_val.json` · `avaliacao_test_val.md` | validação |
| `desenvolvimento/reports/matriz_confusao.csv` · `predicoes.csv` | matriz e predições por imagem |
| `desenvolvimento/reports/erros_classificacao/` | 20 piores casos (imagens) |
| `desenvolvimento/reports/benchmark_latencia.md` · `.json` | latência `.pt` × ONNX |
| `desenvolvimento/reports/validacao_onnx.json` | equivalência ONNX |
| `src/ia_alface/classes.py` · `config.py` · `inferencia.py` | rótulos, configuração, inferência |
| `api/main.py` · `esquemas.py` · `configuracao.py` | contrato HTTP |
| `docs/DIVERGENCIAS.md` | D1–D8 em relação ao projeto original |
| `docs/VERIFICADOR_DOMINIO.md` | verificador de domínio: decisões, dados, treino, avaliação, contrato da API, pendências |
| `desenvolvimento/reports/dataset_dominio_auditoria.md` · `_v1.md` · `dataset_dominio_inventario*.csv` | dataset do verificador (v2 e v1) |
| `desenvolvimento/reports/verificador_val.md` · `verificador_test.md` · `verificador_v2_val.md` · `verificador_v2_test.md` (e `.json`) | avaliação do verificador v1 e v2 |
| `desenvolvimento/reports/predicoes_verificador*.csv` · `cascata_test.csv` · `cascata_v2_test.csv` · `erros_verificador_v2_test/` | predições, cascata e erros do verificador |
| `docs/DECISOES_INFRAESTRUTURA.md` · `docs/IMPLANTACAO.md` | implantação no Azure |
