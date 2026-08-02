# Divergências entre o texto do TCC e a implementação

Este documento registra os pontos em que a implementação do módulo de IA se
afasta do que está escrito em *ENTREGA FINAL corrigido 13_11_2026_1.md*, com a
justificativa técnica de cada decisão. **Cada item precisa ser incorporado ao
texto do TCC** antes da entrega final.

---

## D1 — Classificação de imagens, não detecção de objetos

**O que o TCC diz.** As Seções 5.3.4, 6.1.1, 6.2.6.3 e 6.2.6.8 descrevem
consistentemente detecção de objetos: caixas delimitadoras, anotações em formato
YOLO `.txt` com coordenadas normalizadas, e as colunas `bbox_x`, `bbox_y`,
`bbox_w`, `bbox_h` na tabela `deteccoes`.

**O que foi implementado.** Tarefa de **classificação de imagem inteira**
(`yolo11n-cls` / `yolo11s-cls`).

**Por quê.** O dataset público adotado fornece rótulos em nível de imagem
(diretórios por classe), **sem nenhum arquivo de anotação** e sem `data.yaml`.
Não existe informação de caixa delimitadora a ser aprendida. Treinar detecção
exigiria anotar manualmente as ~2,2 mil imagens, atividade não prevista no
cronograma da Seção 6.2.6.6.

**Texto sugerido para o TCC.**

> O dataset público adotado fornece rótulos em nível de imagem, sem anotações de
> caixas delimitadoras. Optou-se, portanto, pela tarefa de classificação de
> imagens (YOLO11-cls), que atende integralmente ao requisito RF02 — identificar
> se a folha está saudável, com infecção bacteriana ou fúngica — e aos critérios
> de aceitação CT01–CT03. A localização por caixas delimitadoras exigiria
> anotação manual do conjunto, atividade não prevista no cronograma da
> Seção 6.2.6.6.

**Impacto em requisitos.** RF02, CT01, CT02 e CT03 continuam integralmente
atendidos: todos tratam de *identificar a classe* da folha, não de localizar a
lesão.

---

## D2 — Campos `bbox_*` sempre nulos; colunas devem ser `NULLABLE`

**O que o TCC diz.** A Tabela 13 (Seção 6.2.6.8) define `bbox_x`, `bbox_y`,
`bbox_w` e `bbox_h` como `NUMERIC(6,4)` com coordenadas normalizadas em `[0,1]`.

**O que foi implementado.** O `ResultadoDeteccao` preserva os quatro campos para
manter o contrato da tabela, mas eles são **sempre `None`**.

**Ação necessária no banco de dados.** As quatro colunas precisam ser declaradas
`NULLABLE`. A restrição de domínio `[0,1]` deve aceitar `NULL`:

```sql
bbox_x NUMERIC(6,4) NULL CHECK (bbox_x IS NULL OR (bbox_x >= 0 AND bbox_x <= 1)),
```

**Por quê.** Consequência direta de D1: um classificador de imagem inteira não
produz coordenadas.

---

## D3 — `imgsz` de treino igual a 224, não 640

**O que o TCC diz.** A Seção 5.3.4 (etapa 1 do processo de detecção) e a
Seção 6.1.1 especificam redimensionamento para 640×640 pixels, descrito como
"formato nativo do YOLOv11".

**O que foi implementado.** `imgsz=224`, o padrão da tarefa de classificação do
Ultralytics.

**Por quê.** Três razões:

1. **640×640 é o padrão da tarefa de detecção**, não da classificação. Para
   classificação, o pipeline oficial do Ultralytics usa 224.
2. **O dataset não tem resolução para justificar 640.** A mediana medida é
   474×355 px, e os tamanhos mais frequentes são 225×225 e 275×183. Ampliar para
   640 apenas interpola pixels, sem acrescentar informação.
3. **Custo computacional.** O treino roda em CPU neste ambiente; 640 px
   multiplicaria o custo por cerca de oito vezes sem ganho esperado de acurácia.

**Texto sugerido para o TCC.** Substituir a menção a 640×640 na Seção 6.1.1 por:

> As imagens são redimensionadas para 224×224 pixels, resolução nativa da tarefa
> de classificação do Ultralytics. A escolha considera que a resolução mediana do
> conjunto é de 474×355 pixels, de modo que uma entrada maior implicaria
> interpolação sem ganho de informação, além de elevar substancialmente o custo
> computacional de treinamento.

---

## D4 — Aumento de dados em tempo de treino, não gravado em disco

**O que o TCC diz.** A Seção 6.1.1 lista o *data augmentation* (rotação,
espelhamento, brilho, contraste, ruído gaussiano) como uma etapa do preparo do
dataset, sugerindo geração de arquivos.

**O que foi implementado.** O aumento é aplicado **em tempo de treino** pelo
Ultralytics (parâmetros `fliplr`, `flipud`, `degrees`, `scale`, `translate`,
`hsv_h`, `hsv_s`, `hsv_v`, `erasing`), sem gravar nenhuma imagem derivada.

**Por quê.** Gravar variantes em disco foi exatamente a causa dos problemas P1 e
P2 identificados na auditoria (`desenvolvimento/reports/dataset_auditoria.md`): 597 arquivos
byte-idênticos, 104 grupos de duplicatas cruzando a fronteira dos splits e 115
grupos de folha aparecendo em mais de um split. Manter o aumento em memória
elimina a possibilidade de a variante de uma folha de treino reaparecer no
conjunto de teste.

**Texto sugerido para o TCC.**

> As técnicas de aumento de dados são aplicadas dinamicamente a cada época de
> treinamento, sem persistência das imagens derivadas. Essa decisão evita o
> vazamento de dados entre os subconjuntos de treino, validação e teste —
> problema identificado no dataset original, em que variantes aumentadas de uma
> mesma folha estavam distribuídas entre os três subconjuntos.

---

## D5 — Divisão 70/15/15 aplicada sobre grupos de folhas

**O que o TCC diz.** A Seção 6.1.1 especifica "separação estratificada nas
proporções 70% treino, 15% validação e 15% teste, preservando a distribuição de
classes em cada subconjunto".

**O que foi implementado.** Exatamente essas proporções e essa estratificação,
mas a unidade de divisão é o **grupo de folha de origem**, não o arquivo isolado.

**Por quê.** O dataset nomeia variantes de uma mesma folha como `<origem>_<n>`
(por exemplo `ba12_0`, `ba12_2`, `ba12_3`). Dividir por arquivo colocaria
variantes da mesma folha em splits diferentes, inflando artificialmente as
métricas de teste. Dividir por grupo é o procedimento correto quando o conjunto
contém múltiplas amostras da mesma unidade experimental.

**Nota.** A divisão original do dataset baixado era de aproximadamente 84/8/8, e
não 70/15/15.

---

## D6 — Grad-CAM como substituto da localização visual

**O que o TCC diz.** A Seção 7.4 prevê que o dashboard exiba "as detecções de
patógenos com bounding boxes sobrepostos às imagens capturadas".

**O que foi implementado.** Mapas de calor **Grad-CAM**
(`src/ia_alface/explicabilidade.py`), que destacam as regiões da folha que mais
pesaram na decisão do modelo. O `ResultadoDeteccao` traz o campo `heatmap_url`
para o caminho da imagem gerada.

**Por quê.** Atende à intenção do requisito — mostrar ao agricultor *onde* está o
problema — sem exigir a anotação manual de caixas que D1 tornou inviável.

**Texto sugerido para o TCC.**

> Em substituição às caixas delimitadoras, o sistema gera mapas de calor pela
> técnica Grad-CAM (Selvaraju et al., 2017), que destacam as regiões da folha
> mais determinantes para a classificação. A visualização cumpre a função de
> orientar espacialmente o agricultor sem depender de anotação manual de caixas
> delimitadoras.

---

## D7 — `label_smoothing` não é aplicado (removido do Ultralytics)

**O que o spec previa.** A Seção 6 do `spec.md` lista `label_smoothing=0.05`
entre os hiperparâmetros de treino, como técnica de regularização para evitar que
a rede fique excessivamente confiante.

**O que acontece na prática.** A partir do Ultralytics 8.4.x o parâmetro foi
**removido**: ele consta do conjunto `removed_keys` em
`ultralytics/cfg/__init__.py`, ao lado de `save_hybrid` e `crop_fraction`. A
biblioteca aceita o argumento sem erro, mas o descarta — o `args.yaml` da
execução registra `label_smoothing: null`.

**Como foi detectado.** Conferindo o `args.yaml` da execução de treino no Colab
contra os valores esperados do spec. Todos os 23 hiperparâmetros conferiram,
exceto este.

**Ação tomada.** O parâmetro foi removido de `ConfigTreino`
(`src/ia_alface/config.py`) para que o código não sugira uma regularização que
não está ativa. O restante da configuração permanece idêntico ao spec.

**Impacto no resultado.** Nenhum impacto observado: o modelo treinado atingiu
95,06% de acurácia no conjunto de teste, acima da meta de 90% de RNF02/H1. As
demais regularizações (`dropout=0.1`, *early stopping* com `patience=20` e a
augmentation em tempo de treino) permanecem ativas.

**Atenção ao redigir o TCC.** Não afirme que *label smoothing* foi empregado.
Descreva as técnicas de regularização efetivamente aplicadas: dropout, parada
antecipada e aumento de dados.

---

## Resumo do impacto nos requisitos

| Requisito / Caso de teste | Situação | Observação |
| --- | --- | --- |
| RF02 — classificar patógeno | Atendido | Classificação em três classes |
| RNF01 — inferência < 100 ms | Verificável | Ver `desenvolvimento/reports/benchmark_latencia.md` |
| RNF02 — acurácia > 90% | Verificável | Ver `desenvolvimento/reports/avaliacao_test.md` |
| CT01 / CT02 — confiança ≥ 0,85 | Atendido | Classe + confiança, sem caixa |
| CT03 — recall de saudável ≥ 90% | Atendido | — |
| CT04 — latência < 100 ms | Verificável | — |
| Tabela `deteccoes` (bbox) | **Requer alteração** | Colunas `bbox_*` para `NULLABLE` (D2) |
| Seção 7.4 — bounding boxes no dashboard | **Substituído** | Grad-CAM (D6) |
