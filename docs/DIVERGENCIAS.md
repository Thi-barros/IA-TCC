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

## D8 — Verificação de domínio em cascata, resposta `nao_alface` e câmera C270

**O que o TCC diz.**

- Toda imagem recebida pelo endpoint `/analyze` é classificada pelo YOLOv11
  (Seção 6.2.6.7). O texto não prevê etapa que verifique se a imagem contém
  alface: uma foto do solo, da estrutura, da mão do agricultor ou de outra
  planta receberia uma das três classes.
- A coluna `classe` da tabela `deteccoes` é
  `ENUM ('saudavel','bacteriano','fungico')` (Tabela 13, Seção 6.2.6.8).
- A captura das imagens cabe ao módulo ESP32-CAM, ou a um módulo de câmera do
  ESP32 (Seções 3.2, 5.2.4, 5.3.2, 6.2.6.3 — Tabela 7 —, 6.2.6.4 e 6.2.6.7).

**O que foi implementado.**

1. **Cascata com verificador de domínio.** Antes do classificador roda um
   segundo YOLO11n-cls, binário (`alface` / `nao_alface`, rótulos em
   `src/ia_alface/classes.py`). A imagem segue para o classificador quando a
   probabilidade de alface atinge o limiar de 0,50. O classificador de doenças
   não muda. O verificador é ligado pela variável `IA_ALFACE_MODELO_DOMINIO`,
   com o limiar em `IA_ALFACE_LIMIAR_DOMINIO`; sem a variável, a API responde
   exatamente como antes. Pesos a implantar: `modelos/yolo11n_cls_dominio_v2.pt`.
2. **Recusa como resposta, não como erro.** A imagem sem alface recebe HTTP 200
   com `classe = "nao_alface"`, `classe_exibicao = "Alface não encontrada"`,
   `confianca = 1 − probabilidade_alface`, `probabilidades = {}`,
   `patogeno_detectado` e `alerta` falsos e o bloco `verificacao` preenchido.
   Entre 14 e 24/10/2026 a recusa era um HTTP 422, substituído antes da
   implantação. Continuam valendo o 400 para imagem corrompida, antes de
   qualquer modelo, e o 503 quando o verificador está configurado e não
   carrega: o serviço não diagnostica sem ele.
3. **Câmera.** As fotos reais do treino e do teste do verificador são da webcam
   Logitech C270, a câmera do projeto (informado pelo autor em 15/10/2026).
   Fotos de celular entram só no treino. A API não depende da câmera: recebe
   JPEG ou PNG de qualquer origem.

**Por quê.**

1. O classificador é de conjunto fechado: pela *softmax*, sempre escolhe uma
   das três classes, inclusive para imagens sem alface, e pode emitir alerta
   falso. Recusar por confiança baixa não basta: o próprio classificador errou
   com confiança 1,000. Uma quarta classe "não alface" no classificador exigiria
   retreiná-lo e invalidaria os resultados medidos (RNF02, CT01–CT03); o
   verificador separado preserva o classificador.
2. A recusa no mesmo campo `classe` deixa o consumidor tratar todas as respostas
   num só lugar e elimina a ambiguidade do 422, que o FastAPI já usa para
   requisição malformada.
3. A troca de câmera é decisão de hardware do projeto; este módulo a adota nos
   dados e na documentação. A troca refere-se só à câmera: o uso do ESP32 para
   os sensores ambientais está fora deste módulo e não foi alterado aqui.

**Ação necessária no banco de dados e no back-end.**

- O `ENUM` da coluna `classe` precisa aceitar `nao_alface`, ou o back-end (ou a
  Azure Function) não grava essas respostas. A escolha cabe ao grupo do banco e
  está **pendente**. No PostgreSQL, a primeira opção é:

  ```sql
  -- o nome do tipo depende do schema definido pelo grupo do banco
  ALTER TYPE <tipo_da_coluna_classe> ADD VALUE 'nao_alface';
  ```

- Todo consumidor deve decidir pelo `alerta` e pelo `patogeno_detectado` da
  resposta, nunca por "classe diferente de `saudavel`": esse teste
  transformaria a recusa em alerta falso.
- A Function trata `classe = "nao_alface"` como resultado final, sem retry, e
  não grava diagnóstico.

**Situação do critério de implantação.** O verificador v2 **não atingiu** o
critério fixado antes do teste: folhas doentes aceitas 178/184 (mínimo 183) e
"não alface" da C270 recusada 42/48 (mínimo 46). A implantação é decisão do
autor posterior ao resultado, registrada como exceção na Seção 3.8.5 de
`docs/DOCUMENTACAO_IA_TREINAMENTO_RESULTADOS.md` e na Seção 4.8 de
`docs/VERIFICADOR_DOMINIO.md`. Não descreva o verificador como "aprovado" nem
como "dentro das metas".

**Texto sugerido para o TCC.** Na Seção 6.2.6.7 e, resumido, na 6.2.6.3:

> Antes da classificação, um verificador de domínio — um segundo modelo
> YOLO11n-cls, treinado para decidir se a imagem contém alface — examina cada
> imagem capturada pela câmera do sistema, uma webcam Logitech C270. Imagens sem
> alface recebem a resposta “Alface não encontrada” (classe nao_alface), sem
> diagnóstico e sem alerta; as demais seguem para o classificador de doenças,
> inalterado. A etapa evita que o solo, a estrutura, a mão do agricultor ou
> outras plantas recebam diagnóstico. Para que essas respostas sejam
> registradas, a coluna classe da tabela deteccoes deve aceitar o valor
> nao_alface; caso contrário, elas não são gravadas.

Nas Seções 3.2, 5.2.4, 5.3.2, 6.2.6.3 (Tabela 7), 6.2.6.4 e 6.2.6.7, onde o
texto atribui a captura de imagens ao ESP32-CAM ou a um módulo de câmera do
ESP32, substituir pela webcam Logitech C270.

**Impacto em requisitos.**

- RF02, RNF02/H1 e CT01–CT03 são medidos no classificador, que não mudou, e
  continuam valendo: acurácia de 95,06% no teste.
- Sistema completo (verificador + classificador), nas mesmas 324 imagens de
  teste: acurácia 94,1% (305/324), ainda acima dos 90% do RNF02; revocação
  fúngica 80,8% (63/78), contra 84,6% do classificador; 1 alerta verdadeiro
  suprimido. O CT01 do TCC mede confiança nos acertos, não revocação; se for
  lido como revocação fúngica ≥ 85% (critério do contexto geral,
  `CONTEXTO_IA_CONFIAVEL.md`, Seção 6.2), o sistema completo fica em 80,8%.
- RNF01: latência mediana das duas etapas de 51,85 ms (pesos `.pt`, CPU local),
  abaixo de 100 ms. Na nuvem, com a imagem `v2` implantada em 26/10/2026:
  22,62 ms de mediana. A inicialização a frio, de 11,27 s numa única medida,
  fica fora da fronteira do RNF01.
- Imagens sem alface deixam de receber diagnóstico, o que reduz alertas falsos
  (favorável à H3). Outras plantas ainda podem ser aceitas: no teste, 14 de 354
  imagens sem alface foram aceitas, todas de plantas.

---

## Resumo do impacto nos requisitos

| Requisito / Caso de teste | Situação | Observação |
| --- | --- | --- |
| RF02 — classificar patógeno | Atendido | Classificação em três classes |
| RNF01 — inferência < 100 ms | Verificável | Ver `desenvolvimento/reports/benchmark_latencia.md`; com o verificador, duas etapas em 51,85 ms (`desenvolvimento/reports/verificador_v2_test.md`, D8) e 22,62 ms na nuvem |
| RNF02 — acurácia > 90% | Verificável | Ver `desenvolvimento/reports/avaliacao_test.md` |
| CT01 / CT02 — confiança ≥ 0,85 | Atendido | Classe + confiança, sem caixa; medido no classificador. No sistema completo, revocação fúngica de 80,8% (D8) |
| CT03 — recall de saudável ≥ 90% | Atendido | — |
| CT04 — latência < 100 ms | Verificável | — |
| Tabela `deteccoes` (bbox) | **Requer alteração** | Colunas `bbox_*` para `NULLABLE` (D2) |
| Seção 7.4 — bounding boxes no dashboard | **Substituído** | Grad-CAM (D6) |
| Tabela `deteccoes` (classe) | **Requer alteração** | `ENUM` precisa aceitar `nao_alface`, ou essas respostas não são gravadas (D8) |
| Captura por ESP32-CAM | **Substituído** | Webcam Logitech C270 (D8) |
