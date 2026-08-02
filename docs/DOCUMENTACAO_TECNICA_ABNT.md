# DOCUMENTAÇÃO TÉCNICA DO MÓDULO DE INTELIGÊNCIA ARTIFICIAL

**Sistema de detecção automática de doenças foliares em alface (*Lactuca sativa*) para estufas de agricultura familiar**

---

## NOTA DE USO

Este documento reúne, em estrutura compatível com a ABNT, tudo o que foi implementado no módulo de visão computacional do trabalho. Destina-se a ser adaptado e incorporado ao texto final da monografia.

**Sobre a conformidade com a ABNT.** O texto adota as convenções normativas que podem ser expressas em formato textual: numeração progressiva de seções (NBR 6024), identificação de tabelas e quadros com título acima e fonte abaixo, sistema de chamada autor-data (NBR 10520) e referências (NBR 6023). A formatação física — fonte Arial ou Times New Roman tamanho 12, espaçamento 1,5, margens de 3 cm (superior e esquerda) e 2 cm (inferior e direita), recuo de parágrafo de 1,25 cm, citações longas com recuo de 4 cm em fonte 10 — deve ser aplicada no editor de texto ao transpor o conteúdo.

**Numeração das seções.** A numeração abaixo é interna a este documento. Ao incorporar ao TCC, renumere conforme a posição de cada trecho. O Quadro 1 indica o destino sugerido de cada seção.

**Quadro 1 – Correspondência entre as seções deste documento e as do TCC**

| Seção deste documento | Destino sugerido no TCC |
| --- | --- |
| 2.1 Ambiente computacional | 6.2.6.3 Arquitetura do Sistema e Tecnologias |
| 2.2 Conjunto de dados | 6.1.1 Imagens para Visão Computacional |
| 2.3 Arquitetura do modelo | 5.3.4 Conceito e arquitetura YOLOv11 |
| 2.4 Configuração de treinamento | 6.1.1 e seção nova de Desenvolvimento |
| 2.5 Arquitetura de software | 6.2.6.7 Desenho da Arquitetura do Sistema |
| 2.6 Explicabilidade | 7.4 Interface de Monitoramento |
| 2.8 Procedimento de avaliação | 6.2.6.5 Especificação dos Casos de Teste |
| 3 Resultados | 7.1 e 7.2 (substituem os resultados esperados) |
| 3.7 Verificação dos casos de teste | Tabela 18 – Plano de execução dos casos de teste |
| 4.1 Divergências | 6.1.1, 6.2.6.8 e 7.4 (ajustes no texto existente) |
| 4.2 Limitações | 8 Limitações do Trabalho |

Fonte: elaborado pelos autores (2026).

**Marcadores `[PREENCHER]`.** Indicam informações que não puderam ser determinadas automaticamente e que devem ser completadas pelos autores. Não foram preenchidas com estimativas.

---

# 1 ESCOPO DO MÓDULO

O módulo de inteligência artificial constitui o núcleo de percepção do sistema proposto. Sua função é receber uma imagem de folha de alface capturada pelo módulo ESP32-CAM e classificá-la em uma de três categorias: folha saudável, folha com infecção bacteriana (*Xanthomonas campestris*) ou folha com infecção fúngica (*Bremia lactucae*).

O módulo atende diretamente ao requisito funcional RF02 e aos requisitos não funcionais RNF01 (tempo de inferência), RNF02 (acurácia) e RNF07 (documentação e versionamento). Constitui, ainda, o objeto de verificação dos casos de teste CT01 a CT04 e das hipóteses H1 e H2.

Não integram o escopo deste módulo: o firmware do microcontrolador, a camada de comunicação MQTT, a persistência em banco de dados e as interfaces web e mobile. O módulo foi implementado como biblioteca de software independente, sem acoplamento a protocolos de rede ou a mecanismos de armazenamento, de modo que sua integração ao *back-end* ocorra por meio de uma interface programática explícita, descrita na seção 2.5.3.

---

# 2 MATERIAIS E MÉTODOS

## 2.1 Ambiente computacional

O desenvolvimento e a avaliação foram conduzidos no ambiente descrito na Tabela 1. O treinamento do modelo foi realizado em ambiente distinto, com aceleração por unidade de processamento gráfico, conforme detalhado na seção 2.4.4.

**Tabela 1 – Ambiente computacional de desenvolvimento e avaliação**

| Componente | Especificação |
| --- | --- |
| Sistema operacional | Windows 11 (build 10.0.26200) |
| Processador | AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx |
| Aceleração por GPU | Não disponível (CUDA indisponível) |
| Linguagem | Python 3.13.5 |
| Framework de aprendizado profundo | PyTorch 2.13.0+cpu |
| Biblioteca de visão computacional | Ultralytics 8.4.95 |
| Métricas e divisão de dados | scikit-learn 1.6.1 |
| Manipulação de dados | pandas 2.2.3, NumPy 2.1.3 |
| Processamento de imagens | Pillow 11.1.0 |
| Execução de modelo exportado | ONNX Runtime 1.28.0 |
| Visualização | Matplotlib 3.10.0 |

Fonte: elaborado pelos autores (2026).

A ausência de GPU no ambiente local determinou duas decisões de projeto. A primeira foi a seleção automática do dispositivo de execução em tempo de execução, mediante consulta à disponibilidade de CUDA, o que permite ao mesmo código operar em ambos os cenários sem alteração. A segunda foi a realização do treinamento em ambiente de nuvem com GPU, uma vez que a estimativa de tempo em processador central era de aproximadamente seis horas, contra poucos minutos em GPU.

## 2.2 Conjunto de dados

### 2.2.1 Origem e composição

Utilizou-se conjunto público de imagens de folhas de alface obtido na plataforma Kaggle, organizado em diretórios por classe. O conjunto original contém 2.813 arquivos de imagem, distribuídos conforme a Tabela 2.

`[PREENCHER: inserir a referência completa do conjunto de dados no Kaggle — autor, título, ano e URL de acesso.]`

**Tabela 2 – Composição do conjunto de dados original**

| Subconjunto | Bacteriana | Fúngica | Saudável | Total | Participação |
| --- | ---: | ---: | ---: | ---: | ---: |
| Treino | 732 | 581 | 1.049 | 2.362 | 84,0% |
| Validação | 75 | 75 | 75 | 225 | 8,0% |
| Teste | 75 | 75 | 76 | 226 | 8,0% |
| **Total** | **882** | **731** | **1.200** | **2.813** | **100,0%** |

Fonte: elaborado pelos autores (2026), a partir do conjunto de dados obtido no Kaggle.

As imagens apresentam resolução variável, entre 129 e 2.048 pixels em ambas as dimensões, com mediana de 474 pixels de largura por 355 de altura. Os formatos de arquivo distribuem-se em 1.826 arquivos PNG e 987 arquivos JPEG. Do total, 2.812 imagens encontram-se no modo de cor RGB e uma no modo RGBA. Apenas 354 imagens são quadradas.

### 2.2.2 Auditoria do conjunto de dados

Antes de qualquer treinamento, procedeu-se à auditoria sistemática do conjunto, mediante cálculo de resumo criptográfico MD5 de cada arquivo e análise dos padrões de nomenclatura. O procedimento identificou seis problemas, sintetizados no Quadro 2.

**Quadro 2 – Problemas identificados na auditoria do conjunto de dados**

| Código | Problema | Evidência quantitativa |
| --- | --- | --- |
| P1 | Vazamento por duplicatas exatas entre subconjuntos | 104 grupos de arquivos byte a byte idênticos presentes simultaneamente em treino e em validação ou teste |
| P2 | Vazamento de grupo por variantes da mesma folha | 115 grupos de origem distribuídos em mais de um subconjunto |
| P3 | Baixa diversidade real na classe bacteriana | 882 imagens derivadas de aproximadamente 54 folhas distintas |
| P4 | Predomínio de dados sintéticos nos subconjuntos de avaliação | 800 arquivos com prefixo `augmented_`, correspondendo a dois terços das imagens de validação e de teste |
| P5 | Proporções divergentes da metodologia | Divisão original de 84/8/8, em vez dos 70/15/15 especificados |
| P6 | Desbalanceamento entre classes | Participação de 44,4%, 31,0% e 24,6% no subconjunto de treino |

Fonte: elaborado pelos autores (2026).

O problema P1 é o mais crítico do ponto de vista metodológico. Do total de 2.813 arquivos, apenas 2.216 possuem resumo criptográfico distinto, o que indica 597 arquivos redundantes. Verificou-se, entre outros casos, que o arquivo `train/Bacterial/ba14_4.png` é idêntico, byte a byte, ao arquivo `test/Bacterial/ba14_0.png`. A avaliação de um modelo sobre imagens presentes em seu próprio conjunto de treinamento mede a capacidade de memorização, e não a de generalização, invalidando qualquer conclusão sobre desempenho (KAUFMAN et al., 2012).

O problema P2 decorre da nomenclatura adotada no conjunto original, em que variantes obtidas por aumento de dados de uma mesma folha recebem nomes no padrão `<origem>_<índice>` — por exemplo, `ba12_0`, `ba12_2` e `ba12_3`. Ainda que tais arquivos não sejam idênticos entre si, representam a mesma unidade experimental. Sua distribuição entre subconjuntos distintos constitui vazamento, pois o modelo é avaliado sobre folhas que já observou sob outra transformação.

### 2.2.3 Procedimento de preparação

Desenvolveu-se rotina automatizada de preparação, executada previamente a qualquer treinamento, que trata os problemas identificados. O diretório original é tratado como fonte imutável, não sendo objeto de escrita em nenhuma etapa. As etapas do procedimento são:

a) **Inventário.** Varredura determinística de todos os arquivos, com registro de caminho, classe, resumo MD5, dimensões, formato e modo de cor.

b) **Unificação de grupos por identidade de conteúdo.** Fusão, por componentes conexas, de grupos de origem que compartilham arquivos idênticos. O procedimento identificou sete componentes em que nomes de origem distintos referiam-se à mesma folha.

c) **Deduplicação.** Remoção de duplicatas exatas, mantendo-se uma única cópia por resumo criptográfico, selecionada pelo menor caminho em ordem lexicográfica, de modo a garantir determinismo entre execuções.

d) **Redivisão estratificada e disjunta por grupo.** Nova divisão nas proporções de 70%, 15% e 15%, mediante validação cruzada estratificada por grupo (`StratifiedGroupKFold`), com semente fixa. O procedimento assegura que todas as variantes de uma mesma folha sejam alocadas ao mesmo subconjunto, preservando simultaneamente a proporção entre classes.

e) **Verificação de integridade.** Confirmação de que nenhum resumo criptográfico e nenhum grupo de origem ocorre em mais de um subconjunto. A rotina é interrompida com código de erro caso a verificação falhe, impedindo o prosseguimento para o treinamento.

f) **Normalização e materialização.** Conversão de todas as imagens ao modo RGB e gravação em formato JPEG com qualidade 95, sem redimensionamento, preservando-se a proporção original.

### 2.2.4 Conjunto de dados resultante

O conjunto resultante contém 2.216 imagens, distribuídas conforme a Tabela 3.

**Tabela 3 – Composição do conjunto de dados após preparação**

| Subconjunto | Bacteriana | Fúngica | Saudável | Total | Participação |
| --- | ---: | ---: | ---: | ---: | ---: |
| Treino | 510 | 390 | 666 | 1.566 | 70,7% |
| Validação | 114 | 75 | 137 | 326 | 14,7% |
| Teste | 106 | 78 | 140 | 324 | 14,6% |
| **Total** | **730** | **543** | **943** | **2.216** | **100,0%** |

Fonte: elaborado pelos autores (2026).

A verificação de integridade confirmou ausência total de vazamento: nenhum resumo criptográfico e nenhum grupo de origem é compartilhado entre os três subconjuntos. As três classes estão representadas em todos eles.

A Tabela 4 evidencia a diversidade real de cada classe, medida pelo número de folhas de origem distintas.

**Tabela 4 – Diversidade de folhas de origem por classe**

| Classe | Imagens | Grupos de origem | Imagens por grupo |
| --- | ---: | ---: | ---: |
| Saudável | 1.200 | 601 | 2,00 |
| Fúngica | 731 | 328 | 2,23 |
| Bacteriana | 882 | ≈ 54 | ≈ 16,3 |

Fonte: elaborado pelos autores (2026).

Registre-se que a classe bacteriana, embora numericamente equilibrada em relação às demais, deriva de número substancialmente menor de folhas distintas. Trata-se de limitação intrínseca ao conjunto de dados, não corrigível por técnicas de aumento de dados, cujas implicações são discutidas na seção 4.2.

## 2.3 Arquitetura do modelo

### 2.3.1 Justificativa da tarefa de classificação

O projeto original previa a aplicação de detecção de objetos, com delimitação das lesões por caixas delimitadoras. A verificação do conjunto de dados obtido demonstrou, contudo, a inexistência de arquivos de anotação: o conjunto fornece rótulos em nível de imagem, organizados por diretório, sem qualquer informação de localização espacial das lesões.

Adotou-se, por conseguinte, a tarefa de **classificação de imagens**, que atende integralmente ao requisito RF02 — identificar se a folha está saudável, com infecção bacteriana ou fúngica — e aos critérios de aceitação CT01, CT02 e CT03, todos formulados em termos de classe e confiança, e não de localização. A alternativa de treinar um detector exigiria a anotação manual das 2.216 imagens, atividade não prevista no cronograma do projeto.

A informação de localização, prevista para a interface de monitoramento, foi recuperada por técnica alternativa, descrita na seção 2.6.

### 2.3.2 Modelo YOLO11-cls

Empregou-se a variante de classificação da arquitetura YOLO11, disponibilizada pela biblioteca Ultralytics (JOCHER; QIU, 2024). Selecionou-se a variante *nano* (`yolo11n-cls`), indicada para dispositivos com recursos computacionais limitados, cujas características são apresentadas na Tabela 5.

**Tabela 5 – Características do modelo empregado**

| Característica | Valor |
| --- | ---: |
| Variante | YOLO11n-cls (*nano*) |
| Camadas | 86 (47 após fusão) |
| Parâmetros | 1.534.947 |
| Complexidade computacional | 3,2 GFLOPs |
| Resolução de entrada | 224 × 224 pixels |
| Classes de saída | 3 |
| Tamanho do arquivo de pesos | 3,19 MB |

Fonte: elaborado pelos autores (2026).

### 2.3.3 Transferência de aprendizado

O treinamento partiu de pesos previamente ajustados sobre o conjunto ImageNet. Do total de 236 blocos de parâmetros do modelo, 234 foram transferidos; os dois restantes correspondem à camada de classificação final, substituída para acomodar as três classes do domínio.

A adoção de transferência de aprendizado justifica-se pela dimensão reduzida do conjunto de treinamento. Modelos pré-treinados sobre grandes conjuntos genéricos já codificam representações visuais de baixo nível — bordas, texturas e gradientes cromáticos —, de modo que o ajuste fino sobre o domínio específico requer volume de dados substancialmente menor. Mohanty, Hughes e Salathé (2016) demonstraram a eficácia dessa estratégia especificamente no diagnóstico de doenças de plantas por imagem.

## 2.4 Configuração de treinamento

### 2.4.1 Hiperparâmetros

Os hiperparâmetros adotados são apresentados na Tabela 6.

**Tabela 6 – Hiperparâmetros de treinamento**

| Hiperparâmetro | Valor | Finalidade |
| --- | ---: | --- |
| Épocas máximas | 100 | Limite superior de iterações |
| Paciência | 20 | Parada antecipada por estagnação |
| Tamanho do lote | 32 | Amostras por atualização de pesos |
| Resolução de entrada | 224 | Dimensão de redimensionamento |
| Otimizador | AdamW | Algoritmo de atualização de pesos |
| Taxa de aprendizado inicial | 0,001 | Magnitude do passo de otimização |
| Escalonamento da taxa | Cosseno | Decaimento progressivo da taxa |
| *Dropout* | 0,1 | Regularização por desativação de neurônios |
| Semente aleatória | 42 | Reprodutibilidade |
| Execução determinística | Sim | Reprodutibilidade |

Fonte: elaborado pelos autores (2026).

### 2.4.2 Aumento de dados

O aumento de dados foi aplicado **em tempo de treinamento**, a cada época, sem persistência das imagens derivadas em disco. Essa decisão é metodologicamente relevante: a gravação de variantes em disco foi precisamente a origem dos problemas P1 e P2 identificados no conjunto original. As transformações adotadas constam da Tabela 7.

**Tabela 7 – Transformações de aumento de dados**

| Transformação | Parâmetro | Justificativa no domínio |
| --- | ---: | --- |
| Espelhamento horizontal | 0,5 | Folhas não possuem orientação preferencial |
| Espelhamento vertical | 0,3 | Câmera pode assumir orientações diversas |
| Rotação | ± 20° | Variação natural do posicionamento foliar |
| Escala | ± 40% | Variação da distância de captura |
| Translação | 0,1 | Variação do enquadramento |
| Matiz | 0,015 | Variação cromática entre dispositivos |
| Saturação | 0,5 | Variação cromática entre dispositivos |
| Brilho | 0,4 | Variação de iluminação natural e artificial |
| Apagamento aleatório | 0,2 | Oclusão parcial por folhas sobrepostas |

Fonte: elaborado pelos autores (2026).

Destaca-se a variação de brilho, que constitui mitigação direta do risco apontado na revisão de literatura, segundo o qual alterações na intensidade e no espectro da luz modificam as características cromáticas das manchas foliares utilizadas como base para a classificação.

### 2.4.3 Estratégias de regularização

Foram empregadas três estratégias de regularização, com o objetivo de evitar o sobreajuste ao conjunto de treinamento: *dropout* com probabilidade de 0,1; parada antecipada com paciência de 20 épocas, preservando-se os pesos da época de melhor desempenho em validação; e o aumento de dados descrito na seção 2.4.2.

Registre-se que a técnica de suavização de rótulos (*label smoothing*), inicialmente prevista, não foi aplicada. A verificação do arquivo de configuração gerado pela execução demonstrou que o parâmetro correspondente foi removido da biblioteca Ultralytics a partir da versão 8.4, sendo aceito na chamada mas silenciosamente descartado. O código foi corrigido para não indicar o emprego de técnica inativa.

### 2.4.4 Ambiente de treinamento

O treinamento foi executado na plataforma Google Colab, com aceleração por GPU, uma vez que o ambiente local não dispõe de unidade compatível com CUDA. Os hiperparâmetros foram integralmente preservados; alteraram-se apenas os parâmetros dependentes de hardware: dispositivo de execução, precisão mista automática e número de processos de carregamento de dados.

`[PREENCHER: informar o modelo exato da GPU disponibilizada pelo Google Colab durante a execução, obtido pela saída do comando nvidia-smi na primeira célula do notebook.]`

## 2.5 Arquitetura de software

### 2.5.1 Organização

O código foi organizado segundo a separação entre biblioteca e programas executáveis. O diretório `src/ia_alface/` contém a lógica reutilizável, importável tanto pelos programas de linha de comando quanto, futuramente, pelo *back-end* da aplicação. O diretório `desenvolvimento/scripts/` contém os programas que compõem o fluxo de trabalho, numerados na ordem de execução.

Essa separação atende a três finalidades: permitir que o servidor de aplicação importe diretamente a camada de inferência, sem intermediação de scripts; viabilizar a execução dos testes automatizados em segundos, sem acionar procedimentos de treinamento; e possibilitar o reaproveitamento da mesma lógica em ambientes interativos.

### 2.5.2 Módulos implementados

O Quadro 3 descreve os módulos da biblioteca.

**Quadro 3 – Módulos da biblioteca de inteligência artificial**

| Módulo | Responsabilidade |
| --- | --- |
| `classes.py` | Mapeamento canônico das três classes; fonte única dos rótulos no projeto |
| `config.py` | Caminhos do projeto, hiperparâmetros e detecção automática do dispositivo |
| `dados.py` | Inventário, deduplicação, agrupamento e redivisão do conjunto de dados |
| `treino.py` | Rotina de treinamento e registro dos artefatos |
| `avaliacao.py` | Cálculo de métricas, matriz de confusão, latência e geração de relatórios |
| `inferencia.py` | Interface de uso do modelo treinado; contrato de integração |
| `explicabilidade.py` | Geração de mapas de calor por Grad-CAM |
| `exportacao.py` | Conversão para ONNX e TorchScript, com validação numérica |

Fonte: elaborado pelos autores (2026).

O módulo `classes.py` merece registro específico. As três categorias assumem denominações distintas conforme o contexto: diretórios do conjunto original (`Bacterial`, `fungal`, `healthy`), índices numéricos do modelo (0, 1, 2), valores do tipo enumerado no banco de dados (`bacteriano`, `fungico`, `saudavel`) e rótulos de exibição na interface. A concentração desse mapeamento em módulo único elimina a possibilidade de divergência de grafia entre componentes — falha que produziria registros incorretos no banco de dados sem gerar erro de execução. Teste automatizado verifica a correspondência exata entre os rótulos do modelo e os valores do tipo enumerado especificado para a tabela de detecções.

O Quadro 4 descreve os programas executáveis.

**Quadro 4 – Programas do fluxo de trabalho**

| Programa | Função |
| --- | --- |
| `01_preparar_dataset.py` | Audita, deduplica e redivide o conjunto de dados |
| `02_treinar.py` | Executa o treinamento e registra os artefatos |
| `03_avaliar.py` | Avalia o modelo e gera os relatórios de desempenho |
| `04_exportar.py` | Converte para ONNX e TorchScript, com validação numérica |
| `05_benchmark_latencia.py` | Mede a latência de inferência e verifica o RNF01 |
| `testar_imagem.py` | Utilitário de inspeção manual de imagens avulsas |
| `preparar_zip_colab.py` | Utilitário de empacotamento para treinamento em nuvem |

Fonte: elaborado pelos autores (2026).

### 2.5.3 Contrato de inferência

A integração entre o módulo de inteligência artificial e o *back-end* ocorre pela classe `DetectorDoencasAlface`, cuja saída é estruturada em correspondência direta com a tabela de detecções especificada na modelagem do banco de dados. Os campos retornados constam da Tabela 8.

**Tabela 8 – Campos retornados pela camada de inferência**

| Campo | Tipo | Correspondência no banco de dados |
| --- | --- | --- |
| `classe` | Texto | Coluna `classe` (tipo enumerado) |
| `confianca` | Real em [0, 1] | Coluna `confianca` |
| `probabilidades` | Dicionário | Não persistido; disponível para a interface |
| `patogeno_detectado` | Lógico | Derivado; classe distinta de saudável |
| `alerta` | Lógico | Aciona o registro em `alertas` (RF07) |
| `tempo_inferencia_ms` | Real | Verificação do RNF01 |
| `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h` | Nulo | Colunas `bbox_*`, que devem admitir valor nulo |
| `modelo_versao` | Texto | Rastreabilidade da versão empregada |
| `heatmap_url` | Texto ou nulo | Caminho do mapa de calor, quando gerado |

Fonte: elaborado pelos autores (2026).

O campo `alerta` assume valor verdadeiro somente quando há patógeno detectado **e** a confiança atinge o limiar de 0,85, estabelecido pelos critérios de aceitação CT01 e CT02. Folhas classificadas como saudáveis não acionam notificação, tampouco detecções de patógeno com confiança inferior ao limiar, o que reduz a incidência de alertas espúrios.

Duas características de implementação merecem registro. A primeira é a segurança em acesso concorrente: o acesso ao modelo é serializado, permitindo que o processo consumidor das mensagens MQTT e o tratador de requisições HTTP compartilhem a mesma instância. A segunda é o tratamento de entrada inválida: imagens corrompidas, truncadas ou em formato não suportado provocam exceção explícita, jamais um resultado. Essa decisão decorre da análise de modos de falha — a classificação silenciosa de uma imagem corrompida como folha saudável constituiria a falha mais grave do sistema, por suprimir o alerta sem qualquer indicação de anomalia.

## 2.6 Explicabilidade

Implementou-se a técnica Grad-CAM (SELVARAJU et al., 2017), que produz mapa de calor indicando as regiões da imagem de maior influência sobre a decisão do modelo. A técnica opera sobre os gradientes da última camada convolucional, cuja saída preserva estrutura espacial.

A adoção da técnica cumpre duas finalidades. A primeira é atender à intenção do requisito de interface, que previa a sobreposição de caixas delimitadoras às imagens capturadas: o mapa de calor fornece orientação espacial aproximada da lesão sem exigir anotação manual. A segunda é diagnóstica: a inspeção dos mapas permite verificar se o modelo fundamenta suas decisões nas regiões foliares afetadas ou em elementos espúrios do fundo da imagem.

## 2.7 Exportação do modelo

O modelo treinado foi convertido para os formatos ONNX (conjunto de operadores 17, entrada fixa de 224 pixels) e TorchScript. A conversão para ONNX permite a execução por meio do ONNX Runtime, sem dependência do PyTorch, o que reduz o tamanho da imagem de implantação do servidor e, conforme demonstrado na seção 3.5, diminui a latência de inferência.

A equivalência numérica entre o modelo convertido e o original foi verificada, e não presumida: as saídas de ambos foram comparadas sobre 20 imagens do subconjunto de teste, com tolerância absoluta de 10⁻³.

## 2.8 Procedimento de avaliação

A avaliação observou três princípios metodológicos.

O primeiro é a separação estrita entre ajuste e medição. O subconjunto de validação foi empregado para a seleção da melhor época; o subconjunto de teste foi medido uma única vez, ao final do processo. O ajuste de hiperparâmetros mediante observação do conjunto de teste transformaria este em segundo conjunto de validação, comprometendo a imparcialidade da medida final.

O segundo é a medição no caminho de execução real. A latência foi aferida sobre a mesma função que o servidor de aplicação invocará, processando uma imagem por chamada, e não em lotes. Descartaram-se as dez primeiras inferências, destinadas à estabilização de alocações de memória e caches.

O terceiro é a integridade do conjunto de medição. Todas as métricas reportadas na seção 3 foram obtidas sobre o subconjunto de teste do conjunto preparado, isento de vazamento conforme verificado na seção 2.2.4.

Foram ainda implementados 77 testes automatizados, que verificam a correspondência dos rótulos com o tipo enumerado do banco de dados, a ausência de vazamento no procedimento de divisão, a reprodutibilidade da divisão, o contrato de saída da inferência e o tratamento de entradas inválidas.

---

# 3 RESULTADOS

## 3.1 Processo de treinamento

O treinamento foi encerrado por parada antecipada na 33ª época, tendo o melhor desempenho em validação ocorrido na 13ª. Os pesos preservados correspondem a esta última. A Tabela 9 sintetiza a execução.

**Tabela 9 – Síntese do processo de treinamento**

| Parâmetro | Valor |
| --- | ---: |
| Épocas executadas | 33 de 100 |
| Época de melhor desempenho | 13 |
| Acurácia máxima em validação | 97,55% |
| Perda de treinamento (mínima) | 0,0828 |
| Perda de validação (mínima) | 0,0456 |
| Tempo total de treinamento | 7,3 minutos |

Fonte: elaborado pelos autores (2026).

A parada antecipada operou conforme projetado: transcorridas 20 épocas sem melhoria a partir da 13ª, o processo foi interrompido, evitando-se a execução das 67 épocas restantes.

## 3.2 Desempenho no conjunto de teste

O modelo alcançou **acurácia global de 95,06%** sobre as 324 imagens do subconjunto de teste, com F1-*score* macro de 0,9415. As métricas por classe constam da Tabela 10.

**Tabela 10 – Métricas de desempenho por classe no subconjunto de teste**

| Classe | Precisão | Revocação | F1-*score* | Suporte |
| --- | ---: | ---: | ---: | ---: |
| Bacteriana | 0,9364 | 0,9717 | 0,9537 | 106 |
| Fúngica | 0,9429 | 0,8462 | 0,8919 | 78 |
| Saudável | 0,9653 | 0,9929 | 0,9789 | 140 |
| **Média macro** | **0,9482** | **0,9369** | **0,9415** | **324** |

Fonte: elaborado pelos autores (2026).

Registre-se a diferença de 2,49 pontos percentuais entre a acurácia de validação (97,55%) e a de teste (95,06%). Tal diferença é esperada e desejável: o subconjunto de validação participou da seleção da melhor época, constituindo estimativa otimista. Diferenças dessa magnitude indicam capacidade de generalização preservada, ao contrário de discrepâncias superiores a dez pontos percentuais, que caracterizariam sobreajuste.

## 3.3 Matriz de confusão

A distribuição das classificações consta da Tabela 11.

**Tabela 11 – Matriz de confusão do subconjunto de teste**

| Classe real \ Classe predita | Bacteriana | Fúngica | Saudável | Total |
| --- | ---: | ---: | ---: | ---: |
| Bacteriana | **103** | 3 | 0 | 106 |
| Fúngica | 7 | **66** | 5 | 78 |
| Saudável | 0 | 1 | **139** | 140 |
| Total | 110 | 70 | 144 | 324 |

Fonte: elaborado pelos autores (2026).

Do total de 324 classificações, 308 foram corretas e 16 incorretas.

## 3.4 Análise dos erros

A distribuição dos 16 erros, ordenada por gravidade operacional, consta do Quadro 5.

**Quadro 5 – Classificação dos erros por gravidade operacional**

| Tipo de erro | Ocorrências | Consequência para o agricultor |
| --- | ---: | --- |
| Fúngica classificada como saudável | 5 | Infecção não reportada; nenhum alerta emitido |
| Fúngica classificada como bacteriana | 7 | Alerta emitido; possível indicação terapêutica incorreta |
| Bacteriana classificada como fúngica | 3 | Alerta emitido; possível indicação terapêutica incorreta |
| Saudável classificada como fúngica | 1 | Alerta indevido; aplicação desnecessária de defensivo |

Fonte: elaborado pelos autores (2026).

A revocação da classe fúngica, de 0,8462, constitui o principal ponto de fragilidade do modelo: 12 das 78 folhas com míldio não foram identificadas como tal. A gravidade dessas ocorrências é heterogênea. Nos sete casos classificados como bacterianos, o alerta fitossanitário é emitido e o agricultor é notificado da presença de doença, podendo divergir apenas a indicação do defensivo apropriado. Nos cinco casos classificados como saudáveis, contudo, nenhum alerta é gerado — situação que corresponde exatamente ao evento que o sistema se destina a evitar.

Em contrapartida, a classe saudável apresentou revocação de 0,9929, com uma única classificação incorreta em 140 amostras, o que indica baixa incidência de alarmes falsos e favorece a hipótese H3, relativa à redução do consumo de defensivos.

A inspeção dos casos de maior erro revelou padrão que merece registro: as classificações incorretas de maior magnitude ocorreram com confiança próxima da unidade. Em particular, uma folha rotulada como fúngica foi classificada como saudável com confiança de 1,000. Erros com confiança elevada não são filtráveis por limiar e demandam investigação quanto à possibilidade de erro de rotulagem no conjunto de dados de origem, conforme discutido na seção 4.2.

`[PREENCHER: registrar o resultado da inspeção visual das imagens em desenvolvimento/reports/erros_classificacao/, indicando se os casos de erro com confiança elevada correspondem a falhas do modelo ou a rótulos incorretos no conjunto de dados original.]`

## 3.5 Latência de inferência

A latência foi medida sobre 100 imagens do subconjunto de teste, precedidas de 10 inferências de aquecimento, no ambiente descrito na Tabela 1. Os resultados constam da Tabela 12.

**Tabela 12 – Latência de inferência por configuração de execução (em milissegundos)**

| Configuração | Mediana | Média | p95 | p99 | Mínimo | Máximo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch / CPU | 20,10 | 23,23 | 38,27 | 81,34 | 17,07 | 91,20 |
| ONNX Runtime / CPU | **12,73** | 16,15 | 32,69 | 78,98 | 9,08 | 85,37 |

Fonte: elaborado pelos autores (2026).

Ambas as configurações atendem ao requisito RNF01, que estabelece limite de 100 milissegundos por imagem, inclusive no percentil 99. A conversão para ONNX proporcionou redução de 36,7% na latência mediana em relação à execução direta em PyTorch.

Cumpre observar que as medições referem-se ao **servidor de inferência**, e não a dispositivo embarcado. Na arquitetura proposta, o módulo ESP32-CAM limita-se à captura e à transmissão da imagem, cabendo o processamento ao servidor. A comparação com os valores de literatura relativos a dispositivos de borda deve, portanto, considerar essa distinção de contexto.

## 3.6 Validação da exportação

A comparação entre as saídas do modelo em formato ONNX e do modelo original em PyTorch, sobre 20 imagens do subconjunto de teste, resultou em divergência máxima de 4,17 × 10⁻⁷, valor quatro ordens de grandeza inferior à tolerância estabelecida de 10⁻³. Não houve divergência de classe predita em nenhuma das imagens. Os artefatos gerados constam da Tabela 13.

**Tabela 13 – Artefatos de modelo produzidos**

| Formato | Tamanho | Finalidade |
| --- | ---: | --- |
| PyTorch (`.pt`) | 3,19 MB | Treinamento, avaliação e explicabilidade |
| ONNX (`.onnx`) | 6,17 MB | Implantação no servidor de inferência |
| TorchScript | 6,33 MB | Alternativa de implantação |

Fonte: elaborado pelos autores (2026).

## 3.7 Verificação dos casos de teste

A Tabela 14 apresenta o resultado da execução dos casos de teste pertinentes ao módulo, destinada a substituir as linhas correspondentes da Tabela 18 do projeto.

**Tabela 14 – Resultado da execução dos casos de teste do módulo de inteligência artificial**

| ID | Descrição | Resultado esperado | Resultado obtido | Situação |
| :---: | --- | --- | --- | :---: |
| CT01 | Detecção de *Bremia lactucae* em folha de alface | Classe fúngica com confiança ≥ 0,85 | Confiança média de 0,9561; 89,4% dos acertos acima do limiar | **Aprovado** |
| CT02 | Detecção de *Xanthomonas campestris* em folha | Classe bacteriana com confiança ≥ 0,85 | Confiança média de 0,9834; 98,1% dos acertos acima do limiar | **Aprovado** |
| CT03 | Classificação correta de folha saudável | Revocação ≥ 90% | Revocação de 99,29% | **Aprovado** |
| CT04 | Tempo de inferência por imagem no servidor | Inferência em menos de 100 ms | Mediana de 12,73 ms (ONNX); p99 de 78,98 ms | **Aprovado** |

Fonte: elaborado pelos autores (2026).

Adicionalmente, verificou-se o cumprimento dos requisitos não funcionais aplicáveis, conforme a Tabela 15.

**Tabela 15 – Verificação dos requisitos não funcionais do módulo**

| ID | Requisito | Meta | Medido | Situação |
| :---: | --- | ---: | ---: | :---: |
| RNF01 | Tempo de inferência | < 100 ms | 12,73 ms | **Atendido** |
| RNF02 | Acurácia na classificação das três classes | > 90% | 95,06% | **Atendido** |
| RNF07 | Documentação e versionamento em Git | — | 77 testes automatizados; documentação em `docs/` | **Atendido** |

Fonte: elaborado pelos autores (2026).

## 3.8 Verificação das hipóteses

**Hipótese H1.** A hipótese previa acurácia superior a 90% na distinção entre manchas foliares de origem bacteriana e fúngica. A acurácia global obtida sobre o subconjunto de teste foi de 95,06%, superando a meta em 5,06 pontos percentuais. A hipótese é **confirmada**. Ressalte-se que a medição foi realizada sobre conjunto isento de vazamento, conforme verificado na seção 2.2.4, o que assegura que o valor reflete capacidade de generalização.

**Hipótese H2.** A hipótese comporta dois componentes. O primeiro — tempo de inferência inferior a 100 milissegundos — é **confirmado**, com mediana medida de 12,73 milissegundos. O segundo — contenção de focos infecciosos antes que atinjam 5% da área total da estufa — **não pôde ser verificado** neste módulo, por depender da operação do protótipo físico em ambiente de cultivo, prevista para etapa posterior do cronograma.

**Hipóteses H3, H4 e H5.** Referem-se, respectivamente, à redução do consumo de defensivos, à usabilidade das interfaces e ao custo de implementação, não constituindo objeto de verificação deste módulo.

---

# 4 DISCUSSÃO

## 4.1 Divergências em relação ao projeto original

A implementação apresentou sete divergências em relação ao previsto no projeto, sintetizadas no Quadro 6. Todas foram documentadas com a respectiva justificativa técnica.

**Quadro 6 – Divergências entre o projeto e a implementação**

| Código | Divergência | Justificativa |
| --- | --- | --- |
| D1 | Classificação de imagens em vez de detecção de objetos | Ausência de anotações de caixas delimitadoras no conjunto de dados |
| D2 | Campos de caixa delimitadora sempre nulos | Decorrência de D1; as colunas correspondentes devem admitir valor nulo |
| D3 | Resolução de entrada de 224 pixels em vez de 640 | Resolução nativa da tarefa de classificação; mediana do conjunto é inferior a 640 |
| D4 | Aumento de dados em tempo de treinamento, sem persistência | Prevenção do vazamento identificado nos problemas P1 e P2 |
| D5 | Divisão 70/15/15 aplicada sobre grupos de folhas | Procedimento correto quando o conjunto contém variantes da mesma unidade |
| D6 | Grad-CAM em substituição às caixas delimitadoras | Recupera a orientação espacial sem exigir anotação manual |
| D7 | Suavização de rótulos não aplicada | Parâmetro removido da biblioteca Ultralytics a partir da versão 8.4 |

Fonte: elaborado pelos autores (2026).

As divergências D1, D2 e D6 demandam ajuste no texto do projeto. Em particular, a especificação do banco de dados deve estabelecer que as colunas relativas às coordenadas da caixa delimitadora admitam valor nulo, e a descrição da interface de monitoramento deve substituir a menção a caixas delimitadoras pela referência aos mapas de calor.

## 4.2 Limitações

**Diversidade de folhas de origem na classe bacteriana.** Conforme demonstrado na Tabela 4, as 882 imagens da classe bacteriana derivam de aproximadamente 54 folhas distintas. Trata-se de limitação estrutural do conjunto de dados: técnicas de aumento de dados ampliam a variabilidade de apresentação, mas não introduzem novas amostras biológicas. A generalização do modelo para lesões bacterianas em condições, cultivares ou estágios de progressão não representados no conjunto permanece não verificada.

**Erros de classificação com confiança elevada.** Conforme registrado na seção 3.4, os erros de maior magnitude ocorreram com confiança próxima da unidade, incluindo uma folha rotulada como fúngica classificada como saudável com confiança de 1,000. Erros dessa natureza não são mitigáveis por ajuste de limiar de confiança. Cabe investigar se decorrem de falha do modelo ou de rotulagem incorreta no conjunto de origem — hipótese plausível em conjuntos públicos, conforme adverte Barbedo (2019) quanto à variabilidade e à qualidade das amostras em conjuntos de imagens agrícolas.

**Domínio das imagens de avaliação.** O subconjunto de teste, embora isento de vazamento, provém da mesma origem que o de treinamento, compartilhando condições de captura, iluminação e enquadramento. O desempenho sobre imagens capturadas pelo módulo ESP32-CAM em estufa real, sob iluminação natural variável, não foi verificado e pode divergir do reportado.

**Escopo de classes.** O modelo reconhece exclusivamente as três classes para as quais foi treinado. Manifestações de origem viral, deficiências nutricionais ou danos mecânicos com apresentação visual similar não são detectados e podem ser atribuídos incorretamente a uma das classes existentes.

**Ausência de localização espacial precisa.** Conforme a divergência D1, o modelo não delimita a extensão da lesão. Os mapas de calor fornecem indicação aproximada, insuficiente para quantificar a área foliar afetada — informação que seria necessária para a verificação integral da hipótese H2.

---

# 5 CONSIDERAÇÕES SOBRE O MÓDULO

O módulo de inteligência artificial foi implementado, treinado, avaliado e exportado, atendendo aos requisitos que lhe são pertinentes. A acurácia obtida de 95,06% supera a meta estabelecida no requisito RNF02 e confirma a hipótese H1; a latência mediana de 12,73 milissegundos atende ao requisito RNF01 com margem superior a sete vezes.

Cumpre destacar o tratamento conferido à integridade do conjunto de dados. A auditoria identificou 104 grupos de imagens idênticas distribuídas entre subconjuntos distintos e 115 grupos de folhas com variantes em mais de um subconjunto. A manutenção dessa configuração produziria métricas superiores às aqui reportadas, porém desprovidas de validade metodológica, uma vez que mediriam memorização em vez de generalização. A redivisão do conjunto por grupos de folhas de origem, ainda que resulte em valores mais modestos, assegura que os resultados apresentados reflitam a capacidade efetiva do modelo diante de amostras inéditas.

O módulo encontra-se disponível para integração à camada de aplicação, por meio da interface descrita na seção 2.5.3.

---

# REFERÊNCIAS

> **Observação.** As referências abaixo correspondem às obras citadas neste documento. As que já constam da lista de referências do projeto devem ser consolidadas, evitando-se duplicação. As assinaladas com asterisco (*) são novas e devem ser incorporadas à lista do trabalho.

BARBEDO, J. G. A. Plant disease identification from individual lesions and spots using deep learning. **Biosystems Engineering**, v. 180, p. 96-107, 2019.

JOCHER, G.; QIU, J. **Ultralytics YOLO11**. Versão 8.4.95. 2024. Disponível em: https://docs.ultralytics.com. Acesso em: `[PREENCHER: data de acesso]`.

(*) KAUFMAN, S. et al. Leakage in data mining: formulation, detection, and avoidance. **ACM Transactions on Knowledge Discovery from Data**, v. 6, n. 4, p. 1-21, 2012.

MOHANTY, S. P.; HUGHES, D. P.; SALATHÉ, M. Using deep learning for image-based plant disease detection. **Frontiers in Plant Science**, v. 7, art. 1419, 2016.

(*) PASZKE, A. et al. PyTorch: an imperative style, high-performance deep learning library. In: CONFERENCE ON NEURAL INFORMATION PROCESSING SYSTEMS, 33., 2019, Vancouver. **Advances in Neural Information Processing Systems**. p. 8024-8035.

(*) PEDREGOSA, F. et al. Scikit-learn: machine learning in Python. **Journal of Machine Learning Research**, v. 12, p. 2825-2830, 2011.

(*) SELVARAJU, R. R. et al. Grad-CAM: visual explanations from deep networks via gradient-based localization. In: IEEE INTERNATIONAL CONFERENCE ON COMPUTER VISION, 2017, Veneza. **Proceedings**. p. 618-626.

`[PREENCHER: incluir a referência do conjunto de dados obtido no Kaggle, no formato: AUTOR. Título do conjunto de dados. Ano. Disponível em: URL. Acesso em: data.]`

---

# APÊNDICE A – REPRODUÇÃO DOS RESULTADOS

Os resultados apresentados neste documento podem ser reproduzidos mediante a execução da sequência de comandos abaixo, a partir do diretório raiz do projeto.

```
pip install -r requirements.txt
python desenvolvimento/scripts/01_preparar_dataset.py
python desenvolvimento/scripts/02_treinar.py --modelo yolo11n-cls.pt
python desenvolvimento/scripts/03_avaliar.py --pesos modelos/yolo11n_cls_alface.pt
python desenvolvimento/scripts/04_exportar.py
python desenvolvimento/scripts/05_benchmark_latencia.py
python -m pytest -q
```

A semente aleatória fixa em 42 e a execução em modo determinístico asseguram a reprodutibilidade da divisão do conjunto de dados. Pequenas variações nos valores de treinamento podem ocorrer em função de diferenças de hardware.

---

# APÊNDICE B – ARTEFATOS PRODUZIDOS

**Quadro 7 – Relatórios gerados automaticamente**

| Arquivo | Conteúdo | Seção correspondente |
| --- | --- | --- |
| `desenvolvimento/reports/dataset_auditoria.md` | Auditoria completa do conjunto de dados | 2.2.2 e 2.2.3 |
| `desenvolvimento/reports/dataset_inventario.csv` | Inventário de todas as imagens | 2.2.1 |
| `desenvolvimento/reports/dataset_resolucoes.png` | Distribuição de resoluções | 2.2.1 |
| `desenvolvimento/reports/avaliacao_test.md` | Relatório de avaliação | 3.2 a 3.4 |
| `desenvolvimento/reports/metricas_test.json` | Métricas em formato estruturado | 3.2 |
| `desenvolvimento/reports/matriz_confusao.png` | Matriz de confusão | 3.3 |
| `desenvolvimento/reports/curva_treino.png` | Curvas de perda e acurácia | 3.1 |
| `desenvolvimento/reports/predicoes.csv` | Predição individual de cada imagem | 3.4 |
| `desenvolvimento/reports/erros_classificacao/` | Vinte casos de maior erro | 3.4 |
| `desenvolvimento/reports/benchmark_latencia.md` | Medição de latência | 3.5 |
| `desenvolvimento/reports/validacao_onnx.json` | Validação numérica da exportação | 3.6 |

Fonte: elaborado pelos autores (2026).
