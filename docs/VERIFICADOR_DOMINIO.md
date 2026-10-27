# Verificador de domínio — registro de decisões e procedência dos dados

> **Para que serve este arquivo.** Registro de trabalho da etapa "a imagem
> contém alface?" (cascata proposta na Seção 2.9 de
> `DOCUMENTACAO_IA_TREINAMENTO_RESULTADOS.md`). Guarda as decisões, a origem de
> cada dado e os contratempos, para que a redação do TCC não dependa de memória.
> É atualizado a cada passo concluído.
>
> Selos como em `CONTEXTO_IA_CONFIAVEL.md`: **[M]** medido/contado por
> comando · **[D]** documentado pela fonte externa · **[C]** lido no código ·
> **[N]** não medido.
>
> **Situação em 18/10/2026:** verificador **treinado, avaliado no `val` e
> medido no `test` uma única vez** (Seção 4.5). **Metas não atingidas**:
> doentes aceitas 181/184 (98,4%), não alface recusada 338/356 (94,9%);
> latência 53,25 ms (atingida). Alface da C270 aceita: 12/30. Pelo critério
> da Seção 2.9.6, não atende às condições de implantação.
>
> **Nova iteração em preparo (22/10):** 265 fotos de 19–21/10 revisadas (9
> para `duvidosas/`, já movidas); data de corte implementada (sessões até
> 21/10 só no treino), suíte com 205 aprovados. Aguardando sessões novas,
> depois de 21/10, para validação e teste. Código dos
> passos 1–4 testado. Fotos noturnas de 17/10 conferidas (Seção 2.5). Divisão
> da C270 passou a ser por **sessão de captura** (Seção 1, última decisão),
> para que a tarde de 17/10 vá para o treino e a noite para o teste. Suíte
> aprovada com a regra nova (199 testes); 06 reexecutado e aprovado (Seção
> 2.6). Depois, a sessão passou a poder atravessar a meia-noite (10/10–11/10
> numa sessão só): suíte aprovada (200 testes); 06 reexecutado e aprovado
> (Seção 2.7). Depois, +13 fotos da estrutura vazia (Seção 2.8); a jardineira
> vazia ficou como limitação. **Dataset final montado e aprovado** (Seção 2.9,
> 4.417 imagens). **Verificador treinado no Colab** (Seção 4.3: Tesla T4,
> melhor época 6, acurácia de validação 0,99839 — não é o critério de
> aceitação). **Avaliado no `val`** (Seção 4.4): metas atingidas no limiar
> provisório 0,50. **Limiar 0,50 e metas fixados em 18/10**, antes do teste.
> **Teste medido** (Seção 4.5): metas de acurácia não atingidas.
>
> **Retomada da v2 (22/10):** nenhuma sessão da C270 depois de 21/10 em
> `estufa/` (Seção 2.11). Aguardando as sessões novas (passo A). Contagem
> medida por comando: 19/10 tem 64 de alface e 21/10, 29 de "não alface"
> (a Seção 2.10 registrava 65 e 28).
>
> **Sessão de 24/10 à tarde:** `20261024_projeto`, com 105 fotos de alface
> da C270 (15:22:13–15:29:25) e nenhuma de "não alface" (Seção 2.12). As 105
> foram revisadas: todas são alface; 6 movidas para `duvidosas/` (borrão
> forte), restando 99 em `alface/`. A câmera foi usada na mão, num cômodo da casa,
> fora do enquadramento de operação. **Decidido em 24/10: data de corte em
> 24/10** (Seção 1). A sessão vai inteira para o treino; validação e teste
> saem só de sessões iniciadas a partir de 25/10. Padrão alterado no
> `config.py`; suíte aprovada (205 testes, 89,23 s).
>
> **Treino da v2 com as fotos existentes (decidido em 24/10):** sem sessões
> depois do corte, toda a C270 vai para o treino, e validação e teste ficam
> sem C270 — desempenho da v2 no ambiente real: [N] (Seção 1). **06
> reexecutado** (Seção 2.13): 4.772 imagens, integridade aprovada, C270 no
> treino com 350 / 166. **Passo C aplicado** (`--rotulo v2` no 08; caderno com
> `NOME` da v2); suíte aprovada (213 testes, 89,83 s). **v2 treinada no
> Colab** (Seção 4.6): Tesla T4, Ultralytics 8.4.187, melhor época 36 de 56,
> acurácia de validação 0,9984 (não é o critério de aceitação). **Avaliada
> no `val`** (Seção 4.7): 189/189 doentes aceitas e 294/294 "não alface"
> recusadas no limiar 0,50, com intervalo vazio entre 0,327 e 0,735;
> latência de 120,84 ms na primeira medição e **42,70 ms** com os outros
> programas fechados. **Sessões novas da C270 a partir de 25/10** (decisão do
> autor, Seção 1). **Sessão de 25/10 revisada** (Seção 2.14): 79 alface e 19
> "não alface"; 1 rótulo corrigido e 3 fotos movidas para `duvidosas/`. Depois,
> **+28 de "não alface" no quintal** (12:44–12:46), todas corretas, para que
> as duas classes tenham o mesmo ambiente. Sessão final: 75 alface, 48 "não
> alface", 3 duvidosas, toda no teste [C]. **06 reexecutado** (Seção 2.15):
> 4.895 imagens, integridade aprovada; teste com 753 imagens, 123 da C270;
> treino e validação inalterados, a v2 vale sem retreino. **Limiar 0,50,
> metas da Tabela 7 e critério com a C270 fixados em 25/10, antes do teste**
> (Seção 1): 183/184, 337/354, < 100 ms, 72/75 e 46/48.
>
> **Teste da v2 medido uma única vez em 25/10 13:00** (Seção 4.8): folhas
> doentes 178/184 (**não atingida**), não alface 340/354 (atingida),
> latência 51,85 ms (atingida), alface C270 **75/75** (atingida), não alface
> C270 42/48 (**não atingida**). **A v2 não atende ao critério de
> implantação.** Melhora forte na C270 em relação à v1 (12/30 → 75/75) e
> menos alertas suprimidos (3 → 1); recusa 6 folhas fúngicas em macro (3
> imagens de origem) e aceita 6 de ~10 fotos de outras plantas do quintal.
>
> **Decisão do autor (25/10, posterior ao teste): implantar a v2.** Na
> avaliação do autor, os erros não são agravantes (Seção 4.8, "Avaliação da
> gravidade dos erros"); o critério continua reportado como não atingido.
> Melhorias futuras, principalmente com outras plantas: Seção 4.9. Próximos
> passos: recusa como 200 com `classe` = `nao_alface`, documentação (passo
> G) e implantação (passo H).
>
> **Contrato da recusa (decidido em 24/10):** a imagem sem alface passará a
> responder 200 com `classe` = `nao_alface`, no lugar do 422 (Seção 1).
> **Implementado em 25/10** (Seção 4.1); suíte com 214 aprovados e contrato
> estático regenerado.
>
> **Passo G1 (25/10):** documento técnico e `gerar_documento_tcc.py` reescritos
> nas mesmas palavras (Seção 1, 25/10, estrutura). A Seção 2.9 passa a descrever
> o método de uma etapa medida, com Tabela 7 (critérios), Tabela 8 (dados) e
> Tabela 9 (treino). Os resultados da v1 e da v2 ficam numa nova Seção 3.8, e a
> decisão de implantar fica na 3.8.5, como exceção posterior ao teste, com o
> critério reportado como não atingido. As limitações e as melhorias futuras
> ficam no capítulo 4 (4.1 e 4.2). A Tabela 7 ficou sem o `[PREENCHER]`.
> Ajustes no resto do documento: C270 no lugar do ESP32-CAM, 214 testes, CT01
> do sistema completo e 4 referências novas. `gerar_documento_tcc.py`
> executado pelo autor em 25/10 [M]: `.docx` gerado em
> `desenvolvimento/tcc/Modulo_IA_TCC.docx`, 33 épocas lidas; a Figura 4
> (`figura_verificacao_alface.png`) foi conferida.
>
> **Passo G2 (25/10):** a D8 entrou em `docs/DIVERGENCIAS.md`, com o texto
> aprovado pelo autor. Ela cobre a cascata com o verificador, a recusa como 200
> `nao_alface` no lugar de bloqueio e a C270 no lugar do ESP32-CAM nas seis
> seções do TCC que citam a câmera. Traz também a ação pendente no `ENUM` do
> banco, o critério de implantação reportado como não atingido e o texto
> sugerido para as Seções 6.2.6.7 e 6.2.6.3. O resumo de impacto ganhou duas
> linhas (Tabela `deteccoes`, captura) e notas no RNF01 e no CT01.
>
> **Passo G3 (25/10):** `docs/CONTEXTO_IA_CONFIAVEL.md` atualizado. Mudanças:
> - nova Seção 5.7 com o verificador v1 e v2 (dados, treino, validação,
>   teste com IC95, cascata e erros), com selos e o critério da v2 como
>   **não atingido**;
> - o CT01 do sistema completo (80,8%) e o RNF01 com as duas etapas
>   (51,85 ms) na Seção 6;
> - na Seção 8, o contrato `nao_alface` e as variáveis do verificador;
> - na Seção 9, o verificador ainda não implantado;
> - na Seção 10, 214 testes [M];
> - na Seção 11, as limitações 1 e 3 e as pendências novas;
> - nas Seções 12 e 13, a divergência 4 e as fontes.
>
> A divergência 4 (câmera) ficou **parcialmente** resolvida: uma busca mostrou
> o ESP32-CAM ainda citado em docstrings da API, no contrato OpenAPI, em
> READMEs, no `spec.md`, no `DOCUMENTACAO_TECNICA_ABNT.md`, no `inferencia.py`
> e em testes (pendência na Seção 5).
>
> **Passo G4 (25/10):** `CLAUDE.md` atualizado. Mudanças:
> - **Regras:** a 1 inclui o `dataset_dominio_bruto/`; a 2 trata dos rótulos
>   `alface`/`nao_alface`; a 3 diz que o teste vale uma vez por versão; a 7
>   inclui o 06.
> - **Comandos:** 06, zip do Colab, 07, `08 --split val --rotulo v2` e
>   `gerar_documento_tcc.py`; o `pytest` passa a 214 testes (~89 s).
> - **Estado atual:** tabela do verificador v2 com o critério **não atingido**.
> - **Armadilhas novas:** nome `WIN_`, estrutura do bruto, sessões com pausa
>   de mais de 2 h, data de corte `20261024`, caminho duplicado e artefatos do
>   Colab, latência com a máquina carregada, PowerShell e `predicoes.csv`.
> - **Pendências:** a Function com `nao_alface`, o ENUM, o passo H e o
>   ESP32-CAM.
> - **Outras seções:** D1–D8; a frase do Git corrigida (o `.gitignore` lista
>   as pastas pelo nome), com as fotos da estufa entre os não regeneráveis.
>
> **Passo G5 (25/10):** `docs/IMPLANTACAO.md` atualizado. Mudanças:
> - topologia com a C270, deixando o envio ao Blob "a definir pelo grupo",
>   sem presumir caminho;
> - Function tratando `nao_alface` como resultado final;
> - `/health` com `verificador_carregado`;
> - variáveis `IA_ALFACE_MODELO_DOMINIO` e `IA_ALFACE_LIMIAR_DOMINIO=0.50`;
> - contrato da recusa, o ENUM pendente e o critério **não atingido**;
> - RNF01 das duas etapas em 51,85 ms local; na nuvem, não medido.
>
> **Passo G concluído.** Próximo: passo H (implantação).
>
> **Passo H, Dockerfile (25/10):**
> - `COPY modelos/yolo11n_cls_dominio_v2.pt ./modelos/`;
> - `ENV IA_ALFACE_MODELO_DOMINIO=/app/modelos/yolo11n_cls_dominio_v2.pt` e
>   `IA_ALFACE_LIMIAR_DOMINIO=0.50`;
> - exemplo de tag no cabeçalho passou a `v2`, com o aviso de nunca
>   republicar uma tag.
>
> O `.dockerignore` não exclui os `.pt`. Suíte depois da mudança: **214
> aprovados** em 84,79 s [M] (26/10), com 1 aviso de depreciação do Starlette,
> o mesmo de antes.
>
> **Build local (26/10)** [M]: `docker build -t ia-alface:local .` concluiu em
> 337,5 s (17/17 etapas). A etapa 11/12 é o `COPY` de
> `yolo11n_cls_dominio_v2.pt`, sem erro. A primeira tentativa falhou porque o
> Docker Desktop estava parado (`dockerDesktopLinuxEngine` não encontrado). O
> tamanho da imagem não aparece na saída: não medido.
>
> **Container local (26/10)** [M]: `docker run --rm -p 8000:8000
> ia-alface:local`. O `/health` devolveu `status: ok`,
> `modelo_carregado: true` (`yolo11n_cls_alface.pt`, cpu),
> **`verificador_carregado: true`** e
> `modelo_dominio_versao: yolo11n_cls_dominio_v2.pt`.
>
> **`/analyze` no container local (26/10, 22:26 em Brasília)** [M], com duas
> imagens da **validação**; o teste não foi usado:
> - **Folha fúngica** (`fungal_original_01_4…68c12973.jpg`): 200 com
>   `fungico`, confiança 1,0, `alerta: true` e `verificacao.probabilidade_alface`
>   0,999239.
>   - O 08 registrou 0,999191 para o mesmo arquivo, com o mesmo MD5, no
>     ambiente local. A diferença de 0,00005 provavelmente vem das versões de
>     torch e Ultralytics da imagem [N, não investigado].
> - **Foto do COCO** (`coco__000000011813`): 200 com `classe: nao_alface`,
>   "Alface não encontrada", confiança 1,0, `probabilidades: {}`, patógeno e
>   alerta falsos, `tempo_inferencia_ms: 0` e `verificacao` com P = 1e-6.
>   Contrato da Seção 4.1 confirmado.
> - Os tempos dessas chamadas (411 e 1.023 ms na primeira, 342 ms do
>   verificador na segunda) são de requisições isoladas, logo depois de o
>   container subir, ainda em aquecimento. **Não são medição de latência.**
>
> **Grad-CAM no container (26/10)** [M]:
> `python api/cliente.py …/val/bacteriano/augmented_120__60247afe.jpg --mapa`,
> com uma imagem quadrada (474 × 474) da validação. Resultado: `bacteriano`
> com 100,0% e alerta.
> - O mapa ficou em `mapas_de_calor/augmented_120__60247afe_gradcam.png`.
> - Aberto lado a lado com a original: as regiões quentes caem na folha com
>   manchas escuras (centro) e nas manchas marrons da folha de baixo (borda
>   inferior); o solo e as folhas verdes sem lesão ficam frios.
> - Os 2.661 ms de HTTP incluem a montagem do Grad-CAM na primeira chamada:
>   não são medição.
>
> **Correção do Grad-CAM aplicada (26/10; decisão na Seção 1)** [C]:
> - **`src/ia_alface/explicabilidade.py`:**
>   - função nova `regiao_analisada`, com a geometria do `classify_transforms`
>     (lado menor a 224, recorte central, arredondamento do `torchvision`);
>   - função nova `sobrepor_mapa`, que só colore dentro da região e deixa os
>     pixels de fora idênticos;
>   - `GradCAM.gerar` projeta o mapa só sobre a região, com zero fora;
>   - método novo `GradCAM.sobreposicao`;
>   - `salvar_sobreposicao` e o fallback `_redimensionar_e_recortar` usam a
>     mesma geometria.
> - **`api/main.py::_gerar_heatmap`:** usa `sobreposicao`. A cópia da conta de
>   sobreposição foi removida. O contrato não muda (PNG do tamanho da foto),
>   então não é preciso o `gerar_openapi`.
> - **`desenvolvimento/tests/test_explicabilidade.py`:** 7 testes novos:
>   - quatro geometrias: 1280 × 720 → (280, 0, 1000, 720); 315 × 500 →
>     (0, 93, 315, 408); 500 × 315; quadrada;
>   - dimensão inválida;
>   - conferência contra o `classify_transforms` real, com faixas amarelas
>     de 10 px nas bordas calculadas;
>   - faixas fora da região idênticas à original.
>
> Suíte: **221 aprovados** em 101,83 s [M] (26/10), com 1 aviso de depreciação
> do Starlette, o mesmo de antes. Os 7 testes novos passaram, inclusive a
> conferência contra o `classify_transforms`.
>
> **Nova build e conferência visual (26/10)** [M]:
> - Build em 7,5 s: dependências em cache, só as camadas a partir de `src/`
>   refeitas.
> - O container subiu com os dois modelos (logs do `uvicorn`).
> - **Folha quadrada:** mapa idêntico ao anterior.
> - **Folha de 315 × 500:**
>   - as faixas de cima (até y ≈ 93, com o retângulo de ruído) e de baixo
>     (a partir de y ≈ 408) aparecem **sem sobreposição**;
>   - o ponto quente subiu de y ≈ 390 para y ≈ 340, como previsto pela
>     geometria, e continua sobre a área esbranquiçada.
> - Os tempos (3.749 ms na primeira chamada com Grad-CAM e 214 ms na
>   segunda) **não são medição**.
>
> **Documentos atualizados:**
> - Seção 1.8 do documento técnico, no `.md` e no script: o mapa corresponde
>   ao recorte e é projetado só sobre ele;
> - 2.7 e Tabela 22: 221 testes;
> - `CLAUDE.md`: armadilha nova e 221 testes;
> - `CONTEXTO_IA_CONFIAVEL.md`, Seção 10: 221 testes.
>
> **Tag `v2` e push (26/10)** [M]:
> - `docker manifest inspect …:v2` devolveu `manifest unknown`: a tag não
>   existia. Um `denied` anterior era credencial vencida, resolvida com
>   `docker login` usando um token clássico.
> - `docker tag ia-alface:local ghcr.io/thi-barros/ia-alface:v2`.
> - Tamanho da imagem: **2,63 GB em disco, 614 MB de conteúdo
>   (compactado)**. A `v1` tinha 2,66 GB e 615 MB.
> - `docker push` concluído, com digest
>   `sha256:833cc600316dfe62c6d46f1d83a793367923da42cbfdea64c024d69a5b246fd6`,
>   o mesmo da build testada (Grad-CAM corrigido e verificador v2).
>
> **Container Apps atualizado para a `v2` (26/10)** [M]:
> - A primeira tentativa do `az containerapp update` foi recusada com
>   `RequestDisallowedByAzure`: o Azure passou a exigir MFA para alterar
>   recursos, e a sessão do `az` não tinha MFA. O autor resolveu o MFA e
>   repetiu a atualização.
> - `az containerapp show` confirma `imagem: ghcr.io/thi-barros/ia-alface:v2`,
>   com `IA_ALFACE_CHAVE_API` (`secretRef: chave-api`) e
>   `IA_ALFACE_CORS_ORIGENS: http://localhost:5173` mantidos.
> - As variáveis do verificador vêm da própria imagem (`ENV` no
>   `Dockerfile`).
>
> **`/health` na nuvem (26/10)** [M]: HTTP 200, `status: ok`,
> `modelo_carregado: true` (`yolo11n_cls_alface.pt`, cpu),
> **`verificador_carregado: true`** e
> `modelo_dominio_versao: yolo11n_cls_dominio_v2.pt`. **O verificador v2 está
> implantado.** Os 0,56 s da chamada não são medição: a revisão tinha acabado
> de ser criada, e não se sabe se a réplica já estava ativa.
>
> **Latência na nuvem, `v2` (26/10)** [M], com a saída colada pelo autor. O
> loop rodou no PowerShell, sem script nem relatório em `reports/`.
> - **Protocolo:** rota `/analyze/raw` com chave de API; 100 imagens sorteadas
>   de `dataset_clean/val` (`Get-Random -SetSeed 42`); 10 chamadas de
>   aquecimento descartadas. Critério igual ao do 08: duas etapas = tempo de
>   inferência do verificador + do classificador, medidos pelo servidor, sem a
>   leitura do arquivo; percentis com interpolação linear.
> - **Ambiente:** réplica de 1 vCPU e 2 GiB, pesos `.pt`. O HTTP foi medido
>   no computador do autor, com a rede de casa.
>
>   | Etapa | n | Mediana | Média | p95 | p99 | Mín. | Máx. |
>   | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
>   | Só verificador | 100 | 11,27 | 13,07 | 23,46 | 31,19 | 8,22 | 36,67 |
>   | Duas etapas | 100 | **22,62** | 26,78 | 45,15 | 74,22 | 12,43 | 99,06 |
>   | HTTP ponta a ponta (cliente) | 100 | 104,85 | 167,40 | 505,87 | 738,33 | 71,84 | 808,80 |
>
> - **Respostas:** 34 `bacteriano`, 46 `saudavel`, 19 `fungico` e 1
>   `nao_alface`, ou seja, uma folha da validação recusada. No `val`, o 08
>   registrou 325/326 de alface aceita. Não foi conferido se é a mesma imagem
>   (`h60_1`) [N].
> - **Leitura:**
>   - RNF01 atendido na nuvem: mediana das duas etapas de 22,62 ms, contra
>     100 ms; mesmo o máximo, 99,06 ms, ficou abaixo.
>   - O número local de referência é 51,85 ms (CPU Ryzen 5 3500U). É outro
>     hardware; a comparação não mede o modelo.
>   - O HTTP inclui o envio pela internet até o Brazil South, varia com a
>     conexão (p95 de 506 ms) e fica fora do RNF01.
>
> **Inicialização a frio, `v2` (26/10)** [M], rodada 1:
> - Depois de mais de 5 minutos sem chamadas,
>   `az containerapp replica list … --query "[].name" -o tsv` voltou vazio
>   (zero réplicas).
> - A primeira chamada ao `/health` levou **11,27 s** (HTTP 200), medida pelo
>   `curl` no computador do autor. Esse tempo inclui criar a réplica, subir o
>   container, carregar os dois modelos e a rede.
> - Uma única medida, sem faixa.
> - Armadilha: no PowerShell, `--query "length(@)"` chega truncado ao `az`
>   (que é um `.cmd`); usar `--query "[].name" -o tsv`, em que saída vazia
>   quer dizer zero réplicas.
>
> **Documentos atualizados com as medidas da nuvem (26/10, pedido do autor;
> o frio ficou com uma medida):**
> - **Documento técnico, no `.md` e no script:**
>   - 3.6: parágrafo novo com as duas etapas (mediana 22,62 ms, p95 45,15,
>     máx. 99,06), o verificador sozinho (11,27 ms), o HTTP do cliente
>     (104,85 / 505,87 ms) e o frio (11,27 s, uma medição);
>   - 2.10.3: remete à 3.6 e mantém o custo como não medido;
>   - 3.9: RNF01 local e na nuvem;
>   - 4.1 (o): passa a "medições em nuvem restritas".
> - **`CONTEXTO_IA_CONFIAVEL.md`:** 5.5 (tabela da nuvem e o que segue
>   [N]), 5.7, 6.1, 8 e 9 (imagem `v2`, verificador implantado, Grad-CAM) e
>   pendências.
> - **`IMPLANTACAO.md`:** §7 com a tabela da nuvem; §5 (pesos na `v2`);
>   avisos de MFA e de como consultar réplicas.
> - **D8:** RNF01 na nuvem, no texto e no resumo.
> - **`CLAUDE.md`:** estado atual, pendência e armadilhas de MFA, `--query`
>   e GHCR.
> - **`DECISOES_INFRAESTRUTURA.md`:** tamanho da `v2`, frio, imagem
>   implantada, armadilhas e medidas.
>
> **Passo H concluído.** O `.docx` foi regenerado pelo autor em 26/10, com
> 33 épocas lidas e sem erro, já com as Seções 1.8, 2.10 e 3.6 e os 221
> testes.
>
> **Seção 2.10 do documento técnico (26/10, pedido do autor):** a justificativa
> da arquitetura em contêiner entrou em `DOCUMENTACAO_IA_TREINAMENTO_RESULTADOS.md`
> e em `gerar_documento_tcc.py`, com o mesmo texto nos dois.
> - **Conteúdo:**
>   - 2.10.1: por que contêiner (reprodutibilidade, portabilidade, exigência
>     do Container Apps e API sem dependência de nuvem), com a referência
>     MERKEL (2014);
>   - 2.10.2: composição da imagem, em sete decisões, com o tamanho medido da
>     `v1` (615 MB compactada e 2,66 GB em disco) e o da `v2` não medido;
>   - 2.10.3: hospedagem com escala a zero e o Quadro 8 de alternativas;
>   - ressalvas de cold start, latência e custo na nuvem como não medidos.
> - **Figura nova:** Figura 5 (arquitetura de implantação), em
>   `figura_implantacao.png`.
> - **Correções:** a frase da 2.8 dizia que o ONNX reduz "o tamanho da
>   aplicação implantada", o que não vale para a imagem, que mantém o PyTorch
>   e roda com `.pt`. O item (o) do capítulo 4 ganhou o cold start. Também
>   saiu do script a referência "SGDR:stochastic", erro de digitação
>   introduzido no G1.
> - **Renumeração no `.md`:** as figuras 5 e 6 passaram a 6 e 7, e o Quadro 8
>   da 3.5 passou a 9.
> - Preços e cotas do Azure ficaram fora do texto, por falta de referência
>   citável. Falta o autor gerar o `.docx`.

---

## 1. Decisões

| Data | Decisão | Motivo |
| --- | --- | --- |
| 14/10 | Manter o verificador **binário em cascata** (YOLO11n-cls) da Seção 2.9; o classificador de doenças não é alterado | Preserva os resultados medidos e o `ENUM` do banco |
| 14/10 | Descartar a abordagem "só positivas" (distância no espaço de características) mesmo com poucas negativas | Recusaria justamente a alface da estufa, por ser diferente do dataset do Kaggle, suprimindo alertas |
| 14/10 | Recusa por **HTTP 422**, distinta do 400 de imagem inválida | Nenhum corpo de resposta pode ser confundido com diagnóstico; o back-end distingue os dois casos. A Function deve tratar 422 como definitivo, sem retry. **Substituída em 24/10** (última linha) |
| 14/10 | Rótulos `alface` / `nao_alface` em `classes.py`, separados das classes de doença | Regra 2 do `CLAUDE.md`; não entram no `ENUM` |
| 14/10 | Limiar provisório 0,5; o definitivo será escolhido no `val`, favorecendo a aceitação | Recusar folha doente suprime alerta; aceitar imagem alheia custa no máximo um alerta indevido |
| 14/10 | Metas da Tabela 7 podem ser ajustadas, **mas fixadas antes de medir o `test`** | Ajustar depois de ver o resultado invalida a medição |
| 14/10 | Fotos de "não alface" **não** são fabricadas escurecendo/borrando fotos de alface | Regra 6 do `CLAUDE.md` e rótulo conflitante (mesma folha nas duas classes) |
| 15/10 | Câmera do projeto: **Logitech C270** | Informado pelo autor. Resolve a divergência 4 de `CONTEXTO_IA_CONFIAVEL.md` (repositório citava ESP32-CAM) |
| 15/10 | Fotos da C270 entram em treino **e** teste e são a única evidência de desempenho real; fotos de celular só em treino | O celular não é a câmera de operação |
| 15/10 | ~500 imagens por fonte pública negativa (proposta, não contestada) | Negativas na ordem das 2.216 positivas; nenhuma fonte acima de ~1/4 das negativas |
| 17/10 | O 422 de recusa traz `codigo = "imagem_fora_do_dominio"` e `detail` em texto | O FastAPI já usa 422 para requisição malformada (com `detail` em lista); o campo `codigo` distingue as duas. **Substituída em 24/10** (última linha) |
| 17/10 | Verificador configurado e não carregado → **503** na inferência (falha fechada) | Seguir sem ele daria diagnóstico a imagens sem alface, justamente o que a etapa existe para evitar |
| 17/10 | Sem `IA_ALFACE_MODELO_DOMINIO`, a API responde exatamente como antes | Implantação gradual; o verificador só liga depois de medido |
| 17/10 | `tempo_inferencia_ms` da resposta continua sendo só o do classificador; o do verificador vai em `verificacao.tempo_inferencia_ms` | Não muda o significado de um campo existente; o tempo total é a soma |
| 17/10 | Treino do verificador separado do treino do classificador (`treinar_verificador`, `reports/treino_verificador.json`) | Não misturar o verificador no `treino_config.json` nem no comparativo de variantes |
| 17/10 | **Divisão da C270 ajustada** (aprovada pelo autor e implementada em `dados_dominio.dividir_dias_estratificado`): a ordem cronológica é aplicada **separadamente** aos dias que têm "não alface" e aos que só têm alface; cada dia continua inteiro num único split | Os dias antigos só têm alface; pela regra original, toda sessão de "não alface" cairia no teste e nunca no treino. Sozinho, o ajuste não muda o dataset atual (um único dia com "não alface" vai para o teste) |
| 17/10 | **Uma nova sessão de fotos, à noite**, com as duas classes, incluindo a **jardineira com substrato e sem planta**; ~15–20 fotos por classe, sem rajada | Com o ajuste acima, o 17/10 vai para o treino e a sessão nova para o teste: o modelo aprende o ambiente real **e** ele é medido, cobrindo "não alface" noturno e a jardineira vazia — provavelmente o caso de "não alface" mais frequente para uma câmera fixa |
| 17/10 | A sessão de 17/10 18:40 (37 fotos, só "não alface") **fica no 17/10** e vai para o treino junto com a sessão da tarde. A sessão de teste será **noutra data** (18/10 ou depois), com alface, "não alface" e a jardineira vazia. Decisão do autor | A divisão é por data (Seção 2.5); outra sessão no mesmo dia cairia no mesmo split. A regra aprovada e o código não mudam. **Substituída pela decisão seguinte** |
| 17/10 | **Divisão da C270 por sessão de captura**, no lugar do dia (decisão do autor, depois das 20 fotos de alface das 18:57–19:00): fotos do mesmo dia formam uma sessão até uma pausa de **mais de 2 h** entre fotos consecutivas; a sessão nunca atravessa a meia-noite e fica inteira num único split. A estratificação anterior continua, agora por sessão. Resultado previsto [C]: tarde de 17/10 (16:01–16:11) no treino e noite (18:40–19:00) no teste; as demais sessões não mudam. Implementado em `sessoes_de_captura`, `dividir_sessoes` e `dividir_sessoes_estratificado` (antes `dividir_dias*`); `ConfigDatasetDominio.pausa_sessao_horas` = 2,0; `fracao_dias_*` renomeados para `fracao_sessoes_*` | A noite de 17/10 já tem as duas classes no mesmo cenário e luz, que era o objetivo da sessão nova; assim o treino recebe "não alface" da C270 e o teste também o mede, sem outra sessão de fotos. **O limite de 2 h foi escolhido conhecendo os dados**: a pausa de 17/10 é de 2 h 28 min 44 s e a maior pausa dentro de outro dia é de 1 h 20 min (26/09); qualquer valor entre os dois dá o mesmo resultado. Nenhum modelo foi treinado nem medido antes da escolha. Ressalvas: tarde e noite compartilham piso, cabo e ventoinha (a câmera é fixa, como na operação); a jardineira vazia não foi fotografada |
| 17/10 | **A sessão de captura pode atravessar a meia-noite** (decisão do autor, antes do treino): sessão = fotos em sequência com pausas de até 2 h, sem corte na virada do dia. A noite de 10/10 (23:54) e a madrugada de 11/10 (até 00:49) viram uma única sessão, `20261010_2354`. Resultado previsto [C]: sessões só com alface passam de 5 para 4 — 26/09 e 28/09 no treino, 29/09 na validação, 10–11/10 no teste; 17/10 não muda | A última foto de 10/10 (validação) e a primeira de 11/10 (teste) são praticamente a mesma cena, com 2 min de intervalo (Seção 2.6). Como a validação escolhe a melhor época e o limiar, a correção tinha de vir antes do treino. Custo: 29/09 sai do treino (2 fotos) e vai para a validação |
| 22/10 | **Data de corte para a C270** (`ConfigDatasetDominio.data_corte_treino` = `"20261021"`; `--data-corte` no script 06): sessões **iniciadas até 21/10** vão inteiras para o **treino**; só as posteriores entram na divisão cronológica estratificada. Uma sessão que começa no dia do corte e atravessa a meia-noite também vai para o treino. Sem sessão depois do corte, o 06 avisa que validação e teste ficam sem fotos da C270. Decisão do autor. **Data alterada para 24/10 em 24/10** (última linha) | O teste de 18/10 consumiu as sessões de 10–11/10 e da noite de 17/10: elas não podem medir uma nova versão do verificador. Sem o corte, as sessões novas de 19–21/10 cairiam em validação e teste (Seção 2.10), e o vaso redondo, principal falha, ficaria fora do treino. Validação e teste da próxima versão saem só de sessões futuras, planejadas para medição. O teste do Kaggle (324) será reutilizado e já foi visto — limitação a declarar |
| 17/10 | **Seguir sem fotos da jardineira com substrato e sem planta**; fica como limitação declarada. As 13 fotos da estrutura vazia (19:49–19:51) entram no teste, na sessão `20261017_1840` (decisão do autor) | A jardineira vazia é o "não alface" mais parecido com a cena de alface e provavelmente o mais frequente numa câmera fixa; **o desempenho do verificador nesse caso é [N]** e assim deve constar no TCC. A estrutura vazia (piso, cabo, moldura) fica coberta no treino (tarde) e no teste (noite) |
| 18/10 | **Limiar fixado em 0,50** e **metas da Tabela 7 mantidas**: ≥ 99% de alface aceita em folhas doentes, ≥ 95% de não alface recusada, latência mediana das duas etapas < 100 ms. Decisão do autor, **antes de qualquer medição no `test`**, a partir do `val` (Seção 4.4) | No `val`, 0,50 é o menor limiar da grade com a recusa máxima de não alface (293/294) e mantém 189/189 folhas doentes. A regra escrita ("maior limiar que mantém as doentes na meta") daria 0,60, mas acima de 0,50 a recusa de não alface não melhora e só se perdem folhas doentes. 0,40 foi proposto por dar mais folga às doentes (a de menor P tem 0,512) e foi preterido pelo autor; o risco aceito é essa folga pequena, com no máximo 1 recusa admitida entre as 184 doentes do teste. 0,50 equivale a escolher a classe mais provável e já é o padrão do código (`LIMIAR_ALFACE_PADRAO`) e da API |
| 24/10 | **Data de corte passa a 24/10** (`ConfigDatasetDominio.data_corte_treino` = `"20261024"`, padrão no `config.py`): a sessão `20261024_1522` (105 fotos de alface, 15:22–15:29) vai inteira para o **treino**; validação e teste saem só de sessões iniciadas a partir de **25/10**. Uma sessão iniciada ainda em 24/10 também vai para o treino. Decisão do autor (opção A da Seção 2.12), antes de qualquer treino ou medição da v2 | A sessão foi feita com a câmera na mão, num cômodo da casa, fora do enquadramento de operação (Seção 2.12): serve para ensinar (vaso preto de dia, a principal falha da v1), não para medir. Só com alface e sem outra sessão desse tipo, iria sozinha para o teste. Preteridas: "não alface" na mesma sessão até 17:29:25 (só iria para o treino se viesse outra sessão com "não alface") e manter como está (teste com 99 fotos fora da operação). O padrão no `config.py`, e não só `--data-corte`, evita depender de lembrar a opção a cada execução do 06 |
| 24/10 | **A recusa por domínio passa a retornar HTTP 200 com `classe` = `nao_alface`**, no mesmo campo de `bacteriano`, `fungico` e `saudavel`, no lugar do 422 com `codigo = "imagem_fora_do_dominio"`. O rótulo vem de `CLASSE_NAO_ALFACE` em `classes.py` (regra 2 do `CLAUDE.md`). **Implementação antes da implantação** (passo H); até lá, o código segue com o 422 (Seção 4.1). Decisão do autor. Substitui as decisões de 14/10 (422) e 17/10 (`codigo`) | O consumidor recebe a recusa no mesmo campo da classe, e desaparece a armadilha do 422 com dois significados (domínio e requisição malformada). Consequências a resolver na implementação: (1) a coluna `classe` do banco só aceita as três doenças — o grupo do banco precisa aceitar `nao_alface`, ou o back-end não grava essas respostas; (2) hoje `patogeno_detectado` = classe diferente de `saudavel` (`api/esquemas.py`), o que marcaria `nao_alface` como patógeno: `patogeno_detectado` e `alerta` precisam ser falsos explicitamente; (3) definir `confianca`, `probabilidades` e `classe_exibicao` nesse caso; (4) todo consumidor que trate "não é `saudavel`" como doença daria alerta falso — risco que o 422 evitava |
| 25/10 | **Fazer sessões novas da C270 a partir de 25/10**, revertendo a parte da decisão de 24/10 que dispensava as sessões. As fotos vão para `estufa/<AAAAMMDD>_projeto/<alface\|nao_alface\|duvidosas>/`; a data de corte continua 24/10, então essas sessões entram na divisão cronológica estratificada. Decisão do autor, depois da avaliação da v2 no `val` e **antes de qualquer medição no teste** | Sem C270 em validação e teste, o desempenho da v2 no ambiente real ficaria [N] (Seção 1, 24/10). Avaliar a v2 nas fotos da C270 do treino mediria memorização, não desempenho. Consequência a decidir quando as sessões existirem: se alguma sessão nova cair no treino, a v2 foi treinada sem ela — retreinar, ou avaliar os pesos atuais registrando a diferença (validação e teste continuam inéditos para a v2) |
| 25/10 | **v2: limiar fixado em 0,50, metas da Tabela 7 mantidas e critério de implantação ampliado com a C270.** Para implantar, todas as condições no teste da v2: (1) alface aceita em folhas doentes (Kaggle) ≥ 99% — no mínimo **183/184**; (2) "não alface" recusada, todas as fontes ≥ 95% — no mínimo **337/354**; (3) latência mediana das duas etapas < 100 ms, medida com os outros programas fechados; (4) **alface aceita na C270 ≥ 95% — no mínimo 72/75**; (5) **"não alface" recusada na C270 ≥ 95% — no mínimo 46/48**. Comparação pela razão exata, não pela taxa arredondada. Decisão do autor, **antes de qualquer medição da v2 no `test`**, a partir do `val` (Seção 4.7) e da montagem da Seção 2.15 | 0,50: no `val`, nenhuma "não alface" acima de 0,327 e nenhuma folha doente abaixo de 0,735; 0,50 fica nesse intervalo, é o limiar da v1 e o padrão da API (`LIMIAR_ALFACE_PADRAO`). Metas (4) e (5): a Tabela 7 não mede a C270, onde a v1 falhou (12/30 de alface aceita); sem elas, a v2 poderia ser implantada sem funcionar com a câmera do projeto. As metas (4) e (5) não estão em `ConfigAvaliacaoDominio`: o relatório do 08 mostra a C270 "sem meta própria", e a comparação com 72/75 e 46/48 é feita sobre os números dele. Limitação a declarar: amostras pequenas (IC95 largo), uma única sessão, câmera na mão, mesmas plantas do treino |
| 25/10 | **Implantar a v2** (`yolo11n_cls_dominio_v2.pt`, limiar 0,50), **embora o critério de implantação fixado antes do teste não tenha sido atingido** nas condições 1 (folhas doentes 178/184; mínimo 183) e 5 ("não alface" da C270 42/48; mínimo 46). Decisão do autor, **posterior ao teste**, com base na análise dos 20 erros (Seção 4.8). O critério e os números não mudam e continuam reportados como "não atingida". As melhorias para uma próxima versão, principalmente com outras plantas, ficam na Seção 4.9 |
| 25/10 | **Estrutura da documentação do verificador no documento técnico (passo G1).** Seção 2.9 só com o método, nesta ordem: arquitetura (Figura 4 com a C270 e a resposta `nao_alface`), critérios visuais (Quadro 6), alternativas (Quadro 7), critérios de aceitação (Tabela 7, critérios numerados de (1) a (5)), dados (Tabela 8) e treino e limiar (Tabela 9). Resultados das duas versões numa nova **Seção 3.8**: validação (Tabela 18), teste (Tabela 19), cascata (Tabela 20), erros e decisão de implantar (3.8.5). As seções seguintes passam a 3.9 (casos de teste), 3.10 (hipóteses) e 3.11 (síntese); as tabelas antigas 8–17 passam a 10–17 e 21–22. O capítulo 4 passa a "Limitações e melhorias futuras", com 4.1 (limitações, itens novos i–p) e 4.2 (melhorias, resumo da Seção 4.9). O `.md` e o script têm o mesmo texto; no script, `nao_alface` e "Alface não encontrada" vêm de `classes.py`. O parágrafo da decisão de implantar foi aprovado sem alteração, em voz impessoal ("optou-se"). Decisão do autor | Os resultados do classificador já estão no capítulo 3, onde a banca procura resultados; a 3.9 (CT01) e a Tabela 22 (RNF01) precisavam dos números da cascata. A ordem 2.9.4–2.9.7 foi escolhida para que a tabela de critérios continue sendo a Tabela 7, citada na docstring de `ConfigAvaliacaoDominio` (`config.py`) | Avaliação do autor: os erros não são agravantes a ponto de impedir a implantação (justificativa detalhada na Seção 4.8, "Avaliação da gravidade dos erros"): as 6 folhas recusadas vêm de 3 imagens de microscópio ou macro, sem a forma de uma folha, e só 1 tinha alerta; as 14 "não alface" aceitas são todas plantas; as falhas que inutilizavam a v1 (alface da C270 e lente tapada) foram corrigidas. **Ressalva metodológica a declarar no TCC:** a decisão foi tomada depois de ver o resultado do teste, o que a regra de 14/10 procurava evitar; por isso ela é registrada como exceção do autor, e não como atendimento ao critério. O teste da v2 não pode ser reutilizado para avaliar mudanças |
| 26/10 | **Corrigir o alinhamento do Grad-CAM antes de publicar a imagem `v2`.** O mapa passa a ser projetado só sobre a região que a rede analisa (o recorte central); as faixas descartadas ficam sem cor. Não há retreino nem nova medição. Decisão do autor | Todas as fotos da C270 são 1280 × 720, e o modelo descarta cerca de 280 px de cada lado. Com o mapa esticado, todo mapa em operação sairia desalinhado. Corrigir antes do push evita uma `v3` só para isso |
| 24/10 | **Treinar a v2 com as fotos já existentes, sem sessões novas depois do corte.** Toda foto da C270 (sessões até 24/10) vai para o treino; validação e teste da v2 ficam **sem nenhuma foto da C270**. O roteiro de sessões a partir de 25/10 (Seções 2.11 e 2.12) não será feito. Decisão do autor ("já fiz as sessões possíveis") | Consequências a declarar no TCC: (1) o desempenho da v2 na C270 — a principal falha da v1 (12/30 de alface aceita) — fica **[N]**; nenhum número do teste da v2 mostra se ela foi corrigida; (2) o teste do Kaggle (324) e o das fontes públicas são os mesmos da v1, já vistos, com os erros analisados (Seção 4.5); (3) a validação, que escolhe a época e o limiar, também não tem C270, como na v1. As metas da Tabela 7 não incluem a C270, então a v2 pode atingi-las sem evidência do ambiente real: o critério de implantação deve ser decidido no passo E, **antes do teste** |

#### Alternativas consideradas para a sessão única de 17/10 (descartadas)

| Alternativa | Consequência |
| --- | --- |
| A. Seguir sem nova sessão, 17/10 no teste | O modelo aprende "não alface" só com conjuntos públicos; a medição no ambiente real é honesta, mas tende a sair fraca |
| B. Seguir sem nova sessão, 17/10 no treino | O modelo aprende o ambiente real (só de dia), mas a recusa de "não alface" da C270 fica **não medida** no teste |
| Ambas | Sem nenhuma foto da jardineira vazia nem de "não alface" à noite; e mesmo 57/57 acertos daria IC95 ≈ 94%–100%, com incerteza real maior por serem rajadas de ~3 min |

### 1.1 Regra de rotulagem (fotos da estufa)

Se uma pessoa reconhece a folha de alface **e** consegue avaliar se há doença,
a foto é `alface`; caso contrário, `nao_alface`. Casos ambíguos vão para
`duvidosas/` e ficam fora do dataset.

### 1.2 Regras de divisão do script 06 [C] `src/ia_alface/dados_dominio.py`

- Positivas do Kaggle: **mantêm o split** de `dataset_clean/`, para que a
  cascata possa ser medida nas mesmas 324 imagens de teste do classificador.
- Fontes públicas: split por hash do identificador do grupo (estável quando
  novas fontes entram), 70/15/15.
- Grupos: PlantVillage por folha (`leaf-map.json`); DeepWeeds por data de
  captura; PlantDoc e COCO por imagem.
- Estufa, câmera do projeto: por **sessão de captura** (desde 17/10; antes,
  por dia). Sessão = fotos em sequência, pela data e hora do nome do arquivo,
  até uma pausa de mais de `pausa_sessao_horas` = 2 h entre fotos
  consecutivas; **pode atravessar a meia-noite**; identificada por
  `AAAAMMDD_HHMM` da primeira foto. Se alguma foto de um dia não tiver hora
  no nome, esse dia inteiro vira uma sessão à parte. As sessões são divididas em ordem cronológica — as
  mais recentes no teste, as anteriores na validação, o restante no treino —,
  **separadamente** para as que têm "não alface" e as que só têm alface. Cada
  sessão fica inteira num único split. Definido antes do treino, não por
  sorteio.
- Data de corte (desde 22/10): sessões da câmera do projeto **iniciadas até
  `data_corte_treino`** vão inteiras para o treino e ficam fora da divisão
  acima. Valor: `20261021` em 22/10; **`20261024`** desde 24/10 (Seção 1).
- Estufa, celular: somente treino.

Com `fracao_sessoes_teste` = `fracao_sessoes_val` = 0,25, em cada tipo de
sessão: 5 → 3/1/1; 6 → 2/2/2; 2 → 1 treino, 1 teste; 1 → teste.

Cenário previsto [C] (testes `test_sessoes_de_captura_cenario_real` e
`test_dividir_sessoes_estratificado_cenario_real`): sessões só com alface
26/09, 28/09 → treino; 29/09 → validação; 10/10 23:54 – 11/10 00:49
(`20261010_2354`) → teste. Sessões com "não alface": tarde de 17/10
(`20261017_1601`) → **treino**; noite de 17/10 (`20261017_1840`) →
**teste**. C270 por split, previsto: treino 22 alface / 57 "não alface";
validação 2 / 0; teste 30 / 37.

### 1.3 Outras regras do script 06 [C]

- Sorteio por rodízio entre estratos (cultura no PlantVillage, classe no
  PlantDoc, data no DeepWeeds), com semente fixa; até `por_fonte` = 500.
- PlantVillage: **uma imagem por folha**; fotos sem mapeamento ou associadas a
  mais de uma folha no `leaf-map.json` são descartadas.
- Deduplicação por MD5; entre cópias, fica a da fonte prioritária (Kaggle,
  estufa, PlantVillage, PlantDoc, DeepWeeds, COCO). Imagem idêntica nas duas
  classes é erro de rótulo e reprova a montagem.
- JPEG RGB é copiado byte a byte; o resto (PNG, RGBA, cinza, CMYK) é
  reconvertido para JPEG RGB com orientação EXIF aplicada, sem redimensionar.
- Pastas da estufa fora do padrão `<AAAAMMDD>_<projeto|celular>/<alface|nao_alface|duvidosas>`
  interrompem a execução, em vez de rotular fotos errado.
- Exit 1, sem gerar `dataset_dominio/`, se: hash ou grupo em mais de um split;
  imagem nas duas classes; Kaggle fora do split original; celular no teste;
  split sem alguma das classes.

---

## 2. Fontes de dados

Todas em `desenvolvimento/dataset_dominio_bruto/` (fora do Git; ver
`.gitignore`). Contagens [M] feitas após o download, em 14–15/10/2026.

| Fonte | Classe | Imagens | Licença | Referência |
| --- | --- | ---: | --- | --- |
| Kaggle (`dataset_clean/`) | alface | 2.216 | conforme o dataset original | pendente (`[PREENCHER]` do Kaggle) |
| PlantVillage (`raw/color`) | não alface | 54.305 | **não declarada** no repositório | Hughes; Salathé (2015); Mohanty; Hughes; Salathé (2016) |
| PlantDoc | não alface | 2.578 (+1 figura de exemplo do repositório) | CC BY 4.0 [D] | Singh et al. (2020) |
| DeepWeeds | não alface | 17.509 | CC BY 4.0 [D] | Olsen et al. (2019) |
| COCO val2017 | não alface | 5.000 (3.859 após filtro) | anotações CC BY 4.0; imagens sob os termos do Flickr [D] | Lin et al. (2014) |
| Estufa `20261017_projeto/alface` (C270) | alface | 35 arquivos (18 de 26/09–11/10 + 17 de 17/10) | própria | — |
| Estufa `20261017_projeto/nao_alface` (C270) | não alface | 57 arquivos | própria | — |
| Estufa `20261015_celular/alface` | alface | 40 arquivos | própria | — |

Nenhum dos conjuntos públicos negativos contém alface: PlantVillage tem 14
culturas sem alface [D]; as classes do PlantDoc listadas no repositório não
incluem alface [D]; DeepWeeds é vegetação de pastagem australiana [D].

### 2.1 Fotos da estufa

- Em 17/10 as fotos da C270 foram reunidas numa única pasta de sessão,
  `20261017_projeto`; a data de cada foto continua vindo do nome do arquivo.
- Fotos de alface da C270 (antes em `20260930_projeto`): os nomes indicam captura em
  **26, 28 e 29/09 e 10 e 11/10** — cinco dias, não um. Há um provável
  arquivo repetido (`…363109.jpg` e `…363109 (1).jpg`), a confirmar pela
  deduplicação MD5.
- `20261015_celular`: as fotos passaram pelo WhatsApp, que **recomprime** a
  imagem e **apaga a data de captura**; os nomes registram o envio. Nas
  próximas, transferir os originais.
- Convenção de pastas: `estufa/<AAAAMMDD>_<projeto|celular>/<alface|nao_alface>/`.
  Para a câmera do projeto, a data de cada foto vem do nome do arquivo
  (`AAAAMMDDTHHMMSS-…` ou `WIN_AAAAMMDD_HH_MM_SS_Pro`, do app Câmera do Windows);
  a data da pasta só é usada quando o nome não tem data.

### 2.2 Primeira sessão de "não alface" (17/10) — conferência

- 57 fotos da C270, todas de 17/10 entre 16:01:40 e 16:04:48; várias com 1 s
  de intervalo (rajadas, quase idênticas). Conteúdo conferido por amostragem:
  estrutura vazia, mão, teclado do notebook — rótulo correto.
- **Uma única sessão**: pela regra cronológica, o dia inteiro cai no teste e
  nenhuma foto de "não alface" da estufa chega ao treino.
- **Confusão com iluminação**: todas as fotos de alface da C270 foram tiradas
  à noite (19:26–00:49, luz artificial) e todas as de "não alface" de dia
  (16:01–16:04). Nos dados da C270, horário e classe coincidem — o modelo pode
  aprender "luz do dia = não alface", e o teste não mostraria o erro.
- Falta o negativo mais difícil do enquadramento da C270: **a jardineira com
  substrato e sem planta**.
- Encaminhamento: novas sessões com **as duas classes na mesma sessão**, de dia
  e à noite, sem rajadas.
- Atualização (17/10, mesma sessão): +17 fotos de **alface de dia** (16:11:28–16:11:56,
  rajada de 28 s), conferidas por amostragem. A sessão de 17/10 passa a ter as
  duas classes sob a mesma luz, o que atenua a confusão com iluminação **no
  teste**. No treino ela continua: não há "não alface" da C270 à noite.

### 2.3 Primeira execução do script 06 — [M] `reports/dataset_dominio_auditoria.md` (17/10/2026 16:35)

Integridade **APROVADA**. 4.347 imagens; 1 duplicata exata removida (estufa).

| Split | alface | nao_alface | Total | Fotos da C270 (alface / não alface) |
| --- | ---: | ---: | ---: | --- |
| train | 1.611 | 1.400 | 3.011 | 5 / 0 |
| val | 331 | 294 | 625 | 5 / 0 |
| test | 348 | 363 | 711 | 24 / 57 |

PlantVillage: 54.305 imagens, 13.194 sem mapeamento de folha e 4.370 ambíguas
descartadas, 7.078 folhas elegíveis. DeepWeeds: 60 datas de captura.

Leitura: com uma única sessão de "não alface", **nenhuma** foto de "não alface"
da C270 chega ao treino ou à validação. O dataset está tecnicamente pronto, mas
treinar agora ensinaria muito pouco do ambiente real.

### 2.4 Filtro do COCO

Excluídas as imagens com qualquer objeto das categorias: apple, banana, bowl,
broccoli, cake, carrot, dining table, donut, hot dog, orange, pizza, potted
plant, sandwich (supercategoria "food" mais vaso, mesa e tigela) — onde pode
haver alface. Também excluídas as imagens sem nenhuma anotação.

| Item | Imagens [M] |
| --- | ---: |
| Total val2017 | 5.000 |
| Com categoria excluída | 1.093 |
| Sem anotação | 48 |
| **Utilizáveis** | **3.859** |

O filtro é aplicado pelo script 06; as 5.000 continuam no disco, inalteradas.

### 2.5 Segunda sessão de "não alface" (17/10, 18:40–18:47) — conferência

Conferida em 17/10/2026 por listagem de arquivos e leitura de 20 das 37 fotos;
o script 06 **não** foi reexecutado.

- **37 fotos** [M] de "não alface" da C270, de `WIN_20261017_18_40_40_Pro` a
  `WIN_20261017_18_47_26_Pro`, gravadas em `20261017_projeto/nao_alface/` junto
  com as 57 da tarde (total de 94). **Nenhuma foto nova de alface**: `alface/`
  continua com 35. Total na estufa: 169 arquivos [M].
- Estrutura correta: só `20261015_celular` e `20261017_projeto`, sem subpastas
  extras. Conjuntos públicos em `nao_alface/{plantvillage,plantdoc,deepweeds,coco}`,
  nenhum dentro de `estufa/`.
- Rótulo conferido nas 20 abertas, todas `nao_alface`: piso vazio com cabo,
  mão, braço, lata, banana, fita isolante, copo, chinelo, prato de papel verde
  estampado com frutas, flores artificiais em vaso, outra planta verde de
  folha estreita, disco de substrato na mão e lente tapada (quase preta). A
  planta verde, o prato verde e o disco de substrato são bons negativos
  difíceis.
- **Jardineira com substrato e sem planta: não encontrada** nas 20 abertas.
- Rajadas: 18:40:45–18:41:00 (10 fotos em 15 s); grupos de 2–3 fotos com 1–3 s
  de intervalo às 18:41:14, 18:44:14 e 18:47:21.
- Iluminação: luz artificial, no mesmo piso de cerâmica da sessão da tarde.
  Falta o autor confirmar se já estava escuro lá fora.
- **Problema: a sessão caiu no mesmo dia da sessão da tarde.** A divisão usa só
  a data `AAAAMMDD` do nome do arquivo [C] (`listar_estufa`, `dividir_dias`).
  O 17/10 continua sendo **um único dia com "não alface"** e vai inteiro para
  o teste: as 94 de "não alface" e as 17 de alface da tarde. O cenário de 1.2
  (17/10 no treino, sessão nova no teste) **não** acontece. É uma previsão
  pelo código [C]; o relatório do 06 vai confirmar.
- Confusão de cenário: a foto noturna de alface aberta (11/10) foi tirada em
  outro lugar (piso e bancada de madeira), e todas as de "não alface" estão no
  piso de cerâmica. As 17 de alface das 16:11 já estão na cerâmica, o que
  atenua o problema.
- **Atualização (17/10): +20 fotos de alface** [M], de `WIN_20261017_18_57_06_Pro`
  a `WIN_20261017_19_00_22_Pro`, em `20261017_projeto/alface/` (55 de alface no
  total). Conferidas 6: todas alface, na jardineira (18:57) e num vaso redondo
  (18:59–19:00), com algumas folhas amareladas e secas. Mesmo piso de cerâmica
  e mesma luz artificial da sessão "não alface" das 18:40: a sessão da noite
  passa a ter **as duas classes no mesmo cenário**. Rajadas com 1–2 s de
  intervalo em 18:57:30, 18:57:40, 18:57:46, 19:00:07 e 19:00:20.
- Com data de 17/10 no nome, essas fotos **também** entram no dia 17/10. Pelo
  código de então [C], o 17/10 continuaria sendo o único dia com "não alface"
  e iria inteiro para o teste (94 de "não alface" e 37 de alface). A
  jardineira vazia continua faltando.
- Encaminhamento: divisão por sessão de captura (Seção 1, última decisão).
  Previsto [C]: sessão `20261017_1601` (57 "não alface" + 17 alface) no
  treino; `20261017_1840` (37 "não alface" + 20 alface) no teste. Pausa entre
  as duas: 2 h 28 min 44 s (16:11:56 → 18:40:40).

### 2.6 Execução do script 06 com a divisão por sessão — [M] `reports/dataset_dominio_auditoria.md` (17/10/2026 19:28)

Integridade **APROVADA**. 4.404 imagens (antes 4.347; +57 = 37 "não alface" +
20 alface da noite de 17/10). 1 duplicata exata removida (estufa: a cópia
`(1)` de 28/09); as fotos `(2)` do app Câmera não eram cópias. Estufa: 189
arquivos, 7 sessões da C270.

| Split | alface | nao_alface | Total | C270 (alface / não alface) |
| --- | ---: | ---: | ---: | --- |
| train | 1.630 | 1.457 | 3.087 | 24 / 57 |
| val | 329 | 294 | 623 | 3 / 0 |
| test | 351 | 343 | 694 | 27 / 37 |

| Sessão C270 | Classe | Split | Imagens |
| --- | --- | --- | ---: |
| `20260926_1926` | alface | train | 3 |
| `20260928_2154` | alface | train | 2 |
| `20260929_2112` | alface | train | 2 |
| `20261010_2354` | alface | val | 3 |
| `20261011_0001` | alface | test | 7 |
| `20261017_1601` | alface / não alface | train | 17 / 57 |
| `20261017_1840` | alface / não alface | test | 20 / 37 |

Confere com o cenário previsto na Seção 1.2: tarde de 17/10 no treino, noite
no teste, demais sessões inalteradas. Celular: 40 fotos, só no treino.

**Fronteira de meia-noite conferida visualmente:** a última foto de 10/10
(`20261010T235923`, val) e a primeira de 11/10 (`20261011T000122`, test) são
praticamente a mesma cena — mesma jardineira, as mesmas duas plantas,
enquadramento levemente deslocado —, com 2 min de intervalo. São 3 fotos de
alface na validação quase iguais às 7 do teste. O treino usa a validação
para escolher a melhor época, e o limiar também sai dela, então a decisão
precisa vir **antes do treino**. Decidido: a sessão pode atravessar a
meia-noite (Seção 1). **Esta execução fica superada** pela da Seção 2.7.

### 2.7 Execução do script 06 com a sessão atravessando a meia-noite — [M] `reports/dataset_dominio_auditoria.md` (17/10/2026 19:45) — superada pela Seção 2.9

Integridade **APROVADA**. 4.404 imagens; 1 duplicata exata removida (estufa);
6 sessões da C270.

| Split | alface | nao_alface | Total | C270 (alface / não alface) |
| --- | ---: | ---: | ---: | --- |
| train | 1.628 | 1.457 | 3.085 | 22 / 57 |
| val | 328 | 294 | 622 | 2 / 0 |
| test | 354 | 343 | 697 | 30 / 37 |

| Sessão C270 | Dias | Classe | Split | Imagens |
| --- | --- | --- | --- | ---: |
| `20260926_1926` | 26/09 | alface | train | 3 |
| `20260928_2154` | 28/09 | alface | train | 2 |
| `20260929_2112` | 29/09 | alface | val | 2 |
| `20261010_2354` | 10/10 e 11/10 | alface | test | 3 + 7 |
| `20261017_1601` | 17/10 | alface / não alface | train | 17 / 57 |
| `20261017_1840` | 17/10 | alface / não alface | test | 20 / 37 |

Confere com o cenário previsto na Seção 1.2, inclusive as contagens da C270.
Celular: 40 fotos, só no treino. No teste: 324 do Kaggle (as mesmas do
classificador), 67 da C270 e 306 das fontes públicas.

### 2.8 Fotos da estufa vazia (17/10, 19:49–19:51) — conferência

- **13 fotos** [M] de "não alface", de `WIN_20261017_19_49_34_Pro` a
  `WIN_20261017_19_51_03_Pro`, em `20261017_projeto/nao_alface/`. Nenhuma em
  `alface/` nem em `duvidosas/`.
- Abertas 7: todas mostram a **estrutura vazia** — piso de cerâmica, cabo,
  moldura de madeira —, sem planta e sem jardineira. Rótulo correto.
- **A jardineira com substrato e sem planta continua sem nenhuma foto.**
- Pausa desde a última foto da noite (19:00:22): 49 min, menor que 2 h →
  entram na sessão `20261017_1840` (teste), sem criar sessão nova [C]. Com
  elas, a sessão vai até 19:51:03; uma foto nova entraria nela até 21:51:03.
- Rajadas: `19_49_42` e `19_49_42 (2)` praticamente iguais; 19:51:00, :01 e
  :03.
- Incluídas no dataset pela execução da Seção 2.9.

### 2.9 Execução final do script 06 — [M] `reports/dataset_dominio_auditoria.md` (17/10/2026 19:56) — **dataset vigente, usado no treino**

Integridade **APROVADA**. **4.417 imagens** (4.404 + 13 da estrutura vazia);
1 duplicata exata removida (a cópia `(1)` de 28/09; o par `19_49_42` /
`19_49_42 (2)` não era cópia exata). Estufa: 202 arquivos, 6 sessões da C270.

| Split | alface | nao_alface | Total | C270 (alface / não alface) |
| --- | ---: | ---: | ---: | --- |
| train | 1.628 | 1.457 | 3.085 | 22 / 57 |
| val | 328 | 294 | 622 | 2 / 0 |
| test | 354 | 356 | 710 | 30 / 50 |

| Sessão C270 | Dias | Classe | Split | Imagens |
| --- | --- | --- | --- | ---: |
| `20260926_1926` | 26/09 | alface | train | 3 |
| `20260928_2154` | 28/09 | alface | train | 2 |
| `20260929_2112` | 29/09 | alface | val | 2 |
| `20261010_2354` | 10/10 e 11/10 | alface | test | 3 + 7 |
| `20261017_1601` | 17/10 | alface / não alface | train | 17 / 57 |
| `20261017_1840` | 17/10 | alface / não alface | test | 20 / 50 |

Teste: 324 do Kaggle (as mesmas do classificador; 184 folhas doentes),
80 da C270 e 306 das fontes públicas. Celular: 40 fotos, só no treino.
Limitações desta montagem: jardineira vazia sem fotos; validação sem
"não alface" da C270 (Seção 5).

### 2.10 Sessões de 19 a 21/10 (para uma nova iteração) — conferência em 22/10/2026

Contagem por listagem [M]; 22 fotos abertas (amostra). O 06 **não** foi
reexecutado.

| Pasta | Classe | Fotos | Horário | Local (pela amostra) |
| --- | --- | ---: | --- | --- |
| `20261019_projeto` | alface | 67 | 23:01:18–23:05:07 | piso de porcelanato da casa, ao lado de um sofá (fora da estrutura) |
| `20261019_projeto` | nao_alface | **0** | — | — |
| `20261020_projeto` | alface | 102 | 22:22:48–22:28:25 | estrutura da estufa (madeira e cerâmica) |
| `20261020_projeto` | nao_alface | 30 | 22:35:33–22:36:38 | objetos da casa (ferro de passar, estante, mesa com brinquedos) e a estrutura vista por dentro |
| `20261021_projeto` | alface | 38 | 22:48:31–22:50:49 | piso de outro cômodo da casa |
| `20261021_projeto` | nao_alface | 28 | 22:52:08–22:53:43 | cachorro na caminha, cobertor, móvel |

Total: 265 fotos novas (207 alface, 58 não alface). **Corrigido em 22/10**
pela contagem por comando (Seção 2.11): 206 alface e 59 não alface; 19/10
tinha 66 de alface, não 67, e 21/10, 29 de não alface, não 28. Estrutura de pastas
correta, sem `duvidosas/`. As três são sessões separadas (dias diferentes),
**todas à noite**.

**Pontos positivos.** O **vaso redondo preto** aparece nas três noites, e a
jardineira em 19 e 20/10. Era a principal falha do teste. Há também fundos e
ângulos novos.

**Problemas encontrados**

1. **Rótulo:** `WIN_20261019_23_02_27_Pro` é um borrão de movimento
   irreconhecível, rotulado como `alface`. `WIN_20261020_22_28_25_Pro` está
   fortemente borrado. Pela regra 1.1 iriam para `duvidosas/`. A câmera foi
   usada na mão, então pode haver outras. **As 265 não foram revisadas.**
2. **Rajadas:** uma foto a cada ~2–3 s (102 em 5 min 37 s em 20/10). O
   volume real de variedade é bem menor que a contagem.
3. **"Não alface" não cobre as falhas do teste:** a amostra não tem jardineira
   vazia, lente tapada/imagem escura nem estrutura vazia. São objetos da casa,
   fora do enquadramento da câmera em operação. 19/10 não tem "não alface".
4. **Só noite**, e 2 das 3 sessões fora da estufa.
5. **Divisão com a regra atual** (previsão pelo código [C]):
   - só alface: 26/09, 28/09, 29/09 → treino; 10–11/10 → validação;
     **19/10 → teste**;
   - com "não alface": 17/10 tarde e 17/10 noite → treino; **20/10 →
     validação**; **21/10 → teste**.
   - Nenhuma das sessões novas iria para o treino. Os vasos redondos novos
     serviriam só para medir, não para ensinar. O treino ganharia apenas as
     sessões do teste anterior (17/10 noite, com 9 fotos do vaso, e 29/09).
6. O teste do Kaggle (324) seria reutilizado, e já foi visto.

**Revisão foto a foto (22/10, as 265 abertas)**

| Pasta | Classe | Para `duvidosas/` | Motivo |
| --- | --- | --- | --- |
| `20261019_projeto` | alface | `WIN_20261019_23_02_27_Pro`, `WIN_20261019_23_03_32_Pro` | borrão de movimento, folha irreconhecível |
| `20261020_projeto` | alface | `WIN_20261020_22_25_45_Pro`, `WIN_20261020_22_28_13_Pro`, `WIN_20261020_22_28_18_Pro`, `WIN_20261020_22_28_18_Pro (2)`, `WIN_20261020_22_28_25_Pro` | borrão forte: dá para ver que é planta, mas não para avaliar doença |
| `20261021_projeto` | alface | `WIN_20261021_22_48_59_Pro`, `WIN_20261021_22_49_12_Pro` | borrão forte |

Total: **9 fotos** (todas de alface). Motivo para `duvidosas/` e não
`nao_alface`: a C270 em operação é fixa e não produz borrão de câmera na mão,
e uma folha de alface borrada como `nao_alface` ensinaria um rótulo
conflitante.

- Limítrofes **mantidos** como alface (desfoque moderado, folha
  reconhecível): por exemplo `WIN_20261020_22_26_54_Pro`, `…_22_26_56_Pro`,
  `WIN_20261021_22_48_37_Pro`, `…_22_48_56_Pro`, `…_22_50_39_Pro`. Os
  limítrofes de 19/10 e do início de 20/10 não foram anotados por nome;
  todos ficam como alface.
- **"Não alface": os 58 rótulos estão corretos.** Em 20/10 há fotos escuras
  ou com a lente tapada pela mão (22:35:48–22:35:55) e a estrutura vista de
  perto (22:36:00), que cobrem parte das falhas do teste de 18/10. O resto
  são objetos da casa e, em 21/10, o cachorro, a parede, o chão, o teto, a
  perna e tecidos — nada do enquadramento da câmera em operação. Fotos
  borradas de "não alface" continuam corretas (não há alface nelas).
- Cobertura do vaso redondo preto: 19/10 (porcelanato, ao lado do sofá),
  20/10 (cerâmica da estrutura, vista de cima e de lado) e 21/10 (piso da
  casa).

**Encaminhamento (22/10, decisão do autor):** data de corte — sessões até
21/10 só no treino (Seção 1).

### 2.11 Retomada da v2: conferência de `estufa/` em 22/10/2026

Busca de sessões novas pela listagem de arquivos. Contagem [M] pelo comando
abaixo, rodado pelo autor na raiz do projeto em 22/10:

```powershell
Get-ChildItem desenvolvimento\dataset_dominio_bruto\estufa -Recurse -File | Group-Object { "$($_.Directory.Parent.Name)\$($_.Directory.Name)" } | Sort-Object Name | Format-Table Count, Name -AutoSize
```

- **Nenhuma sessão depois do corte.** Não há pasta `2026102[2-9]_*`, nem de
  novembro ou dezembro, e nenhum arquivo `WIN_2026102[2-9]*` ou
  `WIN_20261[12]*` em pasta nenhuma.
- Contagem por pasta [M]:

  | Pasta | alface | nao_alface | duvidosas | Total | Registrado antes (Seção 2.10) |
  | --- | ---: | ---: | ---: | ---: | --- |
  | `20261015_celular` | 40 | — | — | 40 | 40 |
  | `20261017_projeto` | 55 | 107 | — | 162 | 55 + 107 |
  | `20261019_projeto` | **64** | 0 | 2 | 66 | 65 + 2 |
  | `20261020_projeto` | 97 | 30 | 5 | 132 | 97 + 30 + 5 |
  | `20261021_projeto` | 36 | **29** | 2 | 67 | 36 + 28 + 2 |
  | **Total** | 292 | 166 | 9 | 467 | |

- **Correção da Seção 2.10:** 19/10 tem 64 fotos de alface, não 65, e 21/10
  tem 29 de "não alface", não 28. A contagem manual da listagem já apontava
  isso. Em 19–21/10 são 265 fotos: 206 de alface (197 no dataset e 9 em
  `duvidosas/`) e 59 de "não alface", e não 207/58. Essas sessões vão para o
  treino (data de corte); o plano da v2 não muda.
- C270 que entra no dataset, toda no treino, antes da deduplicação do 06:
  252 de alface (55 + 64 + 97 + 36) e 166 de "não alface".

**Roteiro proposto para o passo A** (não decidido; quantidades são
planejamento, não medição)

Como o 06 divide as sessões depois do corte [C] (`dividir_sessoes`, frações
0,25/0,25, arredondamento do Python), com todas as sessões novas tendo as
duas classes:

| Sessões novas | Treino | Validação | Teste |
| ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 1 |
| 2 | 1 | 0 | 1 |
| 3 | 1 | 1 | 1 |
| 4 | 2 | 1 | 1 |
| 5 | 3 | 1 | 1 |
| 6 | 2 | 2 | 2 |
| 7 | 3 | 2 | 2 |
| 8 | 4 | 2 | 2 |

- Mínimo de **3 sessões** para haver C270 na validação e no teste. Com 6
  (3 datas × tarde e noite), cada split recebe uma data inteira: a 1ª no
  treino, a 2ª na validação, a 3ª no teste. Com 8 (4 datas): 2 datas no
  treino, 1 na validação, 1 no teste.
- **Toda sessão precisa ter as duas classes.** Uma sessão só de alface cai
  no outro estrato e, sozinha, vai para o teste [C]
  (`dividir_sessoes_estratificado`), desfazendo o plano.
- **Tarde e noite da mesma data:** mais de 2 h entre a última foto da tarde e
  a primeira da noite; senão viram uma sessão só. Dentro de uma sessão,
  nenhuma pausa acima de 2 h.
- A hora vem do nome `WIN_…`, gravado pelo relógio do Windows: conferir data
  e hora do notebook antes de fotografar.
- C270 **fixa no enquadramento de operação**. As sessões de 19–21/10 foram
  com a câmera na mão; validação e teste precisam parecer com a operação.
  Varia-se o que está na frente da câmera, não a câmera.
- Sem rajada: mudar algo entre uma foto e outra (girar o vaso, trocar a
  planta, mover o objeto, mudar a luz). Arquivos com sufixo `(2)` indicam
  duas fotos no mesmo segundo.
- Por sessão, ~20 alface (metade jardineira, metade vaso preto) e ~20 "não
  alface": jardineira com substrato e sem planta (o caso que nunca foi
  fotografado), vaso preto vazio ou só com substrato, lente tapada ou cena
  escura, estrutura vazia, mão e objetos.
- Casos ambíguos (mão cobrindo parte da planta, planta só no canto) vão para
  `duvidosas/` (regra 1.1).

### 2.12 Primeira sessão depois do corte (24/10/2026, 15:22–15:29) — conferência e revisão foto a foto

Primeiro, listagem de arquivos pela ferramenta de busca, sem abrir as fotos;
depois, revisão foto a foto (abaixo). O 06 **não** foi reexecutado. Contagem por minuto [M] na listagem;
falta confirmá-la pelo comando da Seção 2.11.

- **105 fotos** em `20261024_projeto/alface/`, de `WIN_20261024_15_22_13_Pro`
  a `WIN_20261024_15_29_25_Pro` (7 min 12 s). Por minuto, de 15:22 a 15:29:
  15, 17, 13, 9, 8, 18, 13 e 12.
- **Nenhuma foto de "não alface"** e nenhuma em `duvidosas/`. Todos os
  arquivos seguem o padrão `WIN_…_Pro.jpg`, sem sufixo `(2)`. Não há arquivo
  solto na pasta da sessão, nem outra pasta ou foto com data entre 22/10 e
  24/10.
- **Intervalos entre fotos** (104, calculados pelos nomes): mediana de 3 s;
  63 de até 3 s; 9 de 1 s (15:22:39–40, 15:22:54–55, 15:23:13–14,
  15:25:39–40, 15:27:11–12, 15:27:19–20, 15:27:56–57, 15:29:00–01,
  15:29:18–19); maior pausa de 23 s (15:26:32 → 15:26:55). É o mesmo ritmo
  das sessões de 19–21/10 (Seção 2.10). Só a revisão foto a foto mostra se
  algo mudou entre uma foto e a seguinte [N].
- **À tarde** (15h22): a primeira sessão da C270 nesse horário desde a tarde
  de 17/10. Luz do dia, conferida na revisão abaixo.
- **Volume:** 105 fotos de alface numa sessão de 7 min. O roteiro da Seção
  2.11 previa ~20 por sessão.
- **Divisão prevista pelo código [C], com o corte de 21/10** (superada pela
  decisão de 24/10, abaixo; `sessoes_de_captura`,
  `dividir_sessoes_estratificado`):
  - forma uma única sessão, `20261024_1522`, porque nenhuma pausa passa de
    2 h;
  - começa depois do corte (`"20261024" > "20261021"`) e entra na divisão;
  - **só com alface**, cai no estrato das sessões só de alface. Por ser a
    única, vai inteira para o **teste** (1 sessão → teste). A validação
    ficaria sem nenhuma foto da C270 e o teste, sem "não alface" da C270;
  - o 06 **não avisaria**: o aviso só aparece quando não há nenhuma sessão
    depois do corte (`preparar_dominio`). A conferência do passo B é
    manual.
- **Prazo para "não alface" nesta sessão [C]:** uma sessão só termina numa
  pausa **maior** que 2 h (`momento - anterior > pausa`). A primeira foto de
  "não alface" tem de ser tirada até **17:29:25 de 24/10**; depois, cada foto
  até 2 h depois da anterior. Uma sessão à noite no mesmo dia precisa começar
  **mais de 2 h depois da última foto da tarde**, contando as de "não
  alface".

**Revisão foto a foto (24/10)**

Regra 1.1. Cada foto problemática ou limítrofe é anotada por nome no momento
em que é vista. Nomes abreviados: `…15_22_35` = `WIN_20261024_15_22_35_Pro`.

| Minuto | Fotos | Cena | Rótulo | Anotações por nome |
| --- | ---: | --- | --- | --- |
| 15:22 | 15 | Vaso redondo preto com alface crespa; piso claro de cerâmica de um cômodo da casa (parede amarela e rodapé em `…15_22_50`); luz do dia, com reflexo de janela no piso. **Câmera na mão**: cada foto tem um ângulo (de cima, oblíqua, rente ao chão, de perto) | 15/15 alface, corretos | Quase iguais (1 s): `…15_22_39` / `…15_22_40` e `…15_22_54` / `…15_22_55`. Limítrofe **mantida**: `…15_22_35` (planta pequena no quadro, objeto escuro na borda esquerda). Recipiente retangular verde-azulado com substrato no canto inferior direito de `…15_22_13`, `…15_22_35`, `…15_22_40`, `…15_22_54`, `…15_22_55` e `…15_22_58`; cabo preto na borda esquerda de `…15_22_54` e `…15_22_58` |
| 15:23 | 17 | A mesma cena (vaso preto, piso da casa, luz do dia, câmera na mão); várias fotos rente ao chão | 17/17 alface, corretos | **Proposta para `duvidosas/`: `…15_23_55`** (borrão de movimento forte na imagem toda; a planta é reconhecível, mas a textura da folha não dá para avaliar doença). Limítrofes **mantidas** (desfoque moderado, nervuras visíveis): `…15_23_16`, `…15_23_40`, `…15_23_43`, `…15_23_47`. Pés ou sandálias na borda esquerda de `…15_23_07` e `…15_23_19`; cabo e objeto claro no canto inferior direito de `…15_23_55`. Em `…15_23_07` o vaso aparece inclinado, e o recipiente verde-azulado mostra substrato e a ponta de uma planta |
| 15:24 | 13 | A mesma cena; ao fundo, armário de madeira e parede com interruptor (`…15_24_00`, `…15_24_04`, `…15_24_22`) e pé de móvel de madeira (`…15_24_39` a `…15_24_45`). Folhas secas e marrons na borda do vaso | 13/13 alface, corretos | **Proposta para `duvidosas/`: `…15_24_48`** (borrão de movimento forte, bordas duplicadas, sem textura da folha). Limítrofe **mantida**: `…15_24_00` (desfoque moderado). Sequências quase iguais: `…15_24_11` / `…15_24_13`; `…15_24_32` / `…15_24_34`; `…15_24_39` / `…15_24_42` / `…15_24_45`. Cabo preto em `…15_24_11` e `…15_24_13` |
| 15:25 | 9 | A mesma cena; pé de móvel de madeira e recipiente verde-azulado ao fundo em `…15_25_47` e `…15_25_51` | 9/9 alface, corretos | **Proposta para `duvidosas/`: `…15_25_08`** (borrão de movimento forte: vê-se uma planta verde no vaso, sem nenhuma textura da folha). Limítrofe **mantida**: `…15_25_44` (desfoque moderado). Quase iguais: `…15_25_39` / `…15_25_40` (1 s) e `…15_25_15` / `…15_25_25` |
| 15:26 | 8 | Vaso preto visto rente ao chão, com pés de cadeiras e de mesa de madeira e parede amarela ao fundo (`…15_26_03` a `…15_26_23`); de cima em `…15_26_32`. **`…15_26_55`: primeira foto com a jardineira** (retangular, verde-azulada, com várias alfaces, junto à parede) ao lado do vaso preto. O recipiente verde-azulado dos cantos das fotos anteriores é essa jardineira | 8/8 alface, corretos | Nenhuma problemática. Folhas secas pendendo da borda do vaso em `…15_26_03` a `…15_26_23` |
| 15:27 | 18 | Vaso preto **e** jardineira juntos, junto à parede (`…15_27_02` a `…15_27_27`); depois **só a jardineira**, de cima e oblíqua (`…15_27_40` a `…15_27_57`), com ~4 pés de alface | 18/18 alface, corretos | Nenhuma para `duvidosas/`. Limítrofes **mantidas** (desfoque moderado): `…15_27_27`, `…15_27_49`, `…15_27_53`. Sequências quase iguais: `…15_27_02` a `…15_27_12` (5 fotos da mesma vista geral); `…15_27_19` / `…15_27_20`; `…15_27_40` / `…15_27_42`; `…15_27_56` / `…15_27_57`. Folhas com manchas amarelo-amarronzadas na jardineira em `…15_27_45` e `…15_27_47` (doença ou senescência: não avaliado) |
| 15:28 | 13 | Jardineira de cima e rente ao chão, com a parede ao fundo (`…15_28_28` a `…15_28_36`); vista geral do alto com vaso e jardineira em `…15_28_21` e `…15_28_25` | 13/13 alface, corretos | **Proposta para `duvidosas/`: `…15_28_44`** (borrão de movimento forte, folhas sem textura). Limítrofes **mantidas**: `…15_28_21` (vista geral do alto; vaso e jardineira pequenos no quadro, mas reconhecíveis), `…15_28_36` e `…15_28_49` (desfoque moderado). Quase iguais: `…15_28_28` / `…15_28_30`; `…15_28_34` / `…15_28_36`; `…15_28_51` / `…15_28_53` / `…15_28_55`. Calçado na borda inferior de `…15_28_49` e `…15_28_53` |
| 15:29 | 12 | Jardineira de cima (`…15_29_00`, `…15_29_01`, `…15_29_23`, `…15_29_25`) e rente ao chão, com cadeiras, armário e parede com interruptor ao fundo (`…15_29_06` a `…15_29_21`) | 12/12 alface, corretos | **Propostas para `duvidosas/`: `…15_29_04`** (borrão forte) **e `…15_29_19`** (borrão muito forte: só uma massa verde sobre a jardineira). Limítrofes **mantidas** (desfoque moderado): `…15_29_06`, `…15_29_09`, `…15_29_11`, `…15_29_15`, `…15_29_23`. Quase iguais: `…15_29_00` / `…15_29_01` (1 s; pé de chinelo no canto inferior esquerdo). Folhas esbranquiçadas ou amareladas na base em `…15_29_00`, `…15_29_15`, `…15_29_18` e `…15_29_23` (senescência ou doença: não avaliado) |

**Resultado da revisão (as 105 abertas)**

- **Rótulos:** as 105 mostram alface; nenhuma vai para `nao_alface`.
- **Propostas para `duvidosas/` (6), todas por borrão de movimento forte:**
  `WIN_20261024_15_23_55_Pro`, `WIN_20261024_15_24_48_Pro`,
  `WIN_20261024_15_25_08_Pro`, `WIN_20261024_15_28_44_Pro`,
  `WIN_20261024_15_29_04_Pro` e `WIN_20261024_15_29_19_Pro`. Critério de
  22/10 (Seção 2.10): dá para ver que é planta, mas não para avaliar doença.
  **Aprovadas pelo autor em 24/10** e movidas por ele (só o autor mexe em
  `dataset_dominio_bruto/`). O autor informou 6 fotos em
  `20261024_projeto/duvidosas/`; a saída da contagem não foi registrada.
- **Limítrofes mantidas (18):** 16 por desfoque moderado e 2 por plantas
  pequenas no quadro (`…15_22_35`, `…15_28_21`). Os nomes estão na tabela.
- **Arranjos:** vaso preto como assunto em 61 fotos (15:22:13–15:26:32);
  vaso e jardineira juntos em 13; só a jardineira em 31. Sem as 6
  propostas: 58, 13 e 28 (99 no total).
- **Câmera e local:** câmera **na mão**, cada foto de um ângulo (de cima,
  oblíqua, rente ao chão, de perto). Um cômodo da casa (piso claro de
  cerâmica, parede amarela, mesa, cadeiras e armário de madeira), com luz do
  dia. **Nenhuma foto está no enquadramento de operação** nem na estrutura da
  estufa.
- **Variedade:** as mesmas plantas (1 no vaso, ~4 na jardineira), no mesmo
  cômodo e com a mesma luz, durante 7 min.

**Consequência para a divisão [C]**

O roteiro da Seção 2.11 pede que validação e teste pareçam com a operação,
com a câmera fixa. Esta sessão não atende a isso. Como está — só alface e
única sessão desse tipo depois do corte —, ela iria inteira para o teste.
Opções consideradas; **decidido em 24/10: opção A** (Seção 1):

| Opção | O que acontece [C] |
| --- | --- |
| A. Data de corte em 24/10 (`--data-corte 20261024` no 06, ou o padrão de `ConfigDatasetDominio`) | A sessão vai inteira para o treino. Validação e teste saem só de sessões iniciadas a partir de 25/10; uma sessão à noite de hoje também iria para o treino |
| B. Fotos de "não alface" nesta sessão, com a primeira até 17:29:25 de 24/10 | A sessão passa ao estrato com "não alface". Por ser a mais antiga, vai para o treino se vier pelo menos mais uma sessão com "não alface" depois dela; se ficar sozinha, vai para o teste |
| C. Manter como está | O teste da v2 recebe 99 fotos de câmera na mão, fora da operação |

O padrão de `ConfigDatasetDominio` pode mudar sem quebrar a suíte:
`test_pipeline_com_data_corte_padrao` só exige que as sessões da miniatura
(até 17/10) fiquem no treino [C].

Implementado em 24/10: padrão `"20261024"` em `ConfigDatasetDominio`
(`config.py`), com o motivo na docstring; suíte com 205 aprovados [M]
(89,23 s). O roteiro
da Seção 2.11 passa a valer para as sessões iniciadas a partir de 25/10.
Depois, decidiu-se não fazer essas sessões (Seção 1, 24/10).

### 2.13 Execução do 06 para a v2 — [M] `reports/dataset_dominio_auditoria.md` (24/10/2026 16:44) — **dataset da v2**

Antes da execução, os relatórios da v1 foram copiados para
`reports/dataset_dominio_auditoria_v1.md` (17/10 19:56) e
`reports/dataset_dominio_inventario_v1.csv`. O 06 apagou o
`dataset_dominio/` da v1; o inventário da v1 lista cada imagem com o split.

Integridade **APROVADA**. **4.772 imagens**; 1 duplicata exata removida na
estufa (na v1, era a cópia `(1)` de 28/09; a sessão de 28/09 continua com 2
fotos). Estufa: 557 arquivos fora de `duvidosas/`, 10 sessões da C270. Aviso
esperado: nenhuma sessão depois do corte de 24/10, então validação e teste
ficam sem C270.

| Split | alface | nao_alface | Total | Kaggle | Estufa | Fontes públicas | C270 (alface / não alface) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| train | 1.956 | 1.566 | 3.522 | 1.566 | 556 | 1.400 | 350 / 166 |
| val | 326 | 294 | 620 | 326 | 0 | 294 | 0 / 0 |
| test | 324 | 306 | 630 | 324 | 0 | 306 | 0 / 0 |

Sessões da C270, todas no treino (alface / não alface): `20260926_1926`
3 / 0; `20260928_2154` 2 / 0; `20260929_2112` 2 / 0; `20261010_2354`
10 / 0; `20261017_1601` 17 / 57; `20261017_1840` 20 / 50; `20261019_2301`
64 / 0; `20261020_2222` 97 / 30; `20261021_2248` 36 / 29;
`20261024_1522` 99 / 0. Celular: 40, só no treino.

**Comparação com o dataset da v1 (Seção 2.9)**

- Treino: 3.085 → 3.522. A C270 passa de 22 / 57 para 350 / 166, com o vaso
  preto (19, 20, 21 e 24/10) e a "não alface" noturna (17/10, 20 e 21/10).
- Validação: 622 → 620. Saem as 2 fotos da C270 de 29/09, agora no treino.
- Teste: 710 → 630. Saem as 80 fotos da C270 (30 / 50). Ficam 324 do Kaggle
  e 306 das fontes públicas (COCO 67, PlantVillage 73, DeepWeeds 82,
  PlantDoc 84): as mesmas contagens do teste da v1 (Seção 4.5), como prevê
  a divisão por hash [C]. É o mesmo teste já visto.

### 2.14 Sessão de 25/10/2026 (12:20–12:36) — conferência e revisão foto a foto

Listagem de arquivos pela ferramenta de busca em 25/10; o 06 **não** foi
reexecutado. Contagem [M] na listagem.

- **98 fotos** em `20261025_projeto`: **79 de alface** (`WIN_20261025_12_20_41_Pro`
  a `…12_32_54_Pro`) e **19 de "não alface"** (`…12_35_00_Pro` a
  `…12_36_09_Pro`). Nenhuma em `duvidosas/`. Nenhuma outra pasta ou foto
  com data posterior a 24/10.
- Alface por minuto: 12:20 3, 12:21 10, 12:22 15, 12:23 2, 12:28 5, 12:29 17,
  12:30 4, 12:31 9, 12:32 14. "Não alface": 12:35 15, 12:36 4.
- **Uma única sessão** [C], `20261025_1220`: a maior pausa é de 5 min 33 s
  (12:23:05 → 12:28:38). Começa depois do corte de 24/10 e é a única nessa
  condição; tem as duas classes, então cai no estrato com "não alface" e,
  sozinha, vai inteira para o **teste** (Seção 1.2).
- Ao meio-dia, com luz do dia.
- Duas fotos no mesmo segundo: `…12_35_39_Pro` e `…12_35_39_Pro (2)`.
  Intervalos de 1 s em 12:29:25–26, 12:32:29–30, 12:32:35–37,
  12:35:24–25 e 12:36:08–09.

**Revisão foto a foto (25/10)**

Regra 1.1. Cada foto problemática ou limítrofe é anotada por nome no momento
em que é vista. Nomes abreviados: `…12_20_41` = `WIN_20261025_12_20_41_Pro`.

| Minuto | Classe | Fotos | Cena | Rótulo | Anotações por nome |
| --- | --- | ---: | --- | --- | --- |
| 12:20–12:21 | alface | 13 | Vaso preto e jardineira verde com alfaces, num **piso externo de pedra** (bege e marrom), molhado, com gotas no vaso (`…12_21_56`) e muro escuro com musgo ao fundo; luz do dia. **Câmera na mão**: o ângulo muda a cada foto (de cima, oblíqua, girada, rente ao chão) | 13/13 alface | **Proposta para `duvidosas/`: `…12_21_36`** (quase todo o quadro é piso; a planta aparece só cortada na borda direita — regra 1.1, "planta só no canto"). Cabo preto em `…12_20_41`. Folhas com pontos escuros pequenos em `…12_21_53` |
| 12:22–12:23 | alface | 17 | Mesma cena: vaso preto de cima e de lado, jardineira de cima e de lado; muro com musgo e mangueira verde ao fundo (`…12_22_57`, `…12_23_00`). Câmera na mão | 17/17 alface | Nenhuma problemática. Quase iguais: `…12_22_57` / `…12_23_00` (vaso de lado, mangueira ao fundo); `…12_22_26` / `…12_22_37` / `…12_22_53` / `…12_23_05` (vaso de cima, ângulos próximos) |
| 12:28 | alface | 5 | Depois de uma pausa de 5 min: vaso e jardineira vistos de mais alto, plantas menores no quadro, piso molhado. Câmera na mão | 5/5 alface | **Proposta para `duvidosas/`: `…12_28_54`** (planta cortada no canto superior esquerdo; o resto é piso e a mangueira). Limítrofes **mantidas** (planta pequena no quadro, mas inteira e reconhecível): `…12_28_38`, `…12_28_46` |
| 12:29–12:30 | alface | 21 | Vaso e jardineira de cima, de lado e em vistas gerais do quintal (piso externo, muro, outros vasos ao fundo). Câmera na mão | 21/21 alface | Nenhuma problemática. Outra planta de folha larga, num vaso claro, ao fundo de `…12_30_01` (o assunto continua sendo a alface). Quase iguais: `…12_29_25` / `…12_29_26` (1 s, jardineira de lado); `…12_29_34` / `…12_29_40`; `…12_28_41` / `…12_30_54` (mesma vista geral) |
| 12:31 | alface | 9 | Vaso e jardineira de cima; costela-de-adão e outra planta em vaso na borda esquerda (`…12_31_14`, `…12_31_17`); mangueira e muro (`…12_31_22`, `…12_31_24`). Câmera na mão | 7/9 alface | **Rótulo errado: `…12_31_05`** — só piso, um suporte de arame e a quina de um recipiente; **nenhuma alface**. Proposta: mover para `nao_alface/` (piso vazio). **Proposta para `duvidosas/`: `…12_31_24`** (mangueira e muro ocupam o quadro; a alface aparece só cortada na borda inferior). Costela-de-adão no quadro de `…12_31_14` e `…12_31_17`: as fotos continuam `alface` (a alface é o assunto), mas são casos difíceis |
| 12:32 | alface | 14 | Jardineira e vaso de cima e de lado, de perto e de longe. Câmera na mão | 14/14 alface | Nenhuma problemática. Limítrofe **mantida**: `…12_32_37` (vaso inteiro no canto direito, jardineira cortada no alto). Quase iguais: `…12_32_27` / `…12_32_29`; `…12_32_30` / `…12_32_33` / `…12_32_35`; `…12_32_48` / `…12_32_52` / `…12_32_54` |
| 12:35–12:36 | não alface | 19 | **Dentro de casa** (sala), não no quintal das fotos de alface: pé com chinelo ou crocs sobre tapete (`…12_35_00`, `…12_35_08`, `…12_35_39`), pufe e poltrona (`…12_35_04`), mão em silhueta contra a janela (`…12_35_17`), estante e mesinha (`…12_35_37`, `…12_36_08`), controle remoto no sofá (`…12_35_46`, `…12_35_48`), sofá de perto (`…12_35_52`), TV desligada com reflexo da sala (`…12_35_57`), tapete e tela do notebook (`…12_36_00`), teclado e mão (`…12_36_04`), piso borrado com cabo (`…12_36_09`). Câmera na mão | 19/19 não alface, corretos | **Lente tapada ou cena escura: `…12_35_24`, `…12_35_25` (quase preta), `…12_35_28`, `…12_35_30`**; borrão claro em `…12_35_39_Pro (2)` (diferente de `…12_35_39`, não é cópia). **Negativo difícil:** bambu-da-sorte (planta verde de folha estreita) em `…12_35_37`. Nenhuma foto de jardineira vazia, vaso vazio ou estrutura |

**Fotos de "não alface" no quintal (25/10, 12:44:44–12:46:20)** — tiradas
depois da revisão acima, para atenuar o problema 1. Mudanças anteriores
conferidas na listagem: as 4 fotos saíram de `alface/`; `duvidosas/` tem as
3 propostas e `nao_alface/` recebeu `…12_31_05`.

| Minuto | Classe | Fotos | Cena | Rótulo | Anotações por nome |
| --- | --- | ---: | --- | --- | --- |
| 12:44–12:45:08 | não alface | 7 | **Quintal**, mesmo piso de pedra das fotos de alface: pilar, rodo e grade (`…12_44_44`); vasos de barro com rosas-do-deserto e outras plantas junto ao muro (`…12_44_49`, `…12_44_51`); estante com vasos variados (`…12_44_54`); vaso com planta de folhas largas e onduladas (`…12_44_58`); varal com prendedores e armário (`…12_45_04`); piso vazio (`…12_45_08`) | 7/7 não alface | **Negativos difíceis:** muitas plantas verdes de outras espécies (`…12_44_49` a `…12_44_58`); em `…12_44_58` as folhas são largas e verdes, as mais parecidas com alface até aqui. Nenhuma alface visível nessas fotos (conferido abrindo cada uma) |
| 12:45:11–12:45:31 | não alface | 7 | Quintal: pata-de-elefante e babosa em vasos (`…12_45_11`, `…12_45_14`); piso vazio (`…12_45_17`); vassoura verde e pá junto ao muro com musgo (`…12_45_23`); torneira, galão e ralo (`…12_45_25`); samambaia em vaso suspenso (`…12_45_28`); violetas e aspargo-pluma num vaso sobre tambor azul (`…12_45_31`) | 7/7 não alface | **Negativos difíceis:** `…12_45_31` (folhas arredondadas, verdes e enrugadas das violetas, em roseta), `…12_45_23` (cerdas verde-claras da vassoura). Os demais têm folhas estreitas ou recortadas |
| 12:45:33–12:46:07 | não alface | 7 | Quintal: piso vazio (`…12_45_33`, `…12_45_46`); jiboia no muro e mangueira (`…12_45_37`); mangueira enrolada (`…12_45_41`); saco plástico no muro (`…12_45_43`); lírio-da-paz e antúrio em vasos (`…12_45_45`); dedo tapando a maior parte da lente, com o quintal visível à esquerda (`…12_46_07`) | 7/7 não alface | **Negativo difícil:** `…12_45_45` (folhas largas, verde-escuras e brilhantes do lírio-da-paz). **Lente tapada:** `…12_46_07` (parcial) |
| 12:46:09–12:46:20 | não alface | 7 | Quintal: dedo tapando a lente (`…12_46_09` quase toda; `…12_46_11`, `…12_46_12`, `…12_46_14` parcial, com o piso visível); pé com chinelo no piso (`…12_46_16`, `…12_46_16_Pro (2)`); céu nublado com a silhueta do telhado (`…12_46_20`) | 7/7 não alface | **Lente tapada:** `…12_46_09`, `…12_46_11`, `…12_46_12`, `…12_46_14`. Quase iguais: `…12_46_16` / `…12_46_16_Pro (2)` (mesmo segundo, quase a mesma cena; MD5 só no 06) |

**Resultado das fotos do quintal:** 28/28 rótulos corretos, nenhuma para
`duvidosas/`. Cobrem piso vazio (4), lente tapada no quintal (5), mangueira,
muro, vassoura, torneira e ralo, pé e outras plantas (rosa-do-deserto,
babosa, pata-de-elefante, samambaia, jiboia, violeta, lírio-da-paz, antúrio).
**O problema 1 fica atenuado**: o quintal passa a ter as duas classes, na
mesma luz e no mesmo piso. Continuam valendo os problemas 2 a 4 (câmera na
mão, sem jardineira ou vaso vazios, mesmas plantas do treino).

**Sessão de 25/10 depois das mudanças** (listagem [M]): `alface/` 75,
`nao_alface/` 48 (19 da sala + 1 corrigida + 28 do quintal), `duvidosas/` 3;
total 126. Uma única sessão [C], `20261025_1220`, de 12:20:41 a 12:46:20
(maior pausa: 8 min 35 s, 12:36:09 → 12:44:44). Pela regra do 06 [C], vai
inteira para o **teste**.

### 2.15 Execução do 06 com a sessão de 25/10 — [M] `reports/dataset_dominio_auditoria.md` (25/10/2026 12:55) — **dataset vigente da v2**

Integridade **APROVADA**, **nenhum aviso**. **4.895 imagens** (4.772 da
Seção 2.13 + 123 da sessão de 25/10); 1 duplicata exata removida na estufa,
a mesma de antes. Estufa: 680 arquivos fora de `duvidosas/`, 7 pastas, 11
sessões da C270.

| Split | alface | nao_alface | Total | Kaggle | Estufa | Fontes públicas | C270 (alface / não alface) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| train | 1.956 | 1.566 | 3.522 | 1.566 | 556 | 1.400 | 350 / 166 |
| val | 326 | 294 | 620 | 326 | 0 | 294 | 0 / 0 |
| test | 399 | 354 | 753 | 324 | 123 | 306 | **75 / 48** |

- Sessão `20261025_1220`: 75 alface e 48 "não alface", toda no **teste**,
  como previsto [C]. As demais sessões continuam no treino.
- **Treino e validação idênticos aos da Seção 2.13** (mesmas contagens por
  fonte e por classe). A v2 foi treinada exatamente nesse treino e
  avaliada nessa validação: **não precisa ser retreinada**, e a avaliação
  no `val` da Seção 4.7 continua válida.
- Teste: 324 do Kaggle (184 folhas doentes) e 306 das fontes públicas, os
  mesmos da v1 e já vistos; mais as 123 fotos inéditas da C270.

- **Rótulo errado (1):** `WIN_20261025_12_31_05_Pro`, em `alface/`, não tem
  alface (piso, suporte de arame, quina de um recipiente). Proposta: mover
  para `nao_alface/`.
- **Propostas para `duvidosas/` (3), por planta só no canto ou cortada
  (regra 1.1):** `WIN_20261025_12_21_36_Pro`, `WIN_20261025_12_28_54_Pro`,
  `WIN_20261025_12_31_24_Pro`. Nenhuma por borrão forte.
- **Limítrofes mantidas (4):** `…12_28_38`, `…12_28_46`, `…12_32_37` (planta
  pequena ou parcial no quadro), e as fotos com costela-de-adão
  (`…12_31_14`, `…12_31_17`), anotadas na tabela.
- **Depois das mudanças, se aprovadas:** 75 de alface, 20 de "não alface" e
  3 em `duvidosas/`.
- **Arranjos (alface):** vaso preto e jardineira, sozinhos e juntos, de cima,
  de lado e em vistas gerais. **As mesmas plantas** das sessões de 19, 20,
  21 e 24/10, todas no treino.

**Problemas para a medição**

1. **Cena diferente por classe.** Toda a alface foi fotografada no quintal
   (piso externo de pedra, molhado, muro com musgo, mangueira) e toda a
   "não alface" dentro de casa (sala, tapete, sofá, móveis). No teste, o
   modelo poderia acertar as duas classes reconhecendo o ambiente, e não a
   alface. É a mesma confusão registrada em 17/10 (Seção 2.2), agora com o
   local no lugar do horário.
2. **Câmera na mão**, fora do enquadramento de operação, como em 24/10.
3. **"Não alface" sem os casos do enquadramento real** (jardineira com
   substrato e sem planta, vaso vazio, estrutura); a lente tapada aparece
   em 4 fotos.
4. **Mesmas plantas do treino**, com outro fundo e outra luz. A divisão é
   por sessão, então isso não é vazamento pela regra do 06, mas o teste mede
   plantas já vistas — declarar.

**Prazo para completar a sessão [C]:** fotos de "não alface" **no quintal**
(piso, mangueira, muro, vaso ou jardineira vazios, a costela-de-adão), com a
primeira até **14:36:09 de 25/10** (2 h depois de `…12_36_09`), entram na
mesma sessão `20261025_1220` e reduzem o problema 1.

---

## 3. Procedimento de obtenção

Comandos executados a partir da raiz do projeto, em 14–15/10/2026.

**PlantVillage** — clone parcial, só as fotos coloridas e o agrupamento por
folha (o repositório completo tem ~2,1 GB):

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/spMohanty/PlantVillage-Dataset desenvolvimento/dataset_dominio_bruto/nao_alface/plantvillage
git -C desenvolvimento/dataset_dominio_bruto/nao_alface/plantvillage sparse-checkout set raw/color leaf_grouping
```

**PlantDoc** — o `git clone` **falha no Windows**: 87 dos 2.581 arquivos têm
nomes copiados de URLs com `?`, proibido no NTFS. Solução: baixar o `.zip`
gerado pelo GitHub e extrair com `desenvolvimento/scripts/extrair_plantdoc.py`,
que troca caracteres proibidos por `_`, encurta nomes acima de 80 caracteres
(limite de 260 do caminho no Windows) e desambigua 6 pares de nomes que só
diferem em maiúsculas (imagens diferentes, que o NTFS trataria como o mesmo
arquivo). **183 arquivos renomeados**, com o par original → salvo em
`nao_alface/plantdoc/_renomeados.csv`. O conteúdo das imagens não é alterado.

```bash
curl.exe -L --fail -o plantdoc.zip "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip"
python -I desenvolvimento/scripts/extrair_plantdoc.py plantdoc.zip desenvolvimento/dataset_dominio_bruto/nao_alface/plantdoc
```

**DeepWeeds** — `.zip` do Zenodo (DOI 10.5281/zenodo.7939060). MD5 conferido:
`b7b30f96d466fba86016aa5a26606e0f`, idêntico ao publicado.

```bash
curl.exe -L --fail -o deepweeds_images.zip "https://zenodo.org/records/7939060/files/images.zip?download=1"
python -I -m zipfile -e deepweeds_images.zip desenvolvimento/dataset_dominio_bruto/nao_alface/deepweeds
```

**COCO val2017** — o endereço oficial `images.cocodataset.org` **falha em
HTTPS** (certificado não confere com o nome). Em vez de desligar a verificação,
o mesmo arquivo foi baixado pelo endereço direto da Amazon S3, com certificado
válido; os tamanhos são idênticos aos do endereço oficial (777,8 MB e
241,2 MB). CRC de todos os arquivos internos conferido. Das anotações, só
`instances_val2017.json` foi extraído.

```bash
curl.exe -L --fail -o coco_val2017.zip "https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip"
curl.exe -L --fail -o coco_annotations_trainval2017.zip "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"
```

---

## 4. Estado da implementação

| Passo | Situação | Arquivos |
| --- | --- | --- |
| 1. Rótulos e `VerificadorDominio` | Concluído; suíte de testes passou (a linha com o total não constou da saída registrada) | `src/ia_alface/classes.py`, `inferencia.py` (funções compartilhadas), `verificacao.py`; `desenvolvimento/tests/test_classes.py` (+3), `test_verificacao.py` (14) |
| 2. Script `06_preparar_dataset_dominio.py` | **Executado com dados reais em 17/10** (integridade aprovada; ver 2.3). Em 17/10, divisão da C270 alterada para sessão de captura: biblioteca, relatório (coluna `sessao` no inventário e tabela da C270 por split na Seção 4) e testes (+5); suíte aprovada [M]; o 06 ainda não rodou com a regra nova. Em 24/10, data de corte padrão `20261024` (Seção 1) | `src/ia_alface/config.py` (caminhos e `ConfigDatasetDominio`), `src/ia_alface/dados_dominio.py`, `desenvolvimento/scripts/06_preparar_dataset_dominio.py`, `desenvolvimento/tests/test_dados_dominio.py` (35 testes); gera `desenvolvimento/reports/dataset_dominio_auditoria.md` e `dataset_dominio_inventario.csv` |
| 3. Cascata na biblioteca | Concluído; testes aprovados na suíte (214 em 25/10). Corrigido em 25/10: a linha dizia "testes não executados" | `src/ia_alface/cascata.py` (`AnalisadorCascata`, `ImagemForaDoDominio`, `ResultadoCascata`); `desenvolvimento/tests/test_cascata.py` (8 testes) |
| 4. API | Concluído; desde 25/10, a recusa é 200 com `nao_alface` (Seção 4.1), e o contrato estático foi regenerado em 25/10 (4 rotas, 9 esquemas). Corrigido em 25/10: a linha dizia "testes não executados" e "422" | `api/configuracao.py` (`IA_ALFACE_MODELO_DOMINIO`, `IA_ALFACE_LIMIAR_DOMINIO`), `api/esquemas.py` (`InfoVerificacao`, campos novos em `/health` e `/info`; `RespostaForaDoDominio` removido em 25/10), `api/main.py` (cascata, `_montar_resposta_sem_alface`, 503); `desenvolvimento/tests/test_api.py` |
| 5. Treino | **Executado no Colab em 17/10** (Seção 4.3); pesos em `modelos/yolo11n_cls_dominio.pt` | `src/ia_alface/treino.py` (`treinar_verificador`), `desenvolvimento/scripts/07_treinar_verificador.py`, `desenvolvimento/notebooks/treino_verificador_colab.ipynb` (local, não versionado), `preparar_zip_colab.py --origem desenvolvimento/dataset_dominio` |
| Suíte completa | **214 aprovados** em 25/10 (88,75 s) [M], com a recusa como 200 `nao_alface`. Antes: **213 aprovados** em 24/10 (89,83 s) [M], com o passo C (`--rotulo`, +8 testes). Antes: **205 aprovados** em 24/10 (89,23 s) [M], com a data de corte de 24/10; 1 aviso de depreciação do Starlette, o mesmo de antes. Antes: 205 em 22/10 (97,22 s), com a data de corte de 21/10. Antes: **200 aprovados** em 17/10 (85,84 s), com a sessão atravessando a meia-noite; 1 aviso de depreciação do Starlette (`httpx`), sem relação com o verificador. Antes: 199 (90,35 s) com a divisão por sessão; 194 (73,3 s) antes dela | — |
| 6–7. Avaliação (`val` e `test`) | **Executada em 18/10**: `val` (Seção 4.4) e `test`, medição única (Seção 4.5); metas de acurácia não atingidas no teste | `src/ia_alface/avaliacao_dominio.py` (IC95 de Wilson, métricas por grupo, curva de limiar, cascata sobre as predições oficiais), `desenvolvimento/scripts/08_avaliar_verificador.py`, `desenvolvimento/tests/test_avaliacao_dominio.py` (14 testes); metas em `ConfigAvaliacaoDominio` (`config.py`) |
| 8. Documentação (passo G) | G1 concluído em 25/10 (Seção 2.9, nova 3.8 e capítulo 4), com o `.docx` gerado pelo autor. G2 concluído em 25/10 (D8). G3 concluído em 25/10 (`CONTEXTO_IA_CONFIAVEL.md`). G4 concluído em 25/10 (`CLAUDE.md`). G5 concluído em 25/10 (`IMPLANTACAO.md`). **Passo G concluído** | `docs/DOCUMENTACAO_IA_TREINAMENTO_RESULTADOS.md`, `desenvolvimento/scripts/gerar_documento_tcc.py`, `docs/DIVERGENCIAS.md`, `docs/CONTEXTO_IA_CONFIAVEL.md`, `CLAUDE.md` |
| 9. Implantação (passo H) | **Concluída em 26/10.** Imagem `v2` no Container Apps, `/health` com `verificador_carregado: true`, latência e frio medidos na nuvem | `Dockerfile`, `src/ia_alface/explicabilidade.py`, `api/main.py` (Grad-CAM) |

### 4.3 Treino do verificador (passo E) — [M] log do Colab colado pelo autor, 17/10/2026 ~23:07

Dataset: Seção 2.9 (4.417 imagens), enviado como `dataset_dominio.zip`
(178,1 MB; menor lado reduzido a 320 px por `preparar_zip_colab.py`).
Contagens conferidas no Colab: train 3.085, val 622, test 710.

**Ambiente**

| Item | Valor |
| --- | --- |
| GPU | **Tesla T4** (15.360 MiB no `nvidia-smi`; driver 580.82.07, CUDA 13.0) |
| torch / Ultralytics / Python | 2.11.0+cu130 / **8.4.181** / 3.13.15 |
| Modelo base | `yolo11n-cls.pt` (ImageNet), cabeça trocada para 2 classes; 234/236 tensores transferidos |
| Parâmetros | 1.533.666 (1.528.586 após fusão das camadas) |

O registro da GPU vale **só para o verificador**. A GPU do treino do
classificador continua não registrada [N] (`CONTEXTO_IA_CONFIAVEL.md`, 4.1).

**Hiperparâmetros efetivos** (linha `engine/trainer` do log): `epochs` 100,
`patience` 20, `batch` 32, `nbs` 64, `imgsz` 224, AdamW (`lr0` 0,001, `lrf`
0,01, `momentum` 0,937), `cos_lr`, `warmup_epochs` 3, `weight_decay` 0,0005,
`dropout` 0,1, `seed` 42, `deterministic`, `amp`, `workers` 2, `split` val.
Aumento de dados em memória: `auto_augment` randaugment, `fliplr` 0,5,
`flipud` 0,3, `erasing` 0,2, `scale` 0,4. `degrees`, `translate` e `hsv_*`
foram passados, mas não atuam na tarefa `classify` com RandAugment
(`CONTEXTO_IA_CONFIAVEL.md`, 3.2). São os mesmos valores do classificador.

**Execução**

| Item | Valor |
| --- | --- |
| Épocas executadas | 26 de 100 (parada antecipada, `patience` 20) |
| Melhor época | **6** — acurácia top-1 de validação **0,99839** (621/622) |
| Tempo | 0,130 h (≈ 7,8 min); 465,9 s acumulados na época 26 |
| Oscilação | época 19: 0,947; épocas 20–26 entre 0,981 e 0,995 |

A validação final do `best.pt` usou o split `val` (622 imagens, top-1
0,998). **O split `test` não foi avaliado no Colab**: aparece no log só na
listagem das pastas.

**Leitura.** A acurácia de validação **não** é o critério de aceitação. Com
1 erro em 622, a escolha da melhor época é instável (época 6, logo no
início). A validação tem só 2 fotos da C270 e nenhuma de "não alface" da
C270. O que decide é o passo F: aceitação de alface em folhas doentes e
recusa de "não alface", por fonte e por câmera, com IC95.

Artefatos: `best.pt` baixado como `Downloads\best (1).pt` (contém a classe
`nao_alface`; o `Downloads\best.pt` é o do classificador) e
`artefatos_yolo11n_cls_dominio.zip`. No Colab o Ultralytics gravou em
`runs/classify/runs/classify/yolo11n_cls_dominio` (caminho duplicado pela
versão 8.4.181).

### 4.4 Avaliação no `val` (passo F) — [M] `reports/verificador_val.md` (18/10/2026 17:15)

Os pesos treinados no Ultralytics 8.4.181 carregaram sem erro no 8.4.95
local. CPU local (Ryzen 5 3500U), pesos `.pt` nas duas etapas. **Não é o
resultado oficial**: serve para escolher o limiar.

**No limiar provisório 0,50**

| Critério | Medido | Meta |
| --- | --- | ---: |
| Alface aceita — folhas doentes (Kaggle) | 189/189 (100,0%; IC95 98,0%–100,0%) | ≥ 99% |
| Não alface recusada — todas | 293/294 (99,7%; IC95 98,1%–99,9%) | ≥ 95% |
| Latência mediana das duas etapas (CPU local) | 49,63 ms (p95 139,72; só verificador 24,80) | < 100 ms |
| Alface aceita — todas | 327/328 (99,7%) | — |
| Alface aceita — C270 | 2/2 (IC95 34,2%–100,0%) | — |
| Não alface recusada — C270 | não medido (0 imagens no `val`) | — |

Por grupo: bacteriano 114/114, fúngico 75/75, saudável 136/137; COCO
66/66, DeepWeeds 102/102, PlantDoc 61/62, PlantVillage 64/64.

**Casos perto da fronteira** (`reports/predicoes_verificador_val.csv`):

- Alface com menor P(alface): saudável 0,429 (recusada a 0,50); bacteriano
  0,512 (`augmented_241`); saudáveis 0,536, 0,547, 0,556, 0,594; bacteriano
  0,622; fúngico 0,630. As folhas doentes de menor P são, na maioria,
  arquivos `augmented_*` do dataset original.
- Não alface com maior P(alface): PlantDoc, batata com requeima, **0,946**
  (aceita em qualquer limiar até 0,94); PlantDoc, pimentão, 0,470; PlantDoc,
  tomate com vírus, 0,373; COCO 0,283; PlantDoc 0,231.

**Curva (trecho)**

| Limiar | Doentes aceitas | Alface aceita (todas) | Não alface recusada |
| ---: | ---: | ---: | ---: |
| 0,30 | 100,0% | 100,0% | 99,0% |
| 0,40 | 100,0% | 100,0% | 99,3% |
| 0,45 | 100,0% | 99,7% | 99,3% |
| 0,50 | 100,0% | 99,7% | 99,7% |
| 0,55 | 99,5% | 98,8% | 99,7% |
| 0,60 | 99,5% | 98,2% | 99,7% |
| 0,65 | 98,4% | 97,6% | 99,7% |

**Leitura para a escolha do limiar.** Pela regra escrita ("o maior limiar
que mantém as doentes na meta"), o limiar seria **0,60**, com 188/189 =
99,47%. Mas acima de 0,50 a recusa de "não alface" não melhora: fica em
293/294 até 0,94. Subir o limiar só perde folhas doentes, e 0,60 já recusa
uma no `val`. No teste, com 184 folhas doentes, a meta de 99% admite no
máximo 1 recusa. **Decidido em 18/10: limiar 0,50, metas da Tabela 7
mantidas** (Seção 1).

### 4.5 Teste (passo G) — **medição única** — [M] `reports/verificador_test.md` (18/10/2026 17:33)

Limiar 0,50 e metas fixados antes (Seção 1, 18/10). CPU local, pesos `.pt`.
`reports/verificador_test.json` agora existe: o script recusa uma segunda
medição. **Nenhum ajuste pode ser feito com base nestes números.**

**Metas (Tabela 7)**

| Critério | Medido | Meta | Situação |
| --- | --- | ---: | --- |
| Alface aceita — folhas doentes (Kaggle) | 181/184 (98,4%; IC95 95,3%–99,4%) | ≥ 99% | **não atingida** |
| Não alface recusada — todas | 338/356 (94,9%; IC95 92,2%–96,8%) | ≥ 95% | **não atingida** (faltou 1 imagem: 339/356 atingiria) |
| Latência mediana das duas etapas (CPU local, imagens do `val`) | 53,25 ms (p95 324,16; só verificador 26,19) | < 100 ms | atingida |

**Ambiente real (C270), sem meta própria**

| Grupo | Medido |
| --- | --- |
| Alface aceita — C270 | **12/30 (40,0%; IC95 24,6%–57,7%)** |
| Não alface recusada — C270 | 45/50 (90,0%; IC95 78,6%–95,7%) |

**Por grupo:** bacteriano 105/106, fúngico 76/78, saudável 139/140; COCO
67/67, PlantVillage 73/73, DeepWeeds 81/82, **PlantDoc 72/84 (85,7%)**.

**Cascata sobre o teste oficial do classificador (324 imagens)**

| Classe | Recusadas pelo verificador | Revocação só classificador | Revocação com a cascata |
| --- | ---: | --- | --- |
| bacteriano | 1 | 103/106 (97,2%) | 102/106 (96,2%) |
| fungico | 2 | 66/78 (84,6%) | **64/78 (82,0%)** |
| saudavel | 1 | 139/140 (99,3%) | 138/140 (98,6%) |

Alertas: 167 → 164; **3 alertas verdadeiros suprimidos**. Acurácia: 308/324
(95,1%) só classificador → 304/324 (93,8%) com a cascata.

**Análise dos erros** (`reports/predicoes_verificador_test.csv`; inspeção,
sem alterar nada):

- Folhas doentes recusadas: as 3 são arquivos `augmented_*` do dataset
  original — bacteriano `augmented_489` (P = 0,410), fúngico `augmented_3`
  (0,346) e `augmented_44` (0,250). Todas tinham alerta do classificador.
- **Alface da C270, 18 recusadas de 30:**
  - sessão `20261010_2354`: 7/10 aceitas; as 3 recusadas (11/10,
    00:33–00:49) mostram a planta **num vaso redondo preto**, sobre o piso de
    madeira;
  - sessão `20261017_1840`: 5/20 aceitas. **As 9 fotos do vaso redondo
    (18:59–19:00) foram todas recusadas** (P de 0,005 a 0,246); das 11 da
    jardineira (18:57), 5 aceitas e 6 recusadas, inclusive fotos quase iguais
    a outras aceitas (18:57:30 com P = 0,858 aceita; 18:57:48 com
    P = 0,038 recusada).
  - **Contagem conferida abrindo as 30 fotos** (18/10):

    | Arranjo | Fotos | Aceitas | Recusadas | Taxa de recusa (IC95, calculado) |
    | --- | ---: | ---: | ---: | --- |
    | Vaso redondo preto (11/10 00:33–00:49 e 17/10 18:59–19:00) | 12 | 0 | 12 | **100% (75,8%–100%)** |
    | Jardineira, piso de madeira (10–11/10) | 7 | 7 | 0 | 0% |
    | Jardineira, cerâmica, noite (17/10 18:57) | 11 | 5 | 6 | 54,5% |
    | **Jardineira, total** | 18 | 12 | 6 | 33,3% (16,3%–56,3%) |

  - Treino conferido por amostragem: nas 8 fotos de alface da C270 abertas
    (26/09, 28/09 e 17/10 16:11) só há jardineira. Em 2 fotos do celular a
    jardineira é o assunto e um vaso redondo aparece apenas na borda.
    **Hipótese [N]:** o modelo não generaliza para um arranjo (vaso redondo
    como assunto) ausente do treino. A instabilidade dentro da jardineira
    indica pouca robustez no domínio da C270.
- **Não alface da C270 aceita (5 de 50):** 4 fotos com a lente tapada ou
  quase preta (18:41:28, 18:47:21, :23, :26; P de 0,86 a 0,97) e 1 da
  estrutura vazia (19:50:48; 0,795). **Hipótese [N]:** confusão com
  iluminação: no treino da C270 as fotos escuras/noturnas eram de alface, e
  as de "não alface" eram da tarde.
- **PlantDoc aceita (12 de 84):** folhas largas de outras culturas com
  lesões (batata com requeima, abóbora com oídio, tomate com septoriose ou
  mofo, pimentão), parecidas com alface doente; P de 0,53 a 0,97.
- Observação pós-teste, **não usada para escolher nada**: a folha doente
  recusada mais perto do limiar tem P = 0,410. No limiar 0,40, que foi
  proposto e preterido, ela seria aceita, e o resultado seria 182/184 =
  98,9%, ainda abaixo da meta. A escolha entre 0,40 e 0,50 não mudou o
  resultado dessa meta.

**Consequências**

1. Pelo critério da Seção 2.9.6 do documento técnico ("a implantação da
   etapa estará condicionada ao atendimento dos critérios da Tabela 7"), o
   verificador **não atende às condições de implantação**. A API sem
   `IA_ALFACE_MODELO_DOMINIO` continua respondendo como antes. Decisão do
   autor pendente (Seção 5).
2. O resultado mais grave para a operação é **12/30 de alface aceita na
   C270**: em uso real, a maioria das fotos de alface seria recusada e os
   alertas, suprimidos. A IC95 é larga, mas até o limite superior (57,7%)
   fica longe de qualquer meta aceitável.
3. Um verificador retreinado (mais fotos da C270, vaso redondo, lente
   tapada, jardineira vazia) **precisa de um novo conjunto de teste**, com
   sessões da C270 em datas novas e nunca vistas. O teste atual já foi
   consumido para esta pergunta e não pode servir para escolher nem avaliar
   a próxima versão.

### 4.6 Treino da v2 (passo D) — [M] log do Colab colado pelo autor, 24/10/2026 ~20:45

**Pacote para o Colab** [M] (saída de `preparar_zip_colab.py`, 24/10/2026
17:14): `dataset_dominio_v2.zip`, 4.772 imagens do dataset da Seção 2.13,
menor lado reduzido a 320 px (qualidade 92); 585,2 MB → **192,4 MB** em
212,1 s. Pasta interna `dataset_dominio/`, como o caderno espera. O script
avisou que o arquivo passa de 100 MB e recomendou o envio pelo Google Drive.
A opção Drive do caderno passou a procurar `dataset_dominio_v2.zip`, para não
pegar por engano o zip da v1.

Uma primeira tentativa foi interrompida por um reinício acidental do ambiente,
antes do fim da primeira época (a pasta da execução ficou sem `results.csv`).
O treino foi refeito do início no mesmo ambiente; os números abaixo são dessa
segunda execução. Contagens conferidas na célula 4: train 3.522 (1.956 /
1.566), val 620 (326 / 294), test 630 (324 / 306).

**Ambiente**

| Item | Valor |
| --- | --- |
| GPU | **Tesla T4** (15.360 MiB no `nvidia-smi`; driver 580.82.07, CUDA 13.0) |
| torch / Ultralytics / Python | 2.11.0+cu130 / **8.4.187** / 3.13.16 |
| Modelo base | `yolo11n-cls.pt` (ImageNet), cabeça trocada para 2 classes; 234/236 tensores transferidos |
| Parâmetros | 1.533.666 (1.528.586 após fusão das camadas) |

O Ultralytics passou de 8.4.181 (v1) para **8.4.187**, porque o caderno instala
a versão mais nova a partir de 8.4.95. A versão local continua 8.4.95: o
carregamento dos pesos precisa ser conferido de novo no 08.

**Hiperparâmetros efetivos** (linha `engine/trainer`): **os mesmos da v1**
(Seção 4.3) — `epochs` 100, `patience` 20, `batch` 32, `nbs` 64, `imgsz`
224, AdamW (`lr0` 0,001, `lrf` 0,01, `momentum` 0,937), `cos_lr`,
`warmup_epochs` 3, `weight_decay` 0,0005, `dropout` 0,1, `seed` 42,
`deterministic`, `amp`, `workers` 2, `split` val; `auto_augment`
randaugment, `fliplr` 0,5, `flipud` 0,3, `erasing` 0,2, `scale` 0,4
(`degrees`, `translate` e `hsv_*` passados, sem efeito na tarefa
`classify` com RandAugment).

**Execução**

| Item | Valor |
| --- | --- |
| Épocas executadas | 56 de 100 (parada antecipada, `patience` 20) |
| Melhor época | **36** — acurácia top-1 de validação **0,9984** (99,84% na célula 6; 619/620, calculado) |
| Tempo | 0,328 h (≈ 19,7 min); ~19–20 s por época de treino |
| Oscilação | épocas 1–19 entre 0,973 e 0,992; 0,997 na época 20; 0,998 (arredondado no log) também nas épocas 45 e 47, sem substituir a 36 |

Validação final do `best.pt` no `val` (620 imagens): top-1 0,998. **O split
`test` não foi avaliado no Colab**: aparece no log só na listagem das pastas.
O Ultralytics 8.4.187 também gravou em
`runs/classify/runs/classify/yolo11n_cls_dominio_v2`; o caderno usou a pasta
real (`modelo.trainer.save_dir`) sem erro.

**Leitura.** Como na v1, a acurácia de validação **não** é o critério de
aceitação, e com 1 erro em 620 a escolha da época é instável (três épocas
empatadas no log). A validação não tem nenhuma foto da C270 (decisão de
24/10). O que decide é o passo E (`08 --split val --rotulo v2`).

**Pesos:** baixados como `Downloads\best (2).pt`. Conferido pela busca de
texto no arquivo: contém `8.4.187`, a classe `nao_alface` e o nome
`yolo11n_cls_dominio_v2` (o `best (1).pt` é o da v1 e o `best.pt`, o do
classificador). Destino: `modelos/yolo11n_cls_dominio_v2.pt`.

**Artefatos da execução não guardados** (informado pelo autor em 25/10): o
`artefatos_yolo11n_cls_dominio_v2.zip` não foi baixado, e o ambiente do Colab
já foi encerrado. Perderam-se `results.csv`, `args.yaml`, os gráficos e o
`last.pt`. **O registro do treino da v2 é o log colado** (esta seção): a
linha `engine/trainer` traz os argumentos, e o log, a acurácia de validação
por época. Diferente da v1, não existe
`desenvolvimento/runs/classify/yolo11n_cls_dominio_v2/`. Não houve novo treino
para recuperá-los: os pesos deixariam de corresponder ao log.

### 4.7 Avaliação da v2 no `val` (passo E) — [M] `reports/verificador_v2_val.md` (25/10/2026 10:21)

`08 --split val --rotulo v2`. Os pesos treinados no Ultralytics 8.4.187
carregaram sem erro no 8.4.95 local. CPU local, pesos `.pt`. **Não é o
resultado oficial**: serve para escolher o limiar. A validação não tem fotos
da C270 (Seção 2.13).

**No limiar provisório 0,50, comparado com a v1 (Seção 4.4)**

| Critério | v2 | v1 | Meta |
| --- | --- | --- | ---: |
| Alface aceita — folhas doentes (Kaggle) | 189/189 (100,0%; IC95 98,0%–100,0%) | 189/189 | ≥ 99% |
| Não alface recusada — todas | **294/294** (100,0%; IC95 98,7%–100,0%) | 293/294 | ≥ 95% |
| Latência mediana das duas etapas (CPU local) | **120,84 ms** (p95 368,09; só verificador 64,26) | 49,63 ms (só verificador 24,80) | < 100 ms |
| Alface aceita — todas | 325/326 | 327/328 | — |
| C270 | não medido (0 imagens) | 2/2 alface; "não alface" não medido | — |

Por grupo: bacteriano 114/114, fúngico 75/75, saudável 136/137; COCO 66/66,
DeepWeeds 102/102, PlantDoc 62/62, PlantVillage 64/64.

**Casos perto da fronteira** (`reports/predicoes_verificador_v2_val.csv`):

- Alface com menor P(alface): saudável `h60_1` **0,149** (a única recusada
  a 0,50); fúngico `wl04_2` 0,735; saudável `h83` 0,735; fúngico `wl04_1`
  0,844; fúngico `fungal_original_06_0` 0,884; saudável
  `healthy_original_h12_1` 0,889; bacteriano `augmented_180` 0,898;
  fúngico `augmented_18` 0,899. **Folha doente de menor P: 0,735.**
- Não alface com maior P(alface): COCO `000000549136` **0,327**; PlantDoc,
  pimentão com mancha, 0,307; PlantDoc, uva com podridão-negra, 0,220; COCO
  `000000003553` 0,170; PlantDoc, pimentão, 0,101; PlantDoc, macieira com
  sarna, 0,095. Na v1, a maior era 0,946 (PlantDoc, batata com requeima).
- **Intervalo vazio entre as classes**: nenhuma imagem de "não alface" acima
  de 0,327 e nenhuma folha doente abaixo de 0,735. Qualquer limiar entre
  0,35 e 0,70 dá 189/189 e 294/294 no `val` (curva abaixo).

**Curva (trecho)**

| Limiar | Doentes aceitas | Alface aceita (todas) | Não alface recusada |
| ---: | ---: | ---: | ---: |
| 0,30 | 100,0% | 99,7% | 99,3% |
| 0,35 | 100,0% | 99,7% | 100,0% |
| 0,50 | 100,0% | 99,7% | 100,0% |
| 0,70 | 100,0% | 99,7% | 100,0% |
| 0,75 | 99,5% | 99,1% | 100,0% |
| 0,80 | 99,5% | 99,1% | 100,0% |
| 0,85 | 98,9% | 98,8% | 100,0% |

**Latência acima da meta.** O verificador da v2 tem a mesma arquitetura e o
mesmo número de parâmetros da v1 (1.528.586 após fusão), e o classificador é o
mesmo. Mesmo assim, as duas partes ficaram mais lentas do que em 18/10: o
verificador passou de 24,80 para 64,26 ms, e o classificador, estimado pela
diferença, de ~25 para ~57 ms. **Hipótese [N]:** condição da máquina durante
a medição (bateria ou modo de economia de energia, outros programas), não o
modelo. Pendente: medir de novo em condição controlada e registrada; a
medição oficial é a do teste, também em imagens do `val`.

**Segunda medição** [M] (25/10/2026 10:30, mesmo comando, **com os outros
programas fechados**, condição informada pelo autor; carregador e modo de
energia não informados): mediana das duas etapas **42,70 ms** (média 53,45;
p95 117,19; p99 137,70), só verificador 21,95 ms; **meta atingida**.
Aceitação e recusa idênticas à primeira execução (189/189, 325/326,
294/294), como esperado de pesos fixos. A hipótese da condição da máquina se
sustenta: o verificador da v2 fica na mesma faixa da v1 (24,80 ms em
18/10). O relatório `verificador_v2_val.md` agora traz esta segunda
medição; a primeira (120,84 ms) fica registrada só aqui. **Para o teste:
repetir a mesma condição** (outros programas fechados) e registrá-la.

### 4.8 Teste da v2 — **medição única** — [M] `reports/verificador_v2_test.md` (25/10/2026 13:00)

`08 --split test --rotulo v2 --limiar 0.50`. Limiar, metas e critério de
implantação fixados antes (Seção 1, 25/10). CPU local, pesos `.pt`.
Condição da máquina na medição: o autor foi orientado a fechar os outros
programas; a confirmação não foi registrada. `verificador_v2_test.json`
existe: o script recusa uma segunda medição da v2. **Nenhum ajuste pode ser
feito com base nestes números.**

**Critério de implantação (fixado em 25/10)**

| Critério | Medido | Mínimo | Situação |
| --- | --- | ---: | --- |
| (1) Alface aceita — folhas doentes (Kaggle) | 178/184 (96,7%; IC95 93,1%–98,5%) | 183/184 | **não atingida** |
| (2) Não alface recusada — todas | 340/354 (96,0%; IC95 93,5%–97,6%) | 337/354 | atingida |
| (3) Latência mediana das duas etapas (CPU local, imagens do `val`) | 51,85 ms (p95 122,20; só verificador 25,42) | < 100 ms | atingida |
| (4) Alface aceita — C270 | **75/75** (100,0%; IC95 95,1%–100,0%) | 72/75 | atingida |
| (5) Não alface recusada — C270 | 42/48 (87,5%; IC95 75,3%–94,1%) | 46/48 | **não atingida** |

**Resultado: a v2 não atende ao critério de implantação** (condições 1 e 5).
**Decisão posterior do autor (25/10): implantar a v2 mesmo assim** — ver
"Avaliação da gravidade dos erros", no fim desta seção, e a Seção 1.

**Por grupo:** bacteriano 106/106, fúngico **72/78**, saudável 140/140,
C270 alface 75/75; COCO 67/67, PlantVillage 73/73, DeepWeeds 81/82,
**PlantDoc 77/84**, C270 não alface **42/48**.

**Cascata sobre o teste oficial do classificador (324 imagens)**

| Classe | Recusadas pelo verificador | Revocação só classificador | Revocação com a cascata |
| --- | ---: | --- | --- |
| bacteriano | 0 | 103/106 (97,2%) | 103/106 (97,2%) |
| fungico | 6 | 66/78 (84,6%) | **63/78 (80,8%)** |
| saudavel | 0 | 139/140 (99,3%) | 139/140 (99,3%) |

Alertas: 167 → 166; **1 alerta verdadeiro suprimido**. Acurácia: 308/324
(95,1%) → 305/324 (94,1%).

**Comparação com a v1 (Seção 4.5)** — mesmas 630 imagens do Kaggle e das
fontes públicas; a C270 é de outra sessão em cada versão

| Grupo | v1 | v2 |
| --- | --- | --- |
| Folhas doentes aceitas | 181/184 | **178/184** (pior) |
| Bacteriano / fúngico | 105/106 / 76/78 | 106/106 / **72/78** |
| PlantDoc recusado | 72/84 | **77/84** (melhor) |
| DeepWeeds / COCO / PlantVillage | 81/82 / 67/67 / 73/73 | iguais |
| Alface aceita — C270 | 12/30 (sessões de 10–11/10 e 17/10) | **75/75** (sessão de 25/10) |
| Não alface recusada — C270 | 45/50 | 42/48 |
| Revocação fúngica com a cascata | 64/78 (82,0%) | 63/78 (80,8%) |
| Alertas verdadeiros suprimidos | 3 | **1** |
| Acurácia da cascata | 304/324 (93,8%) | 305/324 (94,1%) |

**Análise dos erros** (`reports/predicoes_verificador_v2_test.csv`,
`cascata_v2_test.csv`; inspeção, sem alterar nada):

- **Folhas fúngicas recusadas (6), de apenas 3 imagens de origem**, todas
  abertas em 25/10:
  - `fungal_original_09_1` (P = 0,084): detalhe ampliado de uma lesão,
    acinzentado, 200 × 200 px, sem forma de folha. Classificador:
    `fungico` 0,915 — **é o alerta suprimido**;
  - `fungal_original_p16_3` (0,490) e `…p16_4` (0,373): superfície verde
    em macro, com hifas e corpos de frutificação pretos, e **um retângulo de
    ruído colorido gravado na imagem** (apagamento aleatório do aumento de
    dados do dataset de origem). Classificador: `fungico` 0,78 (sem alerta)
    e `saudavel`;
  - `fungal_original_p17_1` (0,103), `p17_2` (0,297) e `p17_3` (0,303):
    fragmento de folha sobre fundo branco, em macro, coberto de micélio
    branco. Classificador: `saudavel`, `fungico` 0,533 (sem alerta) e
    `saudavel`.
  - Os três grupos são imagens de **microscópio ou macro**, sem a forma da
    folha. Das 6, o classificador já errava 3 (`saudavel`, sem alerta). Os
    erros estão correlacionados: 6 recusas, mas só 3 imagens de origem.
- **Não alface da C270 aceita (6), todas de outras plantas no quintal:**
  `…12_44_51` (vasos de barro junto ao muro, P = 0,987), `…12_45_14`
  (babosa, 0,976), `…12_45_31` (violetas, 0,979), `…12_45_37` (jiboia e
  mangueira, 0,825), `…12_45_28` (samambaia, 0,805), `…12_45_45`
  (lírio-da-paz, 0,548). Das ~10 fotos com outra planta como assunto, 6
  foram aceitas; **todas as fotos sem planta foram recusadas**, inclusive as
  9 de lente tapada (na v1, 4 dos 5 erros da C270 eram lente tapada).
  **Hipótese [N]:** o modelo aprendeu "planta em vaso no ambiente da casa"
  mais do que "alface"; no treino, a C270 não tinha nenhuma outra planta
  como "não alface".
- **PlantDoc aceita (7), abertas em 25/10** (numeração da pasta
  `erros_verificador_v2_test/`):
  - 13 `apples_apple-scab_02_thm` (0,835): miniatura de 115 × 85 px, folha
    verde-amarelada com manchas escuras de sarna — pouca informação de forma;
  - 14 `P4190299aS` (0,878): muda inteira de pimentão, sadia, folhas largas
    e lisas em roseta sobre terra — o arranjo mais parecido com um pé de
    alface;
  - 15 `500px-Tomato_Early_Blight_Leaf` (0,737): folha amarelada com lesões
    concêntricas marrons, sobre fundo azul (pasta "Potato leaf early blight",
    mas o nome indica tomate; continua "não alface");
  - 16 `14321627273_58d893c3e2_b` (0,760): folha de batata com requeima,
    verde com área parda, nervuras marcadas, fundo preto;
  - 17 `Late-blight-tomato-leaf-VikramBisht-MBag` (0,932): folha de tomate
    com requeima no pé, cercada de folhagem e frutos verdes;
  - 18 `IMG_2686.JPG` (0,796): tomate com vírus do amarelecimento — folhas
    **verde-claras, crespas e enroladas**, muito parecidas com alface
    crespa;
  - 19 `grape-vine-leaf-1327453` (0,576): folha de videira, verde, lobada e
    serrilhada, sobre fundo preto.
- **DeepWeeds aceita (1):** 20 `20170217-111024-0` (0,993): massa de folhas
  ovais verde-acinzentadas de uma planta rasteira, cobrindo todo o quadro.
- Padrão comum às 14 "não alface" aceitas (anotação do autor: todas
  plantas): folhas **largas, verdes e em grupo**, muitas com lesões. Nenhuma
  aceita era objeto, piso, céu ou lente tapada.
- **Alface da C270, 75/75:** inclui vaso preto e jardineira, de cima, de
  lado e rente ao chão. Ressalvas: uma única sessão, câmera na mão, as
  mesmas plantas do treino (com outro fundo e luz) — o resultado
  provavelmente é otimista para outras plantas e para a câmera fixa [N].
- **Os 20 erros** (6 alface recusada + 14 "não alface" aceita) listados pelo
  autor a partir do CSV em 25/10. **Anotação do autor: as 14 "não alface"
  aceitas são todas plantas** (6 da C270, 7 do PlantDoc, 1 do DeepWeeds).
  Cópias das 20 imagens, com índice, em
  `desenvolvimento/reports/erros_verificador_v2_test/` (mesmo padrão de
  `reports/erros_classificacao/`); os originais não são movidos.
- Observações pós-teste, **não usadas para decidir nada**: no limiar 0,40,
  `p16_3` (0,490) seria aceita e o resultado seria 179/184, ainda abaixo
  de 183. As imagens recusadas em (1) não se parecem com o que a C270
  captura em operação, mas o critério foi fixado sobre todas as folhas
  doentes do teste e assim deve ser reportado.

**Avaliação da gravidade dos erros (decisão do autor, 25/10, posterior ao
teste)**

Na avaliação do autor, os erros do teste não são agravantes a ponto de
impedir a implantação. Os fundamentos, todos tirados dos números e das
imagens desta seção:

| Erro | Medido | Por que não foi considerado agravante | O que continua valendo |
| --- | --- | --- | --- |
| Folhas fúngicas recusadas (condição 1) | 6/184 recusadas; 178/184 aceitas | As 6 vêm de **3 imagens de origem** de microscópio ou macro (detalhe de lesão, superfície com hifas, fragmento coberto de micélio), duas com um retângulo de ruído gravado pelo dataset de origem. Nenhuma tem a forma de uma folha ou de uma planta, como as que a C270 captura. Bacteriano 106/106, saudável 140/140 e alface da C270 75/75 foram todas aceitas | **1 alerta verdadeiro suprimido** (`fungal_original_09_1`); revocação fúngica da cascata 84,6% → **80,8%** (63/78); nas outras 5, o classificador já errava (3) ou não emitia alerta (2) |
| "Não alface" da C270 aceita (condição 5) | 42/48 recusadas; 6 aceitas | As 6 aceitas são **outras plantas** do quintal (babosa, violetas, samambaia, jiboia, lírio-da-paz, vasos junto ao muro). Todas as fotos sem planta foram recusadas, inclusive as 9 de lente tapada, que eram a principal falha de "não alface" da v1 | Em operação, outra planta diante da C270 pode receber diagnóstico — frequência [N] |
| PlantDoc e DeepWeeds aceitas | 8 aceitas (7 + 1) | Também plantas de folhas largas, várias com lesões; a meta geral de "não alface" (condição 2) foi atingida: 340/354 | Mesmo risco do item anterior |

Comparação que pesou na decisão: a v1 aceitava 12/30 fotos de alface da
C270 e suprimia 3 alertas verdadeiros; a v2 aceita 75/75 e suprime 1.

Consequências que o TCC deve declarar junto com a decisão:

1. O critério fixado antes do teste **não** foi atingido (178/184 e 42/48);
   a implantação é uma decisão do autor tomada depois do resultado.
2. **O classificador não muda**: sozinho, continua com revocação fúngica de
   84,6% (66/78) e acurácia de 95,1% (308/324), e o CT01 dele continua o
   mesmo. O número novo é o do **sistema completo** (verificador +
   classificador) nas mesmas 324 imagens: 63/78 = **80,8%**. A diferença
   vem só das 6 folhas fúngicas que o verificador recusou: 3 delas o
   classificador acertava (`fungal_original_09_1`, `p16_3`, `p17_2`) e, com
   a cascata, ficam sem diagnóstico (66 − 3 = 63); as outras 3 ele já errava.
   Se o CT01 do contexto geral (revocação fúngica ≥ 85%,
   `CONTEXTO_IA_CONFIAVEL.md` 6.2) for aplicado ao sistema implantado, o
   resultado é 80,8%. A acurácia do sistema completo é 94,1% (305/324).
3. O 75/75 da C270 vem de uma única sessão, com a câmera na mão e as mesmas
   plantas do treino.
4. O teste da v2 foi consumido: uma versão nova (Seção 4.9) exige um
   conjunto de teste inédito.

### 4.9 Melhorias futuras para o refinamento do verificador (registradas em 25/10)

Derivadas dos erros do teste da v2 (Seção 4.8). Nenhuma foi implementada nem
medida [N]. Qualquer versão nova precisa de validação e teste inéditos.

**Principal: outras plantas como "não alface"**

1. **Plantas do ambiente real, com a C270.** Fotografar, no enquadramento de
   operação, plantas que não são alface — as que a v2 aceitou (babosa,
   violeta, samambaia, jiboia, lírio-da-paz, rosa-do-deserto) e outras de
   folha larga (costela-de-adão, bambu-da-sorte) — e incluí-las no treino
   como "não alface". No treino da v2, nenhuma foto de "não alface" da C270
   mostrava outra planta.
2. **Hortaliças folhosas parecidas com alface**, os negativos mais difíceis
   e os mais prováveis numa estufa: rúcula, agrião, couve, espinafre,
   repolho, chicória e escarola, além das plantas espontâneas que crescem
   na estufa. Fontes: fotos próprias com a C270 ou conjuntos públicos
   conferidos para não conter alface.
3. **Mais folhas e mudas de outras culturas**, sadias e com lesões — o tipo
   de imagem do PlantDoc que a v2 aceitou (muda de pimentão, tomate com
   vírus de folhas crespas, batata e tomate com requeima, videira).

**Complementares**

4. **Alface em close-up e com doença avançada, vista pela C270**, para que o
   verificador não dependa da forma da folha inteira. Junto, auditar as
   imagens de microscópio ou macro e as com artefatos gravados do dataset do
   Kaggle (como `fungal_original_p16_3` e `p16_4`), para decidir se pertencem
   ao domínio de operação — sem alterar o teste já medido.
5. **Fotos da C270 das duas classes na validação**, para escolher o limiar
   com negativos do ambiente real; a v1 e a v2 foram calibradas sem eles.
6. **Sessões com a C270 fixa** no enquadramento de operação, em várias
   datas, para validação e teste da próxima versão, incluindo a jardineira
   com substrato e sem planta e o vaso vazio, nunca fotografados.
7. **Coleta em operação**: guardar a probabilidade do verificador com cada
   captura (com a imagem no Blob ou numa coluna do banco), para rotular
   depois as imagens reais e medir o desempenho com a câmera fixa.
8. **Distância no espaço de características** (Quadro 7 do documento
   técnico), como segunda checagem para imagens muito diferentes do treino.

### 4.2 Como a avaliação protege a regra do teste único [C]

- `--split test` recusa-se a rodar se o JSON de teste **da versão** já existir:
  `reports/verificador_test.json` na v1 (sem `--rotulo`) e
  `reports/verificador_<rotulo>_test.json` nas seguintes. Desde 24/10, com
  `--rotulo v2`, pesos (`modelos/yolo11n_cls_dominio_v2.pt`, se `--pesos` for
  omitido) e saídas (`verificador_v2_*`, `predicoes_verificador_v2_*`,
  `cascata_v2_test.csv`) levam a versão no nome, sem sobrescrever a v1.
  `--rotulo v1` é recusado: abriria um segundo teste da v1 com outro nome
  (`avaliacao_dominio.nomes_da_avaliacao`).
- No teste não há curva de limiar: o limiar vem do `val`.
- A cascata usa as predições **oficiais** do classificador (`reports/predicoes.csv`),
  sem executá-lo de novo no teste; o cruzamento é por nome de arquivo e classe,
  porque o CSV guarda o caminho anterior à pasta `desenvolvimento/`.
- A latência das duas etapas é medida em imagens do `val`.
- A comparação com a meta usa a razão exata (acertos/total), não a taxa arredondada.

### 4.1 Contrato da API com o verificador ativo [C]

| Situação | Status | Corpo |
| --- | --- | --- |
| Imagem com alface | 200 | Diagnóstico normal + `verificacao` (`probabilidade_alface`, `limiar`, `tempo_inferencia_ms`, `modelo_versao`) |
| Imagem sem alface | **200** (desde 25/10; antes 422) | `classe = "nao_alface"`, `classe_exibicao = "Alface não encontrada"`, `confianca = 1 − probabilidade_alface`, `probabilidades = {}`, `patogeno_detectado` e `alerta` falsos, `tempo_inferencia_ms = 0`, `modelo_versao` do verificador, `heatmap_base64` nulo, `verificacao` preenchido |
| Requisição malformada | 422 (FastAPI) | `detail` em lista — único uso do 422 |
| Imagem corrompida | 400 | Inalterado: validada antes de qualquer modelo |
| Verificador configurado e não carregado | 503 | `/health` com `status = "erro"` e `verificador_carregado = false` |

A Azure Function (quando existir) deve tratar a resposta com
`classe = "nao_alface"` como resultado final, sem retry, e o 400 como falha
definitiva. Corrigido em 25/10: a frase ainda falava no 422 de domínio,
removido nessa data.

**Implementado em 25/10** [C] (decisão de 24/10, Seção 1): `api/main.py`
(`_montar_resposta_sem_alface`; o tratador de exceção 422 foi removido),
`api/esquemas.py` (`RespostaForaDoDominio` e `CODIGO_FORA_DO_DOMINIO`
removidos; descrições de `classe`, `confianca`, `probabilidades`,
`patogeno_detectado`, `alerta`, `tempo_inferencia_ms` e `modelo_versao`
atualizadas), `api/configuracao.py` (docstring),
`src/ia_alface/classes.py` (`EXIBICAO_NAO_ALFACE`). A biblioteca
(`cascata.py`) não mudou: continua levantando `ImagemForaDoDominio`, e a API a
converte em resposta. Testes de `test_api.py`: os 5 da recusa 422 foram
substituídos por 6 (resposta 200 com `nao_alface`, ausência de alerta e de
diagnóstico, rota bruta, 400 de imagem corrompida, 422 só para requisição
malformada, contrato OpenAPI). Suíte: **214 aprovados** [M] (25/10, 88,75 s);
`docs/openapi.json` e `docs/swagger.html` regenerados (4 rotas, 9 esquemas). A Azure Function, quando
existir, não grava diagnóstico quando `classe = "nao_alface"`, e o ENUM do
banco precisa aceitar o valor para gravar a resposta.

---

## 5. Pendências e não medido

- [x] Primeira sessão de "não alface" da C270 (17/10, 57 fotos, dia)
- [x] Sessão noturna com as duas classes (17/10, 18:40–19:00: 37 "não alface" + 20 alface)
- [x] **Jardineira com substrato e sem planta: não fotografada — limitação declarada** (decisão do autor em 17/10; Seção 1). Desempenho nesse caso: [N]. Histórico: o autor ia fotografar em 17/10, **até 21:00:22** (2 h depois da última foto da noite, 19:00:22), em `20261017_projeto/nao_alface/`. Assim as fotos entram na sessão `20261017_1840` e vão para o **teste**; o modelo não as vê no treino. Fotos depois desse horário formariam uma terceira sessão e empurrariam a noite de 17/10 para a validação. Depois: conferir, rodar o 06 de novo e só então treinar. **Atualização:** as 13 fotos das 19:49–19:51 são da estrutura vazia, não da jardineira (Seção 2.8); a jardineira com substrato continua faltando. Prazo para entrar na mesma sessão passou a ser 21:51:03
- [x] Decidir o que fazer com a sessão de 17/10 18:40 (fica no 17/10, vai para o treino; ver Seção 1)
- [x] Reconfirmar o encaminhamento depois das 20 fotos de alface: dividir a C270 por sessão de captura (Seção 1, última decisão)
- [x] Implementar a divisão por sessão (`sessoes_de_captura`, `dividir_sessoes*`, +5 testes)
- [x] Rodar a suíte com a divisão por sessão (199 aprovados em 17/10, 90,35 s)
- [ ] O `val` fica sem "não alface" da C270 (só há duas sessões com "não alface": treino e teste). O limiar do passo F será escolhido sem negativas do ambiente real — declarar
- [x] Fronteira de meia-noite: a última foto de 10/10 (23:59:23, validação) e a primeira de 11/10 (00:01:22, teste) têm 2 min de intervalo, mesma planta e mesma cena (conferido visualmente; Seção 2.6). Vem da regra por dia já aprovada. Decidido em 17/10: a sessão atravessa a meia-noite (Seção 1)
- [x] Rodar a suíte com a sessão atravessando a meia-noite (200 aprovados em 17/10, 85,84 s)
- [x] Reexecutar o 06 (17/10 19:45, integridade aprovada, conforme o previsto; Seção 2.7)
- [x] Rodar a suíte de testes (176 aprovados em 17/10)
- [x] Rodar a suíte com os testes do script 08 e do ajuste da divisão (194 aprovados em 17/10)
- [x] Decidir o ajuste da divisão (aprovado em 17/10)
- [x] Implementar o ajuste (`dividir_dias_estratificado`, 4 testes novos)
- [x] Reexecutar o 06 e conferir na Seção 4 do relatório: `20261017_1601` no treino, `20261017_1840` no teste, demais sessões como na Seção 1.2 (17/10 19:28, integridade aprovada; Seção 2.6)
- [x] Script 06 com os dados reais (17/10)
- [x] Regenerar `docs/openapi.json` e `docs/swagger.html` (17/10)
- [x] Incluir `modelos/yolo11n_cls_dominio.pt` no `Dockerfile` só depois de treinado — substituído em 25/10: a versão a implantar é a v2 (item "Implantação da v2", abaixo)
- [x] Script de avaliação do verificador (`08_avaliar_verificador.py`)
- [ ] Fora deste escopo: `02_treinar.py` usa `ConfigTreino.label_smoothing`, campo que não existe mais — o script quebra ao iniciar
- [ ] Registrar o commit exato do PlantVillage baixado: `git -C desenvolvimento/dataset_dominio_bruto/nao_alface/plantvillage rev-parse HEAD`
- [ ] Duplicatas **aproximadas** (mesma foto em tamanhos diferentes, comum no PlantDoc) não são detectadas por MD5 — risco avaliado como pequeno, não verificado [N]
- [x] Copiar os pesos para `modelos/yolo11n_cls_dominio.pt` e os artefatos para `desenvolvimento/runs/classify/yolo11n_cls_dominio/` (18/10; `results.csv` confere com o log: melhor época 6, 0,99839)
- [x] Ultralytics do treino (8.4.181, Colab) é mais novo que o local (8.4.95): os pesos carregaram sem erro no 08 (18/10)
- [x] Passo F, avaliação no `val` (18/10; Seção 4.4)
- [x] **Fixar o limiar e as metas antes do teste**: limiar 0,50; metas ≥ 99% / ≥ 95% / < 100 ms (18/10, decisão do autor; Seção 1)
- [x] Passo G: `test` medido **uma única vez**, com `--limiar 0.50` (18/10 17:33; Seção 4.5). Metas de aceitação de doentes e recusa de não alface **não atingidas**; latência atingida
- [x] **Decidir o destino do verificador v1** (autor; resolvido: nova iteração, a v2, e em 25/10 implantar a v2): não implantar e declarar o resultado, ou nova iteração com dados novos (que exige um novo conjunto de teste; Seção 4.5). Opções levantadas em 18/10: documentar sem implantar; modo observação (verificador informa, nunca bloqueia); nova iteração; regras de qualidade da imagem (escura/borrada)
  - Proposta para nova iteração (não decidida; estimativa de planejamento, não medida): **6 a 8 sessões novas**, de preferência 4 datas × (tarde + noite), cada uma com ~20 alface (jardineira e vaso redondo) e ~20 não alface (jardineira vazia, vaso vazio, lente tapada, mão/objetos, estrutura vazia), sem rajada. Todas as sessões até 17/10 passariam a ser **só treino** (exige data de corte no 06), e validação e teste sairiam só das sessões novas. Com 8 sessões novas, as 2 mais recentes vão para o teste e as 2 anteriores para a validação. O teste do Kaggle (324) seria reutilizado e já foi visto: declarar
- [ ] Inspeção visual das 3 folhas doentes recusadas e das 12 PlantDoc aceitas [N]
- [x] Revisar as 265 fotos de 19–21/10 (22/10; Seção 2.10): 9 para `duvidosas/`
- [x] Mover as 9 fotos para `duvidosas/` (conferido em 22/10: 2 em 19/10, 5 em 20/10, 2 em 21/10)
- [x] Implementar a data de corte (`dividir_sessoes_estratificado(data_corte=…)`, `ConfigDatasetDominio.data_corte_treino`, `--data-corte`, aviso no 06; +5 testes)
- [x] Rodar a suíte com a data de corte (205 aprovados em 22/10, 97,22 s)
- [x] Revisar a sessão de 25/10 (Seção 2.14): 98 fotos abertas; rótulos anotados por nome
- [x] Mover `WIN_20261025_12_31_05_Pro` para `nao_alface/` e `…12_21_36`, `…12_28_54`, `…12_31_24` para `duvidosas/` (autor, 25/10; conferido na listagem)
- [x] "Não alface" no quintal na sessão de 25/10 (autor): 28 fotos, 12:44:44–12:46:20, revisadas, 28/28 corretas (Seção 2.14)
- [x] Rodar o 06 com a sessão de 25/10 (25/10 12:55; Seção 2.15): integridade aprovada, sem avisos, `20261025_1220` no teste com 75 / 48; treino e validação inalterados
- [ ] Plano do autor (25/10): **uma sessão em 25/10**; outras só se necessário. Com uma sessão depois do corte, ela vai inteira para o **teste** [C], o treino não muda (a v2 atual vale) e o limiar sai do `val` sem C270. Uma segunda sessão, depois, mandaria a primeira para o treino: decidir isso **antes** de medir o teste
- [ ] **Retomada em 25/10** (Seção 1; antes, "não serão feitas"): sessões novas a partir de 25/10 para validação e teste (autor; antes de 24/10, "depois de 21/10"): C270 **fixa no enquadramento de operação**, com as duas classes, vaso redondo e jardineira, "não alface" do enquadramento real (jardineira vazia, vaso vazio, lente tapada, estrutura vazia), de dia e de noite, sem rajada
- [x] Conferir `estufa/` na retomada da v2 (22/10; Seção 2.11): nenhuma sessão depois de 21/10
- [x] Confirmar por comando a contagem de 19/10 e 21/10 (22/10; Seção 2.11): 64 de alface em 19/10 e 29 de "não alface" em 21/10; Seção 2.10 corrigida
- [x] Copiar os relatórios do 06 da v1 (17/10 19:56) para `dataset_dominio_auditoria_v1.md` e `dataset_dominio_inventario_v1.csv` antes de reexecutar o 06, que os sobrescreve e apaga `dataset_dominio/` (os dois não estão no Git)
- [x] Reexecutar o 06 com as fotos existentes (24/10 16:44; Seção 2.13): integridade APROVADA, 4.772 imagens, toda a C270 no treino, aviso de validação e teste sem C270
- [x] Passo C (24/10): `--rotulo` no 08, com pesos e saídas derivados do rótulo e proteção do teste único por versão (Seção 4.2); função `nomes_da_avaliacao` em `avaliacao_dominio.py` com +8 testes em `test_avaliacao_dominio.py`; no caderno do Colab, `NOME = "yolo11n_cls_dominio_v2"` e leitura da pasta real da execução (`modelo.trainer.save_dir`), porque o Ultralytics 8.4.181 gravou a v1 em `runs/classify/runs/classify/<NOME>` (Seção 4.3) e as células 6 e 7 procuravam em `runs/classify/<NOME>`
- [x] Rodar a suíte com o passo C (213 aprovados em 24/10, 89,83 s)
- [x] Passo D, treino da v2 no Colab (24/10; Seção 4.6): `dataset_dominio_v2.zip`, contagens 3.522 / 620 / 630 conferidas na célula 4; GPU, versões e hiperparâmetros registrados; melhor época 36 de 56, acurácia de validação 0,9984
- [x] Copiar `Downloads\best (2).pt` para `modelos/yolo11n_cls_dominio_v2.pt` (autor; os pesos carregaram no 08 em 25/10). Artefatos da execução não guardados (Seção 4.6)
- [x] Passo E, `08 --split val --rotulo v2` (25/10 10:21; Seção 4.7): pesos do 8.4.187 carregaram no 8.4.95 local
- [x] Medir de novo a latência no `val` com os outros programas fechados (25/10 10:30; Seção 4.7): 42,70 ms, meta atingida
- [x] **Fixar limiar, metas e critério de implantação com o autor antes do teste** (25/10, Seção 1): limiar 0,50; 183/184, 337/354, < 100 ms, 72/75 e 46/48
- [x] Teste da v2, **uma única vez** (25/10 13:00; Seção 4.8): critérios 1 e 5 não atingidos; 2, 3 e 4 atingidos. Condição da máquina não confirmada
- [x] Copiar as 20 imagens dos erros do teste da v2 para `reports/erros_verificador_v2_test/`, com `_indice.csv` (autor, 25/10; 21 arquivos conferidos na listagem); as 20 abertas e descritas na Seção 4.8
- [x] **Decidir o destino da v2** (25/10, autor): implantar, apesar do critério não atingido; justificativa na Seção 4.8 e melhorias futuras na Seção 4.9
- [x] G1 (25/10): Seção 2.9 do documento técnico e `gerar_documento_tcc.py` (iguais), com v1 e v2 medidas, o critério não atingido, a decisão de implantar posterior ao teste e as melhorias futuras; resultados na nova 3.8, limitações e melhorias no capítulo 4 (Seção 1, 25/10)
- [x] Conferir a geração do `.docx` (25/10, autor) [M]: `python desenvolvimento/scripts/gerar_documento_tcc.py` gerou `desenvolvimento/tcc/Modulo_IA_TCC.docx` e as figuras em `desenvolvimento/tcc/figuras_modulo_ia`, com 33 épocas lidas, sem erro. O import de `ia_alface.classes` funcionou. Figura 4 conferida: C270, resposta "Alface não encontrada (classe nao_alface)", sem tracejado. Detalhe estético anterior ao G1: o rótulo "inválida" encosta na borda da caixa de erro
- [x] G2 (25/10): D8 em `DIVERGENCIAS.md` (cascata, 200 `nao_alface`, C270), com as linhas novas no resumo de impacto
- [x] G3 (25/10): `CONTEXTO_IA_CONFIAVEL.md` com o verificador (Seção 5.7), o CT01 do sistema completo, o contrato `nao_alface`, os 214 testes, as limitações 1 e 3 e a divergência 4 (parcial)
- [x] G4 (25/10): `CLAUDE.md` com regras, comandos, estado, armadilhas, pendências, D1–D8 e Git
- [x] G5 (25/10): `IMPLANTACAO.md` com a topologia da C270 (envio ao Blob a definir pelo grupo), as variáveis do verificador, o contrato `nao_alface`, o `/health` e o RNF01 das duas etapas
- [x] Gerar o `.docx` com a Seção 2.10 (26/10): o autor informou que o script rodou, sem colar a saída. A Figura 5 (`figura_implantacao.png`) foi conferida e tem todos os elementos
- [x] **Grad-CAM em imagens não quadradas: desalinhamento confirmado e corrigido** (26/10) [C] + [M]. O autor decidiu corrigir antes da `v2` (Seção 1, 26/10). A correção está em `explicabilidade.py` e `api/main.py`, com 7 testes, 221 aprovados e conferência visual na nova build.
  - **Mecanismo [C]:**
    - o mapa 7 × 7 é calculado sobre o **recorte central** de 224 × 224 (`classify_transforms`: lado menor a 224 e recorte central);
    - `explicabilidade.py::GradCAM.gerar` (usado por `api/main.py::_gerar_heatmap`) o redimensiona para a imagem **inteira**;
    - numa foto retangular, o mapa é esticado sobre as faixas que o modelo não viu.
  - **Teste [M]:** `fungal_original_01_4…68c12973.jpg`, da validação, com 315 × 500 px. Resultado `fungico` com 100,0% e alerta.
    - O modelo só vê y ≈ 93 a 408 (descarta cerca de 93 px no topo e na base). O mapa, porém, pinta as 500 linhas.
    - O ponto mais quente aparece em (65, 390). Pela geometria do recorte, corresponde a y ≈ 339 na foto, cerca de 50 px acima do desenhado.
    - Nesta imagem a leitura não muda: as duas posições caem na mesma área esbranquiçada, de esporulação, à esquerda.
    - Numa lesão pequena perto da borda superior ou inferior, o ponto quente pode cair em tecido sadio ou numa faixa que o modelo não viu.
  - **Observação de passagem:** o retângulo de ruído do aumento de dados do dataset de origem (y ≈ 42–78) fica na faixa descartada, então o modelo não o vê nesta imagem. As lesões marrons do topo ficaram frias, e o mapa concentrou-se na área esbranquiçada.
  - **Alcance:** o desalinhamento já existia na imagem `v1`; vale para a API e para `testar_imagem.py --heatmap`. Não afeta classificação, verificador nem métrica. Afeta só a sobreposição visual (Grad-CAM, D6)
  - **Na C270 [M] + [C]:**
    - As 639 fotos da C270 no inventário do 06 têm **1280 × 720**.
    - O modelo vê só o quadrado central de 720 × 720 e descarta cerca de 280 px de cada lado, ou 44% da largura.
    - Hoje o mapa é esticado sobre as 1.280 colunas. Em operação, **todo** mapa da câmera do projeto sai desalinhado na horizontal, com até cerca de 280 px nas laterais, e as faixas laterais aparecem coloridas como se tivessem sido analisadas.
  - **Correção não exige retreino.** É pós-processamento: muda só a projeção do mapa 7 × 7 sobre a foto, sem tocar em pesos, métricas ou testes consumidos. Não há teste automatizado do Grad-CAM hoje (`test_explicabilidade.py` não existe). A Seção 1.8 do documento técnico diz que o mapa é "ampliado até as dimensões da imagem"
- [ ] Detalhes estéticos a corrigir na próxima geração do `.docx`: o rótulo "envio da imagem" (Figura 5) encosta na caixa da câmera, e o rótulo "inválida" (Figura 4) encosta na caixa de erro
- [ ] ESP32-CAM ainda citado no repositório (busca de 25/10): docstrings de `api/main.py` e contrato gerado (`docs/openapi.json`, `docs/swagger.html`), `docs/referencia_api.html`, `api/README.md`, `api/cliente.py`, `README.md`, `docs/AMBIENTE.md`, `docs/DOCUMENTACAO_TECNICA_ABNT.md`, `docs/README.md`, `.env.exemplo`, `spec.md`, `src/ia_alface/inferencia.py`, `reports/benchmark_latencia.md`, scripts e testes. Mexer em `api/` ou `src/` exige `pytest` e `gerar_openapi.py`. Fora da lista do passo G: decidir com o autor quando fazer
- [x] Implantação da v2 (passo H; concluída em 26/10 com a imagem `v2`, digest `sha256:833cc600…`), depois da recusa como 200 com `nao_alface`: `COPY` de `modelos/yolo11n_cls_dominio_v2.pt` e `ENV IA_ALFACE_MODELO_DOMINIO` no `Dockerfile`; `IA_ALFACE_LIMIAR_DOMINIO=0.50`; nova tag de imagem (a seguinte à v1, nunca republicar); atualizar o Container Apps; conferir `/health` com `verificador_carregado: true`; medir a latência na nuvem [N]
- [x] Desempenho da v2 na C270 (25/10; Seção 4.8): medido no teste, 75/75 de alface aceita e 42/48 de "não alface" recusada; validação sem C270 (limitação)
- [x] Conferir `estufa/` em 24/10 pela listagem (Seção 2.12): uma sessão depois do corte, `20261024_1522`, com 105 fotos de alface e nenhuma de "não alface"
- [x] Decidir o destino da sessão `20261024_1522` (24/10, autor): opção A, data de corte em 24/10; a sessão vai para o treino (Seção 1)
- [x] Mudar o padrão da data de corte para `"20261024"` em `ConfigDatasetDominio` (24/10)
- [x] Rodar a suíte com a data de corte de 24/10 (205 aprovados em 24/10, 89,23 s)
- [ ] Confirmar pelo comando da Seção 2.11 a contagem de `20261024_projeto`
- [x] Revisar foto a foto a sessão de 24/10 (24/10; Seção 2.12): as 105 são alface; 6 propostas para `duvidosas/`; 18 limítrofes mantidas, anotadas por nome
- [x] Mover para `20261024_projeto/duvidosas/` as 6 fotos propostas (24/10, autor; 6 fotos na pasta, segundo o autor; saída da contagem não registrada)
- [ ] Proposta para o passo C, não decidida: o 06 avisar quando a validação ou o teste ficarem sem fotos da C270 de alguma classe depois do corte. Hoje ele só avisa se não houver nenhuma sessão depois do corte
- [x] **Implementar a recusa como 200 com `classe` = `nao_alface`** (25/10; Seção 4.1): código e testes alterados
- [x] Rodar a suíte com a recusa como 200 (25/10: **214 aprovados**, 88,75 s, 1 aviso do Starlette) e regenerar `docs/openapi.json` e `docs/swagger.html` (25/10 16:14: OpenAPI 3.1.0, 4 rotas, 9 esquemas)
- [x] Na documentação (passo G): D8, Seção 2.9 do documento técnico e `CLAUDE.md` com o contrato novo (200 com `nao_alface`; a armadilha do "422 com dois significados" deixa de existir). Seção 2.9 feita no G1 (25/10, Seção 2.9.2 do documento técnico) e D8 no G2 (25/10); `CLAUDE.md` no G4 (25/10). Concluído
- [ ] Combinar com o grupo do banco o valor `nao_alface` na coluna `classe` (ENUM ou CHECK), ou que o back-end não grave essas respostas (autor). Registrado como ação necessária na D8 (25/10), com o `ALTER TYPE` sugerido
- [x] Atualizar a Tabela 7 do documento técnico (tirar o `[PREENCHER]`, registrar que as metas foram fixadas antes do teste): feito no G1 (25/10), com os critérios (1)–(5) e as datas em que foram fixados. A linha dizia "passo H", mas o item pertence ao G
- [x] Desempenho do verificador nas metas de aceitação: medido no teste da v1 (Seção 4.5) e da v2 (Seção 4.8)
- [x] Latência conjunta das duas etapas: medida na CPU local (v2: 51,85 ms no teste) e na nuvem (v2: 22,62 ms de mediana, 26/10). Inicialização a frio: 11,27 s, numa única medida
- [ ] Repetir a medição a frio na nuvem para ter uma faixa (opcional)
- [x] Revocação fúngica ponta a ponta da cascata: v1 82,0% (64/78), v2 80,8% (63/78), contra 84,6% só do classificador

---

## 6. Referências (conferir formatação ABNT na versão final)

HUGHES, D. P.; SALATHÉ, M. An open access repository of images on plant health
to enable the development of mobile disease diagnostics. arXiv:1511.08060,
2015.

LIN, T.-Y. et al. Microsoft COCO: common objects in context. In: EUROPEAN
CONFERENCE ON COMPUTER VISION (ECCV), 2014. arXiv:1405.0312.

MOHANTY, S. P.; HUGHES, D. P.; SALATHÉ, M. Using deep learning for image-based
plant disease detection. **Frontiers in Plant Science**, v. 7, 2016.

OLSEN, A. et al. DeepWeeds: a multiclass weed species image dataset for deep
learning. **Scientific Reports**, v. 9, n. 1, 2019. DOI:
10.1038/s41598-018-38343-3. Dados: DOI 10.5281/zenodo.7939060.

SINGH, D. et al. PlantDoc: a dataset for visual plant disease detection. In:
ACM INDIA JOINT INTERNATIONAL CONFERENCE ON DATA SCIENCE AND MANAGEMENT OF DATA
(CoDS-COMAD), 2020. arXiv:1911.10317.
