# Registro de decisões — infraestrutura e nuvem

Por que a camada de nuvem deste módulo é como é. Cada seção registra a decisão,
a alternativa descartada e o motivo.

Este arquivo não ensina a operar nada — para isso existe
[`IMPLANTACAO.md`](IMPLANTACAO.md), com os comandos. Aqui ficam as razões, que
são o que se perde depois de alguns meses. Divergências em relação ao projeto
original do TCC continuam em [`DIVERGENCIAS.md`](DIVERGENCIAS.md).

---

## O que existe e o que não existe

| Componente | Estado |
| --- | --- |
| Conta de armazenamento e container no Azure Blob Storage | criado e verificado |
| Script de verificação do acesso ao Blob | implementado |
| Imagem de container da API | construída, executada e publicada |
| Autenticação por chave de API | implementada, coberta por testes |
| Documentação de implantação | escrita |
| Serviço no Azure Container Apps | criado e no ar, com escala a zero |
| Azure Function com gatilho de blob | **não implementada** |
| Integração com PostgreSQL e MQTT | fora do escopo deste módulo |

---

## 1. O princípio que orientou tudo

**A API de inferência não conhece a nuvem.** Ela recebe bytes e devolve uma
classificação; de onde vieram os bytes não é problema dela.

Isso não é acidente — é a regra 5 do `CLAUDE.md` (`src/ia_alface/` é biblioteca
pura) estendida à camada HTTP. A consequência prática é que **nenhuma linha de
`api/` ou `src/` mudou** para suportar Azure Blob Storage.

**Alternativa descartada:** criar uma rota `/analyze/url` que baixasse a imagem
do Blob. Foi considerada e recusada por três motivos:

1. **SSRF** — aceitar URL arbitrária de cliente é vetor de ataque conhecido;
   exigiria lista de permissão de domínios
2. **A validação de tamanho perderia o efeito** — hoje o limite de 10 MB é
   conferido sobre bytes já em memória; com URL seria preciso baixar primeiro
3. **A latência do download entraria no caminho crítico**, contaminando a
   medição de RNF01

Quem integra com o Blob é o back-end da aplicação, fase seguinte do TCC. Por
isso o `verificar_blob.py` mora em `desenvolvimento/scripts/` e o
`azure-storage-blob` está em `requirements-dev.txt`, **não** no
`requirements.txt` de produção: a imagem de produção não deve carregar um SDK
que nunca importa.

---

## 2. Armazenamento — Azure Blob Storage

### Configuração adotada

| Escolha | Valor | Motivo |
| --- | --- | --- |
| Conta | `tccimagens` | nome global, só minúsculas e dígitos |
| Container | `imagens-alface` | privado, sem acesso anônimo |
| Região | Brazil South | latência baixa para a estufa e para o autor |
| Redundância | LRS | GRS replica entre regiões e custa o dobro, sem ganho aqui |
| Camada | Hot | escrita frequente, leitura ocasional |
| Exclusão reversível | 7 dias | gratuita; protege de apagar imagem por engano |

Acesso anônimo foi desativado **no nível da conta**, não apenas do container.
Assim o erro de publicar um container por acidente se torna impossível.

### Região: por que não os Estados Unidos

Regiões brasileiras têm preço por GB sensivelmente maior, por conta da carga
tributária local. O argumento de custo é real, mas não morde: sobre algumas
centenas de MB a diferença é de centavos, e trocaria latência 10× maior por
economia irrelevante.

Vale registrar que a assinatura Azure for Students **restringe as regiões
disponíveis**, e a lista varia entre assinaturas. Canada Central foi recusada
com `RequestDisallowedByAzure` nesta conta, apesar de funcionar na de um colega.
Não é erro de configuração e não há como contornar — só escolher outra região.

### Autenticação: identidade em vez de chave

Adotado `DefaultAzureCredential` com `az login`, e não connection string.

| | Identidade (Entra ID) | Connection string |
| --- | --- | --- |
| Segredo em disco | nenhum | sim, e é a chave da conta |
| Granularidade | por pessoa, via RBAC | tudo ou nada |
| Somente leitura | possível | impossível |
| Revogar um usuário | sim | não — só rotacionando a chave, derrubando todos |
| Caminho para produção | Managed Identity, sem credencial | precisa injetar o segredo |

O ganho decisivo é que **a mesma linha de código funciona em três ambientes**:
na máquina do autor encontra a sessão do `az login`; na de um colega, a dele; e
implantado no Azure, encontraria a Managed Identity — sem credencial alguma.

O `verificar_blob.py` aceita connection string como alternativa, para quem não
possa instalar o Azure CLI, mas emite aviso ao usá-la.

**Limite conhecido:** o ESP32-CAM não consegue usar Entra ID — microcontrolador
não executa fluxo OAuth de forma prática. O caminho do dispositivo terá de ser
SAS com validade e permissão mínima. Autenticação é escolha *por consumidor*,
não do sistema inteiro.

### A armadilha que custou uma hora

Ser **Owner da assinatura não dá acesso aos dados** de um blob. São dois planos
de permissão independentes:

| Plano | Endpoint | Governa | Papéis |
| --- | --- | --- | --- |
| Controle | `management.azure.com` | o recurso | Owner, Contributor, Reader |
| Dados | `<conta>.blob.core.windows.net` | o conteúdo | `Storage Blob Data *` |

Observado na prática: a leitura das propriedades do container respondeu **200**,
e a gravação de um blob respondeu **403** com `AuthorizationPermissionMismatch` —
na mesma execução, com a mesma identidade. A primeira operação está sob
`containers/read`, que Owner possui; a segunda sob `containers/blobs/write`, que
só os papéis de dados possuem.

Consequência contraintuitiva: dar **Contributor** a um colega o autoriza a
**apagar a conta de armazenamento inteira** e não o autoriza a baixar uma foto.
`Storage Blob Data Reader`, que soa mais fraco, é o que serve.

O portal disfarça isso porque oferece autenticação por chave de acesso como
alternativa no seletor da listagem — Owner pode ler as chaves, e assim contorna
o RBAC por outro caminho. Por isso o script traduz o 403 nessa causa em vez de
propagar a mensagem original, que não indica o problema.

A atribuição de papel leva alguns minutos para propagar.

### Convenção de nomes de blob

```
{dispositivo}/{ano}/{mes}/{carimbo}.jpg
estufa-01/2026/09/20260906T210637-305211.jpg
```

Três propriedades buscadas:

- **Ordenável lexicograficamente** — a ordem alfabética é a ordem cronológica
- **Listável por prefixo** — recuperar um mês de uma estufa sem varrer o
  container inteiro
- **Chave natural** para a tabela `deteccoes`

A terceira é a que mais importa no futuro: o Event Grid entrega *pelo menos uma
vez*, então o mesmo blob pode ser processado duas vezes. Sem uma chave estável,
isso vira linha duplicada no banco.

O carimbo inclui microssegundos para que duas capturas no mesmo segundo não
colidam.

### O script de verificação

`desenvolvimento/scripts/verificar_blob.py` exercita o ciclo completo — conferir
o container, enviar, listar por prefixo, baixar, **comparar os bytes** e remover.

A comparação existe porque subir sem erro não prova que o conteúdo chegou
íntegro.

O valor real do script, porém, está no **diagnóstico**: ele traduz falha de
credencial e 403 em ação concreta. As mensagens originais do Azure não indicam
a causa, e o objetivo era descobrir problema de permissão isoladamente — não
dentro de uma Azure Function, onde a mesma falha se confundiria com erro de
empacotamento ou de gatilho.

---

## 3. Containerização

### Por que container

O Azure Container Apps **só aceita imagem de container** — não existe a opção de
enviar a pasta e pedir que ele instale as dependências. A alternativa seria
outro serviço de hospedagem, e nenhum servia (ver Seção 4).

### Decisões do `Dockerfile`

| Decisão | Motivo |
| --- | --- |
| `python:3.13-slim` | mesma versão do ambiente de desenvolvimento |
| PyTorch do índice `download.pytorch.org/whl/cpu` | o pacote do PyPI embute bibliotecas CUDA — mais de 2 GB inúteis num serviço que roda em CPU |
| PyTorch instalado **antes** do `requirements.txt` | invertendo a ordem, a resolução do `torch>=2.13.0` iria ao PyPI e traria a variante CUDA |
| `libgl1` e `libglib2.0-0` via apt | o Ultralytics depende do `opencv-python`, que exige bibliotecas de vídeo do sistema mesmo sem interface gráfica |
| `requirements.txt` copiado antes do código | camada em cache: alterar código não reinstala o PyTorch |
| `COPY` seletivo de `api/`, `src/` e pesos | só o que a API importa em execução |
| Ambos os pesos (`.pt` e `.onnx`) | 10 MB somados; permite trocar por variável de ambiente sem reconstruir |
| `.pt` como padrão | mantém o Grad-CAM operante, que o `.onnx` não suporta |
| `YOLO_CONFIG_DIR=/app/.ultralytics` | sem isso o Ultralytics tenta a home do usuário sem privilégios, cai para `/tmp` e avisa a cada inicialização |
| Usuário `alface` sem privilégios | um comprometimento da aplicação não herda root |
| `PYTHONUNBUFFERED=1` | sem isso o log do uvicorn fica retido em buffer e chega aos blocos no painel |

A armadilha do `libgl1` merece destaque: **a ausência dela não quebra a build.**
Quebra o `import` em tempo de execução. Passa em `docker build` e falha em
`docker run` com `libGL.so.1: cannot open shared object file`.

### O `.dockerignore`

Sem ele o Docker envia a pasta inteira como contexto de build **antes** de
executar a primeira instrução — incluindo `desenvolvimento/dataset/`, com cerca
de 966 MB. Os `COPY` seletivos do `Dockerfile` só agem depois disso, e portanto
não evitam o custo.

Também exclui `.env`, que jamais pode entrar numa imagem publicada em registro.

### Resultado medido

| | Valor |
| --- | ---: |
| Tamanho em disco | 2,66 GB |
| **Tamanho compactado** | **615 MB** |
| Carregamento do modelo dentro do container | ~8 s |

O número que importa é o compactado: é o que trafega no push e no pull. Para uma
imagem com PyTorch, 615 MB é bom — sem o índice CPU-only seriam vários GB.

Imagem `v2` (26/10/2026, com o verificador de domínio e o Grad-CAM corrigido):
**2,63 GB em disco e 614 MB compactada**. Praticamente o mesmo tamanho: o
verificador só acrescenta os pesos dele.

**Inicialização a frio no Azure, `v2`:** **11,27 s** na primeira chamada ao
`/health` com zero réplicas, numa **única medida** (26/10/2026). A variação
entre medidas não é conhecida.

---

## 4. Hospedagem — Azure Container Apps

### Alternativas descartadas

| Opção | Por que não |
| --- | --- |
| **App Service F1** (gratuito) | 1 GB de RAM e 1 GB de disco. As dependências estouram o disco antes do modelo entrar; o PyTorch carregado não cabe na memória. No Linux, o F1 também não aceita container próprio |
| **App Service B1** | ~US$13/mês, sempre ligado — cerca de US$156 no ano, contra US$100 de crédito |
| **Container Apps com `--min-replicas 1`** | o container nunca dorme; sai da faixa gratuita |

Sobre o F1 vale um registro extra: **haveria** um caminho em que ele funcionaria
— falar com o `onnxruntime` diretamente, sem Ultralytics nem PyTorch, cabendo
folgadamente em 1 GB. Foi descartado porque exigiria reescrever
`src/ia_alface/inferencia.py`, que é o núcleo do módulo e o objeto do trabalho.
O F1 não economizaria dinheiro em relação à escala a zero; cobraria em
refatoração do código que o TCC avalia.

### A economia da escala a zero

**A cobrança é por tempo acordado, não por requisição.** Essa é a frase que
resume tudo, e ela desfaz a intuição de que "1 milhão de chamadas gratuitas"
resolve o problema — aquele número é de outro serviço (API Management), que
sequer participa deste desenho.

Convertendo a cota gratuita mensal para o que interessa:

| Configuração | Tempo acordado grátis/mês |
| --- | ---: |
| 0,5 vCPU / 1 GiB | ~100 h |
| 1 vCPU / 2 GiB | ~50 h |

Dentro dessa janela, mil requisições em dez minutos custam o mesmo que uma.

### O limiar contraintuitivo

| Frequência de uso | Comportamento | Custo |
| --- | --- | --- |
| 1×/dia | dorme quase sempre | gratuito |
| 1×/hora | acorda 24×/dia | no limite da cota |
| 1×/5 min | **nunca chega a dormir** | equivale a sempre-ligado |

Quanto mais frequente o uso, **menos** a escala a zero compensa. A partir de
alguns minutos entre chamadas, o container permanece acordado e passa a custar
como um serviço fixo, com preço pior.

O que estoura a cota não é volume: é **polling**. Um `setInterval` de 5 segundos
no dashboard consultando `/health` impede o container de dormir enquanto a aba
estiver aberta. Uma aba esquecida da noite para o dia consome dias de cota.

Mitigação de projeto, não de infraestrutura: **o dashboard não depende desta
API.** Ele lê as detecções do PostgreSQL. Só duas situações justificam chamá-la —
`/info`, uma vez por carregamento e cacheável, e `/analyze` em envio manual.

### Registro de container: GHCR, não ACR

O Azure Container Registry Basic custa cerca de US$5/mês — mais da metade do
crédito anual de estudante para guardar um arquivo. O GitHub Container Registry
é gratuito para imagens públicas e o Container Apps puxa dele normalmente.

Como os pesos já são distribuídos publicamente por Release do GitHub, publicar a
imagem não expõe nada novo.

**O pacote precisa ser marcado como público** após o primeiro push — pacotes do
GHCR nascem privados, e o erro de *image pull* resultante não indica a causa.
Visibilidade de pacote é independente da visibilidade do repositório.

### O que foi implantado

| Item | Valor |
| --- | --- |
| URL pública | `https://ia-alface.proudwater-1f30045f.brazilsouth.azurecontainerapps.io` |
| Grupo de recursos | `TCC-imagem` — o mesmo da conta de armazenamento |
| Ambiente | `ia-alface-env`, modo `ConsumptionOnly` |
| Workspace de logs | `workspace-imagemY8N1` (Log Analytics, criado junto) |
| Imagem | `ghcr.io/thi-barros/ia-alface:v2` desde 26/10/2026, com o verificador de domínio (`docs/VERIFICADOR_DOMINIO.md`). Antes, `v1` |
| Réplicas | mínimo 0, máximo 2; `cooldownPeriod` de 300 s |
| Recursos | 1 vCPU / 2 GiB |
| `IA_ALFACE_CHAVE_API` | segredo `chave-api`, referenciado por `secretref:` |
| `IA_ALFACE_CORS_ORIGENS` | `http://localhost:5173` |

O ambiente foi criado com `--environment-mode ConsumptionOnly` porque o padrão
atual do CLI passou a ser *WorkloadProfiles*, que traz perfis dedicados e, em
algumas regiões, exige subnet própria — semântica de cobrança diferente da que a
escala a zero desta seção pressupõe.

A chave entrou como **segredo do Container App**, não como variável em texto
claro. O `az containerapp show` devolve `secretRef: chave-api` e o valor não
aparece no JSON do recurso, que qualquer pessoa com papel de leitura consegue
ler.

O `cooldownPeriod` de 300 s é o que materializa a escala a zero: cinco minutos
sem requisição e a última réplica é removida. `az containerapp replica list`
devolvendo lista vazia é a confirmação de que o serviço está dormindo — e,
portanto, de que não está consumindo cota.

Tempo de inicialização a frio da `v2`: 11,27 s, numa única medida (Seção 3).
Latência das duas etapas na nuvem: 22,62 ms de mediana (Seção 10).

---

## 5. Autenticação da API

### O desenho

Chave no cabeçalho `X-API-Key`, exigida por `POST /analyze` e `/analyze/raw`
**somente quando `IA_ALFACE_CHAVE_API` está definida**.

### Por que ativada por variável, e não sempre

Quatro consequências, todas desejadas:

1. Os 95 testes existentes continuaram passando **sem alteração**
2. Desenvolvimento local segue sem atrito — `uvicorn --reload` e `/teste`
   funcionam como sempre
3. A página `/teste` não precisa de campo para colar chave
4. A proteção liga exatamente onde é necessária: na implantação com ingress
   público

### Por que uma chave compartilhada basta

Porque **nenhum navegador chama esta API**. Os consumidores são a Azure Function
e o back-end — ambos servidores, que guardam segredo com segurança. Num
front-end a chave seria inútil: qualquer visitante a leria no DevTools.

Se o dashboard precisasse chamar diretamente, o desenho teria de ser outro
(login de usuário, JWT), e a complexidade não se pagaria nesta escala.

### Por que só as rotas de inferência

| Rota | Protegida | Motivo |
| --- | :---: | --- |
| `/analyze`, `/analyze/raw` | sim | consomem CPU e provocam inicialização a frio |
| `/health` | não | a sonda do orquestrador precisa alcançá-la |
| `/info` | não | metadados estáticos, custo desprezível |
| `/docs`, `/teste` | não | poder demonstrar o Swagger ao vivo vale mais que o risco |

### Detalhe de implementação

A comparação usa `secrets.compare_digest`, de tempo constante. Comparar com `==`
permitiria descobrir a chave caractere a caractere medindo o tempo das
respostas — a comparação de strings do Python sai no primeiro caractere
diferente.

`APIKeyHeader` é declarado com `auto_error=False` para que a ausência do
cabeçalho chegue à função de verificação como `None`, permitindo devolver 401
com mensagem em português em vez do 403 genérico do FastAPI.

### Limites conhecidos

- **Não identifica quem chamou.** É um segredo só; qualquer portador é aceito
- **Não expira** — vale até ser trocada
- **Não é revogável por chamador** — trocar derruba todos os consumidores
- **Depende de HTTPS**, que o Container Apps fornece por padrão

Ela responde a *"quem chama conhece o segredo?"*, não a *"quem é esse
chamador?"*. Para dois consumidores servidor a servidor, é proporcional.

### Obrigações disparadas

Alterar `api/` obrigou a rodar `pytest` (101 passando) e, por mudar o esquema,
`gerar_openapi.py`. O contrato em `docs/openapi.json` agora declara
`securitySchemes: APIKeyHeader` e o 401 nas duas rotas.

---

## 6. CORS — armadilha registrada, não corrigida

A função `_lista` de `api/configuracao.py` devolve o padrão quando a variável
está ausente **ou vazia**:

```python
bruto = os.environ.get(nome)
if bruto is None or not bruto.strip():
    return padrao      # ["*"]
```

Definir `IA_ALFACE_CORS_ORIGENS=""` na intenção de "não permitir nada" produz o
oposto: cai no padrão e libera tudo. **Valor vazio é mais permissivo que variável
ausente.**

**Decisão: documentar, não corrigir.** A função é compartilhada por outras
configurações, e mudar sua semântica teria alcance maior que o problema. A
orientação de passar valor explícito está registrada em `CLAUDE.md`,
`api/README.md` e `.env.exemplo`.

---

## 7. Planejado e não construído

### Azure Function com gatilho de blob

Decisões já tomadas, para não se perderem:

- **Variante Event Grid, obrigatoriamente.** O gatilho de blob padrão faz
  varredura periódica e pode atrasar **até 10 minutos** conforme o container
  cresce. Funciona nos primeiros testes e degrada depois, o que torna o
  diagnóstico difícil
- **400 é falha definitiva.** A API devolve 400 para imagem corrompida, por
  decisão de projeto. Repetir a requisição jamais funcionaria — registre e siga.
  Retry apenas para 503 e timeout
- **Idempotência pelo nome do blob**, porque o Event Grid entrega ao menos uma
  vez
- **Nunca gravar no mesmo container** que dispara o gatilho: heatmap ou
  resultado gravados ali provocariam recursão infinita
- **Identidade gerenciada** com `Storage Blob Data Reader` — mesma lógica de
  RBAC da Seção 2, agora para a identidade da Function

### Decisões ainda em aberto

1. **Onde o código da Function mora.** Ela é back-end da aplicação, não módulo de
   IA; colocá-la na raiz deste repositório contrariaria a regra de que a raiz
   contém só produção do módulo. Repositório separado é a recomendação
2. **Quem escreve em `deteccoes`** — a Function diretamente, ou um endpoint do
   back-end. Duas aplicações escrevendo na mesma tabela costuma dar problema
3. **O DDL da tabela**, que está com outro integrante. Atenção especial à
   grafia dos valores do ENUM: precisam bater exatamente com `bacteriano`,
   `fungico` e `saudavel`, senão grava dado errado sem erro

---

## 8. Efeito sobre os requisitos

**RNF01 continua atendido.** O requisito, conforme `spec.md`, é *tempo de
inferência por imagem* inferior a 100 ms — medido em 61,81 ms com `.pt` e
12,73 ms com ONNX. A latência acrescentada pelo Event Grid, pela inicialização a
frio e pela rede fica **fora dessa fronteira de medição**.

Isso precisa constar explicitamente do relatório. Sem a ressalva, a distância
entre um requisito de 100 ms e um alerta que chega segundos depois parece
contradição. O `spec.md` já recomenda cuidado equivalente ao comparar o servidor
de inferência com dispositivos de borda.

O fluxo assíncrono muda o comportamento observável em relação ao projeto
original e é candidato a nova entrada em `DIVERGENCIAS.md` quando a Function for
construída.

---

## 9. Índice de armadilhas

Todas custaram tempo nesta etapa.

| Situação | O que acontece | O que fazer |
| --- | --- | --- |
| Owner sem papel de dados | 403 `AuthorizationPermissionMismatch` na escrita, 200 na leitura do container | atribuir `Storage Blob Data Contributor` |
| Papel recém-atribuído | ainda 403 | esperar alguns minutos pela propagação |
| Região recusada | `RequestDisallowedByAzure` | restrição da assinatura; escolher outra região |
| Grupo de recursos com mesmo nome | "localização inválida" | selecionar o existente em vez de "Create new"; a região do grupo é independente da do recurso |
| `libgl1` ausente | build passa, container quebra ao subir | instalar as bibliotecas de sistema |
| PyTorch do PyPI | ~2 GB de CUDA inúteis | índice CPU-only, antes do `requirements.txt` |
| Sem `.dockerignore` | 966 MB de dataset no contexto de build | excluir `desenvolvimento/` |
| Tag de imagem reaproveitada | Azure serve a versão antiga em cache | incrementar sempre (`v1`, `v2`, …) |
| Nome de imagem com maiúscula | `repository name must be lowercase` | usar minúsculas no caminho |
| Pacote GHCR privado | falha de *image pull* no Azure | tornar o pacote público |
| Token fine-grained no GHCR | `does not match expected scopes` | usar token clássico com `write:packages` |
| `IA_ALFACE_CORS_ORIGENS=""` | libera tudo | informar as origens explicitamente |
| Valor com `:` no `--env-vars` do `containerapp create` | a variável é criada **sem valor**, sem erro — e valor vazio cai no `*` da linha acima | conferir com `az containerapp show --query "properties.template.containers[0].env"` e corrigir pelo `update` |
| Aspas simples para escapar esse valor | entram literalmente no valor gravado (`'http://…'`), fechando o CORS para todos | no `update` o texto é gravado literal: passar sem aspas simples |
| Polling do front-end | container nunca dorme, cota esgota | não fazer polling; cachear `/info` |
| `curl` no PowerShell | é apelido de `Invoke-WebRequest`, sintaxe incompatível | usar `curl.exe` |
| `<<EOF` no cmd.exe | heredoc não existe | usar `git commit -F arquivo` |
| WSL sem distribuição | Docker Desktop não inicia | habilitar o recurso do Windows e **reiniciar** |
| Docker Desktop parado | `failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine` | abrir o Docker Desktop e esperar o motor subir |
| `az` sem MFA | `RequestDisallowedByAzure` no `containerapp update`; consultas funcionam | `az logout` + `az login`, confirmando o segundo fator |
| Parênteses no `--query` do `az` pelo PowerShell | `invalid jmespath_type value: 'length(@'` | consultar sem parênteses: `--query "[].name" -o tsv` |
| `denied` no `docker manifest inspect` do GHCR | não diz se a tag existe ou se a credencial venceu | `docker login ghcr.io` com token clássico; tag inexistente passa a responder `manifest unknown` |

---

## 10. O que foi medido e o que foi estimado

Seguindo a regra 4 do `CLAUDE.md`, a distinção fica explícita.

**Medido:**

| Item | Valor |
| --- | ---: |
| Imagem compactada | 615 MB |
| Imagem em disco | 2,66 GB |
| Testes automatizados | 101, em ~26 s |
| Carregamento do modelo no container | ~8 s |
| Primeira inferência (com aquecimento) | 307 ms |
| Inferências seguintes | ~25 ms |
| Ciclo completo no Blob Storage | envio, listagem, download e integridade conferidos |
| Imagem `v2` compactada / em disco | 614 MB / 2,63 GB |
| Duas etapas na nuvem (`v2`), mediana / p95 | 22,62 ms / 45,15 ms |
| HTTP ponta a ponta de um cliente doméstico, mediana / p95 | 104,85 ms / 505,87 ms |
| Inicialização a frio (`v2`) | 11,27 s (uma medida) |

As medidas da `v2` são de 26/10/2026 e estão registradas em
`docs/VERIFICADOR_DOMINIO.md`. Foram tiradas de um loop no PowerShell, sem
relatório em `reports/`.

A diferença entre 307 ms e 25 ms é aquecimento — alocação de buffers e
otimização interna do PyTorch na primeira passada. É por isso que o
`05_benchmark_latencia.py` descarta 10 inferências antes de medir. O valor
válido para o RNF01 é o de regime.

**Não medido, apenas estimado:**

- Variação do tempo de inicialização a frio (só há uma medida)
- Custo mensal efetivo
- Latência ponta a ponta com Event Grid e Function

Nenhum desses deve ser citado como resultado no TCC sem medição.
