# Implantação da API em Azure Container Apps

Como publicar o serviço de inferência na nuvem e por que a configuração é a que
é. As restrições da Seção 4 não são preferências — desrespeitá-las esgota a cota
gratuita mensal em poucos dias.

---

## 1. Topologia

```
Câmera Logitech C270 ──(envio ao Blob: a definir pelo grupo)──► Blob Storage
                              │ evento
                              ▼
                         Event Grid
                              │
                              ▼
                       Azure Function ──POST /analyze/raw──► API (Container App)
                              │                              verificador → classificador
                              │◄────────── JSON classificação ──────┘
                              ▼
                       PostgreSQL + MQTT
```

A API mantém o escopo declarado em `api/main.py`: recebe bytes, devolve
classificação. Não persiste, não fala MQTT, não conhece o Azure. Quem orquestra
é a Function.

A câmera do projeto é a Logitech C270, e não o ESP32-CAM do texto original do
TCC (`docs/DIVERGENCIAS.md`, D8). O caminho da câmera até o Blob ainda não foi
definido pelo grupo e não deve ser presumido.

Com o verificador de domínio ativo, a Function trata a resposta com
`classe = "nao_alface"` como **resultado final**: não faz retry e não grava
diagnóstico (Seção 5).

---

## 2. Construir e testar localmente

Sempre a partir da raiz do projeto:

```bash
docker build -t ia-alface:local .
docker run --rm -p 8000:8000 ia-alface:local
```

Com o container no ar, confira em outro terminal:

```bash
curl http://localhost:8000/health
```

A resposta deve trazer `"modelo_carregado": true`. Se vier `"status": "erro"`, o
modelo não foi encontrado — confira se `modelos/yolo11n_cls_alface.pt` existe no
disco antes da build (o arquivo está no `.gitignore` e é distribuído por Release
do GitHub).

Com o verificador de domínio configurado, o `/health` também deve trazer
`"verificador_carregado": true`. Se vier `false`, com `"status": "erro"`, os
pesos do verificador (`modelos/yolo11n_cls_dominio_v2.pt`) não foram
encontrados, e `/analyze` responde 503 em vez de diagnosticar sem ele. Sem
`IA_ALFACE_MODELO_DOMINIO`, o campo vem `null` e a API funciona como antes.

Vale testar também `http://localhost:8000/teste` com uma foto real. Depurar aqui
é muito mais barato do que depurar um deploy silencioso.

---

## 3. Publicar

O registro recomendado é o **GitHub Container Registry**, gratuito para imagens
públicas. O Azure Container Registry Basic custa cerca de US$5/mês — mais da
metade do crédito Azure for Students em um ano, para guardar um arquivo.

```bash
docker build -t ghcr.io/<usuario>/ia-alface:v1 .
docker push ghcr.io/<usuario>/ia-alface:v1
```

```bash
az containerapp create \
  --name ia-alface \
  --resource-group <grupo> \
  --image ghcr.io/<usuario>/ia-alface:v1 \
  --target-port 8000 \
  --ingress external \
  --min-replicas 0 \
  --max-replicas 2 \
  --cpu 1 --memory 2Gi
```

O comando devolve a URL pública do serviço. É ela que a Function consome.

### Publicar alterações

O ciclo de atualização, depois do serviço no ar:

```bash
python -m pytest -q
```

```bash
docker build -t ghcr.io/<usuario>/ia-alface:v2 .
```

```bash
docker push ghcr.io/<usuario>/ia-alface:v2
```

```bash
az containerapp update --name ia-alface --resource-group <grupo> --image ghcr.io/<usuario>/ia-alface:v2
```

A URL não muda. Leva alguns segundos.

**MFA obrigatório para alterar recursos.** Se o `update` falhar com
`RequestDisallowedByAzure` citando MFA, a sessão do `az` foi aberta sem o
segundo fator. Rode `az logout` e `az login` de novo e confirme o MFA no
navegador. Consultas (`show`, `list`) continuam funcionando sem ele, o que
engana. Visto em 26/10/2026.

**Sempre incremente a tag.** Reconstruir e republicar `:v1` pode fazer o Azure
continuar servindo a imagem antiga em cache — falha difícil de diagnosticar,
porque tudo indica sucesso.

Cada atualização cria uma **revisão**, e as anteriores continuam disponíveis:

```bash
az containerapp revision list --name ia-alface --resource-group <grupo> -o table
```

Se uma versão quebrar, reativar a anterior não exige reconstruir nada.

### Desenvolvimento não precisa de Docker

Para iterar no código, continue com o servidor de desenvolvimento:

```bash
python -m uvicorn api.main:app --reload
```

Recarrega a cada arquivo salvo. O container só entra quando for publicar —
construir imagem a cada alteração de uma linha desperdiça minutos por ciclo.

---

## 4. Escala a zero — a restrição que decide a conta

A cobrança do Container Apps é por **tempo acordado**, não por requisição. A
cota gratuita mensal, convertida:

| Configuração | Tempo acordado grátis/mês | Por dia |
| --- | ---: | ---: |
| 0,5 vCPU / 1 GiB | ~100 h | ~3,3 h |
| 1 vCPU / 2 GiB | ~50 h | ~1,6 h |

Dentro dessa janela o número de chamadas é irrelevante: mil requisições em dez
minutos custam o mesmo que uma.

**O que estoura a cota é polling, não volume.** Um `setInterval` de 5 segundos
no dashboard consultando `/health` impede o container de dormir enquanto a aba
estiver aberta. Uma aba esquecida da noite para o dia consome dias de cota.

Regras:

1. **Nunca faça polling** de `/health` ou `/info` a partir do front-end
2. **Cacheie `/info`** — as classes só mudam com novo deploy
3. **Não aponte monitoramento externo** (UptimeRobot e similares) para a URL:
   mantém o container acordado 24 h por dia
4. **Mantenha `--min-replicas 0`**. Com `1`, o container nunca dorme e o custo
   sai da faixa gratuita
5. Configure um **alerta de orçamento** de US$5 na assinatura

Para conferir se o serviço dormiu, sem acordá-lo, consulte as réplicas pelo
`az`. No PowerShell, `--query "length(@)"` chega truncado ao `az`, que é um
`.cmd`; use a lista de nomes, em que saída vazia quer dizer zero réplicas:

```bash
az containerapp replica list --name ia-alface --resource-group TCC-imagem --query "[].name" -o tsv
```

Note a inversão: quanto mais frequente o uso, menos a escala a zero compensa. A
partir de uma chamada a cada poucos minutos o container deixa de dormir e passa
a custar como um serviço sempre ligado.

O dashboard **não depende desta API** para operar. Ele lê as detecções do
PostgreSQL. Só duas situações justificam chamá-la: `/info`, uma vez por
carregamento, e `/analyze`, em envio manual de foto.

---

## 5. Configuração

| Variável | Padrão | Em produção |
| --- | --- | --- |
| `IA_ALFACE_MODELO` | `.pt` (definido na imagem) | `.onnx` reduz a latência mediana, mas desativa o Grad-CAM |
| `IA_ALFACE_CORS_ORIGENS` | `*` | **defina as origens do dashboard** |
| `IA_ALFACE_LIMIAR` | `0.85` | confira se casa com o esperado pelo back-end |
| `IA_ALFACE_MAX_MB` | `10` | — |
| `IA_ALFACE_MODELO_DOMINIO` | ausente = **sem verificador** | `/app/modelos/yolo11n_cls_dominio_v2.pt` |
| `IA_ALFACE_LIMIAR_DOMINIO` | `0.5` | `0.50`, fixado no `val`. Não mude sem uma nova validação |

Trocar o modelo não exige reconstruir a imagem: os dois arquivos de pesos já
estão dentro dela. Os pesos do verificador entram na imagem a partir da `v2`
(26/10/2026), por `COPY` no `Dockerfile`; a `v1` não os tem.

### Resposta com o verificador ativo

A imagem sem alface **não** gera erro. Ela recebe **200** com
`classe = "nao_alface"`, `classe_exibicao = "Alface não encontrada"`,
`confianca = 1 − probabilidade_alface`, `probabilidades = {}` e
`patogeno_detectado` e `alerta` falsos. O bloco `verificacao` vem preenchido.

- O consumidor decide pelo `alerta`, **nunca** por "classe ≠ `saudavel`". Esse
  teste transformaria a recusa em alerta falso.
- O `ENUM` da coluna `classe` do banco ainda não aceita `nao_alface` (D8). Até
  o grupo do banco decidir, essas respostas não são gravadas.
- O 422 significa só requisição malformada, e o 400, imagem corrompida. Com o
  verificador configurado mas não carregado, a API responde 503.

O verificador v2 **não atingiu** o critério de implantação fixado antes do
teste. A implantação é decisão do autor, posterior ao resultado
(`docs/VERIFICADOR_DOMINIO.md`, Seções 1 e 4.8).

---

## 6. Autenticação

`POST /analyze` e `POST /analyze/raw` exigem a chave no cabeçalho `X-API-Key`
**quando `IA_ALFACE_CHAVE_API` está definida**. Sem a variável a verificação é
inerte, e o serviço com `--ingress external` aceita chamada de qualquer origem —
consumindo CPU e, sob escala a zero, provocando inicializações que gastam a cota.

**Defina a variável ao implantar:**

```bash
az containerapp update --name ia-alface --resource-group <grupo> --set-env-vars IA_ALFACE_CHAVE_API=<chave> IA_ALFACE_CORS_ORIGENS=https://localhost
```

A Function passa a enviar o cabeçalho. Ela roda no servidor e guarda o segredo
com segurança — o que um front-end não faria: qualquer visitante leria a chave no
navegador. Por isso o dashboard consulta o back-end, não este serviço.

`GET /health` e `GET /info` permanecem abertos: a sonda do orquestrador precisa
alcançar o primeiro, e nenhum dos dois executa inferência.

Se um dia for necessário fechar por completo, a alternativa é **ingress interno**
com a Function no mesmo ambiente do Container Apps — ao custo de perder o acesso
externo ao `/docs`.

Alterar `api/` obriga a rodar `python -m pytest -q`. Se a mudança tocar rota ou
esquema, rode também `python desenvolvimento/scripts/gerar_openapi.py`, senão a
cópia estática do contrato em `docs/` passa a mentir sobre o serviço.

---

## 7. Efeito sobre os requisitos

**RNF01 permanece atendido.** O requisito, conforme `spec.md`, é *tempo de
inferência por imagem* inferior a 100 ms — medido em 61,81 ms com `.pt` e
12,73 ms com ONNX. A latência acrescentada pelo Event Grid, pelo cold start e
pela rede está **fora dessa fronteira** de medição.

Com o verificador de domínio, a mediana das duas etapas (verificador +
classificador) foi de **51,85 ms** (`.pt`, CPU local;
`desenvolvimento/reports/verificador_v2_test.md`), também abaixo de 100 ms.

**Na nuvem** (imagem `v2`, 26/10/2026; 100 imagens do `val`, 10 de
aquecimento descartadas; registro em `docs/VERIFICADOR_DOMINIO.md`):

| Medida | Mediana | p95 |
| --- | ---: | ---: |
| Duas etapas, no servidor | 22,62 ms | 45,15 ms |
| HTTP ponta a ponta, de um cliente doméstico | 104,85 ms | 505,87 ms |
| Inicialização a frio (primeira chamada com zero réplicas) | 11,27 s (uma medida) | — |

A inicialização a frio é o preço da escala a zero: a primeira análise depois
de um período ocioso leva cerca de 11 s, e as seguintes voltam ao regime
normal.

Isso precisa constar explicitamente do relatório. Sem a ressalva, a diferença
entre o requisito de 100 ms e um alerta que chega segundos depois parece
contradição. O mesmo cuidado que `spec.md` já recomenda ao comparar o servidor
de inferência com dispositivos de borda vale aqui.

---

## 8. Custos

No regime de uma análise diária mais uso humano esparso do dashboard:

| Componente | Custo |
| --- | --- |
| Blob Storage | frações de centavo |
| Event Grid | zero — 100.000 operações/mês grátis |
| Azure Functions | zero — 1 milhão de execuções/mês grátis |
| Container Apps | zero — dentro da cota, com `--min-replicas 0` |
| GitHub Container Registry | zero para imagem pública |

O que geraria custo real: Azure Container Registry Basic, App Service B1,
`--min-replicas 1`, ou qualquer máquina virtual. Nenhum é necessário.

A assinatura Azure for Students não tem cartão vinculado: esgotado o crédito, os
recursos são desativados, nunca faturados.
