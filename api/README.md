# API de Inferência — Detecção de Doenças Foliares em Alface

Serviço HTTP que expõe o classificador YOLO11-cls treinado neste trabalho.
Permite que o back-end da aplicação, o dashboard e o módulo ESP32-CAM consumam a
IA sem instalar PyTorch nem conhecer detalhes do modelo.

## Escopo

Este serviço faz **uma coisa**: recebe uma imagem de folha e devolve a
classificação.

| Faz | Não faz |
| --- | --- |
| Classifica em `saudavel` / `bacteriano` / `fungico` | Persiste em banco de dados |
| Devolve confiança e probabilidades | Publica ou consome MQTT |
| Indica se o alerta deve disparar | Envia notificação push |
| Gera mapa de calor Grad-CAM (opcional) | Autentica usuários |

A persistência na tabela `deteccoes`, o disparo do push (RF07) e o consumo das
leituras dos sensores são responsabilidade do back-end da aplicação, que consome
esta API.

---

## Como executar

Pré-requisito: o arquivo `modelos/yolo11n_cls_alface.pt` deve existir. Caso
contrário, o serviço sobe mas responde `503` nas rotas de inferência.

```bash
pip install -r requirements.txt
```

Em desenvolvimento, com recarga automática:

```bash
python -m uvicorn api.main:app --reload
```

Acessível a outras máquinas da rede (necessário para o ESP32):

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Com o serviço no ar, a documentação interativa fica em <http://localhost:8000/docs>.
Ela é gerada automaticamente a partir do código e permite testar os endpoints
pelo navegador — é o ponto de partida recomendado para quem for integrar.

Em container, localmente ou em Azure Container Apps:

```bash
docker run --rm -p 8000:8000 ia-alface:local
```

Construção da imagem, publicação e ciclo de atualização em
[`../docs/IMPLANTACAO.md`](../docs/IMPLANTACAO.md).

---

## Testar com suas próprias imagens

Há duas formas prontas, sem escrever código.

### Página web (recomendada)

Com o serviço no ar, abra <http://localhost:8000/teste>.

Arraste fotos para a área indicada — ou clique para escolher — e veja a classe,
a confiança, as três probabilidades e se o alerta dispararia. Marque
*Gerar mapa de calor* para visualizar a região da folha que pesou na decisão.
Cada resultado traz o JSON devolvido pela API, útil para quem estiver integrando.

A página é servida pela própria API, na mesma origem, portanto não há
configuração de CORS nem endereço a ajustar. Aceita várias imagens de uma vez.

### Cliente de linha de comando

Para lotes e medições:

```bash
python api/cliente.py minhas_fotos/
```

```bash
python api/cliente.py foto.jpg --mapa
```

```bash
python api/cliente.py minhas_fotos/ --csv resultados.csv
```

```bash
python api/cliente.py foto.jpg --bruto
```

```bash
python api/cliente.py foto.jpg --url http://192.168.0.100:8000
```

| Opção | Efeito |
| --- | --- |
| `--url` | Endereço da API (padrão `http://localhost:8000`) |
| `--mapa` | Solicita o Grad-CAM e grava os PNG em `mapas_de_calor/` |
| `--bruto` | Usa `/analyze/raw`, a mesma rota do ESP32-CAM |
| `--csv` | Grava uma planilha com todos os resultados |
| `--json` | Imprime as respostas cruas |

O cliente verifica `/health` antes de enviar qualquer imagem, para falhar cedo
com mensagem útil caso o serviço esteja fora do ar. Usa apenas a biblioteca
padrão do Python — não requer `requests`.

> Diferença em relação a `desenvolvimento/scripts/testar_imagem.py`: aquele carrega o modelo
> diretamente na memória; este exercita o **caminho completo de integração**,
> montando a requisição HTTP e interpretando a resposta. É o que valida se o
> serviço está funcionando ponta a ponta.

---

## Endpoints

| Método | Rota | Finalidade |
| --- | --- | --- |
| `GET` | `/teste` | Página de teste interativo |
| `GET` | `/health` | Verificar se o modelo está carregado |
| `GET` | `/info` | Metadados do modelo e lista de classes |
| `POST` | `/analyze` | Classificar imagem enviada como `multipart/form-data` |
| `POST` | `/analyze/raw` | Classificar imagem enviada como corpo bruto |
| `GET` | `/docs` | Documentação interativa (Swagger UI) |
| `GET` | `/redoc` | Mesma documentação em formato de leitura (ReDoc) |
| `GET` | `/openapi.json` | Contrato OpenAPI, para gerar clientes |

Ambas as rotas de inferência aceitam o parâmetro de consulta `?heatmap=true`,
que acrescenta o mapa de calor Grad-CAM à resposta.

### Por que duas rotas de inferência

`/analyze` usa `multipart/form-data`, o formato natural de formulários web e de
clientes HTTP comuns. Use no dashboard e em testes.

`/analyze/raw` recebe os bytes da imagem diretamente no corpo. É a rota indicada
para o **ESP32-CAM**: montar um envelope multipart em microcontrolador consome
memória e tempo, enquanto enviar o buffer JPEG cru é uma única chamada.

---

## Documentação do contrato

Com o serviço no ar, o Swagger UI fica em `/docs` — nele é possível enviar uma
imagem e ver a resposta sem escrever código. O mesmo contrato, em formato de
leitura, fica em `/redoc`, e o arquivo bruto em `/openapi.json`.

Uma cópia estática vive em `docs/`, para consultar sem subir o serviço e para
anexar ao TCC:

| Arquivo | Uso |
| --- | --- |
| [`docs/swagger.html`](../docs/swagger.html) | Swagger UI autocontido; abre com duplo clique |
| [`docs/openapi.json`](../docs/openapi.json) | Contrato para Postman, Insomnia ou geradores de cliente |

Os dois são gerados a partir do código — nunca editados à mão:

```bash
python desenvolvimento/scripts/gerar_openapi.py
```

Rode o comando após alterar rotas ou esquemas, senão a cópia estática diverge
do serviço. A página estática grava `http://localhost:8000` como servidor, de
modo que o botão *Try it out* funcione com o serviço rodando localmente; use
`--servidor` para apontar para outro endereço.

---

## Contrato de resposta

```json
{
  "classe": "fungico",
  "classe_exibicao": "Fúngico (Bremia lactucae)",
  "confianca": 0.956,
  "probabilidades": {
    "bacteriano": 0.021,
    "fungico": 0.956,
    "saudavel": 0.023
  },
  "patogeno_detectado": true,
  "alerta": true,
  "tempo_inferencia_ms": 40.6,
  "detectado_em": "2026-08-02T22:01:37.067680+00:00",
  "modelo_versao": "yolo11n_cls_alface.pt",
  "bbox_x": null,
  "bbox_y": null,
  "bbox_w": null,
  "bbox_h": null,
  "heatmap_base64": null
}
```

### Mapeamento para a tabela `deteccoes`

Os campos correspondem um a um às colunas especificadas no projeto:

| Campo da resposta | Coluna | Observação |
| --- | --- | --- |
| `classe` | `classe` | Valores idênticos ao `ENUM` do banco |
| `confianca` | `confianca` | `NUMERIC(4,3)`, faixa [0, 1] |
| `detectado_em` | `detectado_em` | ISO 8601 em UTC |
| `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h` | `bbox_*` | **Sempre nulo** — ver abaixo |
| — | `estufa_id` | Definido pelo back-end |
| — | `imagem_url` | Definido pelo back-end, após armazenar a imagem |

> **Atenção ao schema.** O modelo é de classificação de imagem inteira e não
> produz caixas delimitadoras. As colunas `bbox_*` precisam ser declaradas
> `NULLABLE`. A justificativa completa está em
> [`../docs/DIVERGENCIAS.md`](../docs/DIVERGENCIAS.md), divergências D1 e D2.

### O campo `alerta`

`alerta` é verdadeiro somente quando **há patógeno E a confiança atinge o limiar**
(0,85 por padrão). É o gatilho da notificação push do RF07.

| Situação | `patogeno_detectado` | `alerta` |
| --- | :---: | :---: |
| `saudavel`, confiança 0,99 | falso | falso |
| `fungico`, confiança 0,95 | verdadeiro | **verdadeiro** |
| `fungico`, confiança 0,60 | verdadeiro | falso |

Use `alerta` para decidir o disparo da notificação; use `patogeno_detectado`
para registrar a ocorrência no histórico, ainda que sem notificar.

---

## Códigos de status

| Código | Significado | Ação do cliente |
| ---: | --- | --- |
| `200` | Classificação concluída | Processar a resposta |
| `400` | Imagem corrompida, truncada, vazia ou em formato não suportado | Recapturar a imagem |
| `413` | Imagem acima do limite configurado | Reduzir a resolução no ESP32 |
| `422` | Campo obrigatório ausente | Corrigir a requisição |
| `503` | Modelo indisponível | Verificar `/health`; tentar novamente |
| `500` | Falha inesperada | Consultar o log do serviço |

O erro `400` é deliberado: uma imagem ilegível **nunca** produz um resultado.
Classificar silenciosamente um JPEG truncado como folha saudável suprimiria o
alerta sem qualquer indicação de anomalia — a pior falha possível deste sistema.

---

## Exemplos de consumo

### cURL

```bash
curl -X POST -F "imagem=@folha.jpg" http://localhost:8000/analyze
```

### Python (back-end da aplicação)

```python
import requests

with open("folha.jpg", "rb") as arquivo:
    resposta = requests.post(
        "http://localhost:8000/analyze",
        files={"imagem": arquivo},
        timeout=10,
    )

resposta.raise_for_status()
deteccao = resposta.json()

if deteccao["alerta"]:
    # persistir em 'deteccoes' e acionar a notificação push (RF07)
    ...
```

### JavaScript (dashboard React)

```javascript
const dados = new FormData();
dados.append("imagem", arquivo);

const resposta = await fetch("http://localhost:8000/analyze", {
  method: "POST",
  body: dados,
});

if (!resposta.ok) {
  const { detail } = await resposta.json();
  throw new Error(detail);
}

const deteccao = await resposta.json();
```

Para exibir o mapa de calor, use `?heatmap=true` e monte a imagem assim:

```javascript
<img src={`data:image/png;base64,${deteccao.heatmap_base64}`} />
```

### ESP32-CAM (Arduino, C++)

```cpp
#include <HTTPClient.h>
#include "esp_camera.h"

const char* SERVIDOR = "http://192.168.0.100:8000/analyze/raw";

void enviarImagem() {
  camera_fb_t* quadro = esp_camera_fb_get();
  if (!quadro) { Serial.println("Falha ao capturar"); return; }

  HTTPClient http;
  http.begin(SERVIDOR);
  http.addHeader("Content-Type", "image/jpeg");
  http.setTimeout(10000);

  int codigo = http.POST(quadro->buf, quadro->len);

  if (codigo == 200) {
    Serial.println(http.getString());   // JSON da deteccao
  } else {
    Serial.printf("Erro HTTP %d\n", codigo);
  }

  http.end();
  esp_camera_fb_return(quadro);
}
```

Substitua `192.168.0.100` pelo IP da máquina que executa o serviço. O servidor
precisa ter sido iniciado com `--host 0.0.0.0` para aceitar conexões externas.

---

## Configuração

Todas as opções têm padrão utilizável; o serviço sobe sem nenhuma variável
definida.

| Variável | Padrão | Descrição |
| --- | --- | --- |
| `IA_ALFACE_MODELO` | `modelos/yolo11n_cls_alface.pt` | Pesos a carregar |
| `IA_ALFACE_DISPOSITIVO` | autodetecção | `cuda` ou `cpu` |
| `IA_ALFACE_LIMIAR` | `0.85` | Confiança mínima para `alerta` |
| `IA_ALFACE_MAX_MB` | `10` | Limite do corpo da requisição |
| `IA_ALFACE_CORS_ORIGENS` | `*` | Origens autorizadas, separadas por vírgula |
| `IA_ALFACE_CHAVE_API` | vazio | Chave exigida em `X-API-Key`. Vazia desativa a autenticação |

Exemplo de produção:

```bash
IA_ALFACE_MODELO=modelos/yolo11n_cls_alface.onnx IA_ALFACE_CORS_ORIGENS=https://estufa.exemplo.br IA_ALFACE_CHAVE_API=<chave> python -m uvicorn api.main:app --host 0.0.0.0
```

> **Antes de implantar publicamente**, restrinja `IA_ALFACE_CORS_ORIGENS` às
> origens do dashboard. O padrão `*` é adequado apenas em desenvolvimento.
>
> Atenção: definir a variável como texto **vazio** não restringe nada — o valor
> em branco cai no padrão `*`, mais permissivo que não defini-la. Informe as
> origens explicitamente.

### Autenticação

Enquanto `IA_ALFACE_CHAVE_API` não é definida, o serviço aceita qualquer
chamada — comportamento adequado ao desenvolvimento local e à página `/teste`.
Definida a variável, `POST /analyze` e `POST /analyze/raw` passam a exigir o
cabeçalho:

```bash
curl -X POST https://<host>/analyze/raw -H "Content-Type: image/jpeg" -H "X-API-Key: <chave>" --data-binary @folha.jpg
```

`GET /health` e `GET /info` permanecem abertos: a sonda do orquestrador precisa
alcançar o primeiro, e nenhum dos dois executa inferência.

A chave destina-se a consumidores **servidor a servidor** — a Function e o
back-end. Não a coloque num front-end: qualquer visitante a leria no navegador.
O dashboard consulta o back-end, não este serviço.

### Escolha do formato do modelo

| Formato | Latência | Grad-CAM | Uso recomendado |
| --- | ---: | :---: | --- |
| `.pt` | ~40 ms | sim | Desenvolvimento; quando o heatmap for necessário |
| `.onnx` | ~30 ms | não | Produção, se o heatmap não for usado |

---

## Desempenho

Medição no ambiente descrito em [`../docs/AMBIENTE.md`](../docs/AMBIENTE.md)
(AMD Ryzen 5 3500U, CPU), 30 requisições após 10 de aquecimento:

| Métrica | Valor |
| --- | ---: |
| Inferência (campo `tempo_inferencia_ms`) | 40,6 ms |
| Ponta a ponta HTTP (127.0.0.1) | 53,3 ms |
| Sobrecarga do HTTP | 12,7 ms |

Ambos os valores atendem ao requisito RNF01 (< 100 ms). O número ponta a ponta é
o mais representativo da experiência real do ESP32, ao qual se soma a latência
da rede Wi-Fi local.

Duas decisões de implementação sustentam esse desempenho: o modelo é carregado
**uma única vez** na inicialização, e a inferência roda em *pool* de threads, de
modo que o laço de eventos permaneça livre para aceitar conexões e responder ao
`/health` enquanto uma imagem é processada.

---

## Testes

```bash
python -m pytest desenvolvimento/tests/test_api.py -v
```

São 15 testes que cobrem as rotas de operação, ambos os caminhos de inferência,
a lógica de alerta e o tratamento de imagem corrompida, truncada, vazia e acima
do limite. Não dependem de pesos treinados — o modelo é substituído por um duplo
de teste.
