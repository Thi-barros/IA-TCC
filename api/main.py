"""Serviço de inferência — API REST para consumo do modelo de IA.

Expõe o classificador de doenças foliares em alface por HTTP, para que o
back-end da aplicação, o dashboard e o módulo ESP32-CAM possam consumi-lo sem
dependerem do PyTorch nem conhecerem detalhes do modelo.

**Escopo deliberadamente restrito à IA.** Este serviço não persiste dados, não
fala MQTT e não envia notificações. Ele recebe uma imagem e devolve a
classificação. A persistência na tabela ``deteccoes``, o disparo do push (RF07) e
o consumo das leituras dos sensores são responsabilidade do back-end da
aplicação, que consome esta API.

O modelo é carregado **uma única vez**, na inicialização, e reutilizado entre
requisições — carregá-lo a cada chamada inviabilizaria o requisito RNF01.

**Verificador de domínio (opcional).** Com ``IA_ALFACE_MODELO_DOMINIO``
definida, a inferência passa a ser em cascata: imagens que não contêm alface
recebem 200 com ``classe = "nao_alface"``, sem diagnóstico nem alerta (decisão
de 25/10/2026, ``docs/VERIFICADOR_DOMINIO.md``). Sem a variável, o serviço se
comporta exatamente como antes.

Execução:
    uvicorn api.main:app --reload
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import base64
import io
import logging
import secrets
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    Security,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.security import APIKeyHeader

from ia_alface.cascata import AnalisadorCascata, ImagemForaDoDominio, ResultadoCascata
from ia_alface.classes import CLASSE_NAO_ALFACE, CLASSES, EXIBICAO_NAO_ALFACE
from ia_alface.inferencia import DetectorDoencasAlface
from ia_alface.verificacao import ResultadoVerificacao, VerificadorDominio

from .configuracao import CONFIGURACAO
from .esquemas import (
    ClasseInfo,
    InfoVerificacao,
    RespostaDeteccao,
    RespostaErro,
    RespostaInfo,
    RespostaSaude,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api")

#: Estado do serviço. Preenchido na inicialização.
_estado: dict[str, Any] = {
    "detector": None,
    "verificador": None,
    "analisador": None,
    "explicador": None,
    "erro_carregamento": None,
    "erro_verificador": None,
}


# --------------------------------------------------------------------------- #
# Ciclo de vida
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def ciclo_de_vida(_app: FastAPI):
    """Carrega o modelo na subida e o libera no encerramento.

    Uma falha de carregamento não derruba o serviço: ela é registrada e passa a
    ser reportada por ``/health``, o que facilita o diagnóstico em implantação.
    """
    logger.info("Carregando modelo de %s", CONFIGURACAO.caminho_modelo)
    try:
        _estado["detector"] = DetectorDoencasAlface(
            caminho_modelo=CONFIGURACAO.caminho_modelo,
            dispositivo=CONFIGURACAO.dispositivo,
            limiar_confianca=CONFIGURACAO.limiar_confianca,
        )
        logger.info(
            "Modelo pronto: %s | dispositivo=%s | limiar=%.2f",
            _estado["detector"].modelo_versao,
            _estado["detector"].dispositivo,
            _estado["detector"].limiar_confianca,
        )
    except Exception as erro:  # noqa: BLE001 - o serviço deve subir mesmo assim
        _estado["erro_carregamento"] = str(erro)
        logger.error("Falha ao carregar o modelo: %s", erro)
        logger.error(
            "O serviço subirá, mas /analyze responderá 503 até que o modelo "
            "esteja disponível. Treine com desenvolvimento/scripts/02_treinar.py."
        )

    if CONFIGURACAO.verifica_dominio:
        logger.info(
            "Carregando verificador de domínio de %s", CONFIGURACAO.caminho_modelo_dominio
        )
        try:
            _estado["verificador"] = VerificadorDominio(
                caminho_modelo=CONFIGURACAO.caminho_modelo_dominio,
                dispositivo=CONFIGURACAO.dispositivo,
                limiar_alface=CONFIGURACAO.limiar_dominio,
            )
            logger.info(
                "Verificador pronto: %s | limiar=%.2f",
                _estado["verificador"].modelo_versao,
                _estado["verificador"].limiar_alface,
            )
        except Exception as erro:  # noqa: BLE001 - o serviço deve subir mesmo assim
            _estado["erro_verificador"] = str(erro)
            logger.error("Falha ao carregar o verificador de domínio: %s", erro)
            logger.error(
                "IA_ALFACE_MODELO_DOMINIO está definida: /analyze responderá 503 "
                "até o verificador carregar. Seguir sem ele daria diagnóstico a "
                "imagens sem alface."
            )

    # A cascata só é montada quando está completa. Com o verificador
    # configurado e não carregado, _obter_analisador responde 503.
    verificador_ok = not CONFIGURACAO.verifica_dominio or _estado["verificador"] is not None
    if _estado["detector"] is not None and verificador_ok:
        _estado["analisador"] = AnalisadorCascata(_estado["detector"], _estado["verificador"])

    yield

    _estado["detector"] = None
    _estado["verificador"] = None
    _estado["analisador"] = None
    _estado["explicador"] = None
    logger.info("Serviço encerrado.")


# --------------------------------------------------------------------------- #
# Metadados do contrato OpenAPI
# --------------------------------------------------------------------------- #

#: Descrição dos grupos de rotas, exibida no Swagger UI e no contrato exportado.
TAGS_OPENAPI: list[dict[str, str]] = [
    {
        "name": "Inferência",
        "description": (
            "Classificação de uma imagem de folha. As duas rotas devolvem o "
            "mesmo corpo e diferem apenas na forma de envio: `/analyze` recebe "
            "um `multipart/form-data`, usado pelo dashboard e pelos testes "
            "manuais; `/analyze/raw` recebe os bytes crus, forma indicada para "
            "o ESP32-CAM.\n\nAmbas aceitam `?heatmap=true`, que acrescenta o "
            "mapa de calor Grad-CAM à resposta — disponível somente quando o "
            "serviço executa pesos `.pt`.\n\nCom o verificador de domínio "
            "ativo (`/info` → `verificacao_dominio`), imagens sem alface "
            "recebem **200** com `classe = \"nao_alface\"`, `alerta` e "
            "`patogeno_detectado` falsos e `probabilidades` vazio: o "
            "classificador não é executado. Trate essa resposta como "
            "definitiva: reenviar a mesma imagem produz o mesmo resultado."
        ),
    },
    {
        "name": "Operação",
        "description": (
            "Disponibilidade do serviço e metadados do modelo em execução. "
            "Consuma `/info` para obter as classes reconhecidas em vez de "
            "duplicá-las no cliente."
        ),
    },
]


def _erro(descricao: str, exemplo: str) -> dict[str, Any]:
    """Declara uma resposta de erro para o contrato OpenAPI.

    Args:
        descricao: Texto exibido ao lado do código de status.
        exemplo: Valor do campo ``detail`` mostrado como exemplo.

    Returns:
        Dicionário no formato aceito pelo parâmetro ``responses`` do FastAPI.
    """
    return {
        "model": RespostaErro,
        "description": descricao,
        "content": {"application/json": {"example": {"detail": exemplo}}},
    }


#: Erros das rotas de inferência. Declarados uma única vez para que as duas
#: descrevam exatamente o mesmo contrato de falha.
RESPOSTAS_DE_ERRO: dict[int | str, dict[str, Any]] = {
    400: _erro(
        "Imagem ausente, corrompida ou em formato não suportado",
        "Não foi possível decodificar a imagem recebida em bytes.",
    ),
    401: _erro(
        "Chave de API ausente ou inválida",
        "Chave de API ausente ou inválida no cabeçalho X-API-Key.",
    ),
    413: _erro(
        "Imagem acima do limite configurado",
        "Imagem de 14.2 MB excede o limite de 10 MB.",
    ),
    503: _erro(
        "Modelo indisponível",
        "Modelo indisponível. Arquivo de pesos não encontrado.",
    ),
}


app = FastAPI(
    title=CONFIGURACAO.titulo,
    summary=(
        "Classifica imagens de folha de alface em saudável, bacteriana ou "
        "fúngica."
    ),
    description=(
        "Serviço de classificação de doenças foliares em alface "
        "(*Lactuca sativa*) baseado em Ultralytics YOLO11-cls.\n\n"
        "Classifica uma imagem de folha em **saudável**, **bacteriana** "
        "(*Xanthomonas campestris*) ou **fúngica** (*Bremia lactucae*).\n\n"
        "Escopo restrito à inferência: não persiste dados, não fala MQTT e não "
        "envia notificações."
    ),
    version="1.0.0",
    openapi_tags=TAGS_OPENAPI,
    # `displayRequestDuration` mostra o tempo de cada chamada no próprio Swagger
    # UI, o que permite conferir o RNF01 sem ferramenta externa. Nenhum `servers`
    # é declarado: sem ele o Swagger UI usa a origem em que está aberto, e a
    # documentação continua funcionando em qualquer host ou porta.
    swagger_ui_parameters={
        "docExpansion": "list",
        "displayRequestDuration": True,
        "defaultModelsExpandDepth": 2,
    },
    lifespan=ciclo_de_vida,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CONFIGURACAO.cors_origens,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def registrar_requisicao(request: Request, call_next):
    """Registra método, rota, status e duração de cada requisição."""
    inicio = time.perf_counter()
    resposta = await call_next(request)
    duracao_ms = (time.perf_counter() - inicio) * 1000
    logger.info(
        "%s %s -> %d (%.1f ms)",
        request.method,
        request.url.path,
        resposta.status_code,
        duracao_ms,
    )
    return resposta


# --------------------------------------------------------------------------- #
# Auxiliares
# --------------------------------------------------------------------------- #


#: Cabeçalho que transporta a chave de API. ``auto_error=False`` faz a ausência
#: chegar à verificação como ``None``, permitindo devolver 401 com mensagem em
#: português em vez do 403 genérico do FastAPI.
_CHAVE_NO_CABECALHO = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verificar_chave(chave: str | None = Security(_CHAVE_NO_CABECALHO)) -> None:
    """Exige a chave de API nas rotas de inferência, quando configurada.

    Sem ``IA_ALFACE_CHAVE_API`` definida a verificação é inerte, e o serviço
    responde como sempre respondeu. A proteção destina-se à implantação, onde o
    ingress é público: sem ela, qualquer origem consome CPU e, sob escala a
    zero, provoca inicializações que consomem a cota gratuita.

    A comparação usa ``secrets.compare_digest``, de tempo constante. Comparar com
    ``==`` permitiria inferir a chave medindo o tempo das respostas.

    Args:
        chave: Conteúdo do cabeçalho ``X-API-Key``, ausente como ``None``.

    Raises:
        HTTPException: 401 quando a chave é exigida e não confere.
    """
    if not CONFIGURACAO.exige_autenticacao:
        return
    if chave is None or not secrets.compare_digest(
        chave.encode("utf-8"), CONFIGURACAO.chave_api.encode("utf-8")
    ):
        raise HTTPException(
            status_code=401,
            detail="Chave de API ausente ou inválida no cabeçalho X-API-Key.",
        )


def _obter_detector() -> DetectorDoencasAlface:
    """Devolve o detector carregado ou interrompe com 503."""
    detector = _estado["detector"]
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Modelo indisponível. "
                f"{_estado['erro_carregamento'] or 'Carregamento não concluído.'}"
            ),
        )
    return detector


def _obter_analisador() -> AnalisadorCascata:
    """Devolve a cascata pronta ou interrompe com 503.

    Com o verificador configurado e não carregado, a inferência é recusada em
    vez de seguir sem ele: imagens sem alface receberiam diagnóstico.
    """
    _obter_detector()
    if CONFIGURACAO.verifica_dominio and _estado["verificador"] is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Verificador de domínio indisponível. "
                f"{_estado['erro_verificador'] or 'Carregamento não concluído.'}"
            ),
        )
    return _estado["analisador"]


def _validar_tamanho(dados: bytes) -> None:
    """Rejeita corpos acima do limite configurado."""
    if len(dados) == 0:
        raise HTTPException(
            status_code=400,
            detail="Corpo da requisição vazio: nenhuma imagem foi recebida.",
        )
    if len(dados) > CONFIGURACAO.tamanho_maximo_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Imagem de {len(dados) / 1e6:.1f} MB excede o limite de "
                f"{CONFIGURACAO.tamanho_maximo_mb:.0f} MB."
            ),
        )


def _gerar_heatmap(dados: bytes) -> str:
    """Gera o mapa de calor Grad-CAM e o devolve em base64.

    O explicador é construído sob demanda, na primeira solicitação, para não
    duplicar o modelo em memória quando o recurso não é utilizado. O mapa cobre
    só a região que a rede analisa (o recorte central); as faixas descartadas
    ficam sem cor.
    """
    if not CONFIGURACAO.suporta_heatmap:
        raise HTTPException(
            status_code=400,
            detail=(
                "O mapa de calor exige pesos no formato .pt. O serviço está "
                f"executando {CONFIGURACAO.caminho_modelo.name}."
            ),
        )

    if _estado["explicador"] is None:
        from ia_alface.explicabilidade import GradCAM

        logger.info("Inicializando Grad-CAM sob demanda.")
        _estado["explicador"] = GradCAM(
            CONFIGURACAO.caminho_modelo, dispositivo=CONFIGURACAO.dispositivo
        )

    from PIL import Image

    with Image.open(io.BytesIO(dados)) as aberta:
        original = aberta.convert("RGB").copy()

    sobreposta, _classe, _confianca = _estado["explicador"].sobreposicao(original)

    buffer = io.BytesIO()
    sobreposta.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _montar_resposta(
    resultado: ResultadoCascata, heatmap: str | None
) -> RespostaDeteccao:
    """Converte o resultado da biblioteca no esquema da API."""
    deteccao = resultado.deteccao
    verificacao = None
    if resultado.verificacao is not None:
        verificacao = InfoVerificacao(
            probabilidade_alface=resultado.verificacao.probabilidade_alface,
            limiar=resultado.verificacao.limiar,
            tempo_inferencia_ms=resultado.verificacao.tempo_inferencia_ms,
            modelo_versao=resultado.verificacao.modelo_versao,
        )
    return RespostaDeteccao(
        classe=deteccao.classe,
        classe_exibicao=deteccao.classe_exibicao,
        confianca=deteccao.confianca,
        probabilidades=deteccao.probabilidades,
        patogeno_detectado=deteccao.patogeno_detectado,
        alerta=deteccao.alerta,
        tempo_inferencia_ms=deteccao.tempo_inferencia_ms,
        detectado_em=datetime.now(timezone.utc).isoformat(),
        modelo_versao=deteccao.modelo_versao,
        heatmap_base64=heatmap,
        verificacao=verificacao,
    )


def _montar_resposta_sem_alface(verificacao: ResultadoVerificacao) -> RespostaDeteccao:
    """Monta a resposta 200 de uma imagem que o verificador não reconheceu como alface.

    O classificador não chegou a ser executado: ``probabilidades`` fica vazio,
    ``tempo_inferencia_ms`` é zero e ``patogeno_detectado`` e ``alerta`` são
    falsos — nenhum consumidor pode tirar dela um diagnóstico.
    """
    return RespostaDeteccao(
        classe=CLASSE_NAO_ALFACE,
        classe_exibicao=EXIBICAO_NAO_ALFACE,
        confianca=round(1.0 - verificacao.probabilidade_alface, 3),
        probabilidades={},
        patogeno_detectado=False,
        alerta=False,
        tempo_inferencia_ms=0.0,
        detectado_em=datetime.now(timezone.utc).isoformat(),
        modelo_versao=verificacao.modelo_versao,
        verificacao=InfoVerificacao(
            probabilidade_alface=verificacao.probabilidade_alface,
            limiar=verificacao.limiar,
            tempo_inferencia_ms=verificacao.tempo_inferencia_ms,
            modelo_versao=verificacao.modelo_versao,
        ),
    )


async def _analisar(dados: bytes, com_heatmap: bool) -> RespostaDeteccao:
    """Executa a inferência sobre os bytes recebidos.

    A inferência é uma operação de CPU que bloqueia por dezenas de milissegundos.
    Executá-la diretamente no laço de eventos travaria o servidor inteiro pelo
    período — inclusive as respostas de ``/health``. Por isso o trabalho pesado é
    delegado ao pool de threads, deixando o laço livre para aceitar conexões.
    O acesso a cada modelo permanece serializado pelo *lock* interno dele.
    """
    analisador = _obter_analisador()
    _validar_tamanho(dados)

    return await run_in_threadpool(_analisar_sincrono, analisador, dados, com_heatmap)


def _analisar_sincrono(
    analisador: AnalisadorCascata, dados: bytes, com_heatmap: bool
) -> RespostaDeteccao:
    """Parte bloqueante da análise, executada fora do laço de eventos.

    A recusa do verificador vira resposta 200 com ``classe = "nao_alface"``;
    imagem inválida continua sendo 400, nunca "não alface". Nenhum mapa de
    calor é gerado para imagem sem alface.
    """

    try:
        resultado = analisador.analisar(dados)
    except ValueError as erro:
        # Imagem corrompida, truncada ou em formato não suportado. Jamais
        # devolver um resultado nesse caso: um falso 'saudavel' suprimiria o
        # alerta sem qualquer indicação de anomalia.
        raise HTTPException(status_code=400, detail=str(erro)) from erro
    except ImagemForaDoDominio as recusa:
        return _montar_resposta_sem_alface(recusa.verificacao)

    heatmap = _gerar_heatmap(dados) if com_heatmap else None

    if resultado.deteccao.alerta:
        logger.warning(
            "ALERTA: %s detectado com confiança %.1f%%",
            resultado.deteccao.classe,
            resultado.deteccao.confianca * 100,
        )

    return _montar_resposta(resultado, heatmap)


# --------------------------------------------------------------------------- #
# Rotas
# --------------------------------------------------------------------------- #


@app.get("/", include_in_schema=False)
async def raiz():
    """Redireciona para a página de teste interativo."""
    return RedirectResponse(url="/teste")


@app.get(
    "/teste",
    include_in_schema=False,
    summary="Página de teste interativo",
)
async def pagina_de_teste():
    """Serve a página de teste manual.

    A página é entregue pela própria API para que as chamadas ocorram na mesma
    origem, dispensando qualquer configuração de CORS no navegador.
    """
    pagina = Path(__file__).resolve().parent / "cliente_web.html"
    if not pagina.exists():
        raise HTTPException(
            status_code=404, detail=f"Página de teste não encontrada: {pagina.name}"
        )
    return FileResponse(pagina, media_type="text/html; charset=utf-8")


@app.get(
    "/health",
    response_model=RespostaSaude,
    summary="Verificar disponibilidade do serviço",
    tags=["Operação"],
    response_description="Estado do serviço",
)
async def saude() -> RespostaSaude:
    """Informa se o modelo está carregado e pronto para inferência.

    Adequado para *health check* de orquestradores e balanceadores.
    """
    detector = _estado["detector"]
    verificador = _estado["verificador"]

    verificador_carregado = None
    if CONFIGURACAO.verifica_dominio:
        verificador_carregado = verificador is not None
    modelo_dominio = verificador.modelo_versao if verificador is not None else None

    if detector is None:
        return RespostaSaude(
            status="erro",
            modelo_carregado=False,
            verificador_carregado=verificador_carregado,
            modelo_dominio_versao=modelo_dominio,
            detalhe=_estado["erro_carregamento"] or "Modelo não carregado.",
        )
    if verificador_carregado is False:
        return RespostaSaude(
            status="erro",
            modelo_carregado=True,
            modelo_versao=detector.modelo_versao,
            dispositivo=detector.dispositivo,
            verificador_carregado=False,
            detalhe=(
                "Verificador de domínio não carregado: "
                f"{_estado['erro_verificador'] or 'carregamento não concluído.'}"
            ),
        )
    return RespostaSaude(
        status="ok",
        modelo_carregado=True,
        modelo_versao=detector.modelo_versao,
        dispositivo=detector.dispositivo,
        verificador_carregado=verificador_carregado,
        modelo_dominio_versao=modelo_dominio,
    )


@app.get(
    "/info",
    response_model=RespostaInfo,
    summary="Consultar metadados do modelo",
    tags=["Operação"],
    response_description="Metadados do modelo e classes reconhecidas",
    responses={503: RESPOSTAS_DE_ERRO[503]},
)
async def info() -> RespostaInfo:
    """Devolve as classes reconhecidas e os parâmetros em vigor.

    Útil para o dashboard montar legendas e rótulos sem duplicar as definições
    de classe no front-end.
    """
    detector = _obter_detector()
    verificador = _estado["verificador"]
    return RespostaInfo(
        modelo_versao=detector.modelo_versao,
        dispositivo=detector.dispositivo,
        imgsz=detector.imgsz,
        limiar_confianca=detector.limiar_confianca,
        suporta_heatmap=CONFIGURACAO.suporta_heatmap,
        tamanho_maximo_mb=CONFIGURACAO.tamanho_maximo_mb,
        verificacao_dominio=CONFIGURACAO.verifica_dominio,
        modelo_dominio_versao=verificador.modelo_versao if verificador else None,
        limiar_dominio=verificador.limiar_alface if verificador else None,
        classes=[
            ClasseInfo(
                indice=classe.indice,
                canonico=classe.canonico,
                exibicao=classe.exibicao,
                patogeno=classe.patogeno_cientifico is not None,
            )
            for classe in CLASSES
        ],
    )


@app.post(
    "/analyze",
    response_model=RespostaDeteccao,
    summary="Classificar imagem enviada como formulário",
    tags=["Inferência"],
    response_description="Classificação da folha",
    responses=RESPOSTAS_DE_ERRO,
    dependencies=[Depends(verificar_chave)],
)
async def analisar(
    imagem: UploadFile = File(
        ..., description="Arquivo de imagem da folha (JPEG ou PNG)."
    ),
    heatmap: bool = Query(
        False, description="Incluir o mapa de calor Grad-CAM na resposta."
    ),
) -> RespostaDeteccao:
    """Classifica uma imagem enviada como `multipart/form-data`.

    Endpoint indicado para o dashboard web, para testes manuais e para clientes
    que já trabalham com formulários. Dispositivos embarcados devem preferir
    `/analyze/raw`, que dispensa a montagem do envelope multipart.
    """
    return await _analisar(await imagem.read(), heatmap)


@app.post(
    "/analyze/raw",
    response_model=RespostaDeteccao,
    summary="Classificar imagem enviada como corpo bruto",
    tags=["Inferência"],
    response_description="Classificação da folha",
    responses=RESPOSTAS_DE_ERRO,
    dependencies=[Depends(verificar_chave)],
    # O corpo é lido de `request.body()`, que o FastAPI não consegue inspecionar.
    # Sem esta declaração o contrato OpenAPI descreveria a rota do ESP32 como se
    # não recebesse corpo algum, e o /docs não ofereceria campo para enviar.
    openapi_extra={
        "requestBody": {
            "required": True,
            "description": "Bytes da imagem, sem envelope multipart.",
            "content": {
                "image/jpeg": {"schema": {"type": "string", "format": "binary"}},
                "image/png": {"schema": {"type": "string", "format": "binary"}},
            },
        }
    },
)
async def analisar_bruto(
    request: Request,
    heatmap: bool = Query(
        False, description="Incluir o mapa de calor Grad-CAM na resposta."
    ),
) -> RespostaDeteccao:
    """Classifica os bytes de imagem enviados diretamente no corpo.

    Endpoint destinado ao ESP32-CAM: o microcontrolador envia o JPEG capturado
    com `Content-Type: image/jpeg`, sem o custo de montar um envelope
    `multipart/form-data`, que consome memória e tempo em hardware limitado.
    """
    return await _analisar(await request.body(), heatmap)


@app.exception_handler(Exception)
async def tratar_erro_inesperado(request: Request, erro: Exception):
    """Converte falhas não previstas em resposta 500 padronizada."""
    logger.exception("Erro não tratado em %s: %s", request.url.path, erro)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Erro interno no serviço de inferência: {erro}"},
    )
