"""Testes do serviço de inferência HTTP.

Como nos demais testes, o modelo Ultralytics é substituído por um duplo, de modo
que a suíte não dependa de pesos treinados e execute em segundos.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from ia_alface.classes import (
    CLASSE_NAO_ALFACE,
    EXIBICAO_NAO_ALFACE,
    ROTULOS_CANONICOS,
    ROTULOS_DOMINIO,
)

# --------------------------------------------------------------------------- #
# Duplo de teste
# --------------------------------------------------------------------------- #


class _ProbsFalso:
    def __init__(self, vetor: list[float]) -> None:
        self.data = torch.tensor(vetor, dtype=torch.float32)


class _ResultadoFalso:
    def __init__(self, vetor: list[float]) -> None:
        self.probs = _ProbsFalso(vetor)


class _YOLOFalso:
    """Duplo de ``ultralytics.YOLO`` para a tarefa de classificação."""

    vetor = [0.02, 0.94, 0.04]  # bacteriano, fungico, saudavel
    task = "classify"
    names = dict(enumerate(ROTULOS_CANONICOS))

    def __init__(self, caminho: str, task: str | None = None, **_kwargs) -> None:
        self.caminho = caminho
        if task:
            self.task = task

    def predict(self, entrada, **_kwargs):
        quantidade = len(entrada) if isinstance(entrada, list) else 1
        return [_ResultadoFalso(list(self.vetor)) for _ in range(quantidade)]


class _YOLOCascataFalso(_YOLOFalso):
    """Responde como verificador quando o arquivo de pesos é o de domínio."""

    vetor_dominio = [0.95, 0.05]  # alface, nao_alface

    def __init__(self, caminho: str, task: str | None = None, **kwargs) -> None:
        super().__init__(caminho, task, **kwargs)
        self.eh_verificador = "dominio" in Path(caminho).name
        if self.eh_verificador:
            self.names = dict(enumerate(ROTULOS_DOMINIO))

    def predict(self, entrada, **kwargs):
        if self.eh_verificador:
            return [_ResultadoFalso(list(self.vetor_dominio))]
        return super().predict(entrada, **kwargs)


def _jpeg(largura: int = 160, altura: int = 120) -> bytes:
    """Gera bytes de um JPEG válido, como os enviados pelo ESP32-CAM."""
    vetor = np.random.default_rng(7).integers(
        0, 255, (altura, largura, 3), dtype=np.uint8
    )
    buffer = io.BytesIO()
    Image.fromarray(vetor).save(buffer, format="JPEG")
    return buffer.getvalue()


#: Chave usada pelo cliente autenticado dos testes.
CHAVE_DE_TESTE = "chave-de-teste-123"


def _subir_aplicacao(
    monkeypatch,
    tmp_path,
    chave_api: str = "",
    modelo_dominio: Path | None = None,
    limiar_dominio: str | None = None,
) -> TestClient:
    """Prepara o ambiente e devolve o cliente da aplicação recarregada.

    Args:
        chave_api: Quando informada, sobe o serviço com autenticação ativa.
        modelo_dominio: Quando informado, liga o verificador de domínio com
            esses pesos (que podem não existir, para testar a falha de carga).
        limiar_dominio: Valor de ``IA_ALFACE_LIMIAR_DOMINIO``, se informado.
    """
    pesos = tmp_path / "modelo_de_teste.pt"
    pesos.write_bytes(b"")

    monkeypatch.setenv("IA_ALFACE_MODELO", str(pesos))
    monkeypatch.setenv("IA_ALFACE_LIMIAR", "0.85")
    monkeypatch.setenv("IA_ALFACE_MAX_MB", "1")

    # Removidas explicitamente para que uma variável presente no ambiente do
    # desenvolvedor não altere o comportamento da suíte.
    if chave_api:
        monkeypatch.setenv("IA_ALFACE_CHAVE_API", chave_api)
    else:
        monkeypatch.delenv("IA_ALFACE_CHAVE_API", raising=False)

    if modelo_dominio is not None:
        monkeypatch.setenv("IA_ALFACE_MODELO_DOMINIO", str(modelo_dominio))
        monkeypatch.setattr("ultralytics.YOLO", _YOLOCascataFalso)
    else:
        monkeypatch.delenv("IA_ALFACE_MODELO_DOMINIO", raising=False)
        monkeypatch.setattr("ultralytics.YOLO", _YOLOFalso)

    if limiar_dominio is not None:
        monkeypatch.setenv("IA_ALFACE_LIMIAR_DOMINIO", limiar_dominio)
    else:
        monkeypatch.delenv("IA_ALFACE_LIMIAR_DOMINIO", raising=False)

    _YOLOFalso.vetor = [0.02, 0.94, 0.04]
    _YOLOCascataFalso.vetor_dominio = [0.95, 0.05]

    # A configuração é lida na importação; recarregar garante que as variáveis
    # de ambiente acima tenham efeito.
    import importlib

    import api.configuracao
    import api.main

    importlib.reload(api.configuracao)
    importlib.reload(api.main)

    return TestClient(api.main.app)


@pytest.fixture
def cliente(monkeypatch, tmp_path):
    """Sobe a aplicação com o modelo substituído pelo duplo de teste."""
    with _subir_aplicacao(monkeypatch, tmp_path) as cliente_teste:
        yield cliente_teste


@pytest.fixture
def cliente_protegido(monkeypatch, tmp_path):
    """Sobe a aplicação com a autenticação por chave de API ativada."""
    with _subir_aplicacao(
        monkeypatch, tmp_path, chave_api=CHAVE_DE_TESTE
    ) as cliente_teste:
        yield cliente_teste


@pytest.fixture
def cliente_cascata(monkeypatch, tmp_path):
    """Sobe a aplicação com o verificador de domínio ativo."""
    pesos_dominio = tmp_path / "verificador_dominio.pt"
    pesos_dominio.write_bytes(b"")
    with _subir_aplicacao(
        monkeypatch, tmp_path, modelo_dominio=pesos_dominio
    ) as cliente_teste:
        yield cliente_teste


# --------------------------------------------------------------------------- #
# Operação
# --------------------------------------------------------------------------- #


def test_health_reporta_modelo_carregado(cliente):
    """O health check confirma que o modelo subiu."""
    resposta = cliente.get("/health")

    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["status"] == "ok"
    assert corpo["modelo_carregado"] is True
    assert corpo["modelo_versao"] == "modelo_de_teste.pt"


def test_info_lista_as_tres_classes(cliente):
    """O endpoint de metadados expõe as classes para o front-end."""
    corpo = cliente.get("/info").json()

    assert {c["canonico"] for c in corpo["classes"]} == set(ROTULOS_CANONICOS)
    assert corpo["limiar_confianca"] == 0.85
    assert corpo["imgsz"] == 224
    # Apenas 'saudavel' não é patógeno.
    assert sum(1 for c in corpo["classes"] if c["patogeno"]) == 2


def test_raiz_redireciona_para_pagina_de_teste(cliente):
    """A raiz leva à página de teste interativo."""
    resposta = cliente.get("/", follow_redirects=False)

    assert resposta.status_code in (307, 308)
    assert resposta.headers["location"] == "/teste"


def test_pagina_de_teste_e_servida(cliente):
    """A página de teste é entregue pela própria API, evitando CORS."""
    resposta = cliente.get("/teste")

    assert resposta.status_code == 200
    assert "text/html" in resposta.headers["content-type"]
    corpo = resposta.text
    # A página chama a API por caminho relativo, na mesma origem.
    assert 'const API = ""' in corpo
    assert "/analyze" in corpo


def test_documentacao_openapi_disponivel(cliente):
    """O contrato OpenAPI é publicado para os consumidores."""
    esquema = cliente.get("/openapi.json").json()

    assert "/analyze" in esquema["paths"]
    assert "/analyze/raw" in esquema["paths"]
    assert {tag["name"] for tag in esquema["tags"]} == {"Inferência", "Operação"}


def test_openapi_descreve_corpo_bruto(cliente):
    """A rota do ESP32-CAM declara que recebe imagem no corpo.

    O corpo é lido de ``request.body()``, que o FastAPI não inspeciona. Sem a
    declaração explícita o contrato diria que a rota não recebe nada, e quem
    gerasse um cliente a partir dele não conseguiria enviar a imagem.
    """
    esquema = cliente.get("/openapi.json").json()

    corpo = esquema["paths"]["/analyze/raw"]["post"]["requestBody"]

    assert corpo["required"] is True
    assert "image/jpeg" in corpo["content"]


def test_openapi_documenta_erros_da_inferencia(cliente):
    """As duas rotas de inferência descrevem o mesmo contrato de falha."""
    esquema = cliente.get("/openapi.json").json()

    for rota in ("/analyze", "/analyze/raw"):
        respostas = esquema["paths"][rota]["post"]["responses"]
        assert {"400", "413", "503"} <= set(respostas)


# --------------------------------------------------------------------------- #
# Inferência
# --------------------------------------------------------------------------- #


def test_analyze_multipart(cliente):
    """Envio como formulário devolve o contrato completo."""
    resposta = cliente.post(
        "/analyze", files={"imagem": ("folha.jpg", _jpeg(), "image/jpeg")}
    )

    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["classe"] == "fungico"
    assert corpo["classe_exibicao"].startswith("Fúngico")
    assert corpo["patogeno_detectado"] is True
    assert corpo["alerta"] is True
    assert corpo["tempo_inferencia_ms"] > 0
    assert corpo["detectado_em"].endswith("+00:00")
    assert sum(corpo["probabilidades"].values()) == pytest.approx(1.0, abs=1e-4)


def test_analyze_raw_para_esp32(cliente):
    """Envio como corpo bruto, o caminho previsto para o ESP32-CAM."""
    resposta = cliente.post(
        "/analyze/raw",
        content=_jpeg(),
        headers={"Content-Type": "image/jpeg"},
    )

    assert resposta.status_code == 200
    assert resposta.json()["classe"] == "fungico"


def test_bbox_sempre_nulo_na_resposta(cliente):
    """O contrato preserva os campos de caixa, sempre nulos."""
    corpo = cliente.post(
        "/analyze", files={"imagem": ("folha.jpg", _jpeg(), "image/jpeg")}
    ).json()

    for campo in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"):
        assert corpo[campo] is None


def test_folha_saudavel_nao_gera_alerta(cliente):
    """Folha sadia nunca aciona notificação."""
    _YOLOFalso.vetor = [0.01, 0.02, 0.97]

    corpo = cliente.post(
        "/analyze", files={"imagem": ("folha.jpg", _jpeg(), "image/jpeg")}
    ).json()

    assert corpo["classe"] == "saudavel"
    assert corpo["patogeno_detectado"] is False
    assert corpo["alerta"] is False


def test_patogeno_abaixo_do_limiar_nao_gera_alerta(cliente):
    """Confiança insuficiente detecta o patógeno mas não dispara o alerta."""
    _YOLOFalso.vetor = [0.55, 0.30, 0.15]

    corpo = cliente.post(
        "/analyze", files={"imagem": ("folha.jpg", _jpeg(), "image/jpeg")}
    ).json()

    assert corpo["classe"] == "bacteriano"
    assert corpo["patogeno_detectado"] is True
    assert corpo["alerta"] is False


# --------------------------------------------------------------------------- #
# Tratamento de erro
# --------------------------------------------------------------------------- #


def test_imagem_corrompida_retorna_400(cliente):
    """Bytes que não são imagem produzem 400, jamais um resultado.

    Um falso 'saudavel' causado por transmissão truncada suprimiria o alerta sem
    qualquer indicação de anomalia — o pior modo de falha do sistema.
    """
    resposta = cliente.post(
        "/analyze", files={"imagem": ("quebrada.jpg", b"nao sou imagem", "image/jpeg")}
    )

    assert resposta.status_code == 400
    assert "decodificar" in resposta.json()["detail"].lower()


def test_jpeg_truncado_retorna_400(cliente):
    """Transmissão interrompida do ESP32 é rejeitada."""
    completo = _jpeg()
    resposta = cliente.post(
        "/analyze/raw",
        content=completo[: len(completo) // 3],
        headers={"Content-Type": "image/jpeg"},
    )

    assert resposta.status_code == 400


def test_corpo_vazio_retorna_400(cliente):
    """Requisição sem imagem é rejeitada com mensagem clara."""
    resposta = cliente.post(
        "/analyze/raw", content=b"", headers={"Content-Type": "image/jpeg"}
    )

    assert resposta.status_code == 400
    assert "vazio" in resposta.json()["detail"].lower()


def test_imagem_acima_do_limite_retorna_413(cliente):
    """O limite de tamanho protege o serviço (configurado em 1 MB no teste)."""
    resposta = cliente.post(
        "/analyze/raw",
        content=b"\xff" * (2 * 1024 * 1024),
        headers={"Content-Type": "image/jpeg"},
    )

    assert resposta.status_code == 413
    assert "excede" in resposta.json()["detail"].lower()


def test_requisicao_sem_arquivo_retorna_422(cliente):
    """A ausência do campo obrigatório é validada pelo FastAPI."""
    assert cliente.post("/analyze").status_code == 422


def test_modelo_indisponivel_retorna_503(monkeypatch, tmp_path):
    """Se o modelo não carrega, o serviço sobe e reporta 503 na inferência."""
    monkeypatch.setenv("IA_ALFACE_MODELO", str(tmp_path / "inexistente.pt"))

    import importlib

    import api.configuracao
    import api.main

    importlib.reload(api.configuracao)
    importlib.reload(api.main)

    with TestClient(api.main.app) as cliente_teste:
        saude = cliente_teste.get("/health").json()
        assert saude["status"] == "erro"
        assert saude["modelo_carregado"] is False

        resposta = cliente_teste.post(
            "/analyze", files={"imagem": ("folha.jpg", _jpeg(), "image/jpeg")}
        )
        assert resposta.status_code == 503


# --------------------------------------------------------------------------- #
# Autenticação por chave de API
# --------------------------------------------------------------------------- #


def test_sem_chave_configurada_a_inferencia_permanece_aberta(cliente):
    """Sem IA_ALFACE_CHAVE_API o serviço responde como sempre respondeu.

    Garante que ativar o recurso não altera o desenvolvimento local nem exige
    credencial da página /teste.
    """
    resposta = cliente.post(
        "/analyze", files={"imagem": ("folha.jpg", _jpeg(), "image/jpeg")}
    )

    assert resposta.status_code == 200


def test_chave_ausente_e_recusada(cliente_protegido):
    """Com autenticação ativa, requisição sem cabeçalho recebe 401."""
    resposta = cliente_protegido.post(
        "/analyze", files={"imagem": ("folha.jpg", _jpeg(), "image/jpeg")}
    )

    assert resposta.status_code == 401
    assert "X-API-Key" in resposta.json()["detail"]


def test_chave_incorreta_e_recusada(cliente_protegido):
    """Uma chave que não confere não autoriza a inferência."""
    resposta = cliente_protegido.post(
        "/analyze",
        files={"imagem": ("folha.jpg", _jpeg(), "image/jpeg")},
        headers={"X-API-Key": "chave-errada"},
    )

    assert resposta.status_code == 401


def test_chave_correta_autoriza(cliente_protegido):
    """Com a chave certa, a classificação ocorre normalmente."""
    resposta = cliente_protegido.post(
        "/analyze",
        files={"imagem": ("folha.jpg", _jpeg(), "image/jpeg")},
        headers={"X-API-Key": CHAVE_DE_TESTE},
    )

    assert resposta.status_code == 200
    assert resposta.json()["classe"] == "fungico"


def test_rota_bruta_tambem_e_protegida(cliente_protegido):
    """O endpoint do ESP32-CAM exige a mesma chave."""
    sem_chave = cliente_protegido.post(
        "/analyze/raw", content=_jpeg(), headers={"Content-Type": "image/jpeg"}
    )
    com_chave = cliente_protegido.post(
        "/analyze/raw",
        content=_jpeg(),
        headers={"Content-Type": "image/jpeg", "X-API-Key": CHAVE_DE_TESTE},
    )

    assert sem_chave.status_code == 401
    assert com_chave.status_code == 200


def test_operacao_permanece_aberta_com_autenticacao_ativa(cliente_protegido):
    """/health e /info seguem acessíveis sem chave.

    A sonda do orquestrador precisa alcançar /health, e nenhuma das duas rotas
    executa inferência ou expõe dado sensível.
    """
    assert cliente_protegido.get("/health").status_code == 200
    assert cliente_protegido.get("/info").status_code == 200


# --------------------------------------------------------------------------- #
# Verificador de domínio (cascata)
# --------------------------------------------------------------------------- #


def _enviar(cliente_teste: TestClient):
    return cliente_teste.post(
        "/analyze", files={"imagem": ("folha.jpg", _jpeg(), "image/jpeg")}
    )


def test_cascata_desligada_por_padrao(cliente):
    """Sem IA_ALFACE_MODELO_DOMINIO o serviço responde exatamente como antes."""
    assert cliente.get("/info").json()["verificacao_dominio"] is False
    assert cliente.get("/health").json()["verificador_carregado"] is None
    corpo = _enviar(cliente).json()
    assert corpo["classe"] == "fungico"
    assert corpo["verificacao"] is None


def test_alface_aceita_traz_o_resultado_do_verificador(cliente_cascata):
    """Imagem aceita recebe o diagnóstico normal e o resultado da etapa 1."""
    resposta = _enviar(cliente_cascata)

    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["classe"] == "fungico"
    assert corpo["alerta"] is True
    assert corpo["verificacao"]["probabilidade_alface"] == pytest.approx(0.95, abs=1e-6)
    assert corpo["verificacao"]["limiar"] == 0.5
    assert corpo["verificacao"]["modelo_versao"] == "verificador_dominio.pt"


def test_imagem_sem_alface_retorna_200_com_classe_nao_alface(cliente_cascata):
    """Recusada, a imagem recebe 'nao_alface' no campo da classe, sem diagnóstico."""
    _YOLOCascataFalso.vetor_dominio = [0.10, 0.90]

    resposta = _enviar(cliente_cascata)

    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["classe"] == CLASSE_NAO_ALFACE
    assert corpo["classe_exibicao"] == EXIBICAO_NAO_ALFACE
    assert corpo["confianca"] == pytest.approx(0.90, abs=1e-3)
    assert corpo["verificacao"]["probabilidade_alface"] == pytest.approx(0.10, abs=1e-6)
    assert corpo["verificacao"]["modelo_versao"] == "verificador_dominio.pt"
    assert corpo["modelo_versao"] == "verificador_dominio.pt"


def test_imagem_sem_alface_nunca_gera_alerta_nem_diagnostico(cliente_cascata):
    """Nenhum campo da resposta 'nao_alface' pode ser lido como doença."""
    _YOLOCascataFalso.vetor_dominio = [0.10, 0.90]

    corpo = _enviar(cliente_cascata).json()

    assert corpo["alerta"] is False
    assert corpo["patogeno_detectado"] is False
    assert corpo["probabilidades"] == {}
    assert corpo["tempo_inferencia_ms"] == 0.0
    assert corpo["heatmap_base64"] is None
    assert corpo["classe"] not in ROTULOS_CANONICOS


def test_rota_bruta_tambem_responde_nao_alface(cliente_cascata):
    """O endpoint de corpo bruto segue o mesmo contrato."""
    _YOLOCascataFalso.vetor_dominio = [0.10, 0.90]

    resposta = cliente_cascata.post(
        "/analyze/raw", content=_jpeg(), headers={"Content-Type": "image/jpeg"}
    )

    assert resposta.status_code == 200
    assert resposta.json()["classe"] == CLASSE_NAO_ALFACE


def test_imagem_corrompida_continua_400_com_a_cascata(cliente_cascata):
    """Arquivo quebrado é erro de imagem (400), nunca 'nao_alface'."""
    resposta = cliente_cascata.post(
        "/analyze", files={"imagem": ("quebrada.jpg", b"nao sou imagem", "image/jpeg")}
    )

    assert resposta.status_code == 400


def test_422_fica_so_para_requisicao_malformada(cliente_cascata):
    """Com o verificador ativo, 422 continua sendo só a validação do FastAPI."""
    resposta = cliente_cascata.post("/analyze")

    assert resposta.status_code == 422
    assert isinstance(resposta.json()["detail"], list)


def test_limiar_do_verificador_vem_da_variavel(monkeypatch, tmp_path):
    """IA_ALFACE_LIMIAR_DOMINIO muda a decisão do verificador."""
    pesos_dominio = tmp_path / "verificador_dominio.pt"
    pesos_dominio.write_bytes(b"")
    with _subir_aplicacao(
        monkeypatch, tmp_path, modelo_dominio=pesos_dominio, limiar_dominio="0.2"
    ) as cliente_teste:
        _YOLOCascataFalso.vetor_dominio = [0.30, 0.70]

        assert _enviar(cliente_teste).status_code == 200
        assert cliente_teste.get("/info").json()["limiar_dominio"] == 0.2


def test_health_e_info_informam_o_verificador(cliente_cascata):
    """O estado da etapa 1 fica visível para a operação e para o dashboard."""
    saude = cliente_cascata.get("/health").json()
    assert saude["status"] == "ok"
    assert saude["verificador_carregado"] is True
    assert saude["modelo_dominio_versao"] == "verificador_dominio.pt"

    info = cliente_cascata.get("/info").json()
    assert info["verificacao_dominio"] is True
    assert info["limiar_dominio"] == 0.5


def test_verificador_configurado_e_ausente_retorna_503(monkeypatch, tmp_path):
    """Configurado e sem carregar, a inferência para em vez de seguir sem ele."""
    with _subir_aplicacao(
        monkeypatch, tmp_path, modelo_dominio=tmp_path / "inexistente_dominio.pt"
    ) as cliente_teste:
        saude = cliente_teste.get("/health").json()
        assert saude["status"] == "erro"
        assert saude["modelo_carregado"] is True
        assert saude["verificador_carregado"] is False

        resposta = _enviar(cliente_teste)
        assert resposta.status_code == 503
        assert "Verificador" in resposta.json()["detail"]


def test_openapi_documenta_nao_alface_no_campo_classe(cliente):
    """O contrato descreve 'nao_alface' na classe e não tem mais o esquema da recusa 422."""
    esquema = cliente.get("/openapi.json").json()

    classe = esquema["components"]["schemas"]["RespostaDeteccao"]["properties"]["classe"]
    assert CLASSE_NAO_ALFACE in classe["description"]
    assert "RespostaForaDoDominio" not in esquema["components"]["schemas"]
