"""Testes do serviço de inferência HTTP.

Como nos demais testes, o modelo Ultralytics é substituído por um duplo, de modo
que a suíte não dependa de pesos treinados e execute em segundos.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from ia_alface.classes import ROTULOS_CANONICOS

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


def _subir_aplicacao(monkeypatch, tmp_path, chave_api: str = "") -> TestClient:
    """Prepara o ambiente e devolve o cliente da aplicação recarregada.

    Args:
        chave_api: Quando informada, sobe o serviço com autenticação ativa.
    """
    pesos = tmp_path / "modelo_de_teste.pt"
    pesos.write_bytes(b"")

    monkeypatch.setenv("IA_ALFACE_MODELO", str(pesos))
    monkeypatch.setenv("IA_ALFACE_LIMIAR", "0.85")
    monkeypatch.setenv("IA_ALFACE_MAX_MB", "1")
    monkeypatch.setattr("ultralytics.YOLO", _YOLOFalso)

    # Removida explicitamente para que uma variável presente no ambiente do
    # desenvolvedor não altere o comportamento da suíte.
    if chave_api:
        monkeypatch.setenv("IA_ALFACE_CHAVE_API", chave_api)
    else:
        monkeypatch.delenv("IA_ALFACE_CHAVE_API", raising=False)

    _YOLOFalso.vetor = [0.02, 0.94, 0.04]

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
