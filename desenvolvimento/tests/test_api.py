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


@pytest.fixture
def cliente(monkeypatch, tmp_path):
    """Sobe a aplicação com o modelo substituído pelo duplo de teste."""
    pesos = tmp_path / "modelo_de_teste.pt"
    pesos.write_bytes(b"")

    monkeypatch.setenv("IA_ALFACE_MODELO", str(pesos))
    monkeypatch.setenv("IA_ALFACE_LIMIAR", "0.85")
    monkeypatch.setenv("IA_ALFACE_MAX_MB", "1")
    monkeypatch.setattr("ultralytics.YOLO", _YOLOFalso)

    _YOLOFalso.vetor = [0.02, 0.94, 0.04]

    # A configuração é lida na importação; recarregar garante que as variáveis
    # de ambiente acima tenham efeito.
    import importlib

    import api.configuracao
    import api.main

    importlib.reload(api.configuracao)
    importlib.reload(api.main)

    with TestClient(api.main.app) as cliente_teste:
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
