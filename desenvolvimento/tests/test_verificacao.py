"""Testes do verificador de domínio (etapa 1 da cascata).

Como em ``test_inferencia.py``, o modelo Ultralytics é substituído por um duplo
de teste: a suíte não depende do verificador treinado.

Referência: docs/DOCUMENTACAO_IA_TREINAMENTO_RESULTADOS.md, Seção 2.9.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ia_alface.classes import CLASSE_ALFACE, ROTULOS_CANONICOS, ROTULOS_DOMINIO
from ia_alface.inferencia import DetectorDoencasAlface
from ia_alface.verificacao import (
    LIMIAR_ALFACE_PADRAO,
    ResultadoVerificacao,
    VerificadorDominio,
)

# --------------------------------------------------------------------------- #
# Duplo de teste do modelo Ultralytics
# --------------------------------------------------------------------------- #


class _ProbsFalso:
    """Imita ``ultralytics.engine.results.Probs``."""

    def __init__(self, vetor: list[float]) -> None:
        self.data = torch.tensor(vetor, dtype=torch.float32)


class _ResultadoFalso:
    """Imita ``ultralytics.engine.results.Results`` para classificação."""

    def __init__(self, vetor: list[float]) -> None:
        self.probs = _ProbsFalso(vetor)


class _YOLODominioFalso:
    """Duplo de ``ultralytics.YOLO`` com as classes do verificador."""

    #: Vetor de probabilidades devolvido; os testes ajustam conforme o cenário.
    vetor = [0.90, 0.10]  # alface, nao_alface
    task = "classify"
    names = dict(enumerate(ROTULOS_DOMINIO))

    def __init__(self, caminho: str, task: str | None = None, **_kwargs) -> None:
        self.caminho = caminho
        if task:
            self.task = task

    def predict(self, entrada, **_kwargs):
        return [_ResultadoFalso(list(self.vetor))]


@pytest.fixture
def pesos_falsos(tmp_path: Path) -> Path:
    """Cria um arquivo de pesos vazio, só para satisfazer a verificação de existência."""
    caminho = tmp_path / "verificador_de_teste.pt"
    caminho.write_bytes(b"")
    return caminho


def _criar_verificador(
    monkeypatch, pesos: Path, vetor: list[float], **kwargs
) -> VerificadorDominio:
    """Instancia o verificador com o duplo devolvendo ``vetor``."""
    monkeypatch.setattr(_YOLODominioFalso, "vetor", vetor)
    monkeypatch.setattr("ultralytics.YOLO", _YOLODominioFalso)
    return VerificadorDominio(pesos, dispositivo="cpu", **kwargs)


def _folha() -> Image.Image:
    """Imagem PIL sintética; o conteúdo é irrelevante para o duplo."""
    return Image.new("RGB", (64, 64), color=(30, 120, 40))


def _bytes_jpeg() -> bytes:
    """Gera os bytes de um JPEG válido, como os enviados pelo ESP32-CAM."""
    vetor = np.random.default_rng(7).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(vetor).save(buffer, format="JPEG")
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Contrato do resultado
# --------------------------------------------------------------------------- #


def test_verificar_retorna_resultado_valido(monkeypatch, pesos_falsos):
    """A verificação devolve um ResultadoVerificacao coerente."""
    verificador = _criar_verificador(monkeypatch, pesos_falsos, [0.90, 0.10])

    resultado = verificador.verificar(_folha())

    assert isinstance(resultado, ResultadoVerificacao)
    assert set(resultado.probabilidades) == set(ROTULOS_DOMINIO)
    assert sum(resultado.probabilidades.values()) == pytest.approx(1.0, abs=1e-4)
    assert resultado.probabilidade_alface == resultado.probabilidades[CLASSE_ALFACE]
    assert resultado.limiar == LIMIAR_ALFACE_PADRAO
    assert resultado.tempo_inferencia_ms > 0
    assert resultado.modelo_versao == "verificador_de_teste.pt"


def test_to_dict_e_serializavel(monkeypatch, pesos_falsos):
    """O dicionário de saída é serializável em JSON."""
    verificador = _criar_verificador(monkeypatch, pesos_falsos, [0.90, 0.10])

    dados = verificador.verificar(_folha()).to_dict()

    assert dados["eh_alface"] is True
    json.dumps(dados)  # não deve levantar


def test_aceita_bytes_de_jpeg(monkeypatch, pesos_falsos):
    """O verificador recebe os mesmos bytes que o classificador."""
    verificador = _criar_verificador(monkeypatch, pesos_falsos, [0.90, 0.10])
    assert verificador.verificar(_bytes_jpeg()).eh_alface is True


# --------------------------------------------------------------------------- #
# Decisão pelo limiar
# --------------------------------------------------------------------------- #


def test_alface_acima_do_limiar_e_aceita(monkeypatch, pesos_falsos):
    """Probabilidade de alface acima do limiar aceita a imagem."""
    verificador = _criar_verificador(monkeypatch, pesos_falsos, [0.90, 0.10])
    assert verificador.verificar(_folha()).eh_alface is True


def test_alface_abaixo_do_limiar_e_recusada(monkeypatch, pesos_falsos):
    """Probabilidade de alface abaixo do limiar recusa a imagem."""
    verificador = _criar_verificador(monkeypatch, pesos_falsos, [0.10, 0.90])
    assert verificador.verificar(_folha()).eh_alface is False


def test_probabilidade_igual_ao_limiar_e_aceita(monkeypatch, pesos_falsos):
    """Na fronteira a imagem é aceita: recusar uma folha doente suprime o alerta."""
    verificador = _criar_verificador(
        monkeypatch, pesos_falsos, [0.50, 0.50], limiar_alface=0.5
    )
    assert verificador.verificar(_folha()).eh_alface is True


def test_limiar_baixo_favorece_a_aceitacao(monkeypatch, pesos_falsos):
    """O limiar configurado é respeitado mesmo contra o argmax."""
    verificador = _criar_verificador(
        monkeypatch, pesos_falsos, [0.30, 0.70], limiar_alface=0.2
    )
    resultado = verificador.verificar(_folha())

    assert resultado.eh_alface is True
    assert resultado.limiar == 0.2


def test_limiar_invalido_e_rejeitado(pesos_falsos):
    """Limiar fora de [0, 1] é erro de configuração."""
    with pytest.raises(ValueError, match="limiar_alface"):
        VerificadorDominio(pesos_falsos, limiar_alface=1.5)
    with pytest.raises(ValueError, match="limiar_alface"):
        VerificadorDominio(pesos_falsos, limiar_alface=-0.1)


# --------------------------------------------------------------------------- #
# Carga do modelo — troca de pesos deve falhar na subida, não na requisição
# --------------------------------------------------------------------------- #


def test_pesos_inexistentes_levantam_filenotfound(tmp_path):
    """Um caminho de pesos inválido falha antes de qualquer inferência."""
    with pytest.raises(FileNotFoundError, match="não encontrados"):
        VerificadorDominio(tmp_path / "sem_verificador.pt")


def test_modelo_de_outra_tarefa_e_rejeitado(monkeypatch, pesos_falsos):
    """Pesos de detecção não servem de verificador."""

    class _YOLODeteccao(_YOLODominioFalso):
        task = "detect"

    monkeypatch.setattr("ultralytics.YOLO", _YOLODeteccao)
    with pytest.raises(ValueError, match="classificação"):
        VerificadorDominio(pesos_falsos)


def test_classificador_de_doencas_nao_serve_de_verificador(monkeypatch, pesos_falsos):
    """Apontar o verificador para os pesos de doença é detectado na carga."""

    class _YOLODoencas(_YOLODominioFalso):
        names = dict(enumerate(ROTULOS_CANONICOS))

    monkeypatch.setattr("ultralytics.YOLO", _YOLODoencas)
    with pytest.raises(ValueError, match="verificador de domínio"):
        VerificadorDominio(pesos_falsos)


def test_verificador_nao_serve_de_classificador_de_doencas(monkeypatch, pesos_falsos):
    """A troca inversa também falha: o detector exige as classes canônicas."""
    monkeypatch.setattr("ultralytics.YOLO", _YOLODominioFalso)
    with pytest.raises(ValueError, match="canônico"):
        DetectorDoencasAlface(pesos_falsos)


# --------------------------------------------------------------------------- #
# Validação de entrada — imagem inválida é erro, nunca 'nao_alface'
# --------------------------------------------------------------------------- #


def test_bytes_corrompidos_levantam_valueerror(monkeypatch, pesos_falsos):
    """Bytes que não são imagem falham antes de chegar ao modelo."""
    verificador = _criar_verificador(monkeypatch, pesos_falsos, [0.10, 0.90])
    with pytest.raises(ValueError, match="decodificar"):
        verificador.verificar(b"isto nao eh uma imagem")


def test_bytes_vazios_levantam_valueerror(monkeypatch, pesos_falsos):
    """Transmissão vazia falha explicitamente."""
    verificador = _criar_verificador(monkeypatch, pesos_falsos, [0.10, 0.90])
    with pytest.raises(ValueError, match="vazios"):
        verificador.verificar(b"")
