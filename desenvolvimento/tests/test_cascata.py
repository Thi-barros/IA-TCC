"""Testes da inferência em cascata (verificador → classificador).

Os dois modelos Ultralytics são substituídos por duplos de teste, cada um com
as suas classes e um contador de chamadas.

Referência: docs/VERIFICADOR_DOMINIO.md.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ia_alface.cascata import AnalisadorCascata, ImagemForaDoDominio, ResultadoCascata
from ia_alface.classes import ROTULOS_CANONICOS, ROTULOS_DOMINIO
from ia_alface.inferencia import DetectorDoencasAlface
from ia_alface.verificacao import VerificadorDominio

# --------------------------------------------------------------------------- #
# Duplos de teste
# --------------------------------------------------------------------------- #


class _ProbsFalso:
    """Imita ``ultralytics.engine.results.Probs``."""

    def __init__(self, vetor: list[float]) -> None:
        self.data = torch.tensor(vetor, dtype=torch.float32)


class _ResultadoFalso:
    """Imita ``ultralytics.engine.results.Results`` para classificação."""

    def __init__(self, vetor: list[float]) -> None:
        self.probs = _ProbsFalso(vetor)


class _YOLOBase:
    """Duplo de ``ultralytics.YOLO`` que conta as inferências."""

    vetor: list[float] = []
    names: dict[int, str] = {}
    chamadas = 0
    task = "classify"

    def __init__(self, caminho: str, task: str | None = None, **_kwargs) -> None:
        self.caminho = caminho

    def predict(self, entrada, **_kwargs):
        type(self).chamadas += 1
        return [_ResultadoFalso(list(self.vetor))]


class _YOLODoencas(_YOLOBase):
    vetor = [0.90, 0.06, 0.04]  # bacteriano, fungico, saudavel
    names = dict(enumerate(ROTULOS_CANONICOS))


class _YOLODominio(_YOLOBase):
    vetor = [0.95, 0.05]  # alface, nao_alface
    names = dict(enumerate(ROTULOS_DOMINIO))


def _pesos(tmp_path: Path, nome: str) -> Path:
    caminho = tmp_path / nome
    caminho.write_bytes(b"")
    return caminho


@pytest.fixture
def detector(monkeypatch, tmp_path) -> DetectorDoencasAlface:
    """Classificador com o duplo de doenças."""
    monkeypatch.setattr(_YOLODoencas, "chamadas", 0)
    monkeypatch.setattr("ultralytics.YOLO", _YOLODoencas)
    return DetectorDoencasAlface(_pesos(tmp_path, "doencas.pt"), dispositivo="cpu")


def _verificador(monkeypatch, tmp_path, vetor: list[float], **kwargs) -> VerificadorDominio:
    """Verificador com o duplo de domínio devolvendo ``vetor``."""
    monkeypatch.setattr(_YOLODominio, "vetor", vetor)
    monkeypatch.setattr(_YOLODominio, "chamadas", 0)
    monkeypatch.setattr("ultralytics.YOLO", _YOLODominio)
    return VerificadorDominio(_pesos(tmp_path, "dominio.pt"), dispositivo="cpu", **kwargs)


def _jpeg() -> bytes:
    """Bytes de um JPEG válido."""
    vetor = np.random.default_rng(7).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(vetor).save(buffer, format="JPEG")
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Imagem aceita
# --------------------------------------------------------------------------- #


def test_alface_segue_para_o_classificador(monkeypatch, tmp_path, detector):
    """Imagem aceita recebe o diagnóstico normal do classificador."""
    analisador = AnalisadorCascata(detector, _verificador(monkeypatch, tmp_path, [0.95, 0.05]))

    resultado = analisador.analisar(_jpeg())

    assert isinstance(resultado, ResultadoCascata)
    assert resultado.verificacao.eh_alface is True
    assert resultado.deteccao.classe in ROTULOS_CANONICOS
    assert _YOLODominio.chamadas == 1
    assert _YOLODoencas.chamadas == 1


def test_alerta_do_classificador_e_preservado(monkeypatch, tmp_path, detector):
    """A cascata não altera a regra de alerta do classificador."""
    analisador = AnalisadorCascata(detector, _verificador(monkeypatch, tmp_path, [0.95, 0.05]))

    deteccao = analisador.analisar(_jpeg()).deteccao

    assert deteccao.patogeno_detectado is True
    assert deteccao.alerta is True


def test_tempo_total_soma_as_duas_etapas(monkeypatch, tmp_path, detector):
    """O tempo total é a soma das inferências executadas."""
    analisador = AnalisadorCascata(detector, _verificador(monkeypatch, tmp_path, [0.95, 0.05]))

    resultado = analisador.analisar(_jpeg())

    esperado = resultado.deteccao.tempo_inferencia_ms + resultado.verificacao.tempo_inferencia_ms
    assert resultado.tempo_total_ms == pytest.approx(esperado, abs=1e-3)
    assert resultado.to_dict()["verificacao"]["eh_alface"] is True


# --------------------------------------------------------------------------- #
# Imagem recusada
# --------------------------------------------------------------------------- #


def test_imagem_sem_alface_e_recusada_sem_diagnostico(monkeypatch, tmp_path, detector):
    """Recusada, a imagem nunca chega ao classificador nem gera alerta."""
    analisador = AnalisadorCascata(detector, _verificador(monkeypatch, tmp_path, [0.10, 0.90]))

    with pytest.raises(ImagemForaDoDominio) as excinfo:
        analisador.analisar(_jpeg())

    assert excinfo.value.verificacao.eh_alface is False
    assert excinfo.value.verificacao.probabilidade_alface == pytest.approx(0.10, abs=1e-6)
    assert _YOLODoencas.chamadas == 0


def test_recusa_nao_e_valueerror():
    """'Não é alface' e 'imagem inválida' são contratos distintos (422 × 400)."""
    assert not issubclass(ImagemForaDoDominio, ValueError)


def test_limiar_do_verificador_e_respeitado(monkeypatch, tmp_path, detector):
    """Com limiar baixo, uma probabilidade modesta de alface já é aceita."""
    verificador = _verificador(monkeypatch, tmp_path, [0.30, 0.70], limiar_alface=0.2)
    resultado = AnalisadorCascata(detector, verificador).analisar(_jpeg())

    assert resultado.verificacao.eh_alface is True
    assert _YOLODoencas.chamadas == 1


# --------------------------------------------------------------------------- #
# Imagem inválida e cascata desligada
# --------------------------------------------------------------------------- #


def test_imagem_corrompida_falha_antes_dos_modelos(monkeypatch, tmp_path, detector):
    """Corrompida é erro (400), nunca 'não alface', e nenhum modelo roda."""
    analisador = AnalisadorCascata(detector, _verificador(monkeypatch, tmp_path, [0.10, 0.90]))

    with pytest.raises(ValueError, match="decodificar"):
        analisador.analisar(b"isto nao eh uma imagem")

    assert _YOLODominio.chamadas == 0
    assert _YOLODoencas.chamadas == 0


def test_sem_verificador_comporta_se_como_o_detector(detector):
    """Cascata desligada: só o classificador roda, como antes da mudança."""
    analisador = AnalisadorCascata(detector)

    resultado = analisador.analisar(_jpeg())

    assert analisador.verifica_dominio is False
    assert resultado.verificacao is None
    assert resultado.tempo_total_ms == resultado.deteccao.tempo_inferencia_ms
    assert _YOLODoencas.chamadas == 1
