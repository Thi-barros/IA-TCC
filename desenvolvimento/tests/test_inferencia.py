"""Testes do contrato de inferência.

A suíte não depende de pesos treinados: o modelo Ultralytics é substituído por um
duplo de teste que expõe a mesma interface. Quando um modelo real existe em
``modelos/``, um teste adicional é executado sobre ele.

Referência: spec.md, Seção 11.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ia_alface.classes import ROTULOS_CANONICOS
from ia_alface.config import CAMINHOS
from ia_alface.inferencia import DetectorDoencasAlface, ResultadoDeteccao

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


class _YOLOFalso:
    """Duplo de teste de ``ultralytics.YOLO`` para a tarefa de classificação."""

    #: Vetor de probabilidades devolvido; os testes ajustam conforme o cenário.
    vetor = [0.05, 0.05, 0.90]  # bacteriano, fungico, saudavel
    task = "classify"
    names = dict(enumerate(ROTULOS_CANONICOS))

    def __init__(self, caminho: str, task: str | None = None, **_kwargs) -> None:
        self.caminho = caminho
        if task:
            self.task = task

    def predict(self, entrada, **_kwargs):
        quantidade = len(entrada) if isinstance(entrada, list) else 1
        return [_ResultadoFalso(list(self.vetor)) for _ in range(quantidade)]


@pytest.fixture
def pesos_falsos(tmp_path: Path) -> Path:
    """Cria um arquivo de pesos vazio, só para satisfazer a verificação de existência."""
    caminho = tmp_path / "modelo_de_teste.pt"
    caminho.write_bytes(b"")
    return caminho


@pytest.fixture
def detector(monkeypatch, pesos_falsos: Path) -> DetectorDoencasAlface:
    """Detector com o modelo substituído pelo duplo de teste."""
    _YOLOFalso.vetor = [0.05, 0.05, 0.90]
    monkeypatch.setattr("ultralytics.YOLO", _YOLOFalso)
    return DetectorDoencasAlface(pesos_falsos, dispositivo="cpu")


@pytest.fixture
def imagem_sintetica(tmp_path: Path) -> Path:
    """Grava uma imagem RGB sintética em disco."""
    caminho = tmp_path / "folha.jpg"
    vetor = np.random.default_rng(42).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    Image.fromarray(vetor).save(caminho)
    return caminho


def _bytes_jpeg(largura: int = 160, altura: int = 120) -> bytes:
    """Gera os bytes de um JPEG válido, como os enviados pelo ESP32-CAM."""
    vetor = np.random.default_rng(7).integers(0, 255, (altura, largura, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(vetor).save(buffer, format="JPEG")
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Contrato do resultado
# --------------------------------------------------------------------------- #


def test_predizer_retorna_resultado_valido(detector, imagem_sintetica):
    """A predição devolve um ResultadoDeteccao coerente."""
    resultado = detector.predizer(imagem_sintetica)

    assert isinstance(resultado, ResultadoDeteccao)
    assert resultado.classe in ROTULOS_CANONICOS
    assert 0.0 <= resultado.confianca <= 1.0
    assert resultado.tempo_inferencia_ms > 0
    assert resultado.modelo_versao == "modelo_de_teste.pt"


def test_probabilidades_somam_um(detector, imagem_sintetica):
    """As probabilidades cobrem as três classes e somam 1."""
    resultado = detector.predizer(imagem_sintetica)

    assert set(resultado.probabilidades) == set(ROTULOS_CANONICOS)
    assert sum(resultado.probabilidades.values()) == pytest.approx(1.0, abs=1e-4)


def test_confianca_corresponde_a_classe_predita(detector, imagem_sintetica):
    """A confiança é a probabilidade da classe predita."""
    resultado = detector.predizer(imagem_sintetica)
    esperada = resultado.probabilidades[resultado.classe]
    assert resultado.confianca == pytest.approx(esperada, abs=1e-3)


def test_bbox_sempre_nulo(detector, imagem_sintetica):
    """O modelo é de classificação: as coordenadas de caixa são sempre nulas."""
    resultado = detector.predizer(imagem_sintetica)

    assert resultado.bbox_x is None
    assert resultado.bbox_y is None
    assert resultado.bbox_w is None
    assert resultado.bbox_h is None


def test_to_dict_e_serializavel(detector, imagem_sintetica):
    """O dicionário de saída inclui o rótulo de exibição e é serializável."""
    import json

    dados = detector.predizer(imagem_sintetica).to_dict()

    assert "classe_exibicao" in dados
    assert dados["classe"] in ROTULOS_CANONICOS
    json.dumps(dados)  # não deve levantar


# --------------------------------------------------------------------------- #
# Lógica de alerta
# --------------------------------------------------------------------------- #


def test_saudavel_nao_gera_alerta(monkeypatch, pesos_falsos, imagem_sintetica):
    """Folha saudável nunca aciona notificação, mesmo com alta confiança."""
    _YOLOFalso.vetor = [0.02, 0.03, 0.95]
    monkeypatch.setattr("ultralytics.YOLO", _YOLOFalso)
    detector = DetectorDoencasAlface(pesos_falsos, dispositivo="cpu")

    resultado = detector.predizer(imagem_sintetica)

    assert resultado.classe == "saudavel"
    assert resultado.patogeno_detectado is False
    assert resultado.alerta is False


def test_patogeno_acima_do_limiar_gera_alerta(monkeypatch, pesos_falsos, imagem_sintetica):
    """Patógeno com confiança >= limiar aciona o alerta (RF07/CT07)."""
    _YOLOFalso.vetor = [0.90, 0.06, 0.04]
    monkeypatch.setattr("ultralytics.YOLO", _YOLOFalso)
    detector = DetectorDoencasAlface(pesos_falsos, dispositivo="cpu", limiar_confianca=0.85)

    resultado = detector.predizer(imagem_sintetica)

    assert resultado.classe == "bacteriano"
    assert resultado.patogeno_detectado is True
    assert resultado.alerta is True


def test_patogeno_abaixo_do_limiar_nao_gera_alerta(
    monkeypatch, pesos_falsos, imagem_sintetica
):
    """Patógeno detectado com confiança insuficiente não aciona o alerta."""
    _YOLOFalso.vetor = [0.60, 0.25, 0.15]
    monkeypatch.setattr("ultralytics.YOLO", _YOLOFalso)
    detector = DetectorDoencasAlface(pesos_falsos, dispositivo="cpu", limiar_confianca=0.85)

    resultado = detector.predizer(imagem_sintetica)

    assert resultado.patogeno_detectado is True
    assert resultado.alerta is False


def test_limiar_invalido_e_rejeitado(pesos_falsos):
    """Limiar fora de [0, 1] é erro de configuração."""
    with pytest.raises(ValueError, match="limiar_confianca"):
        DetectorDoencasAlface(pesos_falsos, limiar_confianca=1.5)
    with pytest.raises(ValueError, match="limiar_confianca"):
        DetectorDoencasAlface(pesos_falsos, limiar_confianca=-0.1)


# --------------------------------------------------------------------------- #
# Formatos de entrada
# --------------------------------------------------------------------------- #


def test_aceita_bytes_de_jpeg(detector):
    """O ESP32-CAM envia bytes de JPEG via HTTP POST."""
    resultado = detector.predizer(_bytes_jpeg())
    assert resultado.classe in ROTULOS_CANONICOS


def test_aceita_ndarray_bgr(detector):
    """Arrays NumPy em BGR são aceitos (convenção do OpenCV/Ultralytics)."""
    vetor = np.random.default_rng(1).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    resultado = detector.predizer(vetor)
    assert resultado.classe in ROTULOS_CANONICOS


def test_aceita_imagem_em_escala_de_cinza(detector):
    """Um array 2-D é replicado para três canais em vez de falhar."""
    vetor = np.random.default_rng(2).integers(0, 255, (80, 80), dtype=np.uint8)
    resultado = detector.predizer(vetor)
    assert resultado.classe in ROTULOS_CANONICOS


def test_aceita_objeto_pil(detector):
    """Imagens PIL são aceitas diretamente."""
    imagem = Image.new("RGB", (64, 64), color=(30, 120, 40))
    resultado = detector.predizer(imagem)
    assert resultado.classe in ROTULOS_CANONICOS


def test_converte_pil_nao_rgb(detector):
    """Modos de cor diferentes (P, L, RGBA) são convertidos para RGB."""
    for modo in ("L", "RGBA", "P"):
        imagem = Image.new(modo, (64, 64))
        resultado = detector.predizer(imagem)
        assert resultado.classe in ROTULOS_CANONICOS


# --------------------------------------------------------------------------- #
# Validação de entrada — um falso 'saudavel' por imagem corrompida é o pior
# modo de falha possível neste sistema.
# --------------------------------------------------------------------------- #


def test_bytes_corrompidos_levantam_valueerror(detector):
    """Bytes que não são imagem devem falhar, nunca produzir um resultado."""
    with pytest.raises(ValueError, match="decodificar"):
        detector.predizer(b"isto nao eh uma imagem")


def test_bytes_vazios_levantam_valueerror(detector):
    """Transmissão vazia deve falhar explicitamente."""
    with pytest.raises(ValueError, match="vazios"):
        detector.predizer(b"")


def test_jpeg_truncado_levanta_valueerror(detector):
    """Um JPEG cortado ao meio (transmissão interrompida) deve falhar."""
    completo = _bytes_jpeg()
    with pytest.raises(ValueError):
        detector.predizer(completo[: len(completo) // 3])


def test_arquivo_inexistente_levanta_valueerror(detector, tmp_path):
    """Caminho inválido produz mensagem clara."""
    with pytest.raises(ValueError, match="não encontrado"):
        detector.predizer(tmp_path / "nao_existe.jpg")


def test_arquivo_vazio_levanta_valueerror(detector, tmp_path):
    """Arquivo de zero byte deve falhar."""
    vazio = tmp_path / "vazio.jpg"
    vazio.write_bytes(b"")
    with pytest.raises(ValueError, match="vazio"):
        detector.predizer(vazio)


def test_array_vazio_levanta_valueerror(detector):
    """Array sem elementos deve falhar."""
    with pytest.raises(ValueError, match="vazio"):
        detector.predizer(np.array([], dtype=np.uint8))


def test_array_com_canais_invalidos_levanta_valueerror(detector):
    """Número de canais inesperado deve falhar."""
    with pytest.raises(ValueError, match="inválido"):
        detector.predizer(np.zeros((10, 10, 7), dtype=np.uint8))


def test_imagem_degenerada_levanta_valueerror(detector):
    """Imagem de 1 pixel não é analisável."""
    with pytest.raises(ValueError, match="degeneradas"):
        detector.predizer(Image.new("RGB", (1, 1)))


def test_tipo_de_entrada_nao_suportado(detector):
    """Tipos fora do contrato produzem mensagem explicativa."""
    with pytest.raises(ValueError, match="não suportado"):
        detector.predizer(12345)


def test_pesos_inexistentes_levantam_filenotfound(tmp_path):
    """Um caminho de pesos inválido falha antes de qualquer inferência."""
    with pytest.raises(FileNotFoundError, match="não encontrados"):
        DetectorDoencasAlface(tmp_path / "sem_modelo.pt")


def test_modelo_de_outra_tarefa_e_rejeitado(monkeypatch, pesos_falsos):
    """Pesos de detecção não podem ser usados neste detector."""

    class _YOLODeteccao(_YOLOFalso):
        task = "detect"

    monkeypatch.setattr("ultralytics.YOLO", _YOLODeteccao)
    with pytest.raises(ValueError, match="classificação"):
        DetectorDoencasAlface(pesos_falsos)


def test_classes_incompativeis_sao_rejeitadas(monkeypatch, pesos_falsos):
    """Um modelo de 3 classes com rótulos errados indica treino sobre outro dataset."""

    class _YOLOOutrasClasses(_YOLOFalso):
        names = {0: "gato", 1: "cachorro", 2: "passaro"}

    monkeypatch.setattr("ultralytics.YOLO", _YOLOOutrasClasses)
    with pytest.raises(ValueError, match="canônico"):
        DetectorDoencasAlface(pesos_falsos)


# --------------------------------------------------------------------------- #
# Lote
# --------------------------------------------------------------------------- #


def test_predizer_lote(detector, imagem_sintetica):
    """O lote devolve um resultado por imagem, na mesma ordem."""
    resultados = detector.predizer_lote([imagem_sintetica] * 3)

    assert len(resultados) == 3
    for resultado in resultados:
        assert resultado.classe in ROTULOS_CANONICOS
        assert resultado.tempo_inferencia_ms > 0


def test_predizer_lote_vazio_levanta_valueerror(detector):
    """Lote vazio é erro de uso."""
    with pytest.raises(ValueError, match="vazia"):
        detector.predizer_lote([])


# --------------------------------------------------------------------------- #
# Teste opcional com o modelo real, quando disponível
# --------------------------------------------------------------------------- #


def _pesos_reais() -> Path | None:
    """Localiza um modelo treinado em ``modelos/``, se existir."""
    if not CAMINHOS.modelos.is_dir():
        return None
    candidatos = sorted(CAMINHOS.modelos.glob("*_alface.pt"))
    return candidatos[0] if candidatos else None


@pytest.mark.skipif(
    _pesos_reais() is None,
    reason="Nenhum modelo treinado em modelos/; execute desenvolvimento/scripts/02_treinar.py.",
)
def test_modelo_real_produz_classe_canonica(imagem_sintetica):
    """Com pesos reais, a classe predita pertence ao conjunto canônico."""
    detector = DetectorDoencasAlface(_pesos_reais(), dispositivo="cpu")

    assert set(detector.nomes_classes.values()) == set(ROTULOS_CANONICOS)

    resultado = detector.predizer(imagem_sintetica)
    assert resultado.classe in ROTULOS_CANONICOS
    assert sum(resultado.probabilidades.values()) == pytest.approx(1.0, abs=1e-3)
    assert resultado.tempo_inferencia_ms > 0
