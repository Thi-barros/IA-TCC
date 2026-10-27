"""Testes da projeção do mapa Grad-CAM sobre a imagem original.

O mapa 7 × 7 descreve só o recorte central que a rede analisa. Estes testes
cobrem a geometria desse recorte e a sobreposição, que são funções puras: não
dependem de pesos treinados.

Referência: docs/VERIFICADOR_DOMINIO.md, Seção 1 (decisão de 26/10/2026).
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from ia_alface.explicabilidade import regiao_analisada, sobrepor_mapa


@pytest.mark.parametrize(
    ("largura", "altura", "esperada"),
    [
        (1280, 720, (280, 0, 1000, 720)),  # foto da C270
        (315, 500, (0, 93, 315, 408)),  # retrato
        (500, 315, (93, 0, 408, 315)),  # paisagem
        (474, 474, (0, 0, 474, 474)),  # quadrada: a imagem inteira
    ],
)
def test_regiao_analisada(largura: int, altura: int, esperada: tuple) -> None:
    """A região é o recorte central do pré-processamento, em pixels originais."""
    assert regiao_analisada(largura, altura) == esperada


def test_regiao_analisada_rejeita_dimensao_invalida() -> None:
    """Dimensão zero ou negativa não tem região analisada."""
    for largura, altura in [(0, 720), (1280, 0), (-5, 10)]:
        with pytest.raises(ValueError):
            regiao_analisada(largura, altura)


def test_regiao_confere_com_classify_transforms() -> None:
    """A conta reproduz o recorte real do Ultralytics numa foto da C270.

    A imagem sintética tem faixas amarelas de 10 px logo dentro das bordas da
    região calculada. Se a conta errasse o corte em mais de ~5 px, a primeira
    ou a última coluna do recorte cairia no vermelho, no azul ou no verde.
    """
    augment = pytest.importorskip("ultralytics.data.augment")
    largura, altura = 1280, 720
    esquerda, _, direita, _ = regiao_analisada(largura, altura)

    faixas = np.zeros((altura, largura, 3), dtype=np.uint8)
    faixas[:, :] = (0, 200, 0)  # verde: miolo do recorte
    faixas[:, :esquerda] = (200, 0, 0)  # vermelho: descartado à esquerda
    faixas[:, esquerda : esquerda + 10] = (200, 200, 0)  # amarelo: borda esquerda
    faixas[:, direita - 10 : direita] = (200, 200, 0)  # amarelo: borda direita
    faixas[:, direita:] = (0, 0, 200)  # azul: descartado à direita

    redimensionar, recortar = augment.classify_transforms(224).transforms[:2]
    recorte = np.asarray(recortar(redimensionar(Image.fromarray(faixas))))

    assert recorte.shape[:2] == (224, 224)
    for coluna in (recorte[:, 1], recorte[:, -2]):
        vermelho, verde, azul = coluna.astype(np.float32).mean(axis=0)
        assert vermelho > 150 and verde > 150 and azul < 50


def test_sobrepor_mapa_nao_colore_fora_da_regiao() -> None:
    """As faixas que a rede não viu ficam idênticas à foto original."""
    original = Image.new("RGB", (1280, 720), (128, 128, 128))
    regiao = regiao_analisada(1280, 720)
    mapa = np.ones((720, 1280), dtype=np.float32)

    resultado = np.asarray(sobrepor_mapa(original, mapa, regiao))
    cinza = np.asarray(original)
    esquerda, topo, direita, base = regiao

    assert resultado.shape == cinza.shape
    assert np.array_equal(resultado[:, :esquerda], cinza[:, :esquerda])
    assert np.array_equal(resultado[:, direita:], cinza[:, direita:])
    assert not np.array_equal(
        resultado[topo:base, esquerda:direita], cinza[topo:base, esquerda:direita]
    )
