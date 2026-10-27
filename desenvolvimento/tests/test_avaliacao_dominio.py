"""Testes das métricas do verificador de domínio e da cascata.

As funções não dependem de modelo: recebem tabelas de previsões sintéticas.

Referência: docs/VERIFICADOR_DOMINIO.md.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ia_alface import avaliacao_dominio as ad
from ia_alface.classes import CLASSE_ALFACE, CLASSE_NAO_ALFACE, ROTULOS_CANONICOS, eh_patogeno
from ia_alface.dados_dominio import CAMERA_PROJETO, FONTE_ESTUFA, FONTE_KAGGLE

SAUDAVEL = next(c for c in ROTULOS_CANONICOS if not eh_patogeno(c))
DOENCAS = [c for c in ROTULOS_CANONICOS if eh_patogeno(c)]

# --------------------------------------------------------------------------- #
# Intervalo de Wilson
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("acertos", "total", "inferior", "superior"),
    [
        (0, 10, 0.0, 0.2775),     # zero acertos ainda tem limite superior útil
        (10, 10, 0.7225, 1.0),
        (5, 10, 0.2366, 0.7634),
    ],
)
def test_intervalo_wilson(acertos, total, inferior, superior):
    """Valores conferidos à mão com z = 1,96."""
    obtido = ad.intervalo_wilson(acertos, total)
    assert obtido[0] == pytest.approx(inferior, abs=1e-3)
    assert obtido[1] == pytest.approx(superior, abs=1e-3)


def test_intervalo_wilson_sem_amostra():
    """Sem imagens não há intervalo — e não se inventa um."""
    assert ad.intervalo_wilson(0, 0) is None
    assert ad.proporcao(0, 0) == {
        "acertos": 0, "total": 0, "taxa": None, "ic95_inf": None, "ic95_sup": None,
    }


def test_intervalo_estreita_com_amostra_maior():
    """Mesma taxa, amostra 10× maior: intervalo mais estreito."""
    pequeno = ad.intervalo_wilson(9, 10)
    grande = ad.intervalo_wilson(90, 100)
    assert (grande[1] - grande[0]) < (pequeno[1] - pequeno[0])


# --------------------------------------------------------------------------- #
# Métricas por grupo e agregados
# --------------------------------------------------------------------------- #


def _previsoes() -> pd.DataFrame:
    """Tabela sintética com resultados conhecidos no limiar 0,5.

    - Kaggle bacteriano: 4 imagens, 1 recusada (0,30).
    - Kaggle saudável: 2 imagens, ambas aceitas.
    - C270 alface: 2 imagens, 1 recusada (0,10).
    - C270 não alface: 3 imagens, 1 aceita por engano (0,80).
    - COCO: 2 imagens, ambas recusadas.
    """
    linhas = [
        (CLASSE_ALFACE, FONTE_KAGGLE, DOENCAS[0], "", p) for p in (0.99, 0.95, 0.60, 0.30)
    ] + [
        (CLASSE_ALFACE, FONTE_KAGGLE, SAUDAVEL, "", p) for p in (0.99, 0.70)
    ] + [
        (CLASSE_ALFACE, FONTE_ESTUFA, CAMERA_PROJETO, CAMERA_PROJETO, p) for p in (0.90, 0.10)
    ] + [
        (CLASSE_NAO_ALFACE, FONTE_ESTUFA, CAMERA_PROJETO, CAMERA_PROJETO, p) for p in (0.05, 0.20, 0.80)
    ] + [
        (CLASSE_NAO_ALFACE, "coco", "coco", "", p) for p in (0.01, 0.02)
    ]
    tabela = pd.DataFrame(linhas, columns=["classe", "fonte", "estrato", "camera", "prob_alface"])
    tabela["grupo"] = [
        ad.grupo_de_avaliacao(f, e, c)
        for f, e, c in zip(tabela["fonte"], tabela["estrato"], tabela["camera"])
    ]
    return tabela


def test_grupo_de_avaliacao():
    """Kaggle por doença, estufa por câmera, demais pela fonte."""
    assert ad.grupo_de_avaliacao(FONTE_KAGGLE, DOENCAS[0], "") == f"{FONTE_KAGGLE}/{DOENCAS[0]}"
    assert ad.grupo_de_avaliacao(FONTE_ESTUFA, "x", CAMERA_PROJETO) == f"{FONTE_ESTUFA}/{CAMERA_PROJETO}"
    assert ad.grupo_de_avaliacao("coco", "coco", "") == "coco"


def test_metricas_por_grupo():
    """Cada grupo conta acerto conforme a sua classe."""
    tabela = ad.metricas_por_grupo(_previsoes(), 0.5).set_index(["classe", "grupo"])

    bacteriano = tabela.loc[(CLASSE_ALFACE, f"{FONTE_KAGGLE}/{DOENCAS[0]}")]
    assert (bacteriano["acertos"], bacteriano["total"]) == (3, 4)

    c270_nao = tabela.loc[(CLASSE_NAO_ALFACE, f"{FONTE_ESTUFA}/{CAMERA_PROJETO}")]
    assert (c270_nao["acertos"], c270_nao["total"]) == (2, 3)

    coco = tabela.loc[(CLASSE_NAO_ALFACE, "coco")]
    assert (coco["acertos"], coco["total"]) == (2, 2)


def test_agregados():
    """Folhas doentes são só as do Kaggle com patógeno; C270 separada por classe."""
    medidas = ad.agregados(_previsoes(), 0.5)

    assert (medidas["alface_folhas_doentes"]["acertos"], medidas["alface_folhas_doentes"]["total"]) == (3, 4)
    assert (medidas["alface_total"]["acertos"], medidas["alface_total"]["total"]) == (6, 8)
    assert (medidas["alface_camera_projeto"]["acertos"], medidas["alface_camera_projeto"]["total"]) == (1, 2)
    assert (medidas["nao_alface_total"]["acertos"], medidas["nao_alface_total"]["total"]) == (4, 5)
    assert (medidas["nao_alface_camera_projeto"]["acertos"],
            medidas["nao_alface_camera_projeto"]["total"]) == (2, 3)


def test_probabilidade_igual_ao_limiar_conta_como_aceita():
    """Mesma regra do verificador: na fronteira, aceita."""
    tabela = pd.DataFrame({
        "classe": [CLASSE_ALFACE], "fonte": [FONTE_KAGGLE], "estrato": [DOENCAS[0]],
        "camera": [""], "prob_alface": [0.5],
    })
    assert ad.agregados(tabela, 0.5)["alface_folhas_doentes"]["acertos"] == 1


def test_curva_limiar_troca_aceitacao_por_rejeicao():
    """Subir o limiar recusa mais: alface aceita cai, não alface recusada sobe."""
    curva = ad.curva_limiar(_previsoes(), [0.05, 0.5, 0.95]).set_index("limiar")

    assert curva.loc[0.05, "alface_total"] >= curva.loc[0.95, "alface_total"]
    assert curva.loc[0.05, "nao_alface_total"] <= curva.loc[0.95, "nao_alface_total"]
    assert curva.loc[0.5, "alface_folhas_doentes"] == pytest.approx(0.75)


def test_limiares_padrao():
    """De 0,05 a 0,95, de 0,05 em 0,05."""
    limiares = ad.limiares_padrao()
    assert limiares[0] == pytest.approx(0.05)
    assert limiares[-1] == pytest.approx(0.95)
    assert len(limiares) == 19


# --------------------------------------------------------------------------- #
# Cascata de ponta a ponta
# --------------------------------------------------------------------------- #


def _predicoes_oficiais() -> pd.DataFrame:
    """Predições no formato de reports/predicoes.csv (caminho do Windows)."""
    doenca = DOENCAS[0]
    return pd.DataFrame({
        "caminho": [
            rf"C:\x\dataset_clean\test\{doenca}\a.jpg",
            rf"C:\x\dataset_clean\test\{doenca}\b.jpg",
            rf"C:\x\dataset_clean\test\{doenca}\c.jpg",
            rf"C:\x\dataset_clean\test\{SAUDAVEL}\d.jpg",
        ],
        "verdadeiro": [doenca, doenca, doenca, SAUDAVEL],
        "predito": [doenca, doenca, SAUDAVEL, SAUDAVEL],
        "confianca": [0.99, 0.95, 0.90, 0.97],
    })


def _verificacoes(prob_b: float = 0.99) -> pd.DataFrame:
    doenca = DOENCAS[0]
    return pd.DataFrame({
        "arquivo": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        "verdadeiro": [doenca, doenca, doenca, SAUDAVEL],
        "prob_alface": [0.99, prob_b, 0.99, 0.99],
    })


def test_cascata_sem_recusas_preserva_o_classificador():
    """Se o verificador aceita tudo, a cascata é idêntica ao classificador."""
    _, resumo = ad.cascata_ponta_a_ponta(_predicoes_oficiais(), _verificacoes(), 0.5, 0.85)

    doenca = resumo["por_classe"][DOENCAS[0]]
    assert doenca["revocacao_classificador"]["acertos"] == 2
    assert doenca["revocacao_cascata"]["acertos"] == 2
    assert resumo["alertas_classificador"] == resumo["alertas_cascata"] == 2
    assert resumo["alertas_verdadeiros_suprimidos"] == 0


def test_cascata_mede_alerta_suprimido():
    """Folha doente recusada perde o diagnóstico e o alerta — e isso é contado."""
    _, resumo = ad.cascata_ponta_a_ponta(
        _predicoes_oficiais(), _verificacoes(prob_b=0.10), 0.5, 0.85
    )

    doenca = resumo["por_classe"][DOENCAS[0]]
    assert doenca["recusadas_pelo_verificador"] == 1
    assert doenca["revocacao_classificador"]["acertos"] == 2
    assert doenca["revocacao_cascata"]["acertos"] == 1
    assert resumo["alertas_cascata"] == 1
    assert resumo["alertas_verdadeiros_suprimidos"] == 1


def test_cascata_exige_verificacao_para_toda_predicao():
    """Comparar conjuntos diferentes invalidaria a comparação: é erro."""
    incompletas = _verificacoes().iloc[:3]
    with pytest.raises(ValueError, match="não têm verificação"):
        ad.cascata_ponta_a_ponta(_predicoes_oficiais(), incompletas, 0.5, 0.85)


# --------------------------------------------------------------------------- #
# Nomes dos arquivos por versão
# --------------------------------------------------------------------------- #


def test_nomes_sem_rotulo_sao_os_da_v1():
    """Sem rótulo, os nomes já gravados pela v1 — e a proteção do teste dela."""
    assert ad.nomes_da_avaliacao("test") == {
        "pesos": "yolo11n_cls_dominio.pt",
        "json": "verificador_test.json",
        "md": "verificador_test.md",
        "predicoes": "predicoes_verificador_test.csv",
        "cascata": "cascata_test.csv",
    }


def test_nomes_com_rotulo_nao_colidem_com_a_v1():
    """Com rótulo, todo arquivo leva a versão; nenhum coincide com os da v1."""
    for split in ("val", "test"):
        v1 = ad.nomes_da_avaliacao(split)
        v2 = ad.nomes_da_avaliacao(split, "v2")
        assert v2["pesos"] == "yolo11n_cls_dominio_v2.pt"
        assert v2["json"] == f"verificador_v2_{split}.json"
        assert v2["predicoes"] == f"predicoes_verificador_v2_{split}.csv"
        assert set(v1.values()).isdisjoint(v2.values())


def test_rotulo_v1_e_recusado():
    """'v1' abriria um segundo teste da primeira versão com outro nome."""
    with pytest.raises(ValueError, match="sem rótulo"):
        ad.nomes_da_avaliacao("test", "v1")


@pytest.mark.parametrize("rotulo", ["", "V2", "v2/..", "v 2"])
def test_rotulo_invalido_falha(rotulo):
    """O rótulo vira parte do nome do arquivo: só minúsculas, dígitos e '_'."""
    with pytest.raises(ValueError, match="minúsculas"):
        ad.nomes_da_avaliacao("val", rotulo)


def test_split_invalido_falha():
    """O Ultralytics exige 'val'; 'valid' não é um split do dataset."""
    with pytest.raises(ValueError, match="split"):
        ad.nomes_da_avaliacao("valid")
