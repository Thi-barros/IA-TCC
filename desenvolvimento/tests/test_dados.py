"""Testes da preparação do dataset.

Os testes usam conjuntos sintéticos pequenos, nunca o dataset real: a suíte deve
rodar em segundos e não depender de ``desenvolvimento/dataset/`` estar presente.

Referência: spec.md, Seção 11.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ia_alface import dados
from ia_alface.classes import CANONICO_PARA_INDICE, ROTULOS_CANONICOS
from ia_alface.dados import (
    _unificar_grupos_por_hash,
    deduplicar,
    dividir_por_grupos,
    normalizar_grupo_origem,
    verificar_integridade,
)


# --------------------------------------------------------------------------- #
# normalizar_grupo_origem
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("nome", "classe", "esperado"),
    [
        # Casos exigidos explicitamente pela Seção 11 do spec.
        ("ba12_3.png", "bacteriano", "ba12"),
        ("ba1_4 - Copy.png", "bacteriano", "ba1"),
        ("h100_2.png", "saudavel", "h100"),
        # Variantes de sufixo de cópia do Windows.
        ("ba12_0 - Copy (2).png", "bacteriano", "ba12"),
        ("ba4_3 - Copy.png", "bacteriano", "ba4"),
        ("ba1_4 - Copy - Copy.png", "bacteriano", "ba1"),
        # Sem sufixo numérico.
        ("h10.jpg", "saudavel", "h10"),
        ("01.png", "fungico", "01"),
        # Sufixo numérico multi-dígito.
        ("01_12.png", "fungico", "01"),
        # Extensões diferentes convergem para o mesmo grupo.
        ("h100.jpg", "saudavel", "h100"),
        ("h100.jpeg", "saudavel", "h100"),
    ],
)
def test_normalizar_grupo_origem(nome, classe, esperado):
    """A chave de agrupamento remove sufixos de cópia e de variante."""
    assert normalizar_grupo_origem(nome, classe) == esperado


def test_h10_e_h100_sao_grupos_distintos():
    """A remoção do sufixo não deve confundir prefixos parecidos."""
    assert normalizar_grupo_origem("h10_1.png", "saudavel") == "h10"
    assert normalizar_grupo_origem("h100_1.png", "saudavel") == "h100"


def test_augmented_forma_grupo_unitario():
    """Arquivos 'augmented_*' recebem um grupo unitário por índice sequencial."""
    assert (
        normalizar_grupo_origem("augmented_0.jpg", "bacteriano", 7)
        == "augmented_bacteriano_7"
    )
    # O índice vem do chamador, não do nome: dois arquivos 'augmented_0.jpg' de
    # splits diferentes são imagens distintas e não devem colapsar no mesmo grupo.
    primeiro = normalizar_grupo_origem("augmented_0.jpg", "saudavel", 0)
    segundo = normalizar_grupo_origem("augmented_0.jpg", "saudavel", 1)
    assert primeiro != segundo


def test_augmented_sem_indice_falha():
    """Omitir o índice sequencial é um erro de programação, não um caso válido."""
    with pytest.raises(ValueError, match="indice_unico"):
        normalizar_grupo_origem("augmented_0.jpg", "bacteriano")


def test_nome_que_reduz_a_vazio_falha():
    """Um nome composto apenas por sufixos deve falhar explicitamente."""
    with pytest.raises(ValueError):
        normalizar_grupo_origem("_3.png", "fungico")


# --------------------------------------------------------------------------- #
# Fixtures sintéticas
# --------------------------------------------------------------------------- #


def _inventario_sintetico(
    grupos_por_classe: int = 40, imagens_por_grupo: int = 2
) -> pd.DataFrame:
    """Cria um inventário sintético com grupos e hashes únicos."""
    registros = []
    contador = 0
    for classe in ROTULOS_CANONICOS:
        for numero_grupo in range(grupos_por_classe):
            grupo = f"{classe}/g{numero_grupo:03d}"
            for numero_imagem in range(imagens_por_grupo):
                contador += 1
                registros.append(
                    {
                        "caminho": f"/fake/{classe}/g{numero_grupo:03d}_{numero_imagem}.jpg",
                        "arquivo": f"g{numero_grupo:03d}_{numero_imagem}.jpg",
                        "classe": classe,
                        "indice_classe": CANONICO_PARA_INDICE[classe],
                        "grupo_origem": f"g{numero_grupo:03d}",
                        "grupo_id": grupo,
                        "hash_md5": f"{contador:032x}",
                        "split_original": "train",
                        "largura": 224,
                        "altura": 224,
                        "formato": "JPEG",
                        "modo": "RGB",
                        "bytes": 1000,
                    }
                )
    return pd.DataFrame(registros)


# --------------------------------------------------------------------------- #
# Divisão sem vazamento
# --------------------------------------------------------------------------- #


def test_divisao_nao_vaza_grupos_nem_hashes():
    """Nenhum grupo e nenhum hash pode aparecer em mais de um split."""
    inventario = _inventario_sintetico()
    dividido = dividir_por_grupos(inventario, seed=42)

    assert verificar_integridade(dividido) == []

    for coluna in ("grupo_id", "hash_md5"):
        por_chave = dividido.groupby(coluna)["split"].nunique()
        assert (por_chave == 1).all(), f"{coluna} vazou entre splits"


def test_divisao_respeita_as_proporcoes():
    """A divisão deve ficar próxima de 70/15/15."""
    inventario = _inventario_sintetico(grupos_por_classe=60)
    dividido = dividir_por_grupos(inventario, seed=42)

    fracoes = dividido["split"].value_counts(normalize=True)
    assert fracoes["train"] == pytest.approx(0.70, abs=0.06)
    assert fracoes["val"] == pytest.approx(0.15, abs=0.06)
    assert fracoes["test"] == pytest.approx(0.15, abs=0.06)


def test_divisao_e_estratificada():
    """Todas as classes devem estar presentes em todos os splits."""
    inventario = _inventario_sintetico()
    dividido = dividir_por_grupos(inventario, seed=42)

    for split in dados.SPLITS_SAIDA:
        classes = set(dividido[dividido["split"] == split]["classe"])
        assert classes == set(ROTULOS_CANONICOS)


def test_divisao_e_reprodutivel():
    """A mesma semente deve produzir a mesma divisão."""
    inventario = _inventario_sintetico()
    primeira = dividir_por_grupos(inventario, seed=42)["split"].tolist()
    segunda = dividir_por_grupos(inventario, seed=42)["split"].tolist()
    assert primeira == segunda


def test_alocador_guloso_tambem_nao_vaza():
    """O caminho alternativo (poucos grupos por classe) mantém as garantias."""
    inventario = _inventario_sintetico(grupos_por_classe=8, imagens_por_grupo=3)
    # n_splits_kfold acima do número de grupos força o alocador guloso.
    dividido = dividir_por_grupos(inventario, seed=42, n_splits_kfold=20)

    assert verificar_integridade(dividido) == []
    por_grupo = dividido.groupby("grupo_id")["split"].nunique()
    assert (por_grupo == 1).all()


def test_proporcoes_invalidas_falham():
    """Proporções que não somam 1,0 são rejeitadas."""
    inventario = _inventario_sintetico(grupos_por_classe=25)
    with pytest.raises(ValueError, match="somar"):
        dividir_por_grupos(inventario, proporcoes=(0.8, 0.15, 0.15))


# --------------------------------------------------------------------------- #
# Deduplicação e fusão de grupos
# --------------------------------------------------------------------------- #


def test_deduplicar_mantem_uma_copia_por_hash():
    """Duplicatas exatas são reduzidas a uma única cópia, de forma determinística."""
    inventario = _inventario_sintetico(grupos_por_classe=5)
    duplicata = inventario.iloc[[0]].copy()
    duplicata["caminho"] = "/fake/zzz_ultimo.jpg"
    com_duplicata = pd.concat([inventario, duplicata], ignore_index=True)

    mantidos, descartados = deduplicar(com_duplicata)

    assert len(mantidos) == len(inventario)
    assert len(descartados) == 1
    assert mantidos["hash_md5"].is_unique
    # Mantém o menor caminho em ordem lexicográfica.
    assert "/fake/zzz_ultimo.jpg" in set(descartados["caminho"])


def test_fusao_de_grupos_por_conteudo_identico():
    """Grupos que compartilham um arquivo idêntico são fundidos em um só.

    Reproduz o caso real em que ``ba15_1.png`` e ``ba6_0.png`` são o mesmo
    arquivo: sem a fusão, outras variantes de ``ba15`` e de ``ba6`` poderiam cair
    em splits diferentes.
    """
    registros = [
        {
            "caminho": "/fake/ba6_0.png",
            "arquivo": "ba6_0.png",
            "classe": "bacteriano",
            "indice_classe": 0,
            "grupo_origem": "ba6",
            "grupo_id": "bacteriano/ba6",
            "hash_md5": "aaaa",
            "split_original": "train",
            "largura": 224, "altura": 224, "formato": "PNG", "modo": "RGB", "bytes": 1,
        },
        {
            "caminho": "/fake/ba15_1.png",
            "arquivo": "ba15_1.png",
            "classe": "bacteriano",
            "indice_classe": 0,
            "grupo_origem": "ba15",
            "grupo_id": "bacteriano/ba15",
            "hash_md5": "aaaa",  # mesmo conteúdo de ba6_0
            "split_original": "train",
            "largura": 224, "altura": 224, "formato": "PNG", "modo": "RGB", "bytes": 1,
        },
        {
            "caminho": "/fake/ba15_2.png",
            "arquivo": "ba15_2.png",
            "classe": "bacteriano",
            "indice_classe": 0,
            "grupo_origem": "ba15",
            "grupo_id": "bacteriano/ba15",
            "hash_md5": "bbbb",
            "split_original": "valid",
            "largura": 224, "altura": 224, "formato": "PNG", "modo": "RGB", "bytes": 1,
        },
    ]
    unificado = _unificar_grupos_por_hash(pd.DataFrame(registros))

    assert unificado["grupo_id"].nunique() == 1
    assert set(unificado["grupo_id_original"]) == {"bacteriano/ba6", "bacteriano/ba15"}


# --------------------------------------------------------------------------- #
# verificar_integridade
# --------------------------------------------------------------------------- #


def test_verificar_integridade_detecta_vazamento_de_grupo():
    """Um grupo em dois splits deve ser reportado."""
    inventario = _inventario_sintetico(grupos_por_classe=25)
    dividido = dividir_por_grupos(inventario, seed=42)

    # Move uma única imagem para outro split, quebrando a disjunção do grupo.
    indice = dividido.index[dividido["split"] == "train"][0]
    dividido.loc[indice, "split"] = "test"

    problemas = verificar_integridade(dividido)
    assert any("grupos de origem" in p for p in problemas)


def test_verificar_integridade_detecta_split_sem_coluna():
    """Um inventário sem a coluna 'split' é reportado, não quebra."""
    inventario = _inventario_sintetico(grupos_por_classe=5)
    problemas = verificar_integridade(inventario)
    assert problemas and "split" in problemas[0]


def test_verificar_integridade_detecta_classe_ausente():
    """Um split sem uma das três classes é reportado."""
    inventario = _inventario_sintetico(grupos_por_classe=25)
    dividido = dividir_por_grupos(inventario, seed=42)
    dividido = dividido[
        ~((dividido["split"] == "test") & (dividido["classe"] == "fungico"))
    ]

    problemas = verificar_integridade(dividido)
    assert any("não contém as classes" in p for p in problemas)


# --------------------------------------------------------------------------- #
# Relatórios auxiliares
# --------------------------------------------------------------------------- #


def test_resumo_por_split_soma_o_total():
    """A tabela de resumo é consistente com o inventário."""
    inventario = _inventario_sintetico(grupos_por_classe=25)
    dividido = dividir_por_grupos(inventario, seed=42)
    resumo = dados.resumo_por_split(dividido)

    assert int(resumo["Total"].sum()) == len(dividido)
    assert list(resumo.index) == list(dados.SPLITS_SAIDA)
    assert resumo["%"].sum() == pytest.approx(100.0, abs=0.5)


def test_contar_grupos_por_classe():
    """A contagem de grupos por classe evidencia a diversidade real."""
    inventario = _inventario_sintetico(grupos_por_classe=10, imagens_por_grupo=4)
    tabela = dados.contar_grupos_por_classe(inventario)

    assert list(tabela.index) == list(ROTULOS_CANONICOS)
    assert (tabela["grupos"] == 10).all()
    assert (tabela["imagens"] == 40).all()
    assert (tabela["imagens_por_grupo"] == 4.0).all()


def test_splits_de_saida_usam_val_e_nao_valid():
    """O Ultralytics exige o diretório 'val' para classificação."""
    assert dados.SPLITS_SAIDA == ("train", "val", "test")
    assert "valid" not in dados.SPLITS_SAIDA
