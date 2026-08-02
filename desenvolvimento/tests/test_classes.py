"""Testes do mapeamento canônico de classes.

Referência: spec.md, Seção 11.
"""

from __future__ import annotations

import pytest

from ia_alface.classes import (
    CANONICO_PARA_INDICE,
    CLASSE_SAUDAVEL,
    CLASSES,
    INDICE_PARA_CANONICO,
    PASTA_ORIGINAL_PARA_CANONICO,
    ROTULOS_CANONICOS,
    canonico_de_pasta,
    eh_patogeno,
    rotulo_exibicao,
    validar_rotulo,
)

#: Valores exatos do ENUM da coluna 'classe' na tabela 'deteccoes' do TCC
#: (Seção 6.2.6.8). Este teste é o contrato entre o modelo e o banco de dados.
ENUM_BANCO_DE_DADOS = {"saudavel", "bacteriano", "fungico"}


def test_rotulos_correspondem_ao_enum_do_banco():
    """Os rótulos canônicos são exatamente os valores do ENUM do banco."""
    assert set(ROTULOS_CANONICOS) == ENUM_BANCO_DE_DADOS


def test_rotulos_sem_acentos_e_em_minusculas():
    """O ENUM do banco exige minúsculas sem acentuação."""
    for rotulo in ROTULOS_CANONICOS:
        assert rotulo == rotulo.lower()
        assert rotulo.isascii(), f"{rotulo!r} contém caracteres não-ASCII"


def test_mapeamento_pasta_para_canonico_e_bijetivo():
    """Cada pasta original corresponde a exatamente um rótulo canônico."""
    assert len(PASTA_ORIGINAL_PARA_CANONICO) == len(CLASSES)
    assert len(set(PASTA_ORIGINAL_PARA_CANONICO.values())) == len(CLASSES)
    assert set(PASTA_ORIGINAL_PARA_CANONICO.values()) == set(ROTULOS_CANONICOS)


def test_mapeamento_indice_e_bijetivo():
    """Índice e rótulo canônico formam uma bijeção."""
    assert len(CANONICO_PARA_INDICE) == len(CLASSES)
    assert len(INDICE_PARA_CANONICO) == len(CLASSES)
    for rotulo, indice in CANONICO_PARA_INDICE.items():
        assert INDICE_PARA_CANONICO[indice] == rotulo


def test_indices_sao_contiguos_a_partir_de_zero():
    """Os índices cobrem 0..n-1 sem lacunas."""
    assert sorted(INDICE_PARA_CANONICO) == list(range(len(CLASSES)))


def test_ordem_dos_indices_segue_ordem_alfabetica():
    """O Ultralytics indexa as classes pela ordem alfabética dos diretórios.

    Como os diretórios de saída de ``desenvolvimento/dataset_clean/`` usam os rótulos canônicos,
    a ordem alfabética destes precisa coincidir com a ordem dos índices — caso
    contrário os rótulos ficariam trocados após o treinamento.
    """
    assert list(ROTULOS_CANONICOS) == sorted(ROTULOS_CANONICOS)


@pytest.mark.parametrize(
    ("pasta", "esperado"),
    [
        ("Bacterial", "bacteriano"),
        ("bacterial", "bacteriano"),
        ("fungal", "fungico"),
        ("Fungal", "fungico"),
        ("healthy", "saudavel"),
        ("HEALTHY", "saudavel"),
        # Aceita reprocessar um dataset já normalizado.
        ("bacteriano", "bacteriano"),
        ("saudavel", "saudavel"),
    ],
)
def test_canonico_de_pasta(pasta, esperado):
    """A conversão de nome de diretório ignora diferenças de caixa."""
    assert canonico_de_pasta(pasta) == esperado


def test_canonico_de_pasta_rejeita_desconhecido():
    """Um diretório inesperado deve falhar de forma explícita."""
    with pytest.raises(KeyError, match="desconhecido"):
        canonico_de_pasta("virose")


def test_eh_patogeno():
    """Somente 'saudavel' representa ausência de patógeno."""
    assert eh_patogeno("bacteriano") is True
    assert eh_patogeno("fungico") is True
    assert eh_patogeno("saudavel") is False
    assert CLASSE_SAUDAVEL == "saudavel"


def test_rotulo_exibicao_menciona_o_patogeno():
    """Os rótulos de interface trazem o nome científico do patógeno."""
    assert "Xanthomonas campestris" in rotulo_exibicao("bacteriano")
    assert "Bremia lactucae" in rotulo_exibicao("fungico")
    assert rotulo_exibicao("saudavel") == "Saudável"


def test_validar_rotulo_rejeita_invalido():
    """Rótulos fora do conjunto canônico são rejeitados."""
    assert validar_rotulo("fungico") == "fungico"
    with pytest.raises(KeyError, match="inválido"):
        validar_rotulo("fúngico")  # com acento — não é o valor do ENUM
    with pytest.raises(KeyError):
        validar_rotulo("Bacterial")  # nome de pasta, não rótulo canônico


def test_classes_expõem_nome_cientifico_coerente():
    """Apenas a classe saudável não tem patógeno associado."""
    for classe in CLASSES:
        if classe.canonico == CLASSE_SAUDAVEL:
            assert classe.patogeno_cientifico is None
        else:
            assert classe.patogeno_cientifico
