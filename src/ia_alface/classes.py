"""Mapeamento canônico das classes de doenças foliares em alface.

Este é o **único** módulo autorizado a declarar os rótulos de classe do projeto.
Qualquer outro módulo deve importar daqui — nunca redeclarar strings de classe.

Os rótulos canônicos correspondem exatamente aos valores do ``ENUM`` da coluna
``classe`` da tabela ``deteccoes`` especificada no TCC (Seção 6.2.6.8):
``('saudavel', 'bacteriano', 'fungico')`` — sem acentos, em minúsculas.

Referência: spec.md, Seção 4.4.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

__all__ = [
    "Classe",
    "CLASSES",
    "PASTA_ORIGINAL_PARA_CANONICO",
    "CANONICO_PARA_INDICE",
    "INDICE_PARA_CANONICO",
    "ROTULOS_CANONICOS",
    "CLASSE_SAUDAVEL",
    "canonico_de_pasta",
    "rotulo_exibicao",
    "eh_patogeno",
    "validar_rotulo",
]


@dataclass(frozen=True)
class Classe:
    """Descreve uma classe do modelo.

    Attributes:
        indice: Índice numérico da classe no vetor de saída do modelo.
        canonico: Rótulo canônico usado na API e no banco de dados.
        pasta_original: Nome do diretório correspondente em ``desenvolvimento/dataset/``.
        exibicao: Rótulo legível para o dashboard web e o app mobile.
        patogeno_cientifico: Nome científico do patógeno, ou ``None`` se saudável.
    """

    indice: int
    canonico: str
    pasta_original: str
    exibicao: str
    patogeno_cientifico: str | None


# A ordem desta tupla é a ordem alfabética dos rótulos canônicos, que por sua vez
# é a ordem em que o Ultralytics indexa os diretórios de classe. Ainda assim, o
# código de inferência deve ler `model.names` em tempo de execução em vez de
# assumir esta ordem (ver spec.md, Seção 4.4).
CLASSES: tuple[Classe, ...] = (
    Classe(
        indice=0,
        canonico="bacteriano",
        pasta_original="Bacterial",
        exibicao="Bacteriano (Xanthomonas campestris)",
        patogeno_cientifico="Xanthomonas campestris",
    ),
    Classe(
        indice=1,
        canonico="fungico",
        pasta_original="fungal",
        exibicao="Fúngico (Bremia lactucae)",
        patogeno_cientifico="Bremia lactucae",
    ),
    Classe(
        indice=2,
        canonico="saudavel",
        pasta_original="healthy",
        exibicao="Saudável",
        patogeno_cientifico=None,
    ),
)

#: Rótulo canônico da classe que representa ausência de patógeno.
CLASSE_SAUDAVEL: str = "saudavel"

#: Rótulos canônicos em ordem de índice.
ROTULOS_CANONICOS: tuple[str, ...] = tuple(c.canonico for c in CLASSES)

#: Nome do diretório em ``desenvolvimento/dataset/`` -> rótulo canônico.
PASTA_ORIGINAL_PARA_CANONICO: Mapping[str, str] = MappingProxyType(
    {c.pasta_original: c.canonico for c in CLASSES}
)

#: Rótulo canônico -> índice numérico.
CANONICO_PARA_INDICE: Mapping[str, int] = MappingProxyType(
    {c.canonico: c.indice for c in CLASSES}
)

#: Índice numérico -> rótulo canônico.
INDICE_PARA_CANONICO: Mapping[int, str] = MappingProxyType(
    {c.indice: c.canonico for c in CLASSES}
)

_POR_CANONICO: Mapping[str, Classe] = MappingProxyType({c.canonico: c for c in CLASSES})


def canonico_de_pasta(nome_pasta: str) -> str:
    """Converte o nome de um diretório de ``desenvolvimento/dataset/`` no rótulo canônico.

    A comparação ignora diferenças de caixa, pois o dataset original mistura
    convenções (``Bacterial`` com maiúscula, ``fungal`` e ``healthy`` sem).

    Args:
        nome_pasta: Nome do diretório de classe, por exemplo ``"Bacterial"``.

    Returns:
        O rótulo canônico correspondente.

    Raises:
        KeyError: Se o diretório não corresponder a nenhuma classe conhecida.
    """
    alvo = nome_pasta.strip().lower()
    for classe in CLASSES:
        if classe.pasta_original.lower() == alvo:
            return classe.canonico
        # Aceita também o próprio rótulo canônico, para permitir reprocessar
        # um dataset já normalizado.
        if classe.canonico == alvo:
            return classe.canonico
    raise KeyError(
        f"Diretório de classe desconhecido: {nome_pasta!r}. "
        f"Esperado um de {[c.pasta_original for c in CLASSES]}."
    )


def rotulo_exibicao(canonico: str) -> str:
    """Retorna o rótulo legível para interface a partir do rótulo canônico.

    Args:
        canonico: Rótulo canônico, por exemplo ``"fungico"``.

    Returns:
        Texto para exibição, por exemplo ``"Fúngico (Bremia lactucae)"``.

    Raises:
        KeyError: Se o rótulo canônico for desconhecido.
    """
    return _POR_CANONICO[validar_rotulo(canonico)].exibicao


def eh_patogeno(canonico: str) -> bool:
    """Informa se o rótulo canônico representa a presença de um patógeno.

    Args:
        canonico: Rótulo canônico.

    Returns:
        ``True`` para ``bacteriano`` e ``fungico``; ``False`` para ``saudavel``.

    Raises:
        KeyError: Se o rótulo canônico for desconhecido.
    """
    return validar_rotulo(canonico) != CLASSE_SAUDAVEL


def validar_rotulo(canonico: str) -> str:
    """Valida que o rótulo pertence ao conjunto canônico.

    Args:
        canonico: Rótulo a validar.

    Returns:
        O próprio rótulo, se válido.

    Raises:
        KeyError: Se o rótulo não for canônico.
    """
    if canonico not in _POR_CANONICO:
        raise KeyError(
            f"Rótulo de classe inválido: {canonico!r}. "
            f"Esperado um de {list(ROTULOS_CANONICOS)}."
        )
    return canonico
