"""Métricas do verificador de domínio e da cascata, sem dependência de modelo.

As funções recebem tabelas de previsões já calculadas, o que permite testá-las
com dados sintéticos. Quem roda os modelos é
``desenvolvimento/scripts/08_avaliar_verificador.py``.

Cada proporção vem com o intervalo de confiança de 95% de Wilson: com amostras
pequenas (as fotos da câmera do projeto), o percentual sozinho sugere uma
precisão que a amostra não tem.

A cascata de ponta a ponta usa as predições **oficiais** do classificador no
teste (``reports/predicoes.csv``), sem executá-lo de novo sobre o split de
teste.

Referência: docs/VERIFICADOR_DOMINIO.md.
"""

from __future__ import annotations

import math
import re
from pathlib import PureWindowsPath
from typing import Iterable

import numpy as np
import pandas as pd

from .classes import CLASSE_ALFACE, CLASSE_NAO_ALFACE, ROTULOS_CANONICOS, eh_patogeno
from .dados_dominio import CAMERA_PROJETO, FONTE_ESTUFA, FONTE_KAGGLE

__all__ = [
    "intervalo_wilson",
    "proporcao",
    "grupo_de_avaliacao",
    "metricas_por_grupo",
    "agregados",
    "curva_limiar",
    "cascata_ponta_a_ponta",
    "limiares_padrao",
    "nomes_da_avaliacao",
]

#: Rótulos de doença que representam patógeno (bacteriano e fungico).
_PATOGENOS: tuple[str, ...] = tuple(c for c in ROTULOS_CANONICOS if eh_patogeno(c))

#: Nome dos pesos da primeira versão do verificador; as seguintes ganham o
#: rótulo como sufixo (``yolo11n_cls_dominio_v2``).
NOME_BASE_VERIFICADOR: str = "yolo11n_cls_dominio"

#: Rótulo de versão aceito: minúsculas, dígitos e ``_`` — vai para nomes de arquivo.
_PADRAO_ROTULO = re.compile(r"[a-z0-9_]+")


def intervalo_wilson(acertos: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    """Intervalo de confiança de Wilson para uma proporção.

    Ao contrário do intervalo normal, continua informativo com 0 ou 100% de
    acertos e amostras pequenas.

    Args:
        acertos: Número de sucessos.
        total: Tamanho da amostra.
        z: Quantil da normal (1,96 para 95%).

    Returns:
        ``(inferior, superior)`` em ``[0, 1]``, ou ``None`` se ``total`` for 0.
    """
    if total <= 0:
        return None
    p = acertos / total
    z2 = z * z
    denominador = 1 + z2 / total
    centro = (p + z2 / (2 * total)) / denominador
    meia_largura = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / denominador
    return max(0.0, centro - meia_largura), min(1.0, centro + meia_largura)


def proporcao(acertos: int, total: int) -> dict:
    """Taxa de acerto com o intervalo de Wilson.

    Args:
        acertos: Número de acertos.
        total: Tamanho da amostra.

    Returns:
        Dicionário com ``acertos``, ``total``, ``taxa``, ``ic95_inf`` e
        ``ic95_sup``; os três últimos são ``None`` quando ``total`` é 0.
    """
    intervalo = intervalo_wilson(acertos, total)
    return {
        "acertos": int(acertos),
        "total": int(total),
        "taxa": round(acertos / total, 4) if total else None,
        "ic95_inf": round(intervalo[0], 4) if intervalo else None,
        "ic95_sup": round(intervalo[1], 4) if intervalo else None,
    }


def grupo_de_avaliacao(fonte: str, estrato: str, camera: str) -> str:
    """Grupo de relatório de uma imagem.

    Args:
        fonte: Fonte da imagem (``kaggle``, ``estufa``, ``plantvillage``…).
        estrato: Estrato do inventário; no Kaggle, a classe de doença.
        camera: Câmera, para fotos da estufa.

    Returns:
        ``kaggle/<doença>``, ``estufa/<câmera>`` ou o nome da fonte pública.
    """
    if fonte == FONTE_KAGGLE:
        return f"{FONTE_KAGGLE}/{estrato}"
    if fonte == FONTE_ESTUFA:
        return f"{FONTE_ESTUFA}/{camera}"
    return fonte


def _acertos(previsoes: pd.DataFrame, limiar: float) -> pd.Series:
    """Acerto por imagem: alface aceita, ou não alface recusada."""
    aceito = previsoes["prob_alface"] >= limiar
    eh_alface = previsoes["classe"] == CLASSE_ALFACE
    return (aceito & eh_alface) | (~aceito & ~eh_alface)


def metricas_por_grupo(previsoes: pd.DataFrame, limiar: float) -> pd.DataFrame:
    """Revocação de cada grupo de imagens, no limiar dado.

    Args:
        previsoes: Colunas ``classe``, ``grupo`` e ``prob_alface``.
        limiar: Probabilidade mínima de ``alface`` para aceitar.

    Returns:
        Uma linha por ``(classe, grupo)``, com acertos, total, taxa e IC95.
    """
    marcadas = previsoes.assign(acerto=_acertos(previsoes, limiar))
    linhas = []
    for (classe, grupo), bloco in marcadas.groupby(["classe", "grupo"], sort=True):
        linhas.append({"classe": classe, "grupo": grupo,
                       **proporcao(int(bloco["acerto"].sum()), len(bloco))})
    return pd.DataFrame(linhas)


def agregados(previsoes: pd.DataFrame, limiar: float) -> dict[str, dict]:
    """Revocações agregadas usadas nas metas e na curva de limiar.

    Args:
        previsoes: Colunas ``classe``, ``fonte``, ``estrato``, ``camera`` e
            ``prob_alface``.
        limiar: Probabilidade mínima de ``alface`` para aceitar.

    Returns:
        Proporções de: alface em folhas doentes (Kaggle, bacteriano e fungico),
        alface no total, alface da câmera do projeto, não alface no total e não
        alface da câmera do projeto.
    """
    acerto = _acertos(previsoes, limiar)
    alface = previsoes["classe"] == CLASSE_ALFACE
    nao_alface = previsoes["classe"] == CLASSE_NAO_ALFACE
    doentes = (previsoes["fonte"] == FONTE_KAGGLE) & previsoes["estrato"].isin(_PATOGENOS)
    camera_projeto = (previsoes["fonte"] == FONTE_ESTUFA) & (previsoes["camera"] == CAMERA_PROJETO)

    def medir(mascara: pd.Series) -> dict:
        return proporcao(int(acerto[mascara].sum()), int(mascara.sum()))

    return {
        "alface_folhas_doentes": medir(alface & doentes),
        "alface_total": medir(alface),
        "alface_camera_projeto": medir(alface & camera_projeto),
        "nao_alface_total": medir(nao_alface),
        "nao_alface_camera_projeto": medir(nao_alface & camera_projeto),
    }


def curva_limiar(previsoes: pd.DataFrame, limiares: Iterable[float]) -> pd.DataFrame:
    """Revocações agregadas para cada limiar candidato.

    Uso exclusivo no split ``val``: é a base para escolher o limiar.

    Args:
        previsoes: Mesmas colunas de :func:`agregados`.
        limiares: Limiares a avaliar.

    Returns:
        Uma linha por limiar, com a taxa de cada agregado.
    """
    linhas = []
    for limiar in limiares:
        medidas = agregados(previsoes, limiar)
        linhas.append({"limiar": round(float(limiar), 4),
                       **{chave: valor["taxa"] for chave, valor in medidas.items()}})
    return pd.DataFrame(linhas)


def cascata_ponta_a_ponta(
    predicoes_classificador: pd.DataFrame,
    verificacoes: pd.DataFrame,
    limiar_alface: float,
    limiar_confianca: float,
) -> tuple[pd.DataFrame, dict]:
    """Combina as predições oficiais do classificador com o verificador.

    Uma imagem só recebe diagnóstico se o verificador a aceitar. Mede quanto a
    cascata muda a revocação de cada doença e quantos alertas o verificador
    suprime.

    Args:
        predicoes_classificador: ``reports/predicoes.csv`` do teste oficial
            (``caminho``, ``verdadeiro``, ``predito``, ``confianca``).
        verificacoes: Saída do verificador para as mesmas imagens
            (``arquivo``, ``verdadeiro``, ``prob_alface``).
        limiar_alface: Limiar do verificador.
        limiar_confianca: Limiar de alerta do classificador.

    Returns:
        ``(tabela por imagem, resumo)``.

    Raises:
        ValueError: Se alguma predição do classificador não tiver verificação
            correspondente — a comparação ficaria sobre conjuntos diferentes.
    """
    oficiais = predicoes_classificador.assign(
        arquivo=predicoes_classificador["caminho"].map(lambda c: PureWindowsPath(c).name)
    )
    unido = oficiais.merge(
        verificacoes[["arquivo", "verdadeiro", "prob_alface"]],
        on=["arquivo", "verdadeiro"],
        how="left",
        validate="one_to_one",
    )
    faltando = int(unido["prob_alface"].isna().sum())
    if faltando:
        raise ValueError(
            f"{faltando} predições do classificador não têm verificação "
            "correspondente; confira se o dataset_dominio foi gerado do mesmo "
            "dataset_clean do teste oficial."
        )

    unido["aceito"] = unido["prob_alface"] >= limiar_alface
    unido["acerto_classificador"] = unido["predito"] == unido["verdadeiro"]
    unido["acerto_cascata"] = unido["acerto_classificador"] & unido["aceito"]
    unido["alerta_classificador"] = unido["predito"].map(eh_patogeno) & (
        unido["confianca"] >= limiar_confianca
    )
    unido["alerta_cascata"] = unido["alerta_classificador"] & unido["aceito"]

    por_classe = {}
    for classe in ROTULOS_CANONICOS:
        bloco = unido[unido["verdadeiro"] == classe]
        por_classe[classe] = {
            "total": len(bloco),
            "recusadas_pelo_verificador": int((~bloco["aceito"]).sum()),
            "revocacao_classificador": proporcao(int(bloco["acerto_classificador"].sum()), len(bloco)),
            "revocacao_cascata": proporcao(int(bloco["acerto_cascata"].sum()), len(bloco)),
        }

    doentes = unido["verdadeiro"].isin(_PATOGENOS)
    resumo = {
        "imagens": len(unido),
        "por_classe": por_classe,
        "alertas_classificador": int(unido["alerta_classificador"].sum()),
        "alertas_cascata": int(unido["alerta_cascata"].sum()),
        "alertas_verdadeiros_suprimidos": int(
            (unido["alerta_classificador"] & ~unido["aceito"] & doentes).sum()
        ),
        "acuracia_classificador": proporcao(int(unido["acerto_classificador"].sum()), len(unido)),
        "acuracia_cascata": proporcao(int(unido["acerto_cascata"].sum()), len(unido)),
    }
    return unido, resumo


def limiares_padrao() -> np.ndarray:
    """Limiares candidatos da curva: 0,05 a 0,95, de 0,05 em 0,05."""
    return np.round(np.arange(0.05, 0.951, 0.05), 2)


def nomes_da_avaliacao(split: str, rotulo: str | None = None) -> dict[str, str]:
    """Nomes dos pesos e dos arquivos de saída da avaliação de uma versão.

    Sem ``rotulo`` valem os nomes da primeira versão, que não levavam a versão
    no nome: os arquivos já gravados continuam válidos e o teste da v1 continua
    protegido. Com ``rotulo`` (por exemplo ``"v2"``), ele entra em todos os
    nomes, e cada versão tem o próprio teste único sem sobrescrever as
    anteriores.

    Args:
        split: ``"val"`` ou ``"test"``.
        rotulo: Versão do verificador: minúsculas, dígitos ou ``_``. ``"v1"``
            é recusado — a v1 usa os nomes sem rótulo, e aceitá-lo abriria um
            segundo teste da mesma versão.

    Returns:
        Mapa com ``pesos``, ``json``, ``md``, ``predicoes`` e ``cascata``.

    Raises:
        ValueError: Se o split ou o rótulo forem inválidos.
    """
    if split not in ("val", "test"):
        raise ValueError(f"split deve ser 'val' ou 'test'; recebido {split!r}.")
    if rotulo is None:
        sufixo = ""
    elif rotulo == "v1":
        raise ValueError("A v1 usa os nomes sem rótulo: omita --rotulo para ela.")
    elif _PADRAO_ROTULO.fullmatch(rotulo):
        sufixo = f"_{rotulo}"
    else:
        raise ValueError(
            f"rotulo deve ter só minúsculas, dígitos ou '_'; recebido {rotulo!r}."
        )
    return {
        "pesos": f"{NOME_BASE_VERIFICADOR}{sufixo}.pt",
        "json": f"verificador{sufixo}_{split}.json",
        "md": f"verificador{sufixo}_{split}.md",
        "predicoes": f"predicoes_verificador{sufixo}_{split}.csv",
        "cascata": f"cascata{sufixo}_{split}.csv",
    }
