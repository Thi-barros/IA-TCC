"""Módulo de IA para detecção automática de doenças foliares em alface.

Biblioteca de visão computacional baseada em Ultralytics YOLO11 (tarefa de
classificação) para identificar folhas de alface (*Lactuca sativa*) saudáveis,
com infecção bacteriana (*Xanthomonas campestris*) ou fúngica
(*Bremia lactucae*).

Este pacote é uma **biblioteca pura**: não expõe rotas HTTP, cliente MQTT nem
acesso a banco de dados. A integração com o back-end FastAPI é feita pelo
consumidor — ver ``docs/README.md``.

Referência: spec.md (raiz do projeto).
"""

from __future__ import annotations

from .classes import (
    CANONICO_PARA_INDICE,
    CLASSE_ALFACE,
    CLASSE_NAO_ALFACE,
    CLASSE_SAUDAVEL,
    CLASSES,
    INDICE_PARA_CANONICO,
    PASTA_ORIGINAL_PARA_CANONICO,
    ROTULOS_CANONICOS,
    ROTULOS_DOMINIO,
    Classe,
    canonico_de_pasta,
    eh_patogeno,
    rotulo_exibicao,
    validar_rotulo,
)
from .config import CAMINHOS, Caminhos, ConfigTreino, detectar_dispositivo

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # classes
    "Classe",
    "CLASSES",
    "ROTULOS_CANONICOS",
    "CLASSE_SAUDAVEL",
    "PASTA_ORIGINAL_PARA_CANONICO",
    "CANONICO_PARA_INDICE",
    "INDICE_PARA_CANONICO",
    "canonico_de_pasta",
    "rotulo_exibicao",
    "eh_patogeno",
    "validar_rotulo",
    "CLASSE_ALFACE",
    "CLASSE_NAO_ALFACE",
    "ROTULOS_DOMINIO",
    # config
    "CAMINHOS",
    "Caminhos",
    "ConfigTreino",
    "detectar_dispositivo",
]


def __getattr__(nome: str):
    """Importação tardia dos módulos que dependem de torch/ultralytics.

    Evita o custo de carregar o PyTorch quando apenas ``classes`` ou ``config``
    são necessários (por exemplo, nos testes de mapeamento de classes).
    """
    if nome == "DetectorDoencasAlface":
        from .inferencia import DetectorDoencasAlface

        return DetectorDoencasAlface
    if nome == "ResultadoDeteccao":
        from .inferencia import ResultadoDeteccao

        return ResultadoDeteccao
    if nome == "VerificadorDominio":
        from .verificacao import VerificadorDominio

        return VerificadorDominio
    if nome == "ResultadoVerificacao":
        from .verificacao import ResultadoVerificacao

        return ResultadoVerificacao
    raise AttributeError(f"módulo {__name__!r} não possui atributo {nome!r}")
