"""Configuração do serviço de inferência, lida de variáveis de ambiente.

Todas as opções possuem padrão utilizável, de modo que o serviço sobe sem
nenhuma variável definida. Em produção, ajuste ao menos ``IA_ALFACE_CORS_ORIGENS``
para não manter a política permissiva de desenvolvimento.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# api/configuracao.py -> api -> raiz do projeto
RAIZ_PROJETO: Path = Path(__file__).resolve().parents[1]


def _texto(nome: str, padrao: str) -> str:
    return os.environ.get(nome, padrao).strip()


def _numero(nome: str, padrao: float) -> float:
    bruto = os.environ.get(nome)
    if bruto is None or not bruto.strip():
        return padrao
    try:
        return float(bruto)
    except ValueError:
        raise ValueError(
            f"A variável de ambiente {nome} deve conter um número; "
            f"recebido {bruto!r}."
        ) from None


def _lista(nome: str, padrao: list[str]) -> list[str]:
    bruto = os.environ.get(nome)
    if bruto is None or not bruto.strip():
        return padrao
    return [item.strip() for item in bruto.split(",") if item.strip()]


@dataclass(frozen=True)
class Configuracao:
    """Parâmetros de execução do serviço.

    Attributes:
        caminho_modelo: Arquivo de pesos. Aceita ``.pt`` (necessário para o
            Grad-CAM) ou ``.onnx`` (mais rápido, sem Grad-CAM).
        dispositivo: ``"cuda"``, ``"cpu"`` ou vazio para autodetecção.
        limiar_confianca: Confiança mínima para que uma detecção de patógeno
            acione o campo ``alerta``.
        tamanho_maximo_mb: Limite do corpo da requisição, em megabytes.
        cors_origens: Origens autorizadas. ``["*"]`` libera todas.
        titulo: Título exibido na documentação interativa.
    """

    caminho_modelo: Path = field(
        default_factory=lambda: Path(
            _texto(
                "IA_ALFACE_MODELO",
                str(RAIZ_PROJETO / "modelos" / "yolo11n_cls_alface.pt"),
            )
        )
    )
    dispositivo: str | None = field(
        default_factory=lambda: _texto("IA_ALFACE_DISPOSITIVO", "") or None
    )
    limiar_confianca: float = field(
        default_factory=lambda: _numero("IA_ALFACE_LIMIAR", 0.85)
    )
    tamanho_maximo_mb: float = field(
        default_factory=lambda: _numero("IA_ALFACE_MAX_MB", 10.0)
    )
    cors_origens: list[str] = field(
        default_factory=lambda: _lista("IA_ALFACE_CORS_ORIGENS", ["*"])
    )
    titulo: str = "API de Detecção de Doenças Foliares em Alface"

    @property
    def tamanho_maximo_bytes(self) -> int:
        """Limite do corpo da requisição, em bytes."""
        return int(self.tamanho_maximo_mb * 1024 * 1024)

    @property
    def suporta_heatmap(self) -> bool:
        """Grad-CAM exige os pesos ``.pt``; modelos exportados não servem."""
        return self.caminho_modelo.suffix.lower() == ".pt"


CONFIGURACAO = Configuracao()
