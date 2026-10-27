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


def _caminho_opcional(nome: str) -> Path | None:
    bruto = _texto(nome, "")
    return Path(bruto) if bruto else None


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
        cors_origens: Origens autorizadas. ``["*"]`` libera todas. **Atenção:**
            definir a variável como texto vazio cai no padrão permissivo; para
            restringir, informe as origens explicitamente.
        chave_api: Chave exigida no cabeçalho ``X-API-Key`` pelas rotas de
            inferência. Vazia desativa a autenticação.
        caminho_modelo_dominio: Pesos do verificador de domínio ("a imagem
            contém alface?"). Vazio desliga a etapa 1 da cascata e o serviço
            responde como antes. Definido, imagens sem alface recebem 200 com
            ``classe = "nao_alface"``, sem diagnóstico nem alerta.
        limiar_dominio: Probabilidade mínima de ``alface`` para a imagem seguir
            ao classificador. Fixado em 0,50 para a v2, antes do teste
            (``docs/VERIFICADOR_DOMINIO.md``, Seção 1).
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
    chave_api: str = field(default_factory=lambda: _texto("IA_ALFACE_CHAVE_API", ""))
    caminho_modelo_dominio: Path | None = field(
        default_factory=lambda: _caminho_opcional("IA_ALFACE_MODELO_DOMINIO")
    )
    # Mesmo padrão de ia_alface.verificacao.LIMIAR_ALFACE_PADRAO. Repetido aqui
    # para que este módulo continue sem depender da biblioteca na importação.
    limiar_dominio: float = field(
        default_factory=lambda: _numero("IA_ALFACE_LIMIAR_DOMINIO", 0.5)
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

    @property
    def exige_autenticacao(self) -> bool:
        """Indica se as rotas de inferência exigem chave de API.

        A autenticação liga apenas quando ``IA_ALFACE_CHAVE_API`` está definida.
        Sem ela o serviço se comporta como sempre — desenvolvimento local, a
        página ``/teste`` e a suíte automatizada seguem sem credencial. A
        proteção existe para a implantação, onde o serviço fica exposto na
        internet.
        """
        return bool(self.chave_api)

    @property
    def verifica_dominio(self) -> bool:
        """Indica se a etapa 1 da cascata (verificador de domínio) está configurada."""
        return self.caminho_modelo_dominio is not None


CONFIGURACAO = Configuracao()
