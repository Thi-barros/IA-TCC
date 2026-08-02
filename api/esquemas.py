"""Esquemas de requisição e resposta da API.

Os campos de :class:`RespostaDeteccao` correspondem um a um às colunas da tabela
``deteccoes`` especificada no projeto, de modo que o serviço consumidor possa
persistir o resultado sem transformação adicional.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = [
    "RespostaDeteccao",
    "RespostaSaude",
    "RespostaInfo",
    "RespostaErro",
    "ClasseInfo",
]


class RespostaDeteccao(BaseModel):
    """Resultado da classificação de uma imagem de folha de alface."""

    classe: str = Field(
        description="Rótulo canônico. Corresponde ao ENUM da coluna 'classe'.",
        examples=["fungico"],
    )
    classe_exibicao: str = Field(
        description="Rótulo legível, para exibição na interface.",
        examples=["Fúngico (Bremia lactucae)"],
    )
    confianca: float = Field(
        ge=0.0, le=1.0,
        description="Confiança da classe predita. Coluna 'confianca'.",
        examples=[0.956],
    )
    probabilidades: dict[str, float] = Field(
        description="Probabilidade atribuída a cada uma das três classes.",
        examples=[{"bacteriano": 0.021, "fungico": 0.956, "saudavel": 0.023}],
    )
    patogeno_detectado: bool = Field(
        description="Verdadeiro quando a classe predita não é 'saudavel'.",
        examples=[True],
    )
    alerta: bool = Field(
        description=(
            "Verdadeiro quando há patógeno E a confiança atinge o limiar. "
            "É o gatilho da notificação push (RF07)."
        ),
        examples=[True],
    )
    tempo_inferencia_ms: float = Field(
        description="Tempo da inferência em milissegundos. Verifica o RNF01.",
        examples=[12.73],
    )
    detectado_em: str = Field(
        description="Instante da detecção em UTC (ISO 8601). Coluna 'detectado_em'.",
        examples=["2026-08-02T18:14:05.123456+00:00"],
    )
    modelo_versao: str = Field(
        description="Arquivo de pesos utilizado, para rastreabilidade.",
        examples=["yolo11n_cls_alface.pt"],
    )
    bbox_x: float | None = Field(
        default=None,
        description=(
            "Sempre nulo: o modelo é de classificação de imagem inteira e não "
            "produz caixas delimitadoras. A coluna correspondente no banco deve "
            "admitir NULL."
        ),
    )
    bbox_y: float | None = Field(default=None, description="Sempre nulo.")
    bbox_w: float | None = Field(default=None, description="Sempre nulo.")
    bbox_h: float | None = Field(default=None, description="Sempre nulo.")
    heatmap_base64: str | None = Field(
        default=None,
        description=(
            "Mapa de calor Grad-CAM em PNG codificado em base64, presente apenas "
            "quando solicitado via 'heatmap=true'. Para exibir no navegador: "
            "data:image/png;base64,<conteudo>."
        ),
    )


class ClasseInfo(BaseModel):
    """Descrição de uma classe reconhecida pelo modelo."""

    indice: int = Field(description="Índice no vetor de saída do modelo.")
    canonico: str = Field(description="Valor gravado no banco de dados.")
    exibicao: str = Field(description="Texto para a interface do usuário.")
    patogeno: bool = Field(description="Indica presença de patógeno.")


class RespostaInfo(BaseModel):
    """Metadados do modelo em execução."""

    modelo_versao: str
    dispositivo: str = Field(examples=["cpu"])
    imgsz: int = Field(description="Resolução de entrada da rede.", examples=[224])
    limiar_confianca: float = Field(examples=[0.85])
    suporta_heatmap: bool = Field(
        description="Grad-CAM disponível apenas com pesos .pt."
    )
    tamanho_maximo_mb: float
    classes: list[ClasseInfo]


class RespostaSaude(BaseModel):
    """Estado do serviço, para verificação de disponibilidade."""

    status: str = Field(
        description="'ok' quando o modelo está carregado; 'erro' caso contrário.",
        examples=["ok"],
    )
    modelo_carregado: bool
    modelo_versao: str | None = None
    dispositivo: str | None = None
    detalhe: str | None = Field(
        default=None, description="Mensagem de erro, quando houver."
    )


class RespostaErro(BaseModel):
    """Corpo padronizado das respostas de erro."""

    detail: str = Field(
        description="Descrição do problema, em português.",
        examples=["Não foi possível decodificar a imagem recebida em bytes."],
    )
