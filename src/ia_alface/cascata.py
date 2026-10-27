"""Inferência em cascata: verificador de domínio → classificador de doenças.

Etapa 1: o :class:`~ia_alface.verificacao.VerificadorDominio` decide se a
imagem contém uma folha de alface. Etapa 2: só as imagens aceitas seguem para o
:class:`~ia_alface.inferencia.DetectorDoencasAlface`, que não é alterado.
Imagem recusada não recebe diagnóstico nem gera alerta.

A imagem é decodificada e validada **uma única vez**, antes das duas etapas. Uma
imagem inválida continua levantando ``ValueError`` — nunca é tratada como "não
alface", que esconderia uma falha de transmissão.

Como os demais módulos de ``ia_alface``, é biblioteca pura: sem HTTP, MQTT ou
banco de dados.

Referência: docs/DOCUMENTACAO_IA_TREINAMENTO_RESULTADOS.md, Seção 2.9;
docs/VERIFICADOR_DOMINIO.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .inferencia import DetectorDoencasAlface, EntradaImagem, ResultadoDeteccao, preparar_entrada
from .verificacao import ResultadoVerificacao, VerificadorDominio

logger = logging.getLogger(__name__)

__all__ = ["ImagemForaDoDominio", "ResultadoCascata", "AnalisadorCascata"]


class ImagemForaDoDominio(Exception):
    """A imagem é válida, mas o verificador não a reconheceu como alface.

    **Não** herda de ``ValueError`` de propósito: ``ValueError`` significa
    imagem inválida (HTTP 400). Misturar os dois faria uma foto de bancada
    parecer arquivo corrompido, e vice-versa.

    Attributes:
        verificacao: Resultado do verificador que motivou a recusa.
    """

    def __init__(self, verificacao: ResultadoVerificacao) -> None:
        self.verificacao = verificacao
        super().__init__(
            "A imagem não foi reconhecida como folha de alface "
            f"(P(alface) = {verificacao.probabilidade_alface:.3f}, limiar "
            f"{verificacao.limiar:.2f}). Nenhum diagnóstico foi emitido."
        )


@dataclass
class ResultadoCascata:
    """Resultado de uma imagem aceita pela cascata.

    Attributes:
        deteccao: Diagnóstico do classificador de doenças, inalterado.
        verificacao: Resultado do verificador, ou ``None`` quando a cascata
            está desligada (sem verificador configurado).
    """

    deteccao: ResultadoDeteccao
    verificacao: ResultadoVerificacao | None = None

    @property
    def tempo_total_ms(self) -> float:
        """Soma dos tempos de inferência das etapas executadas."""
        total = self.deteccao.tempo_inferencia_ms
        if self.verificacao is not None:
            total += self.verificacao.tempo_inferencia_ms
        return round(total, 3)

    def to_dict(self) -> dict[str, Any]:
        """Serializa o resultado para JSON.

        Returns:
            Os campos da detecção, mais ``verificacao`` e ``tempo_total_ms``.
        """
        dados = self.deteccao.to_dict()
        dados["verificacao"] = self.verificacao.to_dict() if self.verificacao else None
        dados["tempo_total_ms"] = self.tempo_total_ms
        return dados


class AnalisadorCascata:
    """Encadeia o verificador de domínio e o classificador de doenças.

    Args:
        detector: Classificador de doenças já carregado.
        verificador: Verificador de domínio já carregado. ``None`` desliga a
            etapa 1, e o analisador se comporta exatamente como o detector.
    """

    def __init__(
        self,
        detector: DetectorDoencasAlface,
        verificador: VerificadorDominio | None = None,
    ) -> None:
        self._detector = detector
        self._verificador = verificador

    @property
    def detector(self) -> DetectorDoencasAlface:
        """Classificador de doenças em uso."""
        return self._detector

    @property
    def verificador(self) -> VerificadorDominio | None:
        """Verificador de domínio em uso, ou ``None`` se a etapa 1 está desligada."""
        return self._verificador

    @property
    def verifica_dominio(self) -> bool:
        """Indica se a etapa 1 está ativa."""
        return self._verificador is not None

    def analisar(self, imagem: EntradaImagem) -> ResultadoCascata:
        """Verifica o domínio e, se aceita, classifica a imagem.

        Args:
            imagem: Mesmos formatos aceitos por
                :meth:`~ia_alface.inferencia.DetectorDoencasAlface.predizer`.

        Returns:
            O :class:`ResultadoCascata` da imagem aceita.

        Raises:
            ValueError: Se a imagem estiver vazia, corrompida ou em formato não
                suportado. Verificado antes de qualquer modelo.
            ImagemForaDoDominio: Se o verificador recusar a imagem. O
                classificador não chega a ser executado.
        """
        entrada = preparar_entrada(imagem)

        verificacao = None
        if self._verificador is not None:
            verificacao = self._verificador.verificar(entrada)
            if not verificacao.eh_alface:
                logger.info(
                    "Imagem recusada pelo verificador: P(alface)=%.3f < %.2f",
                    verificacao.probabilidade_alface,
                    verificacao.limiar,
                )
                raise ImagemForaDoDominio(verificacao)

        deteccao = self._detector.predizer(entrada)
        return ResultadoCascata(deteccao=deteccao, verificacao=verificacao)
