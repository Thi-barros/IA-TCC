"""Verificador de domínio — etapa 1 da inferência em cascata.

Responde à pergunta binária "a imagem contém uma folha de alface?" antes que o
classificador de doenças seja consultado. O classificador é de conjunto
fechado: sem esta etapa, qualquer imagem — solo, bancada, mão do agricultor —
recebe um diagnóstico e pode acionar um alerta falso.

Este módulo apenas **informa** se a imagem pertence ao domínio. A decisão de
recusar a requisição cabe a quem encadeia o verificador e o
:class:`~ia_alface.inferencia.DetectorDoencasAlface`.

Como :mod:`ia_alface.inferencia`, é uma **biblioteca pura**: não expõe rotas
HTTP, não fala MQTT e não acessa banco de dados.

Referência: docs/DOCUMENTACAO_IA_TREINAMENTO_RESULTADOS.md, Seção 2.9.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .classes import CLASSE_ALFACE, ROTULOS_DOMINIO
from .config import CAMINHOS, detectar_dispositivo
from .inferencia import (
    EntradaImagem,
    carregar_modelo_classificacao,
    descobrir_imgsz,
    preparar_entrada,
)

logger = logging.getLogger(__name__)

__all__ = ["ResultadoVerificacao", "VerificadorDominio", "LIMIAR_ALFACE_PADRAO"]

#: Probabilidade mínima de ``alface`` para aceitar a imagem. Em problema
#: binário, 0,5 equivale ao argmax. É um valor provisório: o limiar definitivo
#: deve ser escolhido no split ``val`` — nunca no ``test`` — favorecendo a
#: aceitação, porque recusar uma folha doente suprime o alerta, enquanto aceitar
#: uma imagem alheia custa no máximo um alerta indevido.
LIMIAR_ALFACE_PADRAO: float = 0.5


@dataclass
class ResultadoVerificacao:
    """Resultado do verificador de domínio para uma imagem.

    Attributes:
        eh_alface: ``True`` se ``probabilidade_alface`` atinge o limiar.
        probabilidade_alface: Probabilidade atribuída à classe ``alface``, em
            ``[0.0, 1.0]``.
        probabilidades: Score de cada uma das duas classes do domínio.
        limiar: Limiar aplicado na decisão, registrado para rastreabilidade.
        tempo_inferencia_ms: Tempo medido da inferência, em milissegundos.
        modelo_versao: Nome do arquivo de pesos do verificador.
    """

    eh_alface: bool
    probabilidade_alface: float
    probabilidades: dict[str, float]
    limiar: float
    tempo_inferencia_ms: float
    modelo_versao: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serializa o resultado para JSON.

        Returns:
            Dicionário com todos os campos.
        """
        return asdict(self)

    def __str__(self) -> str:
        decisao = "alface" if self.eh_alface else "NÃO alface"
        return (
            f"{decisao} (P(alface)={self.probabilidade_alface:.1%}, "
            f"limiar {self.limiar:.2f}) em {self.tempo_inferencia_ms:.1f} ms"
        )


class VerificadorDominio:
    """Verificador binário "alface / não alface" baseado em YOLO11-cls.

    Segue o padrão do :class:`~ia_alface.inferencia.DetectorDoencasAlface`:
    deve ser instanciado uma única vez, serializa o acesso ao modelo por um
    *lock* e lê os nomes de classe do próprio modelo. A validação de entrada é
    a mesma, de modo que imagem corrompida continua sendo erro, e não
    ``nao_alface``.

    Args:
        caminho_modelo: Arquivo de pesos. O padrão aponta para
            ``modelos/yolo11n_cls_dominio.pt``.
        dispositivo: ``"cuda"``, ``"cpu"`` ou ``None`` para autodetecção.
        limiar_alface: Probabilidade mínima de ``alface`` para aceitar a imagem.
        imgsz: Tamanho de entrada. Se ``None``, tenta ler do checkpoint.

    Raises:
        FileNotFoundError: Se o arquivo de pesos não existir.
        ValueError: Se o limiar estiver fora de ``[0, 1]``, se o modelo não for
            de classificação, ou se suas classes não forem exatamente as do
            verificador de domínio.
    """

    def __init__(
        self,
        caminho_modelo: str | Path | None = None,
        dispositivo: str | None = None,
        limiar_alface: float = LIMIAR_ALFACE_PADRAO,
        imgsz: int | None = None,
    ) -> None:
        if not 0.0 <= limiar_alface <= 1.0:
            raise ValueError(
                f"limiar_alface deve estar em [0, 1]; recebido {limiar_alface}."
            )

        if caminho_modelo is None:
            caminho_modelo = CAMINHOS.modelos / "yolo11n_cls_dominio.pt"
        caminho_modelo = Path(caminho_modelo)

        # Ao contrário do detector, pesos oficiais do ImageNet não são aceitos:
        # suas 1000 classes não respondem à pergunta do verificador.
        if not caminho_modelo.exists():
            raise FileNotFoundError(
                f"Pesos do verificador não encontrados: {caminho_modelo}. "
                "Treine o verificador de domínio antes de habilitá-lo."
            )

        self._caminho_modelo = caminho_modelo
        self._modelo = carregar_modelo_classificacao(caminho_modelo)
        self._dispositivo = detectar_dispositivo(dispositivo)
        self._limiar_alface = float(limiar_alface)
        self._imgsz = int(imgsz) if imgsz else descobrir_imgsz(self._modelo)
        self._trava = threading.Lock()

        nomes = self._modelo.names
        self._nomes: dict[int, str] = {int(i): str(n) for i, n in nomes.items()}
        self._validar_classes()

        logger.info(
            "Verificador de domínio carregado: %s | dispositivo=%s | imgsz=%d | limiar=%.2f",
            caminho_modelo.name,
            self._dispositivo,
            self._imgsz,
            self._limiar_alface,
        )

    # ------------------------------------------------------------------ #
    # Propriedades
    # ------------------------------------------------------------------ #

    @property
    def modelo_versao(self) -> str:
        """Nome do arquivo de pesos em uso."""
        return self._caminho_modelo.name

    @property
    def dispositivo(self) -> str:
        """Dispositivo efetivo de execução."""
        return self._dispositivo

    @property
    def limiar_alface(self) -> float:
        """Probabilidade mínima de ``alface`` para aceitar a imagem."""
        return self._limiar_alface

    @property
    def imgsz(self) -> int:
        """Tamanho de entrada usado na inferência."""
        return self._imgsz

    @property
    def nomes_classes(self) -> dict[int, str]:
        """Mapa índice -> rótulo, conforme lido do modelo."""
        return dict(self._nomes)

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #

    def verificar(self, imagem: EntradaImagem) -> ResultadoVerificacao:
        """Decide se a imagem contém uma folha de alface.

        Args:
            imagem: Mesmos formatos aceitos por
                :meth:`~ia_alface.inferencia.DetectorDoencasAlface.predizer`.

        Returns:
            O :class:`ResultadoVerificacao` correspondente.

        Raises:
            ValueError: Se a imagem estiver vazia, corrompida, em formato não
                suportado ou com número de canais inválido.
        """
        entrada = preparar_entrada(imagem)

        inicio = time.perf_counter()
        with self._trava:
            resultados = self._modelo.predict(
                entrada,
                imgsz=self._imgsz,
                device=self._dispositivo,
                verbose=False,
            )
        decorrido_ms = (time.perf_counter() - inicio) * 1000.0

        return self._montar_resultado(resultados[0], decorrido_ms)

    # ------------------------------------------------------------------ #
    # Internos
    # ------------------------------------------------------------------ #

    def _validar_classes(self) -> None:
        """Exige exatamente as classes do verificador de domínio.

        Detecta já na carga a troca de arquivos de pesos — por exemplo, o
        classificador de doenças apontado como verificador —, em vez de falhar
        só na primeira requisição.
        """
        nomes = set(self._nomes.values())
        if nomes != set(ROTULOS_DOMINIO):
            raise ValueError(
                f"As classes do modelo {sorted(nomes)} não correspondem às do "
                f"verificador de domínio {sorted(ROTULOS_DOMINIO)}. Verifique se "
                f"{self._caminho_modelo.name} é o verificador, e não o "
                "classificador de doenças."
            )

    def _montar_resultado(self, resultado: Any, tempo_ms: float) -> ResultadoVerificacao:
        """Traduz um ``Results`` do Ultralytics em :class:`ResultadoVerificacao`."""
        probs = getattr(resultado, "probs", None)
        if probs is None:
            raise RuntimeError(
                "O verificador não retornou probabilidades de classe. Verifique "
                "se os pesos correspondem à tarefa de classificação."
            )

        vetor = probs.data.detach().to("cpu").numpy().astype(float)
        probabilidades = {
            self._nomes[int(indice)]: round(float(valor), 6)
            for indice, valor in enumerate(vetor)
        }

        # A decisão usa o mesmo valor arredondado que é reportado, para que um
        # consumidor que compare probabilidade e limiar chegue à mesma resposta.
        probabilidade_alface = probabilidades[CLASSE_ALFACE]

        return ResultadoVerificacao(
            eh_alface=bool(probabilidade_alface >= self._limiar_alface),
            probabilidade_alface=probabilidade_alface,
            probabilidades=probabilidades,
            limiar=self._limiar_alface,
            tempo_inferencia_ms=round(tempo_ms, 3),
            modelo_versao=self.modelo_versao,
        )
