"""Camada de inferência — contrato de integração com o back-end IoT.

Este módulo é uma **biblioteca pura**: não expõe rotas HTTP, não fala MQTT e não
acessa banco de dados. O back-end FastAPI deve instanciar
:class:`DetectorDoencasAlface` **uma única vez** no *startup* e reutilizá-lo
entre requisições — carregar o modelo a cada requisição inviabilizaria o
requisito RNF01 (inferência em menos de 100 ms).

Exemplo mínimo de integração em ``docs/README.md``.

Referência: spec.md, Seção 9.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

from .classes import (
    CLASSE_SAUDAVEL,
    ROTULOS_CANONICOS,
    eh_patogeno,
    rotulo_exibicao,
    validar_rotulo,
)
from .config import CAMINHOS, detectar_dispositivo

logger = logging.getLogger(__name__)

__all__ = [
    "ResultadoDeteccao",
    "DetectorDoencasAlface",
    "IMGSZ_PADRAO",
    "EntradaImagem",
    "preparar_entrada",
    "carregar_modelo_classificacao",
    "descobrir_imgsz",
]

#: Tamanho de entrada padrão da tarefa de classificação (spec.md, Seção 2).
IMGSZ_PADRAO: int = 224

#: Tipos aceitos por :meth:`DetectorDoencasAlface.predizer`.
EntradaImagem = str | Path | bytes | bytearray | np.ndarray | Image.Image


@dataclass
class ResultadoDeteccao:
    """Resultado de uma inferência, alinhado à tabela ``deteccoes`` do TCC.

    Os campos ``bbox_*`` existem para preservar o contrato da tabela
    ``deteccoes`` (Seção 6.2.6.8 do TCC), mas são **sempre** ``None``: o modelo
    é de classificação de imagem inteira e não produz caixas delimitadoras. As
    colunas correspondentes no banco precisam ser ``NULLABLE``
    (ver ``docs/DIVERGENCIAS.md``).

    Attributes:
        classe: Rótulo canônico — ``'saudavel'``, ``'bacteriano'`` ou ``'fungico'``.
        confianca: Confiança da classe predita, em ``[0.0, 1.0]``.
        probabilidades: Score de cada uma das três classes.
        patogeno_detectado: ``True`` se a classe predita não for ``'saudavel'``.
        alerta: ``True`` se há patógeno **e** a confiança atinge o limiar.
            É o gatilho da notificação push (RF07/CT07).
        tempo_inferencia_ms: Tempo medido da inferência, em milissegundos.
        bbox_x: Sempre ``None`` — ver nota acima.
        bbox_y: Sempre ``None``.
        bbox_w: Sempre ``None``.
        bbox_h: Sempre ``None``.
        modelo_versao: Nome do arquivo de pesos, para rastreabilidade.
        heatmap_url: Caminho/URL do mapa de calor Grad-CAM, quando gerado.
    """

    classe: str
    confianca: float
    probabilidades: dict[str, float]
    patogeno_detectado: bool
    alerta: bool
    tempo_inferencia_ms: float
    bbox_x: float | None = None
    bbox_y: float | None = None
    bbox_w: float | None = None
    bbox_h: float | None = None
    modelo_versao: str = ""
    heatmap_url: str | None = None

    @property
    def classe_exibicao(self) -> str:
        """Rótulo legível da classe predita, para o dashboard."""
        return rotulo_exibicao(self.classe)

    def to_dict(self) -> dict[str, Any]:
        """Serializa o resultado para JSON.

        Returns:
            Dicionário com todos os campos, mais ``classe_exibicao``.
        """
        dados = asdict(self)
        dados["classe_exibicao"] = self.classe_exibicao
        return dados

    def __str__(self) -> str:
        return (
            f"{self.classe} ({self.confianca:.1%}) em "
            f"{self.tempo_inferencia_ms:.1f} ms"
            f"{' [ALERTA]' if self.alerta else ''}"
        )


class DetectorDoencasAlface:
    """Detector de doenças foliares em alface baseado em YOLO11-cls.

    A instância é segura para uso concorrente: o acesso ao modelo é serializado
    por um *lock*, pois modelos Ultralytics não são thread-safe. Isso permite
    que o worker MQTT e a rota HTTP do back-end compartilhem o mesmo detector.

    Args:
        caminho_modelo: Arquivo de pesos ``.pt``. O padrão aponta para o modelo
            treinado por ``desenvolvimento/scripts/02_treinar.py``.
        dispositivo: ``"cuda"``, ``"cpu"`` ou ``None`` para autodetecção.
        limiar_confianca: Confiança mínima para que uma detecção de patógeno
            acione ``alerta`` (critérios CT01/CT02 do TCC).
        imgsz: Tamanho de entrada. Se ``None``, tenta ler do checkpoint e cai
            para :data:`IMGSZ_PADRAO`.

    Raises:
        FileNotFoundError: Se o arquivo de pesos não existir.
        ValueError: Se o limiar estiver fora de ``[0, 1]``, se o modelo não for
            de classificação, ou se suas classes não corresponderem ao conjunto
            canônico do projeto.
    """

    def __init__(
        self,
        caminho_modelo: str | Path | None = None,
        dispositivo: str | None = None,
        limiar_confianca: float = 0.85,
        imgsz: int | None = None,
    ) -> None:
        if not 0.0 <= limiar_confianca <= 1.0:
            raise ValueError(
                f"limiar_confianca deve estar em [0, 1]; recebido {limiar_confianca}."
            )

        if caminho_modelo is None:
            caminho_modelo = CAMINHOS.modelos / "yolo11n_cls_alface.pt"
        caminho_modelo = Path(caminho_modelo)

        # Nomes de pesos oficiais (ex.: "yolo11n-cls.pt") são resolvidos e
        # baixados pelo próprio Ultralytics; não exigimos que existam em disco.
        eh_peso_oficial = (
            not caminho_modelo.exists()
            and caminho_modelo.parent == Path(".")
            and caminho_modelo.name.startswith("yolo")
        )
        if not caminho_modelo.exists() and not eh_peso_oficial:
            raise FileNotFoundError(
                f"Pesos não encontrados: {caminho_modelo}. "
                "Execute 'python desenvolvimento/scripts/02_treinar.py' para gerar o modelo."
            )

        self._caminho_modelo = caminho_modelo
        self._modelo = carregar_modelo_classificacao(caminho_modelo)

        self._dispositivo = detectar_dispositivo(dispositivo)
        self._limiar_confianca = float(limiar_confianca)
        self._imgsz = int(imgsz) if imgsz else descobrir_imgsz(self._modelo)
        self._trava = threading.Lock()

        # Lê os nomes de classe do modelo em tempo de execução, em vez de assumir
        # a ordem dos índices (spec.md, Seção 4.4).
        nomes = self._modelo.names
        self._nomes: dict[int, str] = {int(i): str(n) for i, n in nomes.items()}
        self._validar_classes()

        logger.info(
            "Detector carregado: %s | dispositivo=%s | imgsz=%d | limiar=%.2f",
            caminho_modelo.name,
            self._dispositivo,
            self._imgsz,
            self._limiar_confianca,
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
    def limiar_confianca(self) -> float:
        """Confiança mínima para acionar ``alerta``."""
        return self._limiar_confianca

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

    def predizer(self, imagem: EntradaImagem) -> ResultadoDeteccao:
        """Classifica uma única imagem de folha de alface.

        Este é o caminho usado em produção pelo endpoint ``/analyze`` e,
        portanto, o caminho cujo tempo verifica o requisito RNF01.

        Args:
            imagem: Caminho de arquivo, ``bytes`` de um JPEG/PNG (como enviados
                pelo ESP32-CAM), ``numpy.ndarray`` no formato BGR (convenção do
                OpenCV, adotada pelo Ultralytics) ou objeto ``PIL.Image``.

        Returns:
            O :class:`ResultadoDeteccao` correspondente.

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

    def predizer_lote(self, imagens: Sequence[EntradaImagem]) -> list[ResultadoDeteccao]:
        """Classifica várias imagens em uma única passagem pela rede.

        O campo ``tempo_inferencia_ms`` de cada resultado recebe o tempo
        **amortizado** do lote (tempo total dividido pelo número de imagens).
        A verificação do RNF01 deve usar :meth:`predizer`, que reflete o caminho
        real de produção — uma imagem por requisição HTTP.

        Args:
            imagens: Sequência de entradas nos mesmos formatos de :meth:`predizer`.

        Returns:
            Lista de resultados, na mesma ordem da entrada.

        Raises:
            ValueError: Se a sequência estiver vazia ou alguma imagem for inválida.
        """
        if not imagens:
            raise ValueError("A lista de imagens para inferência está vazia.")

        entradas = [preparar_entrada(img) for img in imagens]

        inicio = time.perf_counter()
        with self._trava:
            resultados = self._modelo.predict(
                entradas,
                imgsz=self._imgsz,
                device=self._dispositivo,
                verbose=False,
            )
        decorrido_ms = (time.perf_counter() - inicio) * 1000.0
        por_imagem = decorrido_ms / len(entradas)

        return [self._montar_resultado(r, por_imagem) for r in resultados]

    # ------------------------------------------------------------------ #
    # Internos
    # ------------------------------------------------------------------ #

    def _validar_classes(self) -> None:
        """Garante que as classes do modelo são exatamente as canônicas.

        Um modelo base pré-treinado no ImageNet (1000 classes) é aceito com
        aviso, para permitir testes de fumaça sem pesos treinados; nesse caso a
        inferência não é semanticamente válida para o domínio.
        """
        nomes = set(self._nomes.values())
        canonicos = set(ROTULOS_CANONICOS)

        if nomes == canonicos:
            return

        if len(nomes) > 3:
            logger.warning(
                "O modelo %s possui %d classes (provavelmente pesos base do "
                "ImageNet). Os resultados NÃO são válidos para o domínio de "
                "doenças em alface; treine com desenvolvimento/scripts/02_treinar.py.",
                self._caminho_modelo.name,
                len(nomes),
            )
            return

        raise ValueError(
            f"As classes do modelo {sorted(nomes)} não correspondem ao conjunto "
            f"canônico do projeto {sorted(canonicos)}. Verifique se o modelo foi "
            "treinado sobre desenvolvimento/dataset_clean/."
        )

    def _montar_resultado(self, resultado: Any, tempo_ms: float) -> ResultadoDeteccao:
        """Traduz um ``Results`` do Ultralytics em :class:`ResultadoDeteccao`."""
        probs = getattr(resultado, "probs", None)
        if probs is None:
            raise RuntimeError(
                "O modelo não retornou probabilidades de classe. Verifique se os "
                "pesos correspondem à tarefa de classificação."
            )

        vetor = probs.data.detach().to("cpu").numpy().astype(float)
        probabilidades = {
            self._nomes[int(indice)]: round(float(valor), 6)
            for indice, valor in enumerate(vetor)
        }

        indice_top = int(np.argmax(vetor))
        classe = self._nomes[indice_top]
        confianca = round(float(vetor[indice_top]), 3)

        # Só valida contra o conjunto canônico quando o modelo é do domínio;
        # pesos base do ImageNet passam adiante o rótulo cru (uso em testes).
        if classe in ROTULOS_CANONICOS:
            classe = validar_rotulo(classe)
            patogeno = eh_patogeno(classe)
        else:
            patogeno = classe != CLASSE_SAUDAVEL

        return ResultadoDeteccao(
            classe=classe,
            confianca=confianca,
            probabilidades=probabilidades,
            patogeno_detectado=patogeno,
            alerta=bool(patogeno and confianca >= self._limiar_confianca),
            tempo_inferencia_ms=round(tempo_ms, 3),
            modelo_versao=self.modelo_versao,
        )


# --------------------------------------------------------------------------- #
# Funções compartilhadas com o verificador de domínio (verificacao.py)
# --------------------------------------------------------------------------- #


def carregar_modelo_classificacao(caminho_modelo: Path) -> Any:
    """Carrega pesos Ultralytics e garante que são de classificação.

    Args:
        caminho_modelo: Arquivo de pesos (``.pt``, ``.onnx``, ``.torchscript``…)
            ou nome de peso oficial resolvido pelo próprio Ultralytics.

    Returns:
        A instância ``ultralytics.YOLO`` carregada.

    Raises:
        ValueError: Se os pesos ``.pt`` forem de outra tarefa.
    """
    from ultralytics import YOLO

    # Para pesos .pt o Ultralytics deduz a tarefa do checkpoint, e a dedução
    # serve de verificação. Formatos exportados (.onnx, .torchscript, .engine)
    # não permitem essa dedução no construtor — os metadados só são lidos
    # quando o backend é montado —, então a tarefa é informada explicitamente.
    if caminho_modelo.suffix.lower() == ".pt":
        modelo = YOLO(str(caminho_modelo))
        if getattr(modelo, "task", None) != "classify":
            raise ValueError(
                f"O modelo em {caminho_modelo} tem tarefa "
                f"{getattr(modelo, 'task', 'desconhecida')!r}; "
                "a inferência exige um modelo de classificação ('classify')."
            )
        return modelo
    return YOLO(str(caminho_modelo), task="classify")


def descobrir_imgsz(modelo: Any) -> int:
    """Tenta recuperar o ``imgsz`` de treino a partir do checkpoint.

    Args:
        modelo: Instância ``ultralytics.YOLO`` já carregada.

    Returns:
        O ``imgsz`` registrado no checkpoint, ou :data:`IMGSZ_PADRAO`.
    """
    for acesso in (
        lambda: modelo.ckpt["train_args"]["imgsz"],
        lambda: modelo.overrides["imgsz"],
        lambda: modelo.model.args["imgsz"],
    ):
        try:
            valor = acesso()
            if isinstance(valor, (list, tuple)):
                valor = valor[0]
            if valor:
                return int(valor)
        except Exception:  # noqa: BLE001 - acesso oportunista
            continue
    return IMGSZ_PADRAO


def preparar_entrada(imagem: EntradaImagem) -> Image.Image | np.ndarray:
    """Valida e normaliza a entrada.

    Uma imagem corrompida jamais deve produzir um resultado silencioso: um
    falso ``'saudavel'`` causado por arquivo truncado é o pior modo de falha
    possível neste sistema (spec.md, Seção 9). A mesma validação vale para o
    verificador de domínio: imagem inválida é erro, nunca ``nao_alface``.

    Args:
        imagem: Caminho de arquivo, ``bytes`` de JPEG/PNG, ``numpy.ndarray``
            em BGR ou ``PIL.Image``.

    Returns:
        Imagem PIL em RGB ou array BGR ``uint8`` contíguo, pronta para o
        ``predict`` do Ultralytics.

    Raises:
        ValueError: Se a imagem estiver vazia, corrompida, em formato não
            suportado ou com número de canais inválido.
    """
    if isinstance(imagem, Image.Image):
        return _validar_pil(imagem)

    if isinstance(imagem, (bytes, bytearray)):
        if len(imagem) == 0:
            raise ValueError(
                "Os bytes da imagem estão vazios. Verifique o envio do "
                "ESP32-CAM."
            )
        try:
            with Image.open(io.BytesIO(bytes(imagem))) as aberta:
                aberta.load()
                return _validar_pil(aberta.copy())
        except (UnidentifiedImageError, OSError, ValueError) as erro:
            raise ValueError(
                "Não foi possível decodificar a imagem recebida em bytes "
                f"({len(imagem)} bytes). A transmissão pode estar truncada "
                f"ou o formato não é suportado. Detalhe: {erro}"
            ) from erro

    if isinstance(imagem, (str, Path)):
        caminho = Path(imagem)
        if not caminho.exists():
            raise ValueError(f"Arquivo de imagem não encontrado: {caminho}")
        if caminho.stat().st_size == 0:
            raise ValueError(f"Arquivo de imagem vazio: {caminho}")
        try:
            with Image.open(caminho) as aberta:
                aberta.load()
                return _validar_pil(aberta.copy())
        except (UnidentifiedImageError, OSError, ValueError) as erro:
            raise ValueError(
                f"Não foi possível ler a imagem {caminho}: {erro}"
            ) from erro

    if isinstance(imagem, np.ndarray):
        if imagem.size == 0:
            raise ValueError("O array de imagem está vazio.")
        if imagem.ndim == 2:
            imagem = np.stack([imagem] * 3, axis=-1)
        if imagem.ndim != 3 or imagem.shape[2] not in (3, 4):
            raise ValueError(
                "Array de imagem inválido: esperado formato (altura, largura, 3) "
                f"em BGR; recebido {imagem.shape}."
            )
        if imagem.shape[2] == 4:
            imagem = imagem[:, :, :3]
        if imagem.dtype != np.uint8:
            imagem = np.clip(imagem, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(imagem)

    raise ValueError(
        f"Tipo de entrada não suportado: {type(imagem).__name__}. "
        "Use caminho de arquivo, bytes, numpy.ndarray (BGR) ou PIL.Image."
    )


def _validar_pil(imagem: Image.Image) -> Image.Image:
    """Converte para RGB e rejeita imagens de dimensão degenerada."""
    largura, altura = imagem.size
    if largura < 2 or altura < 2:
        raise ValueError(
            f"Imagem com dimensões degeneradas: {largura}×{altura} px."
        )
    if imagem.mode != "RGB":
        imagem = imagem.convert("RGB")
    return imagem
