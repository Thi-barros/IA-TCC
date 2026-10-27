"""Grad-CAM: mapas de calor de localização aproximada da lesão.

O modelo é de classificação de imagem inteira e não produz caixas delimitadoras.
O dashboard especificado na Seção 7.4 do TCC, porém, prevê a sobreposição visual
da região afetada. O Grad-CAM (Selvaraju et al., 2017) recupera essa informação
de localização a partir dos gradientes da última camada convolucional, sem exigir
anotação manual do dataset.

O mapa descreve só o recorte central de 224 × 224 que a rede analisa. Por isso
ele é projetado apenas sobre essa região da foto original
(:func:`regiao_analisada`); as faixas descartadas pelo recorte ficam sem cor.

Referência: spec.md, Seção 8.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import detectar_dispositivo
from .inferencia import IMGSZ_PADRAO

logger = logging.getLogger(__name__)

__all__ = ["GradCAM", "gerar_heatmap", "regiao_analisada", "sobrepor_mapa"]

#: Média e desvio-padrão do ImageNet, usados pelo pré-processamento do Ultralytics.
MEDIA_IMAGENET = (0.485, 0.456, 0.406)
DESVIO_IMAGENET = (0.229, 0.224, 0.225)


def regiao_analisada(
    largura: int, altura: int, tamanho: int = IMGSZ_PADRAO
) -> tuple[int, int, int, int]:
    """Calcula a região da imagem original que a rede de fato analisa.

    Reproduz a geometria de ``classify_transforms`` do Ultralytics: o lado menor
    é levado a ``tamanho``, o maior acompanha a proporção (truncado, como no
    ``torchvision``) e o quadrado central de ``tamanho`` × ``tamanho`` é
    recortado. O mapa Grad-CAM descreve só esse recorte; esticá-lo sobre a imagem
    inteira o desalinha em fotos não quadradas — numa foto 1280 × 720 da C270,
    em até ~280 px nas laterais.

    Args:
        largura: Largura da imagem original, em pixels.
        altura: Altura da imagem original, em pixels.
        tamanho: Lado do recorte quadrado de entrada da rede.

    Returns:
        Caixa ``(esquerda, topo, direita, base)`` em pixels da imagem original.

    Raises:
        ValueError: Se alguma dimensão não for positiva.
    """
    if largura <= 0 or altura <= 0 or tamanho <= 0:
        raise ValueError(
            f"Dimensões inválidas: {largura} × {altura} (recorte de {tamanho})."
        )
    if largura <= altura:
        nova_largura, nova_altura = tamanho, int(tamanho * altura / largura)
    else:
        nova_largura, nova_altura = int(tamanho * largura / altura), tamanho
    # Mesmo arredondamento do CenterCrop do torchvision.
    esquerda = int(round((nova_largura - tamanho) / 2.0))
    topo = int(round((nova_altura - tamanho) / 2.0))
    escala_x = largura / nova_largura
    escala_y = altura / nova_altura
    return (
        round(esquerda * escala_x),
        round(topo * escala_y),
        round((esquerda + tamanho) * escala_x),
        round((topo + tamanho) * escala_y),
    )


def sobrepor_mapa(
    original: Image.Image,
    mapa: np.ndarray,
    regiao: tuple[int, int, int, int],
    alfa: float = 0.45,
    colormap: str = "jet",
) -> Image.Image:
    """Sobrepõe o mapa de calor à imagem, só dentro da região analisada.

    Fora da região, que a rede não viu, os pixels ficam idênticos aos da imagem
    original: o mapa não sugere que essas faixas foram avaliadas.

    Args:
        original: Imagem original.
        mapa: Mapa 2-D em ``[0, 1]`` com as dimensões da imagem original.
        regiao: Caixa ``(esquerda, topo, direita, base)`` de
            :func:`regiao_analisada`.
        alfa: Opacidade do mapa sobre a imagem.
        colormap: Nome do mapa de cores do matplotlib.

    Returns:
        Nova imagem RGB com a sobreposição.
    """
    from matplotlib import colormaps

    esquerda, topo, direita, base = regiao
    resultado = np.array(original.convert("RGB"), dtype=np.uint8)
    trecho = resultado[topo:base, esquerda:direita].astype(np.float32) / 255.0
    cores = colormaps[colormap](mapa[topo:base, esquerda:direita])[..., :3]
    mistura = np.clip(trecho * (1.0 - alfa) + cores * alfa, 0.0, 1.0)
    resultado[topo:base, esquerda:direita] = (mistura * 255).astype(np.uint8)
    return Image.fromarray(resultado)


class GradCAM:
    """Gera mapas de calor Grad-CAM para um modelo YOLO11-cls.

    A camada alvo padrão é a convolução final do cabeçote de classificação, cuja
    saída ainda preserva estrutura espacial (antes do *pooling* global).

    Note:
        O cabeçote ``Classify`` do Ultralytics aplica *softmax* em modo de
        avaliação, então a retropropagação parte da probabilidade da classe alvo
        (e não do logit bruto). Na prática isso apenas reescala o mapa, que é
        normalizado ao final.

    Args:
        caminho_modelo: Pesos ``.pt`` treinados.
        dispositivo: ``"cuda"``, ``"cpu"`` ou ``None`` para autodetecção.
        imgsz: Tamanho de entrada da rede.

    Raises:
        FileNotFoundError: Se os pesos não existirem.
        RuntimeError: Se nenhuma camada convolucional adequada for localizada.
    """

    def __init__(
        self,
        caminho_modelo: str | Path,
        dispositivo: str | None = None,
        imgsz: int = IMGSZ_PADRAO,
    ) -> None:
        caminho_modelo = Path(caminho_modelo)
        if not caminho_modelo.exists():
            raise FileNotFoundError(f"Pesos não encontrados: {caminho_modelo}")

        import torch
        from ultralytics import YOLO

        self._torch = torch
        self._dispositivo = detectar_dispositivo(dispositivo)
        self._imgsz = int(imgsz)

        modelo = YOLO(str(caminho_modelo))
        self._nomes = {int(i): str(n) for i, n in modelo.names.items()}
        self._rede = modelo.model.to(self._dispositivo).float().eval()
        for parametro in self._rede.parameters():
            parametro.requires_grad_(True)

        self._camada_alvo = self._localizar_camada_alvo()
        self._ativacoes: Any = None
        self._gradientes: Any = None
        self._registrar_hooks()

        logger.info(
            "Grad-CAM pronto: camada alvo=%s | dispositivo=%s",
            type(self._camada_alvo).__name__,
            self._dispositivo,
        )

    # ------------------------------------------------------------------ #
    # Construção
    # ------------------------------------------------------------------ #

    def _localizar_camada_alvo(self):
        """Escolhe a última camada convolucional com saída espacial."""
        import torch.nn as nn

        # Caminho preferencial: a convolução interna do cabeçote Classify.
        try:
            cabecote = self._rede.model[-1]
            if hasattr(cabecote, "conv"):
                return cabecote.conv
        except (AttributeError, IndexError, TypeError):
            pass

        # Alternativa: a última nn.Conv2d na ordem de declaração dos módulos.
        convolucoes = [m for m in self._rede.modules() if isinstance(m, nn.Conv2d)]
        if convolucoes:
            return convolucoes[-1]

        raise RuntimeError(
            "Nenhuma camada convolucional localizada no modelo; o Grad-CAM não "
            "pode ser aplicado a esta arquitetura."
        )

    def _registrar_hooks(self) -> None:
        """Instala os ganchos que capturam ativações e gradientes."""

        def gancho_forward(_modulo, _entrada, saida):
            self._ativacoes = saida
            # A saída não é um tensor-folha; um gancho no próprio tensor é a
            # forma confiável de capturar seu gradiente.
            if saida.requires_grad:
                saida.register_hook(self._guardar_gradiente)

        def _guardar(gradiente):
            self._gradientes = gradiente

        self._guardar_gradiente = _guardar
        self._camada_alvo.register_forward_hook(gancho_forward)

    # ------------------------------------------------------------------ #
    # Pré-processamento
    # ------------------------------------------------------------------ #

    def _preprocessar(self, imagem: Image.Image):
        """Converte uma imagem PIL RGB no tensor de entrada da rede."""
        torch = self._torch

        try:
            from ultralytics.data.augment import classify_transforms

            transformacao = classify_transforms(self._imgsz)
            tensor = transformacao(imagem)
        except Exception as erro:  # noqa: BLE001 - fallback determinístico
            logger.debug("classify_transforms indisponível (%s); usando fallback.", erro)
            redimensionada = self._redimensionar_e_recortar(imagem, self._imgsz)
            vetor = np.asarray(redimensionada, dtype=np.float32) / 255.0
            vetor = (vetor - np.array(MEDIA_IMAGENET, dtype=np.float32)) / np.array(
                DESVIO_IMAGENET, dtype=np.float32
            )
            tensor = torch.from_numpy(vetor.transpose(2, 0, 1))

        return tensor.unsqueeze(0).to(self._dispositivo).float()

    @staticmethod
    def _redimensionar_e_recortar(imagem: Image.Image, tamanho: int) -> Image.Image:
        """Recorta a região analisada e a leva a ``tamanho`` × ``tamanho``.

        Usa a mesma geometria de :func:`regiao_analisada`, para que o recorte do
        caminho alternativo coincida com a região onde o mapa é projetado.
        """
        caixa = regiao_analisada(imagem.width, imagem.height, tamanho)
        return imagem.crop(caixa).resize((tamanho, tamanho), Image.BILINEAR)

    @staticmethod
    def _abrir_rgb(imagem: str | Path | Image.Image) -> Image.Image:
        """Abre o arquivo, ou converte o objeto PIL, em imagem RGB."""
        if isinstance(imagem, (str, Path)):
            with Image.open(imagem) as aberta:
                return aberta.convert("RGB").copy()
        return imagem.convert("RGB")

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #

    def gerar(
        self,
        imagem: str | Path | Image.Image,
        classe_alvo: str | int | None = None,
    ) -> tuple[np.ndarray, str, float]:
        """Calcula o mapa de calor para uma imagem.

        Args:
            imagem: Caminho do arquivo ou objeto ``PIL.Image``.
            classe_alvo: Classe para a qual explicar a decisão. Aceita rótulo
                canônico ou índice. Se ``None``, usa a classe predita.

        Returns:
            Tupla ``(mapa, classe, confianca)``, onde ``mapa`` é um array 2-D em
            ``[0, 1]`` com as dimensões da imagem original, preenchido só na
            região analisada pela rede (:func:`regiao_analisada`) e com zero
            fora dela.

        Raises:
            ValueError: Se ``classe_alvo`` for desconhecida.
            RuntimeError: Se os gradientes não forem capturados.
        """
        torch = self._torch

        original = self._abrir_rgb(imagem)
        entrada = self._preprocessar(original)

        indice_alvo: int | None = None
        if classe_alvo is not None:
            if isinstance(classe_alvo, int):
                indice_alvo = classe_alvo
            else:
                for indice, nome in self._nomes.items():
                    if nome == classe_alvo:
                        indice_alvo = indice
                        break
                if indice_alvo is None:
                    raise ValueError(
                        f"Classe alvo desconhecida: {classe_alvo!r}. "
                        f"Disponíveis: {sorted(self._nomes.values())}."
                    )

        self._ativacoes = None
        self._gradientes = None
        self._rede.zero_grad(set_to_none=True)

        with torch.enable_grad():
            saida = self._rede(entrada)
            if isinstance(saida, (list, tuple)):
                saida = saida[0]
            vetor = saida.reshape(saida.shape[0], -1)
            if indice_alvo is None:
                indice_alvo = int(vetor.argmax(dim=1).item())
            pontuacao = vetor[0, indice_alvo]
            pontuacao.backward()

        if self._ativacoes is None or self._gradientes is None:
            raise RuntimeError(
                "Grad-CAM não capturou ativações/gradientes. A arquitetura do "
                "modelo pode ter mudado; verifique a camada alvo."
            )

        ativacoes = self._ativacoes.detach()[0]  # [C, h, w]
        gradientes = self._gradientes.detach()[0]  # [C, h, w]
        pesos = gradientes.mean(dim=(1, 2), keepdim=True)
        mapa = torch.relu((pesos * ativacoes).sum(dim=0))

        mapa_np = mapa.to("cpu").numpy().astype(np.float32)
        maximo = float(mapa_np.max())
        if maximo > 0:
            mapa_np /= maximo
        else:
            logger.warning(
                "Mapa Grad-CAM totalmente nulo para a classe %s; a saída será "
                "uniforme.",
                self._nomes.get(indice_alvo, indice_alvo),
            )

        # O mapa descreve só o recorte central que a rede analisou: é ampliado
        # para o tamanho dessa região e colocado na posição dela. Esticá-lo
        # sobre a imagem inteira o desalinharia em fotos não quadradas.
        esquerda, topo, direita, base = regiao_analisada(
            original.width, original.height, self._imgsz
        )
        mapa_recorte = np.asarray(
            Image.fromarray((mapa_np * 255).astype(np.uint8)).resize(
                (direita - esquerda, base - topo), Image.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0
        mapa_redimensionado = np.zeros((original.height, original.width), dtype=np.float32)
        mapa_redimensionado[topo:base, esquerda:direita] = mapa_recorte

        confianca = float(vetor[0, indice_alvo].detach().to("cpu").item())
        return mapa_redimensionado, self._nomes.get(indice_alvo, str(indice_alvo)), confianca

    def sobreposicao(
        self,
        imagem: str | Path | Image.Image,
        classe_alvo: str | int | None = None,
        alfa: float = 0.45,
        colormap: str = "jet",
    ) -> tuple[Image.Image, str, float]:
        """Gera o mapa e o devolve sobreposto à imagem, só na região analisada.

        Args:
            imagem: Caminho do arquivo ou objeto ``PIL.Image``.
            classe_alvo: Classe a explicar. ``None`` usa a predita.
            alfa: Opacidade do mapa de calor sobre a imagem original.
            colormap: Nome do mapa de cores do matplotlib.

        Returns:
            Tupla ``(imagem, classe, confianca)``, com a sobreposição em RGB.
        """
        original = self._abrir_rgb(imagem)
        mapa, classe, confianca = self.gerar(original, classe_alvo)
        regiao = regiao_analisada(original.width, original.height, self._imgsz)
        return sobrepor_mapa(original, mapa, regiao, alfa, colormap), classe, confianca

    def salvar_sobreposicao(
        self,
        imagem: str | Path | Image.Image,
        caminho_saida: str | Path,
        classe_alvo: str | int | None = None,
        alfa: float = 0.45,
        colormap: str = "jet",
    ) -> dict:
        """Gera e grava o PNG com o mapa de calor sobreposto à imagem.

        Args:
            imagem: Caminho do arquivo ou objeto ``PIL.Image``.
            caminho_saida: Arquivo PNG de saída.
            classe_alvo: Classe a explicar. ``None`` usa a predita.
            alfa: Opacidade do mapa de calor sobre a imagem original.
            colormap: Nome do mapa de cores do matplotlib.

        Returns:
            Dicionário com ``caminho``, ``classe`` e ``confianca``.
        """
        sobreposta, classe, confianca = self.sobreposicao(
            imagem, classe_alvo, alfa, colormap
        )

        caminho_saida = Path(caminho_saida)
        caminho_saida.parent.mkdir(parents=True, exist_ok=True)
        sobreposta.save(caminho_saida)

        logger.info(
            "Mapa de calor salvo em %s (classe=%s, confiança=%.3f)",
            caminho_saida, classe, confianca,
        )
        return {
            "caminho": str(caminho_saida),
            "classe": classe,
            "confianca": round(confianca, 4),
        }


def gerar_heatmap(
    caminho_modelo: str | Path,
    imagem: str | Path,
    caminho_saida: str | Path,
    classe_alvo: str | None = None,
    dispositivo: str | None = None,
) -> dict:
    """Atalho de uma chamada para gerar um mapa de calor.

    Para uso em lote, instancie :class:`GradCAM` uma vez e reutilize.

    Args:
        caminho_modelo: Pesos ``.pt`` treinados.
        imagem: Imagem de entrada.
        caminho_saida: PNG de saída.
        classe_alvo: Classe a explicar, ou ``None`` para a predita.
        dispositivo: ``"cuda"``, ``"cpu"`` ou ``None``.

    Returns:
        Dicionário com ``caminho``, ``classe`` e ``confianca``.
    """
    explicador = GradCAM(caminho_modelo, dispositivo=dispositivo)
    return explicador.salvar_sobreposicao(imagem, caminho_saida, classe_alvo)
