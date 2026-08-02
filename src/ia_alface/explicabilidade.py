"""Grad-CAM: mapas de calor de localização aproximada da lesão.

O modelo é de classificação de imagem inteira e não produz caixas delimitadoras.
O dashboard especificado na Seção 7.4 do TCC, porém, prevê a sobreposição visual
da região afetada. O Grad-CAM (Selvaraju et al., 2017) recupera essa informação
de localização a partir dos gradientes da última camada convolucional, sem exigir
anotação manual do dataset.

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

__all__ = ["GradCAM", "gerar_heatmap"]

#: Média e desvio-padrão do ImageNet, usados pelo pré-processamento do Ultralytics.
MEDIA_IMAGENET = (0.485, 0.456, 0.406)
DESVIO_IMAGENET = (0.229, 0.224, 0.225)


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
        """Redimensiona pelo menor lado e recorta o centro."""
        largura, altura = imagem.size
        escala = tamanho / min(largura, altura)
        nova = (max(tamanho, round(largura * escala)), max(tamanho, round(altura * escala)))
        imagem = imagem.resize(nova, Image.BILINEAR)
        esquerda = (imagem.width - tamanho) // 2
        topo = (imagem.height - tamanho) // 2
        return imagem.crop((esquerda, topo, esquerda + tamanho, topo + tamanho))

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
            ``[0, 1]`` com as dimensões da imagem original.

        Raises:
            ValueError: Se ``classe_alvo`` for desconhecida.
            RuntimeError: Se os gradientes não forem capturados.
        """
        torch = self._torch

        if isinstance(imagem, (str, Path)):
            with Image.open(imagem) as aberta:
                original = aberta.convert("RGB").copy()
        else:
            original = imagem.convert("RGB")

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

        mapa_redimensionado = np.asarray(
            Image.fromarray((mapa_np * 255).astype(np.uint8)).resize(
                original.size, Image.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0

        confianca = float(vetor[0, indice_alvo].detach().to("cpu").item())
        return mapa_redimensionado, self._nomes.get(indice_alvo, str(indice_alvo)), confianca

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
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import colormaps

        mapa, classe, confianca = self.gerar(imagem, classe_alvo)

        if isinstance(imagem, (str, Path)):
            with Image.open(imagem) as aberta:
                original = aberta.convert("RGB").copy()
        else:
            original = imagem.convert("RGB")

        base = np.asarray(original, dtype=np.float32) / 255.0
        cores = colormaps[colormap](mapa)[..., :3].astype(np.float32)
        mistura = np.clip(base * (1.0 - alfa) + cores * alfa, 0.0, 1.0)

        caminho_saida = Path(caminho_saida)
        caminho_saida.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((mistura * 255).astype(np.uint8)).save(caminho_saida)

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
