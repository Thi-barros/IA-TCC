"""Exportação do modelo para ONNX/TorchScript e validação numérica.

A exportação ONNX viabiliza a execução por ``onnxruntime`` sem dependência do
PyTorch, útil para o servidor de inferência e como caminho de implantação em
borda. A equivalência numérica com o modelo PyTorch é **verificada**, não
presumida.

Referência: spec.md, Seção 10.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .config import CAMINHOS, Caminhos
from .inferencia import IMGSZ_PADRAO

logger = logging.getLogger(__name__)

__all__ = [
    "exportar_onnx",
    "exportar_torchscript",
    "preprocessar_para_tensor",
    "validar_onnx",
]

MEDIA_IMAGENET = np.array((0.485, 0.456, 0.406), dtype=np.float32)
DESVIO_IMAGENET = np.array((0.229, 0.224, 0.225), dtype=np.float32)


def exportar_onnx(
    caminho_pesos: Path,
    imgsz: int = IMGSZ_PADRAO,
    opset: int = 17,
    simplify: bool = True,
    caminhos: Caminhos = CAMINHOS,
) -> Path:
    """Exporta os pesos treinados para ONNX.

    Args:
        caminho_pesos: Arquivo ``.pt`` treinado.
        imgsz: Tamanho de entrada fixo do grafo.
        opset: Versão do conjunto de operadores ONNX.
        simplify: Se ``True``, tenta simplificar o grafo. A simplificação exige
            o pacote ``onnxslim``; se ele não estiver disponível, a exportação é
            repetida sem simplificação em vez de falhar.
        caminhos: Caminhos do projeto.

    Returns:
        Caminho do arquivo ``.onnx`` em ``modelos/``.

    Raises:
        FileNotFoundError: Se os pesos não existirem.
        RuntimeError: Se o Ultralytics não produzir o arquivo.
    """
    from ultralytics import YOLO

    caminho_pesos = Path(caminho_pesos)
    if not caminho_pesos.exists():
        raise FileNotFoundError(f"Pesos não encontrados: {caminho_pesos}")

    def _exportar(usar_simplify: bool) -> Path:
        modelo = YOLO(str(caminho_pesos))
        saida = modelo.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            simplify=usar_simplify,
            dynamic=False,
            verbose=False,
        )
        return Path(saida)

    try:
        gerado = _exportar(simplify)
    except Exception as erro:  # noqa: BLE001 - simplificação é acessória
        if not simplify:
            raise
        logger.warning(
            "Exportação com simplify=True falhou (%s). Repetindo sem "
            "simplificação do grafo.",
            erro,
        )
        gerado = _exportar(False)

    if not gerado.exists():
        raise RuntimeError(f"O Ultralytics não gerou o arquivo ONNX em {gerado}.")

    destino = caminhos.modelos / f"{caminho_pesos.stem}.onnx"
    if gerado.resolve() != destino.resolve():
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(gerado.read_bytes())
        if gerado.parent.resolve() != destino.parent.resolve():
            gerado.unlink(missing_ok=True)

    logger.info(
        "ONNX exportado para %s (%.2f MB)", destino, destino.stat().st_size / 1e6
    )
    return destino


def exportar_torchscript(
    caminho_pesos: Path,
    imgsz: int = IMGSZ_PADRAO,
    caminhos: Caminhos = CAMINHOS,
) -> Path:
    """Exporta os pesos para TorchScript, como alternativa de implantação.

    Args:
        caminho_pesos: Arquivo ``.pt`` treinado.
        imgsz: Tamanho de entrada fixo.
        caminhos: Caminhos do projeto.

    Returns:
        Caminho do arquivo ``.torchscript`` em ``modelos/``.

    Raises:
        FileNotFoundError: Se os pesos não existirem.
    """
    from ultralytics import YOLO

    caminho_pesos = Path(caminho_pesos)
    if not caminho_pesos.exists():
        raise FileNotFoundError(f"Pesos não encontrados: {caminho_pesos}")

    modelo = YOLO(str(caminho_pesos))
    gerado = Path(
        modelo.export(format="torchscript", imgsz=imgsz, verbose=False)
    )

    destino = caminhos.modelos / f"{caminho_pesos.stem}.torchscript"
    if gerado.resolve() != destino.resolve():
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(gerado.read_bytes())
        if gerado.parent.resolve() != destino.parent.resolve():
            gerado.unlink(missing_ok=True)

    logger.info(
        "TorchScript exportado para %s (%.2f MB)",
        destino,
        destino.stat().st_size / 1e6,
    )
    return destino


def preprocessar_para_tensor(
    caminho_imagem: str | Path, imgsz: int = IMGSZ_PADRAO
) -> np.ndarray:
    """Converte uma imagem no tensor NCHW esperado pela rede.

    Usa o pré-processamento oficial do Ultralytics quando disponível; caso
    contrário, aplica um equivalente determinístico (redimensionar pelo menor
    lado, recortar o centro, normalizar com média/desvio do ImageNet). Como o
    **mesmo** tensor alimenta o modelo PyTorch e o ONNX, a comparação numérica é
    válida em qualquer um dos dois caminhos.

    Args:
        caminho_imagem: Arquivo de imagem.
        imgsz: Lado do recorte quadrado.

    Returns:
        Array ``float32`` de forma ``(1, 3, imgsz, imgsz)``.
    """
    with Image.open(caminho_imagem) as aberta:
        imagem = aberta.convert("RGB").copy()

    try:
        from ultralytics.data.augment import classify_transforms

        tensor = classify_transforms(imgsz)(imagem)
        return tensor.unsqueeze(0).numpy().astype(np.float32)
    except Exception as erro:  # noqa: BLE001 - fallback determinístico
        logger.debug("classify_transforms indisponível (%s); usando fallback.", erro)

    largura, altura = imagem.size
    escala = imgsz / min(largura, altura)
    imagem = imagem.resize(
        (max(imgsz, round(largura * escala)), max(imgsz, round(altura * escala))),
        Image.BILINEAR,
    )
    esquerda = (imagem.width - imgsz) // 2
    topo = (imagem.height - imgsz) // 2
    imagem = imagem.crop((esquerda, topo, esquerda + imgsz, topo + imgsz))

    vetor = np.asarray(imagem, dtype=np.float32) / 255.0
    vetor = (vetor - MEDIA_IMAGENET) / DESVIO_IMAGENET
    return vetor.transpose(2, 0, 1)[None, ...].astype(np.float32)


def validar_onnx(
    caminho_pesos: Path,
    caminho_onnx: Path,
    imagens: Sequence[Path],
    imgsz: int = IMGSZ_PADRAO,
    atol: float = 1e-3,
) -> dict:
    """Compara as saídas do modelo ONNX e do modelo PyTorch.

    Args:
        caminho_pesos: Pesos ``.pt`` de referência.
        caminho_onnx: Modelo ONNX exportado.
        imagens: Imagens de validação (a Seção 10 do spec exige ao menos 20).
        imgsz: Tamanho de entrada.
        atol: Tolerância absoluta máxima aceita.

    Returns:
        Dicionário com ``aprovado``, ``n_imagens``, ``divergencia_maxima``,
        ``divergencia_media``, ``classes_divergentes`` e ``atol``.

    Raises:
        FileNotFoundError: Se algum dos modelos não existir.
        ValueError: Se nenhuma imagem for fornecida.
    """
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO

    caminho_pesos, caminho_onnx = Path(caminho_pesos), Path(caminho_onnx)
    for caminho in (caminho_pesos, caminho_onnx):
        if not caminho.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {caminho}")
    if not imagens:
        raise ValueError("Nenhuma imagem fornecida para a validação do ONNX.")

    rede = YOLO(str(caminho_pesos)).model.float().eval()
    sessao = ort.InferenceSession(
        str(caminho_onnx), providers=["CPUExecutionProvider"]
    )
    nome_entrada = sessao.get_inputs()[0].name

    divergencias: list[float] = []
    classes_divergentes = 0

    for caminho_imagem in imagens:
        tensor = preprocessar_para_tensor(caminho_imagem, imgsz)

        with torch.no_grad():
            saida_torch = rede(torch.from_numpy(tensor))
        if isinstance(saida_torch, (list, tuple)):
            saida_torch = saida_torch[0]
        vetor_torch = saida_torch.reshape(-1).cpu().numpy().astype(np.float64)

        saida_onnx = sessao.run(None, {nome_entrada: tensor})[0]
        vetor_onnx = np.asarray(saida_onnx).reshape(-1).astype(np.float64)

        if vetor_torch.shape != vetor_onnx.shape:
            raise RuntimeError(
                "As saídas têm formatos diferentes: "
                f"PyTorch {vetor_torch.shape} vs ONNX {vetor_onnx.shape}."
            )

        divergencias.append(float(np.abs(vetor_torch - vetor_onnx).max()))
        if int(np.argmax(vetor_torch)) != int(np.argmax(vetor_onnx)):
            classes_divergentes += 1

    maxima = max(divergencias)
    relatorio = {
        "aprovado": bool(maxima <= atol and classes_divergentes == 0),
        "n_imagens": len(divergencias),
        "divergencia_maxima": float(f"{maxima:.3e}"),
        "divergencia_media": float(f"{float(np.mean(divergencias)):.3e}"),
        "classes_divergentes": classes_divergentes,
        "atol": atol,
    }

    if relatorio["aprovado"]:
        logger.info(
            "Validação ONNX APROVADA em %d imagens (divergência máxima %.2e ≤ %.0e).",
            relatorio["n_imagens"], maxima, atol,
        )
    else:
        logger.error(
            "Validação ONNX REPROVADA: divergência máxima %.2e (tolerância %.0e), "
            "%d imagens com classe predita diferente.",
            maxima, atol, classes_divergentes,
        )
    return relatorio
