"""Etapa 4 — Exportação para ONNX e TorchScript, com validação numérica.

Exporta os pesos treinados e **verifica** que a saída do ONNX coincide com a do
modelo PyTorch em ao menos 20 imagens do split de teste. Falha se divergir.

Uso:
    python desenvolvimento/scripts/04_exportar.py
    python desenvolvimento/scripts/04_exportar.py --pesos modelos/yolo11s_cls_alface.pt --n-validacao 40

Referência: spec.md, Seção 10.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ia_alface import exportacao
from ia_alface.avaliacao import listar_imagens
from ia_alface.config import CAMINHOS
from ia_alface.inferencia import IMGSZ_PADRAO

logger = logging.getLogger("exportar")


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada do script."""
    parser = argparse.ArgumentParser(
        description="Exporta o modelo treinado para ONNX e TorchScript."
    )
    parser.add_argument(
        "--pesos",
        type=Path,
        default=CAMINHOS.modelos / "yolo11n_cls_alface.pt",
        help="Arquivo de pesos treinados.",
    )
    parser.add_argument("--imgsz", type=int, default=IMGSZ_PADRAO)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--sem-simplify",
        action="store_true",
        help="Não tentar simplificar o grafo ONNX (dispensa o pacote onnxslim).",
    )
    parser.add_argument(
        "--sem-torchscript", action="store_true", help="Não exportar TorchScript."
    )
    parser.add_argument(
        "--n-validacao",
        type=int,
        default=20,
        help="Número de imagens do split de teste usadas na validação numérica.",
    )
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    CAMINHOS.garantir_diretorios()

    try:
        caminho_onnx = exportacao.exportar_onnx(
            args.pesos,
            imgsz=args.imgsz,
            opset=args.opset,
            simplify=not args.sem_simplify,
        )
    except (FileNotFoundError, RuntimeError) as erro:
        logger.error("Exportação ONNX falhou: %s", erro)
        return 1

    caminho_torchscript = None
    if not args.sem_torchscript:
        try:
            caminho_torchscript = exportacao.exportar_torchscript(
                args.pesos, imgsz=args.imgsz
            )
        except Exception as erro:  # noqa: BLE001 - TorchScript é alternativa
            logger.warning("Exportação TorchScript falhou: %s", erro)

    try:
        itens = listar_imagens(CAMINHOS.dataset_limpo / "test")
    except (FileNotFoundError, RuntimeError) as erro:
        logger.error(
            "Não foi possível validar o ONNX porque o split de teste não está "
            "disponível: %s",
            erro,
        )
        return 1

    # Amostra distribuída ao longo da lista, para cobrir as três classes.
    passo = max(1, len(itens) // args.n_validacao)
    imagens = [caminho for caminho, _ in itens[::passo]][: args.n_validacao]

    relatorio = exportacao.validar_onnx(
        args.pesos,
        caminho_onnx,
        imagens,
        imgsz=args.imgsz,
        atol=args.atol,
    )
    relatorio["pesos_pytorch"] = str(args.pesos)
    relatorio["modelo_onnx"] = str(caminho_onnx)
    relatorio["modelo_torchscript"] = (
        str(caminho_torchscript) if caminho_torchscript else None
    )
    relatorio["imgsz"] = args.imgsz
    relatorio["opset"] = args.opset

    caminho_json = CAMINHOS.relatorios / "validacao_onnx.json"
    caminho_json.write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info("=" * 70)
    logger.info("ONNX:        %s", caminho_onnx)
    if caminho_torchscript:
        logger.info("TorchScript: %s", caminho_torchscript)
    logger.info(
        "Validação numérica: %s (%d imagens, divergência máxima %.2e, tolerância %.0e)",
        "APROVADA" if relatorio["aprovado"] else "REPROVADA",
        relatorio["n_imagens"],
        relatorio["divergencia_maxima"],
        relatorio["atol"],
    )
    logger.info("Relatório: %s", caminho_json)
    logger.info("=" * 70)

    return 0 if relatorio["aprovado"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
