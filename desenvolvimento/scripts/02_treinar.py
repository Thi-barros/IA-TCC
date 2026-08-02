"""Etapa 2 — Treinamento do modelo YOLO11-cls.

Treina sobre ``desenvolvimento/dataset_clean/`` (gerado pela etapa 1) e copia os melhores pesos
para ``modelos/``.

Uso:
    python desenvolvimento/scripts/02_treinar.py
    python desenvolvimento/scripts/02_treinar.py --modelo yolo11s-cls.pt
    python desenvolvimento/scripts/02_treinar.py --epocas 1 --batch 16    # teste de fumaça

Referência: spec.md, Seção 6.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ia_alface.config import CAMINHOS, ConfigTreino, descrever_ambiente
from ia_alface.treino import treinar

logger = logging.getLogger("treinar")


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada do script."""
    padrao = ConfigTreino()
    parser = argparse.ArgumentParser(
        description="Treina o classificador YOLO11-cls de doenças foliares em alface."
    )
    parser.add_argument("--modelo", default=padrao.modelo_base,
                        help="Pesos base (ex.: yolo11n-cls.pt, yolo11s-cls.pt).")
    parser.add_argument("--epocas", type=int, default=padrao.epocas)
    parser.add_argument("--imgsz", type=int, default=padrao.imgsz)
    parser.add_argument("--batch", type=int, default=padrao.batch,
                        help="Reduza para 16 se houver falta de memória.")
    parser.add_argument("--patience", type=int, default=padrao.patience)
    parser.add_argument("--lr0", type=float, default=padrao.lr0)
    parser.add_argument("--optimizer", default=padrao.optimizer)
    parser.add_argument("--dropout", type=float, default=padrao.dropout)
    parser.add_argument("--label-smoothing", type=float, default=padrao.label_smoothing)
    parser.add_argument("--seed", type=int, default=padrao.seed)
    parser.add_argument("--workers", type=int, default=padrao.workers,
                        help="Mantenha 0 no Windows.")
    parser.add_argument("--dispositivo", default=None,
                        help="'cuda' ou 'cpu'. O padrão autodetecta.")
    parser.add_argument("--nome", default=None,
                        help="Nome da execução. O padrão deriva do modelo base.")
    parser.add_argument("--sem-latencia", action="store_true",
                        help="Não medir a latência indicativa ao final.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    ambiente = descrever_ambiente()
    logger.info(
        "Ambiente: Python %s | torch %s | ultralytics %s | CUDA disponível: %s",
        ambiente["python"], ambiente["torch"], ambiente["ultralytics"],
        ambiente["cuda_disponivel"],
    )

    config = ConfigTreino(
        modelo_base=args.modelo,
        epocas=args.epocas,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        lr0=args.lr0,
        optimizer=args.optimizer,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        seed=args.seed,
        workers=args.workers,
        dispositivo=args.dispositivo,
        nome_execucao=args.nome,
    )

    try:
        resumo = treinar(
            config,
            caminhos=CAMINHOS,
            medir_latencia_indicativa=not args.sem_latencia,
        )
    except (FileNotFoundError, RuntimeError) as erro:
        logger.error("Treinamento interrompido: %s", erro)
        return 1

    acuracia = resumo["melhor_acuracia_val"]
    logger.info("=" * 70)
    logger.info("Treinamento concluído em %s", resumo["duracao_formatada"])
    logger.info("Pesos: %s", resumo["pesos"])
    logger.info("Parâmetros: %.2f M", resumo["parametros_milhoes"])
    logger.info(
        "Melhor acurácia top-1 de validação: %s",
        "não medida" if acuracia is None else f"{acuracia:.2%}",
    )
    if resumo["latencia_cpu_ms"]:
        logger.info(
            "Latência indicativa em CPU (mediana): %.2f ms",
            resumo["latencia_cpu_ms"]["mediana"],
        )
    logger.info(
        "Próximo passo: python desenvolvimento/scripts/03_avaliar.py --pesos %s", resumo["pesos"]
    )
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
