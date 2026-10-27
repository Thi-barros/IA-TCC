"""Etapa 7 — Treinamento do verificador de domínio (alface / não alface).

Treina um YOLO11n-cls binário sobre ``desenvolvimento/dataset_dominio/`` (gerado
pela etapa 6) com os mesmos hiperparâmetros do classificador de doenças, e copia
os melhores pesos para ``modelos/yolo11n_cls_dominio.pt``.

Em CPU o treino leva horas; prefira
``desenvolvimento/notebooks/treino_verificador_colab.ipynb`` com GPU.

Rode só **depois** que as fotos de "não alface" da estufa estiverem no dataset
(reexecute a etapa 6 antes).

Uso:
    python desenvolvimento/scripts/07_treinar_verificador.py
    python desenvolvimento/scripts/07_treinar_verificador.py --epocas 1 --batch 16    # teste de fumaça

Referência: docs/VERIFICADOR_DOMINIO.md.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ia_alface.config import CAMINHOS, ConfigTreino, descrever_ambiente
from ia_alface.treino import treinar_verificador

logger = logging.getLogger("treinar_verificador")


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada do script."""
    padrao = ConfigTreino()
    parser = argparse.ArgumentParser(
        description="Treina o verificador de domínio YOLO11-cls (alface / não alface)."
    )
    parser.add_argument("--modelo", default=padrao.modelo_base, help="Pesos base.")
    parser.add_argument("--epocas", type=int, default=padrao.epocas)
    parser.add_argument("--imgsz", type=int, default=padrao.imgsz)
    parser.add_argument("--batch", type=int, default=padrao.batch,
                        help="Reduza para 16 se houver falta de memória.")
    parser.add_argument("--patience", type=int, default=padrao.patience)
    parser.add_argument("--seed", type=int, default=padrao.seed)
    parser.add_argument("--workers", type=int, default=padrao.workers,
                        help="Mantenha 0 no Windows.")
    parser.add_argument("--dispositivo", default=None,
                        help="'cuda' ou 'cpu'. O padrão autodetecta.")
    parser.add_argument("--nome", default=None,
                        help="Nome da execução (padrão: <modelo>_dominio).")
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

    nome = args.nome or f"{Path(args.modelo).stem.replace('-', '_')}_dominio"
    config = ConfigTreino(
        modelo_base=args.modelo,
        epocas=args.epocas,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        seed=args.seed,
        workers=args.workers,
        dispositivo=args.dispositivo,
        nome_execucao=nome,
    )

    try:
        resumo = treinar_verificador(config, caminhos=CAMINHOS)
    except (FileNotFoundError, RuntimeError) as erro:
        logger.error("Treinamento interrompido: %s", erro)
        return 1

    acuracia = resumo["melhor_acuracia_val"]
    logger.info("=" * 70)
    logger.info("Treinamento concluído em %s", resumo["duracao_formatada"])
    logger.info("Pesos: %s", resumo["pesos"])
    logger.info(
        "Melhor acurácia top-1 de validação: %s",
        "não medida" if acuracia is None else f"{acuracia:.2%}",
    )
    logger.info(
        "A acurácia geral não é o critério de aceitação: o que importa é a "
        "revocação de 'alface' em folhas doentes, medida na avaliação."
    )
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
