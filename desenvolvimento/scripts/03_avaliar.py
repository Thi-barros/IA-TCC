"""Etapa 3 — Avaliação no split de teste.

Avalia **sempre** ``desenvolvimento/dataset_clean/test``, jamais ``desenvolvimento/dataset/test`` (que contém
imagens vazadas do treino).

Produz em ``desenvolvimento/reports/``: ``metricas_test.json``, ``matriz_confusao.png/.csv``,
``curva_treino.png``, ``erros_classificacao/`` e ``avaliacao_test.md``.

Uso:
    python desenvolvimento/scripts/03_avaliar.py
    python desenvolvimento/scripts/03_avaliar.py --pesos modelos/yolo11s_cls_alface.pt
    python desenvolvimento/scripts/03_avaliar.py --split val    # ajuste de hiperparâmetros

Referência: spec.md, Seção 7.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ia_alface import avaliacao
from ia_alface.config import CAMINHOS, descrever_ambiente
from ia_alface.inferencia import DetectorDoencasAlface

logger = logging.getLogger("avaliar")


def _localizar_results_csv(pesos: Path) -> Path | None:
    """Procura o ``results.csv`` da execução que produziu estes pesos."""
    nome = pesos.stem  # ex.: "yolo11n_cls_alface"
    candidato = CAMINHOS.runs / "classify" / nome / "results.csv"
    if candidato.exists():
        return candidato

    diretorio_classify = CAMINHOS.runs / "classify"
    if not diretorio_classify.is_dir():
        return None
    encontrados = sorted(
        diretorio_classify.glob("*/results.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return encontrados[0] if encontrados else None


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada do script."""
    parser = argparse.ArgumentParser(
        description="Avalia o classificador de doenças foliares em alface."
    )
    parser.add_argument(
        "--pesos",
        type=Path,
        default=CAMINHOS.modelos / "yolo11n_cls_alface.pt",
        help="Arquivo de pesos treinados.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "val", "train"],
        help=(
            "Split a avaliar. Use 'val' para ajustar hiperparâmetros; 'test' deve "
            "ser medido uma única vez, ao final."
        ),
    )
    parser.add_argument("--dispositivo", default=None, help="'cuda' ou 'cpu'.")
    parser.add_argument(
        "--limiar", type=float, default=0.85, help="Limiar de confiança para alerta."
    )
    parser.add_argument(
        "--piores", type=int, default=20, help="Quantidade de piores casos a exportar."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    CAMINHOS.garantir_diretorios()

    if args.split != "test":
        logger.warning(
            "Avaliando o split '%s'. Lembre-se: nenhum hiperparâmetro deve ser "
            "ajustado observando o split 'test'.",
            args.split,
        )

    try:
        detector = DetectorDoencasAlface(
            caminho_modelo=args.pesos,
            dispositivo=args.dispositivo,
            limiar_confianca=args.limiar,
        )
    except (FileNotFoundError, ValueError) as erro:
        logger.error("Não foi possível carregar o modelo: %s", erro)
        return 1

    raiz_split = CAMINHOS.dataset_limpo / args.split
    try:
        resultado = avaliacao.avaliar(detector, raiz_split)
    except (FileNotFoundError, RuntimeError) as erro:
        logger.error("Avaliação interrompida: %s", erro)
        return 1

    metricas = avaliacao.calcular_metricas(resultado)
    metricas["split_avaliado"] = args.split
    metricas["pesos"] = str(args.pesos)
    metricas["dispositivo"] = detector.dispositivo

    sufixo = "" if args.split == "test" else f"_{args.split}"
    avaliacao.salvar_metricas_json(
        metricas, CAMINHOS.relatorios / f"metricas_test{sufixo}.json"
    )

    absoluta, normalizada = avaliacao.matriz_confusao(resultado)
    nome_matriz = f"matriz_confusao{sufixo}.png"
    avaliacao.salvar_matriz_confusao(
        absoluta,
        normalizada,
        CAMINHOS.relatorios / nome_matriz,
        CAMINHOS.relatorios / f"matriz_confusao{sufixo}.csv",
    )

    nomes_figuras = {"matriz": nome_matriz}
    results_csv = _localizar_results_csv(args.pesos)
    if results_csv and avaliacao.salvar_curva_treino(
        results_csv, CAMINHOS.relatorios / "curva_treino.png"
    ):
        nomes_figuras["curva"] = "curva_treino.png"

    piores = avaliacao.salvar_erros_classificacao(
        resultado,
        CAMINHOS.relatorios / f"erros_classificacao{sufixo}",
        quantidade=args.piores,
    )

    resultado.para_dataframe().to_csv(
        CAMINHOS.relatorios / f"predicoes{sufixo}.csv", index=False, encoding="utf-8"
    )

    avaliacao.gerar_relatorio_markdown(
        metricas=metricas,
        absoluta=absoluta,
        normalizada=normalizada,
        piores=piores,
        caminho_md=CAMINHOS.relatorios / f"avaliacao_test{sufixo}.md",
        modelo_versao=detector.modelo_versao,
        dispositivo=detector.dispositivo,
        ambiente=descrever_ambiente(),
        nomes_figuras=nomes_figuras,
    )

    acuracia = metricas["acuracia_global"]
    latencia = metricas["latencia_ms"]["mediana"]
    logger.info("=" * 70)
    logger.info("Split avaliado: %s (%d imagens)", args.split, metricas["n_imagens"])
    logger.info(
        "Acurácia global: %.2f%%  (meta RNF02: > 90%%)  -> %s",
        acuracia * 100,
        "APROVADO" if acuracia > avaliacao.META_ACURACIA else "REPROVADO",
    )
    logger.info(
        "Latência mediana: %.2f ms  (meta RNF01: < 100 ms) -> %s",
        latencia,
        "APROVADO" if latencia < avaliacao.META_LATENCIA_MS else "REPROVADO",
    )
    for classe, dados in metricas["por_classe"].items():
        logger.info(
            "  %-12s precisão=%.3f recall=%.3f f1=%.3f (n=%d)",
            classe, dados["precisao"], dados["recall"], dados["f1_score"],
            dados["suporte"],
        )
    logger.info("Relatório: %s", CAMINHOS.relatorios / f"avaliacao_test{sufixo}.md")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
