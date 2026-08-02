"""Utilitário — classifica imagens avulsas com o modelo treinado.

Ferramenta de inspeção manual: aponta o modelo para uma imagem, uma pasta ou
vários arquivos e mostra a classe prevista, as probabilidades e o tempo de
inferência. Útil para testar fotos próprias e para demonstrações.

Não faz parte do pipeline numerado (01–05).

Uso:
    python desenvolvimento/scripts/testar_imagem.py foto.jpg
    python desenvolvimento/scripts/testar_imagem.py minhas_fotos/ --heatmap
    python desenvolvimento/scripts/testar_imagem.py foto.jpg --como-bytes
    python desenvolvimento/scripts/testar_imagem.py foto.jpg --pesos modelos/yolo11n_cls_alface.onnx
    python desenvolvimento/scripts/testar_imagem.py foto.jpg --json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ia_alface.classes import ROTULOS_CANONICOS, rotulo_exibicao
from ia_alface.config import CAMINHOS
from ia_alface.inferencia import DetectorDoencasAlface, ResultadoDeteccao

logger = logging.getLogger("testar_imagem")

EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _coletar_imagens(entradas: list[Path]) -> list[Path]:
    """Expande diretórios e filtra por extensão de imagem."""
    encontradas: list[Path] = []
    for entrada in entradas:
        if entrada.is_dir():
            encontradas.extend(
                sorted(p for p in entrada.rglob("*") if p.suffix.lower() in EXTENSOES)
            )
        else:
            encontradas.append(entrada)
    return encontradas


def _barra(fracao: float, largura: int = 28) -> str:
    """Desenha uma barra horizontal proporcional."""
    cheios = round(fracao * largura)
    return "█" * cheios + "·" * (largura - cheios)


def _imprimir(caminho: Path, resultado: ResultadoDeteccao, limiar: float) -> None:
    """Mostra o resultado de uma imagem em formato legível."""
    print(f"\n{'=' * 60}")
    print(f"Imagem: {caminho.name}")
    print(f"{'-' * 60}")

    for classe in ROTULOS_CANONICOS:
        probabilidade = resultado.probabilidades.get(classe, 0.0)
        marca = "->" if classe == resultado.classe else "  "
        print(f" {marca} {classe:<11} {_barra(probabilidade)} {probabilidade:7.2%}")

    print(f"{'-' * 60}")
    print(f" Classe:     {rotulo_exibicao(resultado.classe)}")
    print(f" Confianca:  {resultado.confianca:.1%}")
    print(f" Tempo:      {resultado.tempo_inferencia_ms:.1f} ms")

    if resultado.alerta:
        print(f" ALERTA:     SIM — dispara notificacao push (RF07)")
    elif resultado.patogeno_detectado:
        print(
            f" ALERTA:     nao — patogeno detectado, mas confianca "
            f"{resultado.confianca:.1%} < limiar {limiar:.0%}"
        )
    else:
        print(" ALERTA:     nao — folha saudavel")


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada do script."""
    parser = argparse.ArgumentParser(
        description="Classifica imagens de folhas de alface com o modelo treinado."
    )
    parser.add_argument(
        "imagens", type=Path, nargs="+",
        help="Arquivos de imagem ou diretórios.",
    )
    parser.add_argument(
        "--pesos", type=Path,
        default=CAMINHOS.modelos / "yolo11n_cls_alface.pt",
        help="Modelo a usar (.pt ou .onnx).",
    )
    parser.add_argument("--limiar", type=float, default=0.85)
    parser.add_argument("--dispositivo", default=None, help="'cuda' ou 'cpu'.")
    parser.add_argument(
        "--heatmap", action="store_true",
        help="Gera o mapa de calor Grad-CAM em desenvolvimento/reports/heatmaps/.",
    )
    parser.add_argument(
        "--como-bytes", action="store_true",
        help=(
            "Lê o arquivo como bytes e o envia assim ao detector, simulando o "
            "envio do ESP32-CAM por HTTP POST."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Saída em JSON.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s: %(message)s"
    )

    imagens = _coletar_imagens(args.imagens)
    if not imagens:
        logger.error("Nenhuma imagem encontrada nos caminhos informados.")
        return 1

    try:
        detector = DetectorDoencasAlface(
            caminho_modelo=args.pesos,
            dispositivo=args.dispositivo,
            limiar_confianca=args.limiar,
        )
    except (FileNotFoundError, ValueError) as erro:
        logger.error("Nao foi possivel carregar o modelo: %s", erro)
        return 1

    explicador = None
    if args.heatmap:
        if args.pesos.suffix.lower() != ".pt":
            logger.error("O Grad-CAM exige os pesos .pt, nao um modelo exportado.")
            return 1
        from ia_alface.explicabilidade import GradCAM

        explicador = GradCAM(args.pesos, dispositivo=args.dispositivo)

    if not args.json:
        print(f"\nModelo:      {detector.modelo_versao}")
        print(f"Dispositivo: {detector.dispositivo}")
        print(f"Limiar:      {detector.limiar_confianca:.0%}")
        print(f"Imagens:     {len(imagens)}")

    saida_json: list[dict] = []
    falhas = 0
    alertas = 0

    for caminho in imagens:
        try:
            entrada = caminho.read_bytes() if args.como_bytes else caminho
            resultado = detector.predizer(entrada)
        except ValueError as erro:
            falhas += 1
            if args.json:
                saida_json.append({"imagem": str(caminho), "erro": str(erro)})
            else:
                print(f"\n{'=' * 60}\nImagem: {caminho.name}\n  ERRO: {erro}")
            continue

        alertas += int(resultado.alerta)

        if explicador is not None:
            destino = CAMINHOS.relatorios / "heatmaps" / f"{caminho.stem}_gradcam.png"
            info = explicador.salvar_sobreposicao(caminho, destino)
            resultado.heatmap_url = info["caminho"]

        if args.json:
            registro = resultado.to_dict()
            registro["imagem"] = str(caminho)
            saida_json.append(registro)
        else:
            _imprimir(caminho, resultado, detector.limiar_confianca)
            if resultado.heatmap_url:
                print(f" Heatmap:    {resultado.heatmap_url}")

    if args.json:
        print(json.dumps(saida_json, indent=2, ensure_ascii=False))
    elif len(imagens) > 1:
        print(f"\n{'=' * 60}")
        print(f"Total: {len(imagens) - falhas} analisadas, {falhas} com erro")
        print(f"Alertas disparados: {alertas}")

    return 1 if falhas == len(imagens) else 0


if __name__ == "__main__":
    raise SystemExit(main())
