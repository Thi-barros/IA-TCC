"""Utilitário — empacota ``desenvolvimento/dataset_clean/`` para envio ao Google Colab.

Não faz parte do pipeline numerado (01–05): é apenas um auxiliar para viabilizar
o treinamento em GPU na nuvem.

O ``desenvolvimento/dataset_clean/`` completo ocupa cerca de 220 MB, tamanho em que o
``files.upload()`` do Colab fica lento e sujeito a falhas. Por padrão este script
gera uma versão **reduzida**, em que o menor lado de cada imagem é limitado a
320 px.

Por que 320 px não prejudica o treino: o modelo treina em ``imgsz=224``, e a
augmentation recorta entre 60% e 100% da área antes de redimensionar para 224.
Partindo de um menor lado de 320 px, o recorte mais agressivo ainda tem cerca de
248 px — acima dos 224 finais. Ou seja, nenhum pixel útil é perdido. Imagens que
já são menores que o limite não são tocadas.

Uso:
    python desenvolvimento/scripts/preparar_zip_colab.py
    python desenvolvimento/scripts/preparar_zip_colab.py --sem-reducao      # cópia fiel, ~220 MB
    python desenvolvimento/scripts/preparar_zip_colab.py --max-lado 448     # mais margem
    python desenvolvimento/scripts/preparar_zip_colab.py --origem desenvolvimento/dataset_dominio   # verificador
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from PIL import Image

from ia_alface.config import CAMINHOS

logger = logging.getLogger("zip_colab")


def _reduzir(origem: Path, destino: Path, max_lado: int, qualidade: int) -> None:
    """Grava uma cópia com o menor lado limitado a ``max_lado`` px."""
    with Image.open(origem) as imagem:
        imagem = imagem.convert("RGB")
        menor = min(imagem.size)
        if menor > max_lado:
            escala = max_lado / menor
            novo = (
                max(1, round(imagem.width * escala)),
                max(1, round(imagem.height * escala)),
            )
            imagem = imagem.resize(novo, Image.LANCZOS)
        imagem.save(destino, format="JPEG", quality=qualidade)


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada do script."""
    parser = argparse.ArgumentParser(
        description="Empacota desenvolvimento/dataset_clean/ para upload no Google Colab."
    )
    parser.add_argument(
        "--origem", type=Path, default=CAMINHOS.dataset_limpo,
        help="Diretório do dataset preparado (padrão: desenvolvimento/dataset_clean/).",
    )
    parser.add_argument(
        "--saida", type=Path, default=None,
        help="Arquivo .zip de saída (padrão: <nome da origem>.zip na raiz do projeto).",
    )
    parser.add_argument(
        "--max-lado", type=int, default=320,
        help="Limite do menor lado, em pixels (padrão: 320).",
    )
    parser.add_argument("--qualidade", type=int, default=92)
    parser.add_argument(
        "--sem-reducao", action="store_true",
        help="Empacota as imagens sem redimensionar.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.saida is None:
        args.saida = CAMINHOS.raiz / f"{args.origem.name}.zip"

    if not args.origem.is_dir():
        logger.error(
            "%s não existe. Gere-o antes: 01_preparar_dataset.py (dataset_clean/) "
            "ou 06_preparar_dataset_dominio.py (dataset_dominio/).",
            args.origem,
        )
        return 1

    imagens = sorted(args.origem.rglob("*.jpg"))
    if not imagens:
        logger.error("Nenhuma imagem .jpg encontrada em %s.", args.origem)
        return 1

    bytes_origem = sum(f.stat().st_size for f in imagens)
    logger.info(
        "%d imagens em %s (%.1f MB).", len(imagens), args.origem, bytes_origem / 1e6
    )

    inicio = time.time()

    if args.sem_reducao:
        with zipfile.ZipFile(args.saida, "w", zipfile.ZIP_STORED) as pacote:
            for arquivo in imagens:
                pacote.write(arquivo, arquivo.relative_to(args.origem.parent))
    else:
        logger.info(
            "Reduzindo o menor lado para no máximo %d px (qualidade %d).",
            args.max_lado, args.qualidade,
        )
        with tempfile.TemporaryDirectory() as temporario:
            raiz_temp = Path(temporario) / args.origem.name
            for indice, arquivo in enumerate(imagens, start=1):
                relativo = arquivo.relative_to(args.origem)
                saida_imagem = raiz_temp / relativo
                saida_imagem.parent.mkdir(parents=True, exist_ok=True)
                _reduzir(arquivo, saida_imagem, args.max_lado, args.qualidade)
                if indice % 500 == 0:
                    logger.info("  %d/%d imagens processadas", indice, len(imagens))

            with zipfile.ZipFile(args.saida, "w", zipfile.ZIP_STORED) as pacote:
                for arquivo in sorted(raiz_temp.rglob("*.jpg")):
                    pacote.write(arquivo, arquivo.relative_to(raiz_temp.parent))

    tamanho_mb = args.saida.stat().st_size / 1e6
    logger.info("=" * 70)
    logger.info("Arquivo gerado: %s", args.saida)
    logger.info(
        "Tamanho: %.1f MB (origem: %.1f MB) em %.1fs",
        tamanho_mb, bytes_origem / 1e6, time.time() - inicio,
    )
    if tamanho_mb > 100:
        logger.warning(
            "Acima de 100 MB: prefira enviar via Google Drive em vez do "
            "files.upload() do Colab."
        )
    else:
        logger.info("Tamanho adequado para o files.upload() do Colab.")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
