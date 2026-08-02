"""Etapa 5 — Benchmark de latência de inferência.

Mede a latência de ``DetectorDoencasAlface.predizer`` — o caminho real de
produção, uma imagem por chamada — sobre 100 imagens do split de teste, após
10 inferências de aquecimento. Compara PyTorch/CPU, ONNX/CPU e, quando há GPU,
PyTorch/GPU.

A medição vale para o **servidor de inferência**. A Tabela 16 do TCC compara
latências em dispositivos de borda; são contextos distintos e o relatório
gerado deixa isso explícito.

Uso:
    python desenvolvimento/scripts/05_benchmark_latencia.py
    python desenvolvimento/scripts/05_benchmark_latencia.py --n-imagens 200 --aquecimento 20

Referência: spec.md, Seção 10.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ia_alface.avaliacao import META_LATENCIA_MS, listar_imagens, medir_latencia
from ia_alface.config import CAMINHOS, descrever_ambiente
from ia_alface.inferencia import DetectorDoencasAlface

logger = logging.getLogger("benchmark")


def _medir(
    rotulo: str,
    caminho_modelo: Path,
    dispositivo: str,
    imagens: list[Path],
    aquecimento: int,
) -> dict | None:
    """Mede uma configuração, devolvendo ``None`` se ela não estiver disponível."""
    if not caminho_modelo.exists():
        logger.warning(
            "%s: modelo não encontrado em %s — ignorado.", rotulo, caminho_modelo
        )
        return None
    try:
        detector = DetectorDoencasAlface(caminho_modelo, dispositivo=dispositivo)
        estatisticas = medir_latencia(detector, imagens, aquecimento=aquecimento)
    except Exception as erro:  # noqa: BLE001 - configuração pode não estar disponível
        logger.warning("%s: medição falhou (%s) — ignorado.", rotulo, erro)
        return None

    estatisticas["rotulo"] = rotulo
    estatisticas["modelo"] = caminho_modelo.name
    logger.info(
        "%-18s mediana=%6.2f ms  média=%6.2f ms  p95=%6.2f ms  p99=%6.2f ms",
        rotulo,
        estatisticas["mediana"],
        estatisticas["media"],
        estatisticas["p95"],
        estatisticas["p99"],
    )
    return estatisticas


def _escrever_relatorio(
    medicoes: list[dict],
    caminho_md: Path,
    ambiente: dict[str, str],
    n_imagens: int,
    aquecimento: int,
) -> None:
    """Monta ``desenvolvimento/reports/benchmark_latencia.md``."""
    linhas: list[str] = []
    ad = linhas.append

    ad("# Benchmark de latência de inferência")
    ad("")
    ad(f"*Gerado por `desenvolvimento/scripts/05_benchmark_latencia.py` em "
       f"{datetime.now().strftime('%d/%m/%Y %H:%M')}.*")
    ad("")
    ad("| Parâmetro | Valor |")
    ad("| --- | --- |")
    ad(f"| Imagens medidas | {n_imagens} (split `test` de `dataset_clean`) |")
    ad(f"| Inferências de aquecimento | {aquecimento} (descartadas) |")
    ad("| Caminho medido | `DetectorDoencasAlface.predizer` — 1 imagem por chamada |")
    ad(f"| CPU | {ambiente.get('processador', 'desconhecido')} |")
    ad(f"| Sistema operacional | {platform.platform()} |")
    ad(f"| Python / torch / ultralytics | {ambiente.get('python')} / "
       f"{ambiente.get('torch')} / {ambiente.get('ultralytics')} |")
    ad(f"| CUDA disponível | {ambiente.get('cuda_disponivel')} |")
    if "gpu" in ambiente:
        ad(f"| GPU | {ambiente['gpu']} |")
    ad("")

    ad("## Resultados")
    ad("")
    if not medicoes:
        ad("Nenhuma configuração pôde ser medida. Treine o modelo com "
           "`desenvolvimento/scripts/02_treinar.py` e exporte com `desenvolvimento/scripts/04_exportar.py`.")
        ad("")
    else:
        ad("| Configuração | Mediana | Média | p95 | p99 | Mínimo | Máximo |")
        ad("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for m in medicoes:
            ad(f"| {m['rotulo']} | {m['mediana']:.2f} ms | {m['media']:.2f} ms | "
               f"{m['p95']:.2f} ms | {m['p99']:.2f} ms | {m['minimo']:.2f} ms | "
               f"{m['maximo']:.2f} ms |")
        ad("")

        ad("## Conclusão sobre o requisito RNF01")
        ad("")
        ad(f"RNF01 e o caso de teste CT04 exigem inferência em menos de "
           f"**{META_LATENCIA_MS:.0f} ms** por imagem no servidor.")
        ad("")
        ad("| Configuração | Mediana medida | Meta | Status |")
        ad("| --- | ---: | ---: | --- |")
        for m in medicoes:
            atende = m["mediana"] < META_LATENCIA_MS
            ad(f"| {m['rotulo']} | {m['mediana']:.2f} ms | < "
               f"{META_LATENCIA_MS:.0f} ms | "
               f"{'**APROVADO**' if atende else '**REPROVADO**'} |")
        ad("")
        referencia = min(medicoes, key=lambda m: m["mediana"])
        ad(f"A configuração mais rápida medida é **{referencia['rotulo']}**, com "
           f"mediana de **{referencia['mediana']:.2f} ms** no hardware descrito "
           "acima.")
        ad("")

    ad("## Observação metodológica")
    ad("")
    ad("Estes números foram medidos no servidor de inferência (CPU/GPU do host "
       "que executou o script), **não** em dispositivo embarcado. A Tabela 16 do "
       "TCC reúne latências de literatura em ESP32 e em modelos de borda "
       "quantizados; os dois contextos não devem ser comparados diretamente. Na "
       "arquitetura proposta (Seção 6.2.6.3 do TCC), o ESP32-CAM apenas captura "
       "e transmite a imagem via HTTP; a inferência ocorre no servidor.")
    ad("")

    caminho_md.write_text("\n".join(linhas), encoding="utf-8")
    logger.info("Relatório escrito em %s", caminho_md)


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada do script."""
    parser = argparse.ArgumentParser(
        description="Mede a latência de inferência e verifica o requisito RNF01."
    )
    parser.add_argument(
        "--pesos",
        type=Path,
        default=CAMINHOS.modelos / "yolo11n_cls_alface.pt",
        help="Pesos PyTorch treinados.",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="Modelo ONNX. O padrão deriva do nome dos pesos.",
    )
    parser.add_argument("--n-imagens", type=int, default=100)
    parser.add_argument("--aquecimento", type=int, default=10)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    CAMINHOS.garantir_diretorios()

    caminho_onnx = args.onnx or (CAMINHOS.modelos / f"{args.pesos.stem}.onnx")

    try:
        itens = listar_imagens(CAMINHOS.dataset_limpo / "test")
    except (FileNotFoundError, RuntimeError) as erro:
        logger.error("Split de teste indisponível: %s", erro)
        return 1

    passo = max(1, len(itens) // args.n_imagens)
    imagens = [caminho for caminho, _ in itens[::passo]][: args.n_imagens]
    logger.info(
        "Medindo latência sobre %d imagens (%d de aquecimento).",
        len(imagens),
        args.aquecimento,
    )

    ambiente = descrever_ambiente()
    medicoes: list[dict] = []

    for rotulo, caminho, dispositivo in (
        ("PyTorch / CPU", args.pesos, "cpu"),
        ("ONNX / CPU", caminho_onnx, "cpu"),
    ):
        medicao = _medir(rotulo, caminho, dispositivo, imagens, args.aquecimento)
        if medicao:
            medicoes.append(medicao)

    if ambiente.get("cuda_disponivel") == "True":
        medicao = _medir(
            "PyTorch / GPU", args.pesos, "cuda", imagens, args.aquecimento
        )
        if medicao:
            medicoes.append(medicao)
    else:
        logger.info("CUDA indisponível: medição em GPU ignorada.")

    (CAMINHOS.relatorios / "benchmark_latencia.json").write_text(
        json.dumps(
            {
                "ambiente": ambiente,
                "n_imagens": len(imagens),
                "aquecimento": args.aquecimento,
                "meta_rnf01_ms": META_LATENCIA_MS,
                "medicoes": medicoes,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _escrever_relatorio(
        medicoes,
        CAMINHOS.relatorios / "benchmark_latencia.md",
        ambiente,
        len(imagens),
        args.aquecimento,
    )

    if not medicoes:
        logger.error("Nenhuma configuração pôde ser medida.")
        return 1

    melhor = min(medicoes, key=lambda m: m["mediana"])
    atende = melhor["mediana"] < META_LATENCIA_MS
    logger.info("=" * 70)
    logger.info(
        "RNF01 (< %.0f ms): %s — melhor mediana %.2f ms em %s",
        META_LATENCIA_MS,
        "APROVADO" if atende else "REPROVADO",
        melhor["mediana"],
        melhor["rotulo"],
    )
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
