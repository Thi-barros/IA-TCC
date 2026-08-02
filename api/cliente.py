"""Cliente de linha de comando para consumir a API de inferência.

Diferente de ``desenvolvimento/scripts/testar_imagem.py``, que carrega o modelo diretamente,
este cliente exercita o **caminho completo de integração**: monta a requisição
HTTP, envia à API e interpreta a resposta. É o que valida se o serviço está
funcionando ponta a ponta.

Usa apenas a biblioteca padrão — não exige ``requests`` instalado.

Uso:
    python api/cliente.py foto.jpg
    python api/cliente.py minhas_fotos/ --mapa
    python api/cliente.py foto.jpg --bruto              # rota do ESP32-CAM
    python api/cliente.py minhas_fotos/ --csv saida.csv
    python api/cliente.py foto.jpg --url http://192.168.0.100:8000
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def coletar_imagens(entradas: list[Path]) -> list[Path]:
    """Expande diretórios e filtra por extensão de imagem."""
    encontradas: list[Path] = []
    for entrada in entradas:
        if entrada.is_dir():
            encontradas.extend(
                sorted(p for p in entrada.rglob("*") if p.suffix.lower() in EXTENSOES)
            )
        elif entrada.is_file():
            encontradas.append(entrada)
        else:
            print(f"  aviso: caminho inexistente, ignorado: {entrada}")
    return encontradas


def montar_multipart(caminho: Path) -> tuple[bytes, str]:
    """Monta um corpo ``multipart/form-data`` com um único arquivo.

    Implementado à mão para manter o cliente livre de dependências externas.

    Args:
        caminho: Arquivo de imagem a enviar.

    Returns:
        Tupla ``(corpo, content_type)``.
    """
    fronteira = f"----ia-alface-{uuid.uuid4().hex}"
    tipo = mimetypes.guess_type(caminho.name)[0] or "application/octet-stream"

    corpo = b"".join(
        [
            f"--{fronteira}\r\n".encode(),
            f'Content-Disposition: form-data; name="imagem"; '
            f'filename="{caminho.name}"\r\n'.encode(),
            f"Content-Type: {tipo}\r\n\r\n".encode(),
            caminho.read_bytes(),
            f"\r\n--{fronteira}--\r\n".encode(),
        ]
    )
    return corpo, f"multipart/form-data; boundary={fronteira}"


def enviar(
    url_base: str, caminho: Path, com_mapa: bool, bruto: bool, tempo_limite: float
) -> tuple[dict | None, str | None, float]:
    """Envia uma imagem à API.

    Args:
        url_base: Endereço do serviço, sem barra final.
        caminho: Imagem a enviar.
        com_mapa: Solicita o mapa de calor Grad-CAM.
        bruto: Usa a rota ``/analyze/raw`` (corpo bruto, como o ESP32-CAM).
        tempo_limite: Tempo máximo de espera, em segundos.

    Returns:
        Tupla ``(resposta, erro, duracao_ms)``. Exatamente um entre resposta e
        erro é diferente de ``None``.
    """
    rota = "/analyze/raw" if bruto else "/analyze"
    url = f"{url_base}{rota}" + ("?heatmap=true" if com_mapa else "")

    if bruto:
        corpo = caminho.read_bytes()
        tipo = mimetypes.guess_type(caminho.name)[0] or "image/jpeg"
    else:
        corpo, tipo = montar_multipart(caminho)

    requisicao = urllib.request.Request(
        url, data=corpo, headers={"Content-Type": tipo}, method="POST"
    )

    inicio = time.perf_counter()
    try:
        with urllib.request.urlopen(requisicao, timeout=tempo_limite) as resposta:
            dados = json.load(resposta)
        return dados, None, (time.perf_counter() - inicio) * 1000
    except urllib.error.HTTPError as erro:
        try:
            detalhe = json.load(erro).get("detail", erro.reason)
        except Exception:  # noqa: BLE001 - corpo pode não ser JSON
            detalhe = erro.reason
        return None, f"HTTP {erro.code}: {detalhe}", (time.perf_counter() - inicio) * 1000
    except urllib.error.URLError as erro:
        return None, f"falha de conexão: {erro.reason}", 0.0


def barra(fracao: float, largura: int = 26) -> str:
    """Desenha uma barra horizontal proporcional."""
    cheios = round(fracao * largura)
    return "#" * cheios + "." * (largura - cheios)


def imprimir(caminho: Path, dados: dict, duracao_ms: float) -> None:
    """Mostra o resultado de uma imagem em formato legível."""
    print(f"\n{'=' * 62}")
    print(f"Imagem: {caminho.name}")
    print("-" * 62)

    for classe in ("bacteriano", "fungico", "saudavel"):
        p = dados["probabilidades"].get(classe, 0.0)
        marca = "->" if classe == dados["classe"] else "  "
        print(f" {marca} {classe:<11} {barra(p)} {p:7.2%}")

    print("-" * 62)
    print(f" Classe:      {dados['classe_exibicao']}")
    print(f" Confianca:   {dados['confianca']:.1%}")
    print(f" Inferencia:  {dados['tempo_inferencia_ms']:.1f} ms")
    print(f" Total HTTP:  {duracao_ms:.1f} ms")
    print(f" Detectado:   {dados['detectado_em']}")

    if dados["alerta"]:
        print(" ALERTA:      SIM - dispararia a notificacao push (RF07)")
    elif dados["patogeno_detectado"]:
        print(" ALERTA:      nao - confianca abaixo do limiar")
    else:
        print(" ALERTA:      nao - folha saudavel")


def salvar_mapa(caminho: Path, dados: dict, destino: Path) -> Path | None:
    """Grava o mapa de calor recebido em base64, se presente."""
    if not dados.get("heatmap_base64"):
        return None
    destino.mkdir(parents=True, exist_ok=True)
    saida = destino / f"{caminho.stem}_gradcam.png"
    saida.write_bytes(base64.b64decode(dados["heatmap_base64"]))
    return saida


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada do cliente."""
    parser = argparse.ArgumentParser(
        description="Envia imagens à API de inferência e mostra o resultado."
    )
    parser.add_argument("imagens", type=Path, nargs="+",
                        help="Arquivos de imagem ou diretórios.")
    parser.add_argument("--url", default="http://localhost:8000",
                        help="Endereço da API (padrão: http://localhost:8000).")
    parser.add_argument("--mapa", action="store_true",
                        help="Solicitar o mapa de calor Grad-CAM.")
    parser.add_argument("--saida-mapas", type=Path, default=Path("mapas_de_calor"),
                        help="Diretório onde gravar os mapas de calor.")
    parser.add_argument("--bruto", action="store_true",
                        help="Usar /analyze/raw, a rota do ESP32-CAM.")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Gravar os resultados em arquivo CSV.")
    parser.add_argument("--json", action="store_true",
                        help="Imprimir as respostas cruas em JSON.")
    parser.add_argument("--tempo-limite", type=float, default=30.0)
    args = parser.parse_args(argv)

    url_base = args.url.rstrip("/")

    # Verifica o serviço antes de enviar qualquer imagem, para falhar cedo com
    # uma mensagem útil em vez de um erro de conexão por arquivo.
    try:
        with urllib.request.urlopen(f"{url_base}/health", timeout=10) as resposta:
            saude = json.load(resposta)
    except Exception as erro:  # noqa: BLE001
        print(f"Nao foi possivel contatar a API em {url_base}: {erro}")
        print("Inicie o servico com:  python -m uvicorn api.main:app")
        return 1

    if not saude.get("modelo_carregado"):
        print(f"A API respondeu, mas o modelo nao esta carregado: "
              f"{saude.get('detalhe')}")
        return 1

    imagens = coletar_imagens(args.imagens)
    if not imagens:
        print("Nenhuma imagem encontrada nos caminhos informados.")
        return 1

    if not args.json:
        print(f"API:      {url_base}")
        print(f"Modelo:   {saude['modelo_versao']} ({saude['dispositivo']})")
        print(f"Rota:     {'/analyze/raw' if args.bruto else '/analyze'}")
        print(f"Imagens:  {len(imagens)}")

    linhas: list[dict] = []
    respostas_cruas: list[dict] = []
    alertas = falhas = 0
    tempos: list[float] = []

    for caminho in imagens:
        dados, erro, duracao = enviar(
            url_base, caminho, args.mapa, args.bruto, args.tempo_limite
        )

        if erro is not None:
            falhas += 1
            if args.json:
                respostas_cruas.append({"imagem": str(caminho), "erro": erro})
            else:
                print(f"\n{'=' * 62}\nImagem: {caminho.name}\n  ERRO: {erro}")
            linhas.append({"imagem": caminho.name, "erro": erro})
            continue

        alertas += int(dados["alerta"])
        tempos.append(dados["tempo_inferencia_ms"])

        if args.json:
            respostas_cruas.append({"imagem": str(caminho), **dados})
        else:
            imprimir(caminho, dados, duracao)
            salvo = salvar_mapa(caminho, dados, args.saida_mapas)
            if salvo:
                print(f" Mapa:        {salvo}")

        linhas.append(
            {
                "imagem": caminho.name,
                "classe": dados["classe"],
                "confianca": dados["confianca"],
                "alerta": dados["alerta"],
                "prob_bacteriano": dados["probabilidades"].get("bacteriano"),
                "prob_fungico": dados["probabilidades"].get("fungico"),
                "prob_saudavel": dados["probabilidades"].get("saudavel"),
                "tempo_inferencia_ms": dados["tempo_inferencia_ms"],
                "tempo_http_ms": round(duracao, 2),
                "detectado_em": dados["detectado_em"],
                "erro": "",
            }
        )

    if args.json:
        print(json.dumps(respostas_cruas, indent=2, ensure_ascii=False))

    if args.csv:
        campos = [
            "imagem", "classe", "confianca", "alerta", "prob_bacteriano",
            "prob_fungico", "prob_saudavel", "tempo_inferencia_ms",
            "tempo_http_ms", "detectado_em", "erro",
        ]
        with args.csv.open("w", newline="", encoding="utf-8") as arquivo:
            escritor = csv.DictWriter(arquivo, fieldnames=campos)
            escritor.writeheader()
            for linha in linhas:
                escritor.writerow({c: linha.get(c, "") for c in campos})
        print(f"\nCSV gravado em {args.csv}")

    if not args.json and len(imagens) > 1:
        print(f"\n{'=' * 62}")
        print(f"Analisadas: {len(imagens) - falhas}   Falhas: {falhas}   "
              f"Alertas: {alertas}")
        if tempos:
            print(f"Inferencia mediana: {statistics.median(tempos):.1f} ms")

    return 1 if falhas == len(imagens) else 0


if __name__ == "__main__":
    raise SystemExit(main())
