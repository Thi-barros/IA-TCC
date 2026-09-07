"""Gera o contrato OpenAPI (Swagger) da API a partir do código.

Escreve dois arquivos em ``docs/``:

* ``openapi.json`` — contrato OpenAPI 3.1, para gerar clientes, importar no
  Postman/Insomnia e anexar ao TCC;
* ``swagger.html`` — Swagger UI autocontido, com o contrato embutido. Abre com
  duplo clique, sem servidor: útil para consultar e imprimir a documentação
  sem subir o serviço nem carregar o modelo.

O contrato é lido de ``api.main.app``; nada é escrito à mão. Importar o módulo
não carrega o modelo — o carregamento acontece no ciclo de vida da aplicação,
que só roda quando o servidor sobe. Rode este script após qualquer alteração em
rotas ou esquemas, para que os arquivos não divirjam do código.

Uso:
    python desenvolvimento/scripts/gerar_openapi.py
    python desenvolvimento/scripts/gerar_openapi.py --servidor http://192.168.0.10:8000

Referência: spec.md, Seção 9.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# desenvolvimento/scripts/gerar_openapi.py -> scripts -> desenvolvimento -> raiz
RAIZ_PROJETO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ_PROJETO))

from api.main import app  # noqa: E402 - depende do sys.path acima
from ia_alface.config import CAMINHOS  # noqa: E402

logger = logging.getLogger("openapi")

#: Rotas que precisam constar do contrato. Se alguma sumir, a exportação falha
#: em vez de gravar um contrato incompleto para os consumidores.
ROTAS_ESPERADAS = ("/health", "/info", "/analyze", "/analyze/raw")

#: Versão fixa do Swagger UI. Fixar evita que uma atualização de terceiros mude
#: a aparência da documentação impressa no TCC.
VERSAO_SWAGGER_UI = "5.17.14"

MODELO_HTML = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{titulo} — Swagger</title>
<link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@{versao}/swagger-ui.css">
<style>
  body {{ margin: 0; background: #fafafa; }}
  #aviso {{
    display: none;
    margin: 3rem auto;
    max-width: 44rem;
    padding: 1.5rem 1.75rem;
    border: 1px solid #d9c47a;
    border-left: 4px solid #b8860b;
    border-radius: 4px;
    background: #fdf8e6;
    color: #3b3524;
    font: 15px/1.6 "Segoe UI", system-ui, sans-serif;
  }}
  #aviso code {{ background: #efe7cb; padding: 0.1em 0.35em; border-radius: 3px; }}
</style>
</head>
<body>
<div id="aviso">
  <strong>Não foi possível carregar o Swagger UI.</strong>
  <p>Esta página baixa a interface de um CDN e precisa de conexão com a internet
  na primeira vez. Sem conexão, use a documentação servida pela própria API:</p>
  <p><code>python -m uvicorn api.main:app</code> e abra
  <code>http://localhost:8000/docs</code>.</p>
  <p>O contrato em si continua disponível em <code>docs/openapi.json</code>.</p>
</div>
<div id="swagger-ui"></div>
<script src="https://unpkg.com/swagger-ui-dist@{versao}/swagger-ui-bundle.js" crossorigin></script>
<script>
// O contrato é embutido em vez de buscado com fetch: aberto por file://, o
// navegador bloquearia a leitura de openapi.json por política de origem.
const ESPECIFICACAO = {especificacao};

window.addEventListener("load", function () {{
  if (typeof SwaggerUIBundle === "undefined") {{
    document.getElementById("aviso").style.display = "block";
    return;
  }}
  SwaggerUIBundle({{
    spec: ESPECIFICACAO,
    dom_id: "#swagger-ui",
    presets: [SwaggerUIBundle.presets.apis],
    layout: "BaseLayout",
    docExpansion: "list",
    defaultModelsExpandDepth: 2,
    displayRequestDuration: true,
    tryItOutEnabled: true
  }});
}});
</script>
</body>
</html>
"""


def montar_especificacao(servidor: str) -> dict:
    """Extrai o contrato da aplicação e acrescenta o servidor de teste.

    A aplicação não declara ``servers``: servida em ``/docs``, ela usa a própria
    origem e funciona em qualquer host. O arquivo estático, porém, é aberto do
    disco e não tem origem útil — sem um ``servers`` explícito o botão
    *Try it out* não teria para onde enviar a requisição.

    Args:
        servidor: URL base onde a API responde durante os testes.

    Returns:
        O contrato OpenAPI como dicionário.

    Raises:
        RuntimeError: Se alguma rota esperada estiver ausente do contrato.
    """
    especificacao = app.openapi()

    ausentes = [rota for rota in ROTAS_ESPERADAS if rota not in especificacao["paths"]]
    if ausentes:
        raise RuntimeError(f"Rotas ausentes no contrato: {', '.join(ausentes)}")

    corpo_bruto = especificacao["paths"]["/analyze/raw"]["post"].get("requestBody")
    if not corpo_bruto:
        raise RuntimeError(
            "/analyze/raw não declara requestBody. O contrato descreveria a "
            "rota do ESP32-CAM como se não recebesse imagem alguma."
        )

    especificacao["servers"] = [
        {"url": servidor, "description": "Serviço em execução local"}
    ]
    return especificacao


def escrever_contrato(especificacao: dict, destino: Path) -> Path:
    """Grava o contrato em JSON legível, preservando os acentos."""
    arquivo = destino / "openapi.json"
    arquivo.write_text(
        json.dumps(especificacao, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return arquivo


def escrever_pagina(especificacao: dict, destino: Path) -> Path:
    """Gera a página autocontida do Swagger UI com o contrato embutido."""
    embutido = json.dumps(especificacao, ensure_ascii=False, indent=2)
    # Uma ocorrência de "</script>" dentro da string encerraria o bloco de
    # script no analisador de HTML, quebrando a página.
    embutido = embutido.replace("</", "<\\/")

    arquivo = destino / "swagger.html"
    arquivo.write_text(
        MODELO_HTML.format(
            titulo=especificacao["info"]["title"],
            versao=VERSAO_SWAGGER_UI,
            especificacao=embutido,
        ),
        encoding="utf-8",
    )
    return arquivo


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada do script."""
    parser = argparse.ArgumentParser(
        description="Gera docs/openapi.json e docs/swagger.html a partir da API."
    )
    parser.add_argument(
        "--destino",
        type=Path,
        default=CAMINHOS.docs,
        help="Diretório de saída (padrão: docs/).",
    )
    parser.add_argument(
        "--servidor",
        default="http://localhost:8000",
        help=(
            "URL base gravada no contrato, usada pelo botão 'Try it out' da "
            "página estática."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args.destino.mkdir(parents=True, exist_ok=True)

    try:
        especificacao = montar_especificacao(args.servidor)
    except RuntimeError as erro:
        logger.error("%s", erro)
        return 1

    contrato = escrever_contrato(especificacao, args.destino)
    pagina = escrever_pagina(especificacao, args.destino)

    rotas = len(especificacao["paths"])
    esquemas = len(especificacao.get("components", {}).get("schemas", {}))
    logger.info(
        "Contrato OpenAPI %s: %d rotas, %d esquemas.",
        especificacao["openapi"],
        rotas,
        esquemas,
    )
    logger.info("Gravado %s (%.1f KB)", contrato, contrato.stat().st_size / 1024)
    logger.info("Gravado %s (%.1f KB)", pagina, pagina.stat().st_size / 1024)
    logger.info("Documentação viva: python -m uvicorn api.main:app -> /docs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
