"""Verificação da integração com o Azure Blob Storage.

Exercita o ciclo completo — enviar, listar, baixar, conferir integridade e
remover — contra o container real, isoladamente. O objetivo é validar
credencial, permissão e conectividade **antes** que esse código seja embutido
no back-end ou numa Azure Function, onde uma falha de autorização se confunde
com erro de empacotamento ou de gatilho.

Não pertence ao módulo de IA. A API de inferência recebe bytes e não conhece a
origem deles; quem integra com o Blob é o back-end da aplicação, fase seguinte
do TCC. Este script vive em ``desenvolvimento/`` justamente por isso, e
``src/ia_alface/`` permanece sem qualquer dependência de nuvem.

Autenticação, em ordem de preferência:

1. **Identidade do Entra ID** (recomendado). Rode ``az login`` uma vez. Nenhum
   segredo em disco, e a permissão é governada pelo papel
   ``Storage Blob Data Contributor`` atribuído à sua conta.
2. **Connection string**, se ``AZURE_STORAGE_CONNECTION_STRING`` estiver
   definida. Alternativa para quem não pode instalar o Azure CLI.

Uso:
    python desenvolvimento/scripts/verificar_blob.py
    python desenvolvimento/scripts/verificar_blob.py --imagem foto.jpg
    python desenvolvimento/scripts/verificar_blob.py --container outro --manter

Requer ``pip install -r desenvolvimento/requirements-dev.txt``.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# desenvolvimento/scripts/verificar_blob.py -> scripts -> desenvolvimento -> raiz
RAIZ_PROJETO: Path = Path(__file__).resolve().parents[2]

logger = logging.getLogger("verificar_blob")

#: Padrões usados quando nem o `.env` nem os argumentos definem o valor.
CONTA_PADRAO = "tccimagens"
CONTAINER_PADRAO = "imagens-alface"
DISPOSITIVO_PADRAO = "estufa-01"

try:
    from azure.core.exceptions import (
        ClientAuthenticationError,
        HttpResponseError,
        ResourceNotFoundError,
    )
    from azure.storage.blob import BlobServiceClient, ContainerClient

    _FALHA_DE_IMPORTACAO: str | None = None
except ImportError as erro:  # dependência ausente: orientar em vez de estourar
    _FALHA_DE_IMPORTACAO = str(erro)


# --------------------------------------------------------------------------- #
# Configuração
# --------------------------------------------------------------------------- #


def carregar_env(caminho: Path | None = None) -> None:
    """Carrega pares ``CHAVE=valor`` de um arquivo ``.env`` para o ambiente.

    Implementado à mão para não acrescentar dependência por dez linhas. Aceita
    comentários iniciados por ``#`` e ignora linhas em branco. Variáveis já
    definidas no ambiente têm precedência e não são sobrescritas.

    Args:
        caminho: Arquivo a ler. Por padrão, ``.env`` na raiz do projeto.
    """
    arquivo = caminho or (RAIZ_PROJETO / ".env")
    if not arquivo.exists():
        return

    for linha in arquivo.read_text(encoding="utf-8").splitlines():
        limpa = linha.strip()
        if not limpa or limpa.startswith("#") or "=" not in limpa:
            continue
        chave, _, valor = limpa.partition("=")
        os.environ.setdefault(chave.strip(), valor.strip().strip("\"'"))


def nome_de_blob(dispositivo: str, momento: datetime | None = None) -> str:
    """Monta o caminho do blob segundo a convenção do projeto.

    O formato ``{dispositivo}/{ano}/{mes}/{carimbo}.jpg`` foi escolhido para
    que o nome seja **ordenável lexicograficamente** e **listável por prefixo**:
    é possível recuperar tudo de uma estufa num mês sem varrer o container
    inteiro. O carimbo inclui microssegundos para que duas capturas no mesmo
    segundo não colidam.

    O nome também serve de chave natural na tabela ``deteccoes``. Isso importa
    porque o Event Grid entrega *pelo menos uma vez*: o mesmo blob pode ser
    processado duas vezes, e é a chave que torna a gravação idempotente.

    Args:
        dispositivo: Identificador da câmera de origem, ex. ``"estufa-01"``.
        momento: Instante da captura. Por padrão, agora em UTC.

    Returns:
        Caminho relativo do blob dentro do container.
    """
    agora = momento or datetime.now(timezone.utc)
    carimbo = agora.strftime("%Y%m%dT%H%M%S-%f")
    return f"{dispositivo}/{agora:%Y}/{agora:%m}/{carimbo}.jpg"


def imagem_sintetica() -> bytes:
    """Gera um JPEG pequeno em memória, para não depender do dataset.

    Returns:
        Bytes de um JPEG 224x224 válido.
    """
    from PIL import Image

    imagem = Image.new("RGB", (224, 224))
    imagem.putdata(
        [
            ((x * 7) % 256, (y * 5) % 256, ((x + y) * 3) % 256)
            for y in range(224)
            for x in range(224)
        ]
    )
    buffer = io.BytesIO()
    imagem.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Conexão
# --------------------------------------------------------------------------- #


def criar_cliente(conta: str) -> tuple[BlobServiceClient, str]:
    """Constrói o cliente do Blob Storage pelo melhor método disponível.

    Args:
        conta: Nome da conta de armazenamento, sem sufixo de domínio.

    Returns:
        O cliente conectado e o rótulo do método de autenticação usado.

    Raises:
        RuntimeError: Se a connection string estiver malformada.
    """
    conexao = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "").strip()
    if conexao:
        logger.warning(
            "Usando connection string: ela embute a chave da conta e concede "
            "acesso total. Prefira `az login` assim que possível."
        )
        try:
            return (
                BlobServiceClient.from_connection_string(conexao),
                "connection string",
            )
        except ValueError as erro:
            raise RuntimeError(
                f"AZURE_STORAGE_CONNECTION_STRING está malformada: {erro}"
            ) from erro

    from azure.identity import DefaultAzureCredential

    cliente = BlobServiceClient(
        account_url=f"https://{conta}.blob.core.windows.net",
        credential=DefaultAzureCredential(),
    )
    return cliente, "identidade do Entra ID (DefaultAzureCredential)"


# --------------------------------------------------------------------------- #
# Etapas da verificação
# --------------------------------------------------------------------------- #


def _conferir_container(cliente: BlobServiceClient, container: str) -> ContainerClient:
    """Confirma que o container existe e está acessível."""
    alvo = cliente.get_container_client(container)
    if not alvo.exists():
        raise RuntimeError(
            f"O container {container!r} não existe na conta. Crie-o no portal "
            "em Armazenamento de dados > Contentores, com nível de acesso "
            "Privado."
        )
    return alvo


def _enviar(alvo: ContainerClient, caminho: str, dados: bytes) -> None:
    """Grava o blob, falhando se já existir."""
    alvo.upload_blob(name=caminho, data=dados, overwrite=False)


def _listar(alvo: ContainerClient, prefixo: str) -> list[str]:
    """Lista os blobs sob um prefixo."""
    return [blob.name for blob in alvo.list_blobs(name_starts_with=prefixo)]


def _baixar(alvo: ContainerClient, caminho: str) -> bytes:
    """Recupera o conteúdo do blob."""
    return alvo.get_blob_client(caminho).download_blob().readall()


def verificar(
    conta: str, container: str, dispositivo: str, dados: bytes, manter: bool
) -> int:
    """Executa o ciclo completo e relata cada etapa.

    Args:
        conta: Nome da conta de armazenamento.
        container: Container de destino.
        dispositivo: Identificador usado no prefixo do blob.
        dados: Bytes da imagem de teste.
        manter: Se ``True``, não remove o blob ao final.

    Returns:
        ``0`` se todas as etapas passaram, ``1`` caso contrário.
    """
    cliente, metodo = criar_cliente(conta)
    logger.info("Conta ......: %s", conta)
    logger.info("Container ..: %s", container)
    logger.info("Autenticação: %s", metodo)
    logger.info("-" * 70)

    alvo = _conferir_container(cliente, container)
    logger.info("[1/5] Container acessível.")

    caminho = nome_de_blob(dispositivo)
    _enviar(alvo, caminho, dados)
    logger.info("[2/5] Enviado: %s (%d bytes)", caminho, len(dados))

    encontrados = _listar(alvo, f"{dispositivo}/")
    if caminho not in encontrados:
        logger.error("[3/5] O blob enviado não apareceu na listagem.")
        return 1
    logger.info(
        "[3/5] Listagem por prefixo %r: %d blob(s), incluindo o enviado.",
        f"{dispositivo}/",
        len(encontrados),
    )

    recuperados = _baixar(alvo, caminho)
    if recuperados != dados:
        logger.error(
            "[4/5] Integridade falhou: enviados %d bytes, recuperados %d.",
            len(dados),
            len(recuperados),
        )
        return 1
    logger.info("[4/5] Baixado e conferido: %d bytes idênticos.", len(recuperados))

    if manter:
        logger.info("[5/5] Blob mantido a pedido (--manter).")
    else:
        alvo.delete_blob(caminho)
        logger.info("[5/5] Removido.")

    logger.info("-" * 70)
    logger.info("Integração com o Blob Storage verificada com sucesso.")
    return 0


# --------------------------------------------------------------------------- #
# Diagnóstico de falhas
# --------------------------------------------------------------------------- #


def _explicar_autenticacao(erro: Exception) -> None:
    """Traduz falha de credencial em ação concreta."""
    logger.error("Falha de autenticação: %s", erro)
    logger.error("")
    logger.error("Nenhuma credencial foi encontrada. Escolha um caminho:")
    logger.error("  1. Instale o Azure CLI e rode `az login` (recomendado).")
    logger.error("  2. Defina AZURE_STORAGE_CONNECTION_STRING no arquivo .env.")


def _explicar_autorizacao(erro: HttpResponseError) -> None:
    """Traduz 403 na causa que quase sempre a produz."""
    logger.error("Acesso negado (HTTP 403): %s", getattr(erro, "reason", erro))
    logger.error("")
    logger.error(
        "Sua identidade autenticou, mas não tem permissão sobre os DADOS. "
        "Ser Owner da assinatura não basta:"
    )
    logger.error("")
    logger.error("  Plano de controle  -> Owner, Contributor  (gerencia o recurso)")
    logger.error("  Plano de dados     -> Storage Blob Data * (lê e grava blobs)")
    logger.error("")
    logger.error(
        "No portal: a conta de armazenamento > Controlo de acesso (IAM) > "
        "Adicionar atribuição de função > `Storage Blob Data Contributor` > "
        "sua conta."
    )
    logger.error("A atribuição leva alguns minutos para propagar.")


def main() -> int:
    """Ponto de entrada."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # O SDK do Azure registra cada requisição e cada cabeçalho HTTP em INFO.
    # Com o logger raiz em INFO isso sepulta o relatório das cinco etapas sob
    # dezenas de linhas de protocolo. Suba para WARNING: erros continuam
    # visíveis, o tráfego normal não.
    logging.getLogger("azure").setLevel(logging.WARNING)

    if _FALHA_DE_IMPORTACAO is not None:
        logger.error("SDK do Azure ausente: %s", _FALHA_DE_IMPORTACAO)
        logger.error("Instale com:")
        logger.error("    pip install -r desenvolvimento/requirements-dev.txt")
        return 1

    carregar_env()

    analisador = argparse.ArgumentParser(
        description="Verifica a integração com o Azure Blob Storage."
    )
    analisador.add_argument(
        "--conta",
        default=os.environ.get("AZURE_STORAGE_ACCOUNT", CONTA_PADRAO),
        help="Nome da conta de armazenamento.",
    )
    analisador.add_argument(
        "--container",
        default=os.environ.get("AZURE_STORAGE_CONTAINER", CONTAINER_PADRAO),
        help="Container de destino.",
    )
    analisador.add_argument(
        "--dispositivo",
        default=os.environ.get("AZURE_DISPOSITIVO", DISPOSITIVO_PADRAO),
        help="Identificador da câmera, usado como prefixo do blob.",
    )
    analisador.add_argument(
        "--imagem",
        type=Path,
        help="Imagem a enviar. Sem isto, um JPEG sintético é gerado em memória.",
    )
    analisador.add_argument(
        "--manter",
        action="store_true",
        help="Não remover o blob ao final, para inspecioná-lo no portal.",
    )
    args = analisador.parse_args()

    if args.imagem is not None:
        if not args.imagem.exists():
            logger.error("Imagem não encontrada: %s", args.imagem)
            return 1
        dados = args.imagem.read_bytes()
    else:
        dados = imagem_sintetica()

    try:
        return verificar(
            args.conta, args.container, args.dispositivo, dados, args.manter
        )
    except ClientAuthenticationError as erro:
        _explicar_autenticacao(erro)
        return 1
    except ResourceNotFoundError as erro:
        logger.error("Recurso não encontrado: %s", erro)
        logger.error(
            "Confira se a conta %r e o container existem, e se o nome está "
            "escrito corretamente.",
            args.conta,
        )
        return 1
    except HttpResponseError as erro:
        if erro.status_code == 403:
            _explicar_autorizacao(erro)
        else:
            logger.error("Erro HTTP %s: %s", erro.status_code, erro)
        return 1
    except RuntimeError as erro:
        logger.error("%s", erro)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
