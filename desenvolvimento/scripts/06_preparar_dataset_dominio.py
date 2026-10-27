"""Etapa 6 — Dataset do verificador de domínio (alface / não alface).

Lê ``desenvolvimento/dataset_clean/`` (positivas do Kaggle, split mantido) e
``desenvolvimento/dataset_dominio_bruto/`` (fontes públicas e fotos da
estufa), sorteia, deduplica, divide por grupo e materializa
``desenvolvimento/dataset_dominio/``. Nenhuma pasta de origem é escrita.

Falha com código de saída 1 se restar vazamento entre splits ou conflito de
classe; nesse caso ``dataset_dominio/`` não é gerado.

Reexecute sempre que novas fotos da estufa forem adicionadas.

Uso:
    python desenvolvimento/scripts/06_preparar_dataset_dominio.py
    python desenvolvimento/scripts/06_preparar_dataset_dominio.py --por-fonte 300
    python desenvolvimento/scripts/06_preparar_dataset_dominio.py --somente-auditoria
    python desenvolvimento/scripts/06_preparar_dataset_dominio.py --data-corte nenhuma

Referência: docs/VERIFICADOR_DOMINIO.md.
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd

from ia_alface import dados_dominio as dd
from ia_alface.classes import CLASSE_ALFACE, CLASSE_NAO_ALFACE, ROTULOS_DOMINIO
from ia_alface.config import CAMINHOS, ConfigDatasetDominio
from ia_alface.dados import SPLITS_SAIDA

logger = logging.getLogger("preparar_dataset_dominio")

#: Ordem de apresentação das fontes no relatório.
_FONTES = (
    dd.FONTE_KAGGLE,
    dd.FONTE_ESTUFA,
    dd.FONTE_PLANTVILLAGE,
    dd.FONTE_PLANTDOC,
    dd.FONTE_DEEPWEEDS,
    dd.FONTE_COCO,
)


def _tabela_markdown(df: pd.DataFrame, indice_titulo: str = "") -> str:
    """Renderiza um DataFrame como tabela Markdown, incluindo o índice."""
    copia = df.copy()
    copia.index.name = indice_titulo or copia.index.name or ""
    plano = copia.reset_index()
    cabecalho = [str(c) for c in plano.columns]
    linhas = [
        "| " + " | ".join(cabecalho) + " |",
        "| " + " | ".join("---" for _ in cabecalho) + " |",
    ]
    for registro in plano.itertuples(index=False):
        linhas.append("| " + " | ".join(str(v) for v in registro) + " |")
    return "\n".join(linhas)


def _contar_por_fonte(df: pd.DataFrame) -> pd.Series:
    """Conta linhas por fonte, tolerando DataFrame vazio."""
    if df.empty or "fonte" not in df.columns:
        return pd.Series(dtype=int)
    return df["fonte"].value_counts()


def _pivo(df: pd.DataFrame, colunas: str, ordem_colunas: list[str]) -> pd.DataFrame:
    """Tabela split × ``colunas`` com totais."""
    tabela = (
        df.pivot_table(index="split", columns=colunas, values="caminho", aggfunc="count", fill_value=0)
        .reindex(index=list(SPLITS_SAIDA), columns=ordem_colunas, fill_value=0)
        .astype(int)
    )
    tabela["Total"] = tabela.sum(axis=1)
    return tabela


def _escrever_auditoria(artefatos: dict, caminho_md: Path) -> None:
    """Monta ``desenvolvimento/reports/dataset_dominio_auditoria.md``."""
    config: ConfigDatasetDominio = artefatos["config"]
    dividido: pd.DataFrame = artefatos["dividido"]
    contagens: dict = artefatos["contagens"]
    estatisticas: dict = artefatos["estatisticas"]
    ilegiveis = _contar_por_fonte(artefatos["ilegiveis"])
    descartados = _contar_por_fonte(artefatos["descartados"])
    finais = _contar_por_fonte(dividido)
    problemas: list[str] = artefatos["problemas"]

    linhas: list[str] = []
    ad = linhas.append

    ad("# Auditoria do dataset do verificador de domínio")
    ad("")
    ad("*Gerado automaticamente por `desenvolvimento/scripts/06_preparar_dataset_dominio.py` em "
       f"{datetime.now().strftime('%d/%m/%Y %H:%M')}.*")
    ad("")
    ad("| Parâmetro | Valor |")
    ad("| --- | --- |")
    ad(f"| Positivas | `{artefatos['raiz_limpo'].name}/` (split mantido) e fotos da estufa |")
    ad(f"| Fontes brutas | `{artefatos['raiz_bruto'].name}/` (somente leitura) |")
    ad(f"| Dataset gerado | `{artefatos['destino'].name}/` |")
    ad(f"| Máximo por fonte pública | {config.por_fonte} |")
    ad(f"| Proporções (fontes públicas) | {config.proporcoes[0]:.0%} / "
       f"{config.proporcoes[1]:.0%} / {config.proporcoes[2]:.0%} |")
    ad(f"| Sessões da câmera do projeto | fotos em sequência até uma pausa de mais de "
       f"{config.pausa_sessao_horas:g} h; {config.fracao_sessoes_teste:.0%} mais recentes no "
       f"teste, {config.fracao_sessoes_val:.0%} seguintes na validação — separadamente para "
       "as sessões com \"não alface\" e as só com alface |")
    corte = config.data_corte_treino
    ad(f"| Data de corte (C270) | {corte + ': sessões iniciadas até ela só no treino' if corte else 'desligada'} |")
    ad(f"| Semente | {config.seed} |")
    ad(f"| Python | {platform.python_version()} |")
    ad("")
    ad("Decisões, licenças e procedência: `docs/VERIFICADOR_DOMINIO.md`.")
    ad("")

    ad("## 1. Fontes")
    ad("")
    ad("| Fonte | Classe | Disponíveis | Sorteadas | Ilegíveis | Duplicatas removidas | No dataset |")
    ad("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for fonte in _FONTES:
        if fonte == dd.FONTE_KAGGLE:
            classe = CLASSE_ALFACE
        elif fonte == dd.FONTE_ESTUFA:
            classe = "ambas"
        else:
            classe = CLASSE_NAO_ALFACE
        c = contagens.get(fonte, {"disponiveis": 0, "sorteadas": 0})
        ad(f"| {fonte} | {classe} | {c['disponiveis']} | {c['sorteadas']} | "
           f"{int(ilegiveis.get(fonte, 0))} | {int(descartados.get(fonte, 0))} | "
           f"{int(finais.get(fonte, 0))} |")
    ad("")

    ad("## 2. Detalhes por fonte")
    ad("")
    for fonte in _FONTES:
        if fonte in estatisticas:
            itens = ", ".join(f"{k} = {v}" for k, v in estatisticas[fonte].items())
            ad(f"- **{fonte}**: {itens}")
    ad("")
    ad("PlantVillage: uma imagem por folha; fotos sem mapeamento no `leaf-map.json` "
       "ou associadas a mais de uma folha são descartadas. COCO: excluídas as imagens "
       "com comida, vaso de planta, mesa de refeição ou tigela, e as sem anotação.")
    ad("")

    ad("## 3. Distribuição final")
    ad("")
    ad(_tabela_markdown(_pivo(dividido, "classe", list(ROTULOS_DOMINIO)), "split"))
    ad("")
    ad(_tabela_markdown(_pivo(dividido, "fonte", list(_FONTES)), "split"))
    ad("")

    ad("## 4. Fotos da estufa")
    ad("")
    estufa = dividido[dividido["fonte"] == dd.FONTE_ESTUFA]
    if estufa.empty:
        ad("Nenhuma foto da estufa.")
    else:
        tabela = (
            estufa.groupby(["camera", "dia", "sessao", "classe", "split"])
            .size()
            .rename("imagens")
            .to_frame()
        )
        ad(_tabela_markdown(tabela.reset_index().set_index("camera"), "camera"))
        ad("")
        projeto = estufa[estufa["camera"] == dd.CAMERA_PROJETO]
        if not projeto.empty:
            ad("Fotos da câmera do projeto (C270) por split:")
            ad("")
            ad(_tabela_markdown(_pivo(projeto, "classe", list(ROTULOS_DOMINIO)), "split"))
            ad("")
        ad("Câmera do projeto (C270): sessões de captura — fotos em sequência até uma pausa "
           f"de mais de {config.pausa_sessao_horas:g} h, podendo atravessar a meia-noite, "
           "identificadas por `AAAAMMDD_HHMM` da primeira foto — em ordem cronológica, as "
           "mais recentes no teste, aplicada "
           "separadamente às sessões com \"não alface\" e às só com alface; cada sessão fica "
           "inteira num único split. Celular: somente treino — não é a câmera de operação.")
        if config.data_corte_treino:
            ad("")
            ad(f"Data de corte {config.data_corte_treino}: as sessões iniciadas até essa data "
               "vão inteiras para o treino e não entram na divisão cronológica.")
    ad("")

    ad("## 5. Verificação de integridade")
    ad("")
    if problemas:
        ad("**REPROVADO** — `dataset_dominio/` não foi gerado:")
        ad("")
        for problema in problemas:
            ad(f"- {problema}")
    else:
        ad("**APROVADO** — zero hashes MD5 e zero grupos compartilhados entre `train`, "
           "`val` e `test`; nenhuma imagem nas duas classes; Kaggle no split original; "
           "nenhuma foto de celular no teste; as duas classes presentes em todos os splits.")
    ad("")

    ad("## 6. Avisos")
    ad("")
    for aviso in artefatos["avisos"] or ["Nenhum aviso."]:
        ad(f"- {aviso}")
    ad("")

    ad("## 7. Limitações desta montagem")
    ad("")
    ad("- A deduplicação é por MD5: cópias **aproximadas** (mesma foto em outro tamanho, "
       "comum em imagens da web como as do PlantDoc) não são detectadas.")
    ad("- As fontes públicas negativas têm fundos e enquadramentos próprios; o "
       "desempenho no ambiente real só é evidenciado pelas fotos da câmera do projeto.")
    ad("")

    caminho_md.write_text("\n".join(linhas), encoding="utf-8")
    logger.info("Relatório de auditoria escrito em %s", caminho_md)


def _data_corte(texto: str) -> str | None:
    """Converte o argumento ``--data-corte``: ``AAAAMMDD`` ou ``nenhuma``."""
    if texto.strip().lower() in {"", "nenhuma"}:
        return None
    try:
        datetime.strptime(texto, "%Y%m%d")
    except ValueError as erro:
        raise argparse.ArgumentTypeError(
            f"data de corte inválida: {texto!r} (use AAAAMMDD ou 'nenhuma')"
        ) from erro
    return texto


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada do script."""
    padrao = ConfigDatasetDominio()
    parser = argparse.ArgumentParser(
        description="Monta o dataset do verificador de domínio sem vazamento entre splits."
    )
    parser.add_argument("--bruto", type=Path, default=CAMINHOS.dataset_dominio_bruto,
                        help="Fontes brutas (padrão: desenvolvimento/dataset_dominio_bruto/).")
    parser.add_argument("--limpo", type=Path, default=CAMINHOS.dataset_limpo,
                        help="Positivas do Kaggle (padrão: desenvolvimento/dataset_clean/).")
    parser.add_argument("--destino", type=Path, default=CAMINHOS.dataset_dominio,
                        help="Saída (padrão: desenvolvimento/dataset_dominio/).")
    parser.add_argument("--por-fonte", type=int, default=padrao.por_fonte,
                        help=f"Máximo por fonte pública negativa (padrão: {padrao.por_fonte}).")
    parser.add_argument("--seed", type=int, default=padrao.seed, help="Semente.")
    parser.add_argument("--proporcoes", type=float, nargs=3, default=list(padrao.proporcoes),
                        metavar=("TREINO", "VAL", "TESTE"),
                        help="Frações para as fontes públicas (padrão: 0.70 0.15 0.15).")
    parser.add_argument("--data-corte", type=_data_corte, default=padrao.data_corte_treino,
                        metavar="AAAAMMDD",
                        help="Sessões da câmera do projeto iniciadas até esta data vão só "
                             f"para o treino (padrão: {padrao.data_corte_treino}; "
                             "'nenhuma' desliga).")
    parser.add_argument("--somente-auditoria", action="store_true",
                        help="Gera os relatórios sem escrever desenvolvimento/dataset_dominio/.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    CAMINHOS.garantir_diretorios()

    config = ConfigDatasetDominio(
        por_fonte=args.por_fonte,
        seed=args.seed,
        proporcoes=tuple(args.proporcoes),
        data_corte_treino=args.data_corte,
    )
    artefatos = dd.preparar_dominio(
        raiz_bruto=args.bruto,
        raiz_limpo=args.limpo,
        destino=args.destino,
        config=config,
        escrever=not args.somente_auditoria,
    )

    colunas = [
        "caminho", "fonte", "classe", "estrato", "grupo", "camera", "dia", "sessao",
        "split", "hash_md5", "largura", "altura", "formato", "modo",
    ]
    dividido: pd.DataFrame = artefatos["dividido"]
    if "caminho_destino" in dividido.columns:
        colunas.append("caminho_destino")
    caminho_csv = CAMINHOS.relatorios / "dataset_dominio_inventario.csv"
    dividido[colunas].to_csv(caminho_csv, index=False, encoding="utf-8")
    logger.info("Inventário escrito em %s", caminho_csv)

    _escrever_auditoria(artefatos, CAMINHOS.relatorios / "dataset_dominio_auditoria.md")

    for aviso in artefatos["avisos"]:
        logger.warning(aviso)

    problemas: list[str] = artefatos["problemas"]
    if problemas:
        logger.error("Verificação de integridade REPROVADA:")
        for problema in problemas:
            logger.error("  - %s", problema)
        logger.error("dataset_dominio/ NÃO foi gerado. Corrija os problemas acima.")
        return 1

    resumo = dividido.pivot_table(
        index="split", columns="classe", values="caminho", aggfunc="count", fill_value=0
    ).reindex(list(SPLITS_SAIDA))
    logger.info("Integridade APROVADA — nenhum vazamento entre splits.")
    logger.info("Distribuição final:\n%s", resumo.to_string())
    if args.somente_auditoria:
        logger.info("Modo somente-auditoria: dataset_dominio/ não foi escrito.")
    else:
        logger.info("Dataset pronto em %s", args.destino)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
