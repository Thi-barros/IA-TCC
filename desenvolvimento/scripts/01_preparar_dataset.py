"""Etapa 1 — Preparação do dataset: auditoria, deduplicação e re-divisão.

Lê ``desenvolvimento/dataset/`` (fonte imutável), remove duplicatas exatas, agrupa as variantes
de cada folha de origem e re-divide o conjunto em 70/15/15 de forma
estratificada por classe e **disjunta por grupo**, materializando o resultado em
``desenvolvimento/dataset_clean/``.

Falha com código de saída 1 se restar qualquer vazamento entre splits.

Uso:
    python desenvolvimento/scripts/01_preparar_dataset.py
    python desenvolvimento/scripts/01_preparar_dataset.py --seed 7 --proporcoes 0.7 0.15 0.15
    python desenvolvimento/scripts/01_preparar_dataset.py --somente-auditoria

Referência: spec.md, Seção 4.3.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ia_alface import dados
from ia_alface.classes import CLASSES, ROTULOS_CANONICOS
from ia_alface.config import CAMINHOS, descrever_ambiente

logger = logging.getLogger("preparar_dataset")


def _tabela_markdown(df: pd.DataFrame, indice_titulo: str = "") -> str:
    """Renderiza um DataFrame como tabela Markdown, incluindo o índice.

    Implementado à mão para não exigir a dependência ``tabulate``, que não
    consta da lista de dependências do spec.
    """
    copia = df.copy()
    copia.index.name = indice_titulo or copia.index.name or ""
    plano = copia.reset_index()

    def formatar(valor: object) -> str:
        if isinstance(valor, float):
            return f"{valor:.1f}" if valor % 1 else f"{valor:.0f}"
        return str(valor)

    cabecalho = [str(c) for c in plano.columns]
    linhas = [
        "| " + " | ".join(cabecalho) + " |",
        "| " + " | ".join("---" for _ in cabecalho) + " |",
    ]
    for registro in plano.itertuples(index=False):
        linhas.append("| " + " | ".join(formatar(v) for v in registro) + " |")
    return "\n".join(linhas)


def _grafico_resolucoes(inventario: pd.DataFrame, saida: Path) -> None:
    """Gera o histograma de resoluções citado no relatório de auditoria."""
    figura, eixos = plt.subplots(1, 2, figsize=(12, 4.5))

    eixos[0].scatter(
        inventario["largura"], inventario["altura"], s=6, alpha=0.25, color="#2f7d32"
    )
    limite = max(inventario["largura"].max(), inventario["altura"].max())
    eixos[0].plot([0, limite], [0, limite], "--", color="#999", linewidth=1)
    eixos[0].set_xlabel("Largura (px)")
    eixos[0].set_ylabel("Altura (px)")
    eixos[0].set_title("Dispersão de resoluções\n(linha tracejada = imagens quadradas)")

    maior_lado = inventario[["largura", "altura"]].max(axis=1)
    eixos[1].hist(maior_lado, bins=40, color="#2f7d32", edgecolor="white")
    eixos[1].axvline(
        224, color="#c62828", linestyle="--", label="imgsz de treino = 224 px"
    )
    eixos[1].set_xlabel("Maior lado da imagem (px)")
    eixos[1].set_ylabel("Nº de imagens")
    eixos[1].set_title("Distribuição do maior lado")
    eixos[1].legend()

    figura.suptitle(
        "Auditoria de resoluções do dataset de folhas de alface", fontsize=12
    )
    figura.tight_layout()
    figura.savefig(saida, dpi=150)
    plt.close(figura)


def _escrever_auditoria(artefatos: dict, caminho_md: Path, caminho_grafico: Path) -> None:
    """Monta ``desenvolvimento/reports/dataset_auditoria.md``.

    O conteúdo é material direto para as Seções 6.1.1 e 8 do TCC.
    """
    bruto: pd.DataFrame = artefatos["inventario_bruto"]
    cruzadas: pd.DataFrame = artefatos["duplicatas_cruzadas"]
    descartados: pd.DataFrame = artefatos["descartados"]
    dividido: pd.DataFrame = artefatos["dividido"]
    problemas: list[str] = artefatos["problemas"]

    resolucao = dados.estatisticas_resolucao(bruto)
    ambiente = descrever_ambiente()

    aumentadas = bruto[bruto["arquivo"].str.lower().str.startswith("augmented_")]
    pivot_aumentadas = (
        aumentadas.pivot_table(
            index="split_original",
            columns="classe",
            values="caminho",
            aggfunc="count",
            fill_value=0,
        )
        if not aumentadas.empty
        else pd.DataFrame()
    )

    grupos_antes = (
        bruto.groupby("classe")["grupo_id"].nunique().reindex(list(ROTULOS_CANONICOS))
    )
    grupos_depois = (
        dividido.groupby("classe")["grupo_id"].nunique().reindex(list(ROTULOS_CANONICOS))
    )
    tabela_grupos = pd.DataFrame(
        {
            "imagens (original)": bruto.groupby("classe").size().reindex(list(ROTULOS_CANONICOS)),
            "grupos (por nome)": grupos_antes,
            "imagens (após dedup)": dividido.groupby("classe").size().reindex(list(ROTULOS_CANONICOS)),
            "grupos (após fusão por conteúdo)": grupos_depois,
        }
    )
    tabela_grupos["imagens por grupo"] = (
        tabela_grupos["imagens (após dedup)"]
        / tabela_grupos["grupos (após fusão por conteúdo)"]
    ).round(2)

    linhas: list[str] = []
    ad = linhas.append

    ad("# Auditoria do dataset de folhas de alface")
    ad("")
    ad(f"*Gerado automaticamente por `desenvolvimento/scripts/01_preparar_dataset.py` em "
       f"{datetime.now().strftime('%d/%m/%Y %H:%M')}.*")
    ad("")
    ad("| Parâmetro | Valor |")
    ad("| --- | --- |")
    ad(f"| Dataset de origem | `{artefatos['raiz_dataset'].name}/` (imutável) |")
    ad(f"| Dataset gerado | `{artefatos['destino'].name}/` |")
    ad(f"| Proporções | {artefatos['proporcoes'][0]:.0%} treino / "
       f"{artefatos['proporcoes'][1]:.0%} validação / {artefatos['proporcoes'][2]:.0%} teste |")
    ad(f"| Semente | {artefatos['seed']} |")
    ad(f"| Python / Ultralytics / torch | {ambiente['python']} / "
       f"{ambiente['ultralytics']} / {ambiente['torch']} |")
    ad("")
    ad("Este relatório fundamenta as Seções 6.1.1 (Coleta de dados) e 8 "
       "(Limitações do trabalho) do TCC.")
    ad("")

    ad("## 1. Inventário do dataset original")
    ad("")
    ad(f"- Imagens catalogadas: **{len(bruto)}**")
    ad(f"- Imagens ilegíveis (ignoradas): **{len(bruto.attrs.get('ilegiveis', []))}**")
    ad(f"- Formatos: {resolucao['formatos']}")
    ad(f"- Modos de cor: {resolucao['modos_de_cor']} "
       "(todos convertidos para RGB na saída)")
    ad(f"- Largura: mín. {resolucao['largura_min']} px, mediana "
       f"{resolucao['largura_mediana']} px, máx. {resolucao['largura_max']} px")
    ad(f"- Altura: mín. {resolucao['altura_min']} px, mediana "
       f"{resolucao['altura_mediana']} px, máx. {resolucao['altura_max']} px")
    ad(f"- Imagens quadradas: {resolucao['quadradas']} de {len(bruto)}")
    ad(f"- Tamanhos mais frequentes: "
       + ", ".join(f"{w}×{h} ({n})" for (w, h), n in resolucao["mais_frequentes"]))
    ad("")
    ad(f"![Histograma de resoluções]({caminho_grafico.name})")
    ad("")

    ad("## 2. Distribuição original por split e classe")
    ad("")
    ad(_tabela_markdown(dados.resumo_por_split(bruto, "split_original"), "split original"))
    ad("")
    ad("> **Problema P5** — a divisão original corresponde a aproximadamente "
       "84/8/8, e não aos 70/15/15 especificados na Seção 6.1.1 do TCC.")
    ad("")

    ad("## 3. Problema P1 — vazamento por duplicatas exatas")
    ad("")
    n_hashes = bruto["hash_md5"].nunique()
    ad(f"- Arquivos: **{len(bruto)}**")
    ad(f"- Hashes MD5 distintos: **{n_hashes}**")
    ad(f"- Arquivos redundantes (cópias byte-idênticas): **{len(bruto) - n_hashes}**")
    ad(f"- Grupos de duplicatas que **cruzam a fronteira dos splits**: "
       f"**{len(cruzadas)}**")
    ad("")
    ad("Métricas medidas sobre o split `test` original seriam cientificamente "
       "inválidas: o modelo estaria sendo avaliado em imagens vistas durante o "
       "treino, invalidando a verificação de RNF02 e da hipótese H1.")
    ad("")
    if len(cruzadas):
        ad("### 3.1 Lista completa dos grupos de duplicatas cruzadas")
        ad("")
        ad("| # | Classe | Splits afetados | Arquivos |")
        ad("| ---: | --- | --- | --- |")
        for numero, registro in enumerate(cruzadas.itertuples(), start=1):
            ad(f"| {numero} | {registro.classe} | {registro.splits} | "
               f"`{registro.arquivos}` |")
        ad("")

    ad("## 4. Problemas P2 e P3 — vazamento de grupo e diversidade real")
    ad("")
    ad("As variantes aumentadas de uma mesma folha seguem o padrão "
       "`<origem>_<n>` (por exemplo `ba12_0`, `ba12_2`, `ba12_3`). Agrupá-las "
       "impede que a mesma folha apareça em splits diferentes. Adicionalmente, "
       "grupos que compartilham arquivos idênticos foram fundidos: o dataset "
       "contém nomes de origem distintos apontando para a mesma folha.")
    ad("")
    ad(_tabela_markdown(tabela_grupos, "classe"))
    ad("")
    ad("> **Problema P3** — a classe `bacteriano` deriva de um número muito "
       "pequeno de folhas originais. Esse é um teto real de generalização e não "
       "pode ser corrigido por augmentation; deve constar da Seção 8 do TCC.")
    ad("")

    ad("## 5. Problema P4 — participação de dados sintéticos")
    ad("")
    ad(f"- Arquivos nomeados `augmented_*`: **{len(aumentadas)}** de {len(bruto)}")
    if not pivot_aumentadas.empty:
        ad("")
        ad(_tabela_markdown(pivot_aumentadas, "split original"))
    ad("")
    ad("Nos splits `valid` e `test` originais, esses arquivos representavam a "
       "maior parte das imagens, o que tornava a avaliação dependente de dados "
       "sintéticos de procedência não rastreável. Cada um foi tratado como "
       "grupo unitário na re-divisão.")
    ad("")

    ad("## 6. Deduplicação")
    ad("")
    ad(f"- Imagens descartadas por duplicidade exata: **{len(descartados)}**")
    ad(f"- Imagens remanescentes: **{len(dividido)}**")
    ad("- Critério de retenção: menor caminho em ordem lexicográfica "
       "(determinístico entre execuções).")
    ad("")

    ad("## 7. Nova divisão estratificada e disjunta por grupo")
    ad("")
    ad(_tabela_markdown(dados.resumo_por_split(dividido), "split"))
    ad("")
    percentuais = (
        dividido.pivot_table(
            index="split", columns="classe", values="caminho", aggfunc="count", fill_value=0
        )
        .reindex(list(dados.SPLITS_SAIDA))
        .pipe(lambda d: (d.T / d.sum(axis=1)).T * 100)
        .round(1)
    )
    ad("Composição percentual de classes dentro de cada split (verificação da "
       "estratificação):")
    ad("")
    ad(_tabela_markdown(percentuais, "split"))
    ad("")

    ad("## 8. Verificação de integridade")
    ad("")
    if problemas:
        ad("**REPROVADO** — foram encontrados os seguintes problemas:")
        ad("")
        for problema in problemas:
            ad(f"- {problema}")
    else:
        ad("**APROVADO** — zero hashes MD5 e zero grupos de origem "
           "compartilhados entre `train`, `val` e `test`. As três classes estão "
           "presentes em todos os splits.")
    ad("")

    ad("## 9. Mapeamento de classes aplicado")
    ad("")
    ad("| Pasta original | Índice | Rótulo canônico | Exibição |")
    ad("| --- | :---: | --- | --- |")
    for classe in CLASSES:
        ad(f"| `{classe.pasta_original}` | {classe.indice} | `{classe.canonico}` | "
           f"{classe.exibicao} |")
    ad("")

    ad("## 10. Observações para a redação do TCC")
    ad("")
    ad("1. A Seção 6.1.1 descreve *data augmentation* aplicada na preparação do "
       "dataset. Nesta implementação, o aumento de dados é feito **em tempo de "
       "treino** pelo Ultralytics, sem gravar imagens em disco. Gravar variantes "
       "em disco foi justamente a causa do vazamento identificado nos problemas "
       "P1 e P2.")
    ad("2. A Seção 8 deve registrar a baixa diversidade de folhas originais na "
       "classe `bacteriano` (ver item 4 acima) como limitação de generalização.")
    ad("3. A divisão 70/15/15 exigida pela Seção 6.1.1 foi aplicada sobre os "
       "**grupos de folhas**, não sobre arquivos isolados — o que é o "
       "procedimento correto quando o conjunto contém variantes da mesma amostra.")
    ad("")

    caminho_md.write_text("\n".join(linhas), encoding="utf-8")
    logger.info("Relatório de auditoria escrito em %s", caminho_md)


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada do script."""
    parser = argparse.ArgumentParser(
        description=(
            "Audita, deduplica e re-divide o dataset de folhas de alface em "
            "70/15/15 sem vazamento entre splits."
        )
    )
    parser.add_argument(
        "--origem",
        type=Path,
        default=CAMINHOS.dataset_original,
        help="Diretório do dataset original (padrão: desenvolvimento/dataset/).",
    )
    parser.add_argument(
        "--destino",
        type=Path,
        default=CAMINHOS.dataset_limpo,
        help="Diretório de saída (padrão: desenvolvimento/dataset_clean/).",
    )
    parser.add_argument(
        "--proporcoes",
        type=float,
        nargs=3,
        default=list(dados.PROPORCOES_PADRAO),
        metavar=("TREINO", "VAL", "TESTE"),
        help="Frações da divisão. Devem somar 1,0 (padrão: 0.70 0.15 0.15).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Semente de reprodutibilidade."
    )
    parser.add_argument(
        "--somente-auditoria",
        action="store_true",
        help="Executa a análise e gera os relatórios sem escrever desenvolvimento/dataset_clean/.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    CAMINHOS.garantir_diretorios()

    logger.info("Iniciando preparação do dataset a partir de %s", args.origem)
    artefatos = dados.preparar(
        raiz_dataset=args.origem,
        destino=args.destino,
        proporcoes=tuple(args.proporcoes),
        seed=args.seed,
        escrever=not args.somente_auditoria,
    )

    caminho_csv = CAMINHOS.relatorios / "dataset_inventario.csv"
    colunas = [
        "caminho",
        "arquivo",
        "split_original",
        "split",
        "classe",
        "indice_classe",
        "grupo_origem",
        "grupo_id",
        "hash_md5",
        "largura",
        "altura",
        "formato",
        "modo",
        "bytes",
    ]
    if "caminho_destino" in artefatos["dividido"].columns:
        colunas.append("caminho_destino")
    artefatos["dividido"][colunas].to_csv(caminho_csv, index=False, encoding="utf-8")
    logger.info("Inventário escrito em %s", caminho_csv)

    caminho_grafico = CAMINHOS.relatorios / "dataset_resolucoes.png"
    _grafico_resolucoes(artefatos["inventario_bruto"], caminho_grafico)

    _escrever_auditoria(
        artefatos,
        CAMINHOS.relatorios / "dataset_auditoria.md",
        caminho_grafico,
    )

    problemas: list[str] = artefatos["problemas"]
    if problemas:
        logger.error("Verificação de integridade REPROVADA:")
        for problema in problemas:
            logger.error("  - %s", problema)
        logger.error(
            "desenvolvimento/dataset_clean/ NÃO foi gerado. Corrija os problemas acima antes de treinar."
        )
        return 1

    resumo = dados.resumo_por_split(artefatos["dividido"])
    logger.info("Integridade APROVADA — nenhum vazamento entre splits.")
    logger.info("Distribuição final:\n%s", resumo.to_string())
    if args.somente_auditoria:
        logger.info("Modo somente-auditoria: desenvolvimento/dataset_clean/ não foi escrito.")
    else:
        logger.info("Dataset pronto em %s", args.destino)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
