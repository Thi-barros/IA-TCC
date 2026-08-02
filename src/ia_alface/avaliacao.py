"""Avaliação do modelo: métricas, matriz de confusão, latência e relatórios.

A avaliação é sempre feita sobre o split ``test`` de ``desenvolvimento/dataset_clean/`` — nunca
sobre ``desenvolvimento/dataset/test``, que contém imagens vazadas do treino (spec.md, Seção 7).

Regra metodológica: **nenhum hiperparâmetro deve ser ajustado observando o split
``test``**. Todo ajuste usa o split ``val``; o ``test`` é medido uma única vez,
ao final.

Referência: spec.md, Seção 7.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .classes import ROTULOS_CANONICOS, rotulo_exibicao
from .inferencia import DetectorDoencasAlface

logger = logging.getLogger(__name__)

__all__ = [
    "ResultadoAvaliacao",
    "listar_imagens",
    "avaliar",
    "calcular_metricas",
    "matriz_confusao",
    "medir_latencia",
    "salvar_matriz_confusao",
    "salvar_curva_treino",
    "salvar_erros_classificacao",
    "gerar_relatorio_markdown",
]

#: Confiança mínima exigida pelos casos de teste CT01 e CT02.
LIMIAR_CT: float = 0.85

#: Metas numéricas do TCC.
META_ACURACIA: float = 0.90
META_RECALL_SAUDAVEL: float = 0.90
META_LATENCIA_MS: float = 100.0


@dataclass
class ResultadoAvaliacao:
    """Predições brutas de uma passagem completa sobre um split.

    Attributes:
        caminhos: Caminhos das imagens avaliadas.
        y_verdadeiro: Rótulos canônicos verdadeiros.
        y_predito: Rótulos canônicos preditos.
        confiancas: Confiança da classe predita.
        prob_classe_verdadeira: Probabilidade que o modelo atribuiu à classe
            correta — base para ranquear os piores erros.
        tempos_ms: Tempo de inferência de cada imagem.
    """

    caminhos: list[str]
    y_verdadeiro: list[str]
    y_predito: list[str]
    confiancas: list[float]
    prob_classe_verdadeira: list[float]
    tempos_ms: list[float]

    def para_dataframe(self) -> pd.DataFrame:
        """Converte as predições em um DataFrame."""
        return pd.DataFrame(
            {
                "caminho": self.caminhos,
                "verdadeiro": self.y_verdadeiro,
                "predito": self.y_predito,
                "confianca": self.confiancas,
                "prob_classe_verdadeira": self.prob_classe_verdadeira,
                "tempo_ms": self.tempos_ms,
                "acertou": [
                    v == p for v, p in zip(self.y_verdadeiro, self.y_predito)
                ],
            }
        )


def listar_imagens(raiz_split: Path) -> list[tuple[Path, str]]:
    """Lista as imagens de um split, com o rótulo verdadeiro de cada uma.

    Args:
        raiz_split: Diretório do split (por exemplo ``desenvolvimento/dataset_clean/test``),
            contendo um subdiretório por classe.

    Returns:
        Lista de pares ``(caminho, classe_canonica)``, em ordem determinística.

    Raises:
        FileNotFoundError: Se o diretório não existir.
        RuntimeError: Se nenhuma imagem for encontrada.
    """
    raiz_split = Path(raiz_split)
    if not raiz_split.is_dir():
        raise FileNotFoundError(
            f"Split não encontrado: {raiz_split}. "
            "Execute 'python desenvolvimento/scripts/01_preparar_dataset.py' primeiro."
        )

    itens: list[tuple[Path, str]] = []
    for classe in ROTULOS_CANONICOS:
        pasta = raiz_split / classe
        if not pasta.is_dir():
            logger.warning("Classe ausente no split: %s", pasta)
            continue
        for arquivo in sorted(pasta.iterdir()):
            if arquivo.is_file() and arquivo.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                itens.append((arquivo, classe))

    if not itens:
        raise RuntimeError(f"Nenhuma imagem encontrada em {raiz_split}.")
    return itens


def avaliar(
    detector: DetectorDoencasAlface,
    raiz_split: Path,
    registrar_a_cada: int = 50,
) -> ResultadoAvaliacao:
    """Executa a inferência sobre todas as imagens de um split.

    Usa :meth:`DetectorDoencasAlface.predizer` (uma imagem por chamada), que é o
    caminho real de produção — os tempos coletados aqui são comparáveis ao
    requisito RNF01.

    Args:
        detector: Detector já carregado.
        raiz_split: Diretório do split a avaliar.
        registrar_a_cada: Frequência das mensagens de progresso.

    Returns:
        O :class:`ResultadoAvaliacao` com as predições brutas.
    """
    itens = listar_imagens(raiz_split)
    logger.info("Avaliando %d imagens de %s", len(itens), raiz_split)

    caminhos: list[str] = []
    y_verdadeiro: list[str] = []
    y_predito: list[str] = []
    confiancas: list[float] = []
    prob_verdadeira: list[float] = []
    tempos: list[float] = []

    for indice, (caminho, classe_real) in enumerate(itens, start=1):
        resultado = detector.predizer(caminho)
        caminhos.append(str(caminho))
        y_verdadeiro.append(classe_real)
        y_predito.append(resultado.classe)
        confiancas.append(resultado.confianca)
        prob_verdadeira.append(resultado.probabilidades.get(classe_real, 0.0))
        tempos.append(resultado.tempo_inferencia_ms)

        if indice % registrar_a_cada == 0:
            logger.info("  %d/%d imagens processadas", indice, len(itens))

    return ResultadoAvaliacao(
        caminhos=caminhos,
        y_verdadeiro=y_verdadeiro,
        y_predito=y_predito,
        confiancas=confiancas,
        prob_classe_verdadeira=prob_verdadeira,
        tempos_ms=tempos,
    )


def calcular_metricas(resultado: ResultadoAvaliacao) -> dict:
    """Calcula acurácia global e métricas por classe.

    Também calcula, por classe, a confiança média das predições corretas e a
    fração dessas predições que atinge o limiar de 0,85 — os números exigidos
    pelos casos de teste CT01 e CT02.

    Args:
        resultado: Saída de :func:`avaliar`.

    Returns:
        Dicionário serializável em JSON com as métricas.
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    classes = list(ROTULOS_CANONICOS)
    y_true = resultado.y_verdadeiro
    y_pred = resultado.y_predito

    acuracia = float(accuracy_score(y_true, y_pred))
    precisao, recall, f1, suporte = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )

    df = resultado.para_dataframe()

    por_classe: dict[str, dict] = {}
    for indice, classe in enumerate(classes):
        corretas = df[(df["verdadeiro"] == classe) & (df["acertou"])]
        por_classe[classe] = {
            "exibicao": rotulo_exibicao(classe),
            "precisao": round(float(precisao[indice]), 4),
            "recall": round(float(recall[indice]), 4),
            "f1_score": round(float(f1[indice]), 4),
            "suporte": int(suporte[indice]),
            "confianca_media_acertos": (
                round(float(corretas["confianca"].mean()), 4)
                if len(corretas)
                else None
            ),
            "fracao_acertos_acima_limiar": (
                round(float((corretas["confianca"] >= LIMIAR_CT).mean()), 4)
                if len(corretas)
                else None
            ),
        }

    tempos = np.asarray(resultado.tempos_ms, dtype=float)
    macro_f1 = float(np.mean(f1))

    return {
        "n_imagens": len(y_true),
        "acuracia_global": round(acuracia, 4),
        "f1_macro": round(macro_f1, 4),
        "por_classe": por_classe,
        "latencia_ms": {
            "mediana": round(float(np.median(tempos)), 2),
            "media": round(float(np.mean(tempos)), 2),
            "p95": round(float(np.percentile(tempos, 95)), 2),
            "minimo": round(float(tempos.min()), 2),
            "maximo": round(float(tempos.max()), 2),
        },
        "limiar_ct": LIMIAR_CT,
        "metas": {
            "acuracia": META_ACURACIA,
            "recall_saudavel": META_RECALL_SAUDAVEL,
            "latencia_ms": META_LATENCIA_MS,
        },
    }


def matriz_confusao(resultado: ResultadoAvaliacao) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monta a matriz de confusão 3×3.

    Args:
        resultado: Saída de :func:`avaliar`.

    Returns:
        Tupla ``(absoluta, normalizada_por_linha)``. Linhas são a classe
        verdadeira, colunas a predita.
    """
    from sklearn.metrics import confusion_matrix

    classes = list(ROTULOS_CANONICOS)
    matriz = confusion_matrix(
        resultado.y_verdadeiro, resultado.y_predito, labels=classes
    )
    absoluta = pd.DataFrame(matriz, index=classes, columns=classes)
    absoluta.index.name = "verdadeiro"
    absoluta.columns.name = "predito"

    somas = absoluta.sum(axis=1).replace(0, np.nan)
    normalizada = (absoluta.T / somas).T.fillna(0.0).round(4)
    return absoluta, normalizada


def medir_latencia(
    detector: DetectorDoencasAlface,
    imagens: Sequence[Path],
    aquecimento: int = 10,
) -> dict:
    """Mede a latência de :meth:`DetectorDoencasAlface.predizer`.

    Args:
        detector: Detector carregado.
        imagens: Imagens a usar na medição.
        aquecimento: Número de inferências descartadas antes da medição, para
            estabilizar caches e alocações do PyTorch.

    Returns:
        Dicionário com ``n``, ``mediana``, ``media``, ``p95``, ``p99``,
        ``minimo``, ``maximo`` (em ms) e ``dispositivo``.

    Raises:
        ValueError: Se a lista de imagens estiver vazia.
    """
    if not imagens:
        raise ValueError("Nenhuma imagem fornecida para a medição de latência.")

    for indice in range(min(aquecimento, len(imagens))):
        detector.predizer(imagens[indice % len(imagens)])

    tempos: list[float] = []
    for caminho in imagens:
        tempos.append(detector.predizer(caminho).tempo_inferencia_ms)

    vetor = np.asarray(tempos, dtype=float)
    return {
        "n": len(vetor),
        "aquecimento": min(aquecimento, len(imagens)),
        "dispositivo": detector.dispositivo,
        "mediana": round(float(np.median(vetor)), 2),
        "media": round(float(np.mean(vetor)), 2),
        "p95": round(float(np.percentile(vetor, 95)), 2),
        "p99": round(float(np.percentile(vetor, 99)), 2),
        "minimo": round(float(vetor.min()), 2),
        "maximo": round(float(vetor.max()), 2),
    }


def salvar_matriz_confusao(
    absoluta: pd.DataFrame,
    normalizada: pd.DataFrame,
    caminho_png: Path,
    caminho_csv: Path,
) -> None:
    """Grava a matriz de confusão em CSV e como figura.

    Args:
        absoluta: Matriz com contagens.
        normalizada: Matriz normalizada por linha.
        caminho_png: Arquivo de imagem de saída.
        caminho_csv: Arquivo CSV de saída (contém as duas matrizes).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with caminho_csv.open("w", encoding="utf-8", newline="") as arquivo:
        arquivo.write("# Matriz de confusão — valores absolutos\n")
        absoluta.to_csv(arquivo)
        arquivo.write("\n# Matriz de confusão — normalizada por linha\n")
        normalizada.to_csv(arquivo)

    figura, eixos = plt.subplots(1, 2, figsize=(12, 5))
    for eixo, matriz, titulo, formato in (
        (eixos[0], absoluta, "Valores absolutos", "{:.0f}"),
        (eixos[1], normalizada, "Normalizada por linha", "{:.2f}"),
    ):
        imagem = eixo.imshow(matriz.to_numpy(), cmap="Greens")
        eixo.set_xticks(range(len(matriz.columns)), matriz.columns, rotation=20, ha="right")
        eixo.set_yticks(range(len(matriz.index)), matriz.index)
        eixo.set_xlabel("Classe predita")
        eixo.set_ylabel("Classe verdadeira")
        eixo.set_title(titulo)
        limite = matriz.to_numpy().max() / 2 if matriz.to_numpy().max() else 0.5
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                valor = matriz.iat[i, j]
                eixo.text(
                    j,
                    i,
                    formato.format(valor),
                    ha="center",
                    va="center",
                    color="white" if valor > limite else "black",
                    fontsize=11,
                )
        figura.colorbar(imagem, ax=eixo, fraction=0.046)

    figura.suptitle("Matriz de confusão — split de teste (dataset_clean)", fontsize=13)
    figura.tight_layout()
    figura.savefig(caminho_png, dpi=150)
    plt.close(figura)
    logger.info("Matriz de confusão salva em %s", caminho_png)


def salvar_curva_treino(caminho_results_csv: Path, caminho_png: Path) -> bool:
    """Plota a evolução de perda e acurácia por época.

    Args:
        caminho_results_csv: ``results.csv`` produzido pelo Ultralytics na pasta
            da execução de treino.
        caminho_png: Figura de saída.

    Returns:
        ``True`` se a figura foi gerada; ``False`` se o CSV não existe.
    """
    caminho_results_csv = Path(caminho_results_csv)
    if not caminho_results_csv.exists():
        logger.warning(
            "results.csv não encontrado em %s; a curva de treino não será gerada.",
            caminho_results_csv,
        )
        return False

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    historico = pd.read_csv(caminho_results_csv)
    historico.columns = [c.strip() for c in historico.columns]

    def coluna(*candidatas: str) -> pd.Series | None:
        for nome in candidatas:
            if nome in historico.columns:
                return historico[nome]
        return None

    epocas = coluna("epoch")
    perda_treino = coluna("train/loss")
    perda_val = coluna("val/loss")
    acuracia = coluna("metrics/accuracy_top1")

    figura, eixos = plt.subplots(1, 2, figsize=(12, 4.5))

    if perda_treino is not None:
        eixos[0].plot(epocas, perda_treino, label="treino", color="#2f7d32")
    if perda_val is not None:
        eixos[0].plot(epocas, perda_val, label="validação", color="#c62828")
    eixos[0].set_xlabel("Época")
    eixos[0].set_ylabel("Perda")
    eixos[0].set_title("Perda por época")
    eixos[0].legend()
    eixos[0].grid(alpha=0.3)

    if acuracia is not None:
        eixos[1].plot(epocas, acuracia, color="#2f7d32", label="acurácia top-1 (val)")
        eixos[1].axhline(
            META_ACURACIA, color="#c62828", linestyle="--", label="meta RNF02 = 90%"
        )
        eixos[1].set_ylim(0, 1.02)
    eixos[1].set_xlabel("Época")
    eixos[1].set_ylabel("Acurácia")
    eixos[1].set_title("Acurácia de validação por época")
    eixos[1].legend()
    eixos[1].grid(alpha=0.3)

    figura.suptitle("Curvas de treinamento — YOLO11-cls / alface", fontsize=13)
    figura.tight_layout()
    figura.savefig(caminho_png, dpi=150)
    plt.close(figura)
    logger.info("Curva de treino salva em %s", caminho_png)
    return True


def salvar_erros_classificacao(
    resultado: ResultadoAvaliacao,
    destino: Path,
    quantidade: int = 20,
) -> pd.DataFrame:
    """Copia as imagens com maior erro de confiança para inspeção qualitativa.

    O erro é definido como ``1 - probabilidade atribuída à classe verdadeira``,
    de modo que o ranking inclui tanto erros grosseiros quanto acertos frágeis.

    Args:
        resultado: Saída de :func:`avaliar`.
        destino: Diretório de saída (é recriado a cada execução).
        quantidade: Número de imagens a exportar.

    Returns:
        DataFrame com as imagens exportadas e suas métricas.
    """
    destino = Path(destino)
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True, exist_ok=True)

    df = resultado.para_dataframe()
    df["erro"] = 1.0 - df["prob_classe_verdadeira"]
    piores = df.sort_values(["erro", "caminho"], ascending=[False, True]).head(quantidade)

    linhas = []
    for posicao, registro in enumerate(piores.itertuples(), start=1):
        origem = Path(registro.caminho)
        nome = (
            f"{posicao:02d}_erro-{registro.erro:.3f}"
            f"_verdadeiro-{registro.verdadeiro}"
            f"_predito-{registro.predito}"
            f"_conf-{registro.confianca:.3f}{origem.suffix}"
        )
        shutil.copy2(origem, destino / nome)
        linhas.append(
            {
                "posicao": posicao,
                "arquivo_exportado": nome,
                "origem": str(origem),
                "verdadeiro": registro.verdadeiro,
                "predito": registro.predito,
                "confianca": registro.confianca,
                "prob_classe_verdadeira": registro.prob_classe_verdadeira,
                "erro": round(registro.erro, 4),
            }
        )

    tabela = pd.DataFrame(linhas)
    tabela.to_csv(destino / "_indice.csv", index=False, encoding="utf-8")
    logger.info("%d piores casos exportados para %s", len(tabela), destino)
    return tabela


def _situacao(atingiu: bool) -> str:
    """Formata o status de um critério de aceitação."""
    return "**APROVADO**" if atingiu else "**REPROVADO**"


def gerar_relatorio_markdown(
    metricas: dict,
    absoluta: pd.DataFrame,
    normalizada: pd.DataFrame,
    piores: pd.DataFrame,
    caminho_md: Path,
    modelo_versao: str,
    dispositivo: str,
    ambiente: dict[str, str],
    nomes_figuras: dict[str, str],
) -> None:
    """Escreve ``desenvolvimento/reports/avaliacao_test.md``.

    A tabela de verificação é preenchida com os valores **realmente medidos**.
    Nenhum número é estimado: quando um valor não pôde ser medido, o relatório
    registra ``não medido``.

    Args:
        metricas: Saída de :func:`calcular_metricas`.
        absoluta: Matriz de confusão absoluta.
        normalizada: Matriz de confusão normalizada.
        piores: Tabela dos piores casos.
        caminho_md: Arquivo de saída.
        modelo_versao: Nome do arquivo de pesos avaliado.
        dispositivo: Dispositivo usado na medição.
        ambiente: Saída de ``config.descrever_ambiente``.
        nomes_figuras: Nomes de arquivo das figuras a referenciar.
    """
    from datetime import datetime

    acuracia = metricas["acuracia_global"]
    latencia = metricas["latencia_ms"]["mediana"]
    por_classe = metricas["por_classe"]

    conf_fungico = por_classe["fungico"]["confianca_media_acertos"]
    conf_bacteriano = por_classe["bacteriano"]["confianca_media_acertos"]
    recall_saudavel = por_classe["saudavel"]["recall"]

    def texto(valor: float | None, sufixo: str = "") -> str:
        return "não medido" if valor is None else f"{valor:.4g}{sufixo}"

    linhas: list[str] = []
    ad = linhas.append

    ad("# Avaliação do modelo no split de teste")
    ad("")
    ad(f"*Gerado por `desenvolvimento/scripts/03_avaliar.py` em "
       f"{datetime.now().strftime('%d/%m/%Y %H:%M')}.*")
    ad("")
    ad("| Parâmetro | Valor |")
    ad("| --- | --- |")
    ad(f"| Pesos avaliados | `{modelo_versao}` |")
    ad(f"| Split | `desenvolvimento/dataset_clean/test` ({metricas['n_imagens']} imagens) |")
    ad(f"| Dispositivo | {dispositivo} |")
    ad(f"| CPU / SO | {ambiente.get('processador', '—')} / "
       f"{ambiente.get('plataforma', '—')} |")
    ad(f"| torch / ultralytics | {ambiente.get('torch', '—')} / "
       f"{ambiente.get('ultralytics', '—')} |")
    ad("")
    ad("> O split de teste provém de `desenvolvimento/dataset_clean/`, gerado com deduplicação e "
       "divisão disjunta por folha de origem. Nenhuma imagem aqui avaliada "
       "aparece no treino — ver `desenvolvimento/reports/dataset_auditoria.md`.")
    ad("")

    ad("## 1. Verificação dos critérios de aceitação")
    ad("")
    ad("| Critério | Meta | Medido | Status |")
    ad("| --- | --- | --- | --- |")
    ad(f"| RNF02 / H1 — acurácia global | > 90% | {acuracia:.2%} | "
       f"{_situacao(acuracia > META_ACURACIA)} |")
    ad(f"| CT01 — confiança média em `fungico` (acertos) | ≥ 0,85 | "
       f"{texto(conf_fungico)} | "
       f"{_situacao(conf_fungico is not None and conf_fungico >= LIMIAR_CT)} |")
    ad(f"| CT02 — confiança média em `bacteriano` (acertos) | ≥ 0,85 | "
       f"{texto(conf_bacteriano)} | "
       f"{_situacao(conf_bacteriano is not None and conf_bacteriano >= LIMIAR_CT)} |")
    ad(f"| CT03 — recall de `saudavel` | ≥ 90% | {recall_saudavel:.2%} | "
       f"{_situacao(recall_saudavel >= META_RECALL_SAUDAVEL)} |")
    ad(f"| CT04 / RNF01 — latência mediana por imagem | < 100 ms | "
       f"{latencia:.2f} ms | {_situacao(latencia < META_LATENCIA_MS)} |")
    ad("")
    ad("A latência acima foi medida no caminho de produção "
       "(`DetectorDoencasAlface.predizer`, uma imagem por chamada) durante esta "
       "avaliação. Para a medição dedicada, com aquecimento e percentis, ver "
       "`desenvolvimento/reports/benchmark_latencia.md`.")
    ad("")

    ad("## 2. Métricas por classe")
    ad("")
    ad("| Classe | Precisão | Recall | F1-score | Suporte | "
       "Confiança média (acertos) | % acertos ≥ 0,85 |")
    ad("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for classe in ROTULOS_CANONICOS:
        dados = por_classe[classe]
        fracao = dados["fracao_acertos_acima_limiar"]
        ad(
            f"| `{classe}` | {dados['precisao']:.4f} | {dados['recall']:.4f} | "
            f"{dados['f1_score']:.4f} | {dados['suporte']} | "
            f"{texto(dados['confianca_media_acertos'])} | "
            f"{'não medido' if fracao is None else f'{fracao:.1%}'} |"
        )
    ad("")
    ad(f"- Acurácia global: **{acuracia:.2%}**")
    ad(f"- F1-score macro: **{metricas['f1_macro']:.4f}**")
    ad("")

    ad("## 3. Matriz de confusão")
    ad("")
    ad(f"![Matriz de confusão]({nomes_figuras.get('matriz', 'matriz_confusao.png')})")
    ad("")
    ad("Valores absolutos (linhas = classe verdadeira, colunas = classe predita):")
    ad("")
    ad("| verdadeiro \\ predito | " + " | ".join(f"`{c}`" for c in absoluta.columns) + " |")
    ad("| --- | " + " | ".join("---:" for _ in absoluta.columns) + " |")
    for classe in absoluta.index:
        ad(f"| `{classe}` | " + " | ".join(str(int(v)) for v in absoluta.loc[classe]) + " |")
    ad("")
    ad("Normalizada por linha:")
    ad("")
    ad("| verdadeiro \\ predito | " + " | ".join(f"`{c}`" for c in normalizada.columns) + " |")
    ad("| --- | " + " | ".join("---:" for _ in normalizada.columns) + " |")
    for classe in normalizada.index:
        ad(f"| `{classe}` | "
           + " | ".join(f"{v:.3f}" for v in normalizada.loc[classe])
           + " |")
    ad("")

    ad("## 4. Curvas de treinamento")
    ad("")
    if "curva" in nomes_figuras:
        ad(f"![Curvas de treinamento]({nomes_figuras['curva']})")
    else:
        ad("Curva não gerada: `results.csv` da execução de treino não foi "
           "encontrado.")
    ad("")

    ad("## 5. Piores casos (inspeção qualitativa)")
    ad("")
    ad("As imagens estão em `desenvolvimento/reports/erros_classificacao/`. O erro é "
       "`1 − probabilidade atribuída à classe verdadeira`.")
    ad("")
    if len(piores):
        ad("| # | Verdadeiro | Predito | Confiança | Erro |")
        ad("| ---: | --- | --- | ---: | ---: |")
        for registro in piores.itertuples():
            ad(f"| {registro.posicao} | `{registro.verdadeiro}` | "
               f"`{registro.predito}` | {registro.confianca:.3f} | "
               f"{registro.erro:.3f} |")
    else:
        ad("Nenhum caso exportado.")
    ad("")

    ad("## 6. Diagnóstico")
    ad("")
    if acuracia > META_ACURACIA:
        ad(f"A acurácia global de **{acuracia:.2%}** supera a meta de 90% "
           "estabelecida em RNF02 e na hipótese H1.")
    else:
        ad(f"A acurácia global de **{acuracia:.2%}** **não** atinge a meta de 90% "
           "de RNF02/H1. Hipóteses diagnósticas a investigar, em ordem de "
           "probabilidade:")
        ad("")
        ad("1. **Diversidade insuficiente da classe `bacteriano`** — o conjunto "
           "deriva de poucas dezenas de folhas originais (problema P3 em "
           "`desenvolvimento/reports/dataset_auditoria.md`). É o fator mais provável e não é "
           "corrigível por ajuste de hiperparâmetros.")
        ad("2. **Confusão entre lesão bacteriana e fúngica em estágio inicial** — "
           "verificar as células fora da diagonal na matriz de confusão acima e "
           "as imagens em `desenvolvimento/reports/erros_classificacao/`.")
        ad("3. **Variação de iluminação e de escala** — considerar ampliar as "
           "faixas de `hsv_v` e `scale` no treino.")
        ad("4. **Desbalanceamento** — se o recall de `fungico` (classe "
           "minoritária) estiver notavelmente abaixo dos demais, aplicar "
           "sobre-amostragem ou perda ponderada, documentando a intervenção.")
        ad("")
        ad("> Nenhuma métrica deste relatório foi obtida com dados de treino. "
           "Reintroduzir imagens vazadas elevaria o número, mas invalidaria a "
           "verificação da hipótese.")
    ad("")

    minoritaria = min(
        ROTULOS_CANONICOS, key=lambda c: por_classe[c]["suporte"]
    )
    recall_minoritaria = por_classe[minoritaria]["recall"]
    if recall_minoritaria < 0.90:
        ad(f"**Atenção:** a classe `{minoritaria}` apresenta recall de "
           f"{recall_minoritaria:.2%}, abaixo de 0,90. Conforme a Seção 6 do "
           "spec, este é o gatilho para aplicar sobre-amostragem da classe "
           "minoritária ou perda ponderada.")
        ad("")

    caminho_md.write_text("\n".join(linhas), encoding="utf-8")
    logger.info("Relatório de avaliação escrito em %s", caminho_md)


def salvar_metricas_json(metricas: dict, caminho: Path) -> None:
    """Grava as métricas em JSON com acentuação preservada."""
    caminho.write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Métricas salvas em %s", caminho)
