"""Gera o texto do módulo de IA para o TCC em formato Word (.docx), nas normas ABNT.

Produz ``desenvolvimento/tcc/Modulo_IA_TCC.docx`` e as figuras em
``desenvolvimento/tcc/figuras_modulo_ia/``. Usa apenas a biblioteca padrão, o
matplotlib e os rótulos de ``src/ia_alface/classes.py`` (sem torch) — nenhuma
dependência de geração de documentos é necessária.

O histórico por época é lido de ``desenvolvimento/runs/.../results.csv``; as
demais métricas reproduzem os relatórios em ``desenvolvimento/reports/``.

Formatação aplicada (NBR 14724, NBR 6024, NBR 6023, NBR 10520, IBGE):
papel A4; margens 3 cm (superior e esquerda) e 2 cm (inferior e direita);
fonte 12 no texto e 10 em tabelas, legendas, fontes e notas; espaçamento 1,5
no texto e simples em tabelas, notas e referências; recuo de parágrafo de
1,25 cm; numeração progressiva das seções; tabelas abertas nas laterais e
quadros fechados; título acima e fonte abaixo de cada ilustração.

Uso:
    python desenvolvimento/scripts/gerar_documento_tcc.py
"""

from __future__ import annotations

import csv
import re
import struct
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

# Rótulos da resposta de recusa vêm de classes.py (regra 2 do CLAUDE.md).
from ia_alface.classes import CLASSE_NAO_ALFACE, EXIBICAO_NAO_ALFACE  # noqa: E402

SAIDA_DIR = RAIZ / "desenvolvimento" / "tcc"
FIG_DIR = SAIDA_DIR / "figuras_modulo_ia"
SAIDA_DOCX = SAIDA_DIR / "Modulo_IA_TCC.docx"
RESULTS_CSV = (
    RAIZ / "desenvolvimento" / "runs" / "classify" / "yolo11n_cls_alface" / "results.csv"
)

FONTE = "Arial"  # ABNT admite Arial ou Times New Roman
LARGURA_TEXTO = 9071  # 16 cm em twips (21 - 3 - 2)
FONTE_PADRAO = "Fonte: elaborado pelos autores (2026)."

# ====================================================================== #
# Figuras
# ====================================================================== #

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    }
)

COR_PRINCIPAL = "#E8F1E4"
COR_BORDA = "#2F5D2A"
COR_ALERTA = "#FBE9E7"
COR_BORDA_ALERTA = "#9C2B1E"
COR_NEUTRA = "#EEF1F5"
COR_BORDA_NEUTRA = "#37474F"


def _caixa(ax, cx, cy, w, h, texto, cor=COR_PRINCIPAL, borda=COR_BORDA,
           tracejada=False, tamanho=9, negrito=False):
    caixa = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=cor, edgecolor=borda, linewidth=1.2,
        linestyle="--" if tracejada else "-",
    )
    ax.add_patch(caixa)
    ax.text(cx, cy, texto, ha="center", va="center", fontsize=tamanho,
            fontweight="bold" if negrito else "normal")


def _seta(ax, x1, y1, x2, y2, rotulo=None, lado="right"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops={"arrowstyle": "-|>", "color": "#333333", "lw": 1.1},
    )
    if rotulo:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx = 0.12 if lado == "right" else -0.12
        ax.text(mx + dx, my + 0.08, rotulo, fontsize=8, style="italic",
                ha="left" if lado == "right" else "right", va="bottom")


def figura_fluxo_inferencia(destino: Path) -> None:
    etapas = [
        "Recepção da imagem capturada\n(JPEG ou PNG)",
        "Validação da imagem",
        "Conversão para o padrão de cor RGB",
        "Redimensionamento do lado menor para 224 px\ne recorte central de 224 × 224 px",
        "Normalização dos pixels para o intervalo [0, 1]",
        "Rede neural convolucional YOLO11n-cls",
        "Função softmax: uma probabilidade por classe",
        "Classe = maior probabilidade\nConfiança = valor dessa probabilidade",
        "Classe de doença e confiança ≥ 0,85?",
        "Emissão de alerta fitossanitário",
    ]
    fig, ax = plt.subplots(figsize=(7.2, 9.2))
    xc, w, h, passo = 3.2, 4.6, 0.62, 0.98
    ys = [len(etapas) * passo - i * passo for i in range(len(etapas))]
    for i, (texto, y) in enumerate(zip(etapas, ys)):
        decisao = i in (1, 8)
        alerta = i == 9
        _caixa(ax, xc, y, w, h, texto,
               cor=COR_ALERTA if alerta else (COR_NEUTRA if decisao else COR_PRINCIPAL),
               borda=COR_BORDA_ALERTA if alerta else (COR_BORDA_NEUTRA if decisao else COR_BORDA))
        if i > 0:
            _seta(ax, xc, ys[i - 1] - h / 2, xc, y + h / 2,
                  rotulo="válida" if i == 2 else ("sim" if i == 9 else None))
    xs = 8.0
    _caixa(ax, xs, ys[1], 3.0, h, "Rejeição da imagem\n(nenhuma classificação)",
           cor=COR_ALERTA, borda=COR_BORDA_ALERTA)
    _seta(ax, xc + w / 2, ys[1], xs - 1.5, ys[1], rotulo="inválida")
    _caixa(ax, xs, ys[8], 3.0, h, "Registro do resultado\nsem alerta")
    _seta(ax, xc + w / 2, ys[8], xs - 1.5, ys[8], rotulo="não")
    ax.set_xlim(0.6, 9.8)
    ax.set_ylim(ys[-1] - 0.6, ys[0] + 0.6)
    ax.axis("off")
    fig.savefig(destino)
    plt.close(fig)


def figura_arquitetura(destino: Path) -> None:
    blocos = [
        ("Entrada", "3×224×224", 224, "e"),
        ("Conv", "16×112×112", 112, "c"),
        ("Conv", "32×56×56", 56, "c"),
        ("C3k2", "64×56×56", 56, "k"),
        ("Conv", "64×28×28", 28, "c"),
        ("C3k2", "128×28×28", 28, "k"),
        ("Conv", "128×14×14", 14, "c"),
        ("C3k2", "128×14×14", 14, "k"),
        ("Conv", "256×7×7", 7, "c"),
        ("C3k2", "256×7×7", 7, "k"),
        ("C2PSA", "256×7×7", 7, "a"),
        ("Conv 1×1", "1280×7×7", 7, "h"),
        ("GAP", "1280", 1, "h"),
        ("Linear", "3", 1, "h"),
        ("Softmax", "3", 1, "h"),
    ]
    cores = {"e": "#D9D9D9", "c": "#A8CFA0", "k": "#6FAF63", "a": "#3E7F35",
             "h": "#F2C38B"}
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, (nome, dim, res, tipo) in enumerate(blocos):
        altura = 0.35 + 2.6 * (res / 224) ** 0.55
        x = i * 1.0
        ax.add_patch(plt.Rectangle((x - 0.32, -altura / 2), 0.64, altura,
                                   facecolor=cores[tipo], edgecolor="#333333", lw=0.8))
        ax.text(x, altura / 2 + 0.12, nome, ha="center", va="bottom", fontsize=8,
                rotation=45)
        ax.text(x, -1.75, dim, ha="center", va="top", fontsize=7.5, rotation=90)
        if i > 0:
            ax.annotate("", xy=(x - 0.34, 0), xytext=(x - 0.66, 0),
                        arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": "#333"})
    ax.annotate("", xy=(0.65, 2.55), xytext=(10.35, 2.55),
                arrowprops={"arrowstyle": "-", "lw": 1.0})
    ax.text(5.5, 2.65, "Extrator de características (backbone)", ha="center",
            va="bottom", fontsize=9, fontweight="bold")
    ax.annotate("", xy=(10.65, 2.55), xytext=(14.35, 2.55),
                arrowprops={"arrowstyle": "-", "lw": 1.0})
    ax.text(12.5, 2.65, "Cabeça de classificação", ha="center", va="bottom",
            fontsize=9, fontweight="bold")
    ax.text(-0.4, -3.35, "Dimensões de saída: canais × altura × largura. "
            "A altura dos blocos é proporcional à resolução espacial.",
            fontsize=8, style="italic", ha="left")
    ax.set_xlim(-0.8, 14.8)
    ax.set_ylim(-3.6, 3.4)
    ax.axis("off")
    fig.savefig(destino)
    plt.close(fig)


def figura_fluxo_colab(destino: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 8.4))
    xl, xr, w, h = 2.6, 8.2, 4.3, 0.68
    ax.text(xl, 11.0, "Ambiente local (CPU)", ha="center", fontsize=10.5, fontweight="bold")
    ax.text(xr, 11.0, "Google Colab (GPU)", ha="center", fontsize=10.5, fontweight="bold")
    ax.axvline(5.4, color="#999999", lw=0.8, ls=":", ymin=0.03, ymax=0.95)
    esquerda_1 = [
        (10.1, "1. Auditoria, deduplicação e\nredivisão do conjunto de dados"),
        (9.0, "2. Compactação do conjunto\npreparado (≈ 86 MB)"),
    ]
    direita = [
        (9.0, "3. Verificação da disponibilidade\nda GPU"),
        (7.9, "4. Instalação da biblioteca\nUltralytics"),
        (6.8, "5. Extração e conferência do\nconjunto (2.216 imagens)"),
        (5.7, "6. Treinamento do YOLO11n-cls"),
        (4.6, "7. Conferência da melhor época\n(subconjunto de validação)"),
    ]
    esquerda_2 = [
        (3.5, "8. Recebimento dos pesos\nda melhor época"),
        (2.4, "9. Avaliação no subconjunto\nde teste (execução única)"),
        (1.3, "10. Exportação para ONNX e\nmedição de latência"),
    ]
    for y, t in esquerda_1 + esquerda_2:
        _caixa(ax, xl, y, w, h, t, cor=COR_NEUTRA, borda=COR_BORDA_NEUTRA)
    for y, t in direita:
        _caixa(ax, xr, y, w, h, t)
    _seta(ax, xl, 10.1 - h / 2, xl, 9.0 + h / 2)
    _seta(ax, xl + w / 2, 9.0, xr - w / 2, 9.0, rotulo="envio")
    for (y1, _), (y2, _) in zip(direita, direita[1:]):
        _seta(ax, xr, y1 - h / 2, xr, y2 + h / 2)
    ax.annotate("", xy=(xl + w / 2, 3.5), xytext=(xr, 4.6 - h / 2),
                arrowprops={"arrowstyle": "-|>", "lw": 1.1, "color": "#333",
                            "connectionstyle": "angle,angleA=-90,angleB=180"})
    ax.text(6.4, 3.62, "download", fontsize=8, style="italic")
    for (y1, _), (y2, _) in zip(esquerda_2, esquerda_2[1:]):
        _seta(ax, xl, y1 - h / 2, xl, y2 + h / 2)
    ax.set_xlim(0.1, 10.7)
    ax.set_ylim(0.6, 11.4)
    ax.axis("off")
    fig.savefig(destino)
    plt.close(fig)


def figura_cascata(destino: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    xc, w, h = 3.0, 4.4, 0.7
    etapas = [
        (7.2, "Imagem capturada pela câmera Logitech C270", COR_NEUTRA, COR_BORDA_NEUTRA, False),
        (6.0, "Validação técnica da imagem", COR_NEUTRA, COR_BORDA_NEUTRA, False),
        (4.8, "Etapa 1 — Verificador de domínio:\na imagem contém alface?\nP(alface) ≥ 0,50", "#FFF8E1", "#8D6E00", False),
        (3.6, "Etapa 2 — Classificador de doenças", COR_PRINCIPAL, COR_BORDA, False),
        (2.4, "Saudável  |  Bacteriana  |  Fúngica", COR_PRINCIPAL, COR_BORDA, False),
        (1.2, "Regra de alerta\n(doença e confiança ≥ 0,85)", COR_ALERTA, COR_BORDA_ALERTA, False),
    ]
    for y, t, c, b, tr in etapas:
        _caixa(ax, xc, y, w, h, t, cor=c, borda=b, tracejada=tr)
    for (y1, *_), (y2, *_) in zip(etapas, etapas[1:]):
        _seta(ax, xc, y1 - h / 2, xc, y2 + h / 2,
              rotulo="sim" if y2 == 3.6 else ("válida" if y2 == 4.8 else None))
    xs = 8.1
    _caixa(ax, xs, 6.0, 3.2, h, "Erro: imagem inválida\n(nenhuma classificação)",
           COR_ALERTA, COR_BORDA_ALERTA)
    _seta(ax, xc + w / 2, 6.0, xs - 1.6, 6.0, rotulo="inválida")
    _caixa(ax, xs, 4.8, 3.2, 1.0,
           f"Resposta “{EXIBICAO_NAO_ALFACE}”\n(classe {CLASSE_NAO_ALFACE}):\n"
           "sem diagnóstico e sem alerta",
           COR_ALERTA, COR_BORDA_ALERTA)
    _seta(ax, xc + w / 2, 4.8, xs - 1.6, 4.8, rotulo="não")
    ax.set_xlim(0.6, 9.9)
    ax.set_ylim(0.1, 7.8)
    ax.axis("off")
    fig.savefig(destino)
    plt.close(fig)


def figura_implantacao(destino: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 10.2))
    xc, w, h = 3.6, 4.6, 0.8
    # Contêiner e imagem desenhados antes das caixas internas, que ficam por cima.
    for x0, y0, x1, y1, cor in ((0.6, 0.8, 6.6, 7.3, "#F4F6F9"),
                                (1.0, 1.7, 6.2, 7.0, "#FFFFFF")):
        ax.add_patch(FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.02,rounding_size=0.15",
            facecolor=cor, edgecolor=COR_BORDA_NEUTRA, linewidth=1.2))
    ax.text(xc, 1.2, "Azure Container Apps — escala a zero (0 a 2 réplicas)",
            ha="center", va="center", fontsize=8.5, fontweight="bold")
    ax.text(xc, 2.2, "Imagem Docker: python:3.13-slim e PyTorch para CPU",
            ha="center", va="center", fontsize=8.5, style="italic")
    _caixa(ax, xc, 12.8, w, h, "Câmera Logitech C270", COR_NEUTRA, COR_BORDA_NEUTRA)
    _caixa(ax, xc, 11.1, w, h, "Azure Blob Storage\n(armazenamento das imagens)",
           COR_NEUTRA, COR_BORDA_NEUTRA)
    _caixa(ax, xc, 9.4, w, h, "Azure Function — orquestração\n(não implementada)",
           COR_NEUTRA, COR_BORDA_NEUTRA, tracejada=True)
    _caixa(ax, xc, 6.3, 4.2, h, "API HTTP (FastAPI)", COR_NEUTRA, COR_BORDA_NEUTRA)
    _caixa(ax, xc, 4.8, 4.2, h, "Etapa 1 — Verificador de domínio\n(YOLO11n-cls, versão 2)",
           "#FFF8E1", "#8D6E00")
    _caixa(ax, xc, 3.3, 4.2, h, "Etapa 2 — Classificador de doenças\n(YOLO11n-cls)")
    _seta(ax, xc, 12.8 - h / 2, xc, 11.1 + h / 2,
          rotulo="envio da imagem\n(a definir pelo grupo)")
    _seta(ax, xc, 11.1 - h / 2, xc, 9.4 + h / 2, rotulo="evento de nova imagem")
    _seta(ax, xc - 0.5, 9.4 - h / 2, xc - 0.5, 6.3 + h / 2,
          rotulo="imagem, com\nchave de API", lado="left")
    _seta(ax, xc + 0.5, 6.3 + h / 2, xc + 0.5, 9.4 - h / 2, rotulo="resposta JSON")
    _seta(ax, xc, 6.3 - h / 2, xc, 4.8 + h / 2)
    _seta(ax, xc, 4.8 - h / 2, xc, 3.3 + h / 2, rotulo="contém alface")
    xd = 9.3
    _caixa(ax, xd, 9.4, 3.0, 1.0, "Banco de dados e\nnotificações\n(fora deste módulo)",
           COR_NEUTRA, COR_BORDA_NEUTRA, tracejada=True)
    _seta(ax, xc + w / 2, 9.4, xd - 1.5, 9.4, rotulo="resultado")
    _caixa(ax, xd, 4.8, 3.0, 1.0, "GitHub Container\nRegistry (imagem\nversionada: v1, v2, …)",
           COR_NEUTRA, COR_BORDA_NEUTRA)
    ax.annotate("", xy=(6.6, 4.8), xytext=(xd - 1.5, 4.8),
                arrowprops={"arrowstyle": "-|>", "color": "#333333", "lw": 1.1,
                            "linestyle": "--"})
    ax.text((6.6 + xd - 1.5) / 2, 4.92, "implantação", fontsize=8, style="italic",
            ha="center", va="bottom")
    ax.set_xlim(0.3, 11.0)
    ax.set_ylim(0.4, 13.4)
    ax.axis("off")
    fig.savefig(destino)
    plt.close(fig)


def figura_curvas(historico: list[dict], destino: Path) -> None:
    ep = [r["epoch"] for r in historico]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot(ep, [r["train/loss"] for r in historico], color="#2E7D32", lw=1.6,
            label="Treinamento")
    a1.plot(ep, [r["val/loss"] for r in historico], color="#C62828", lw=1.6, ls="--",
            label="Validação")
    a1.set_xlabel("Época")
    a1.set_ylabel("Perda (entropia cruzada)")
    a1.set_title("(a) Perda", fontsize=10)
    a2.plot(ep, [100 * r["metrics/accuracy_top1"] for r in historico], color="#2E7D32",
            lw=1.6, label="Acurácia de validação")
    a2.axhline(90, color="#555555", ls=":", lw=1.1, label="Meta do RNF02 (90%)")
    a2.set_xlabel("Época")
    a2.set_ylabel("Acurácia (%)")
    a2.set_ylim(65, 100)
    a2.set_title("(b) Acurácia no subconjunto de validação", fontsize=10)
    for ax in (a1, a2):
        ax.axvline(13, color="#1565C0", lw=1.0, ls="-.")
        ax.axvspan(13, 33, color="#1565C0", alpha=0.06)
        ax.grid(alpha=0.3)
        ax.set_xlim(0.5, 33.5)
    a1.text(13.4, a1.get_ylim()[1] * 0.93, "melhor época (13)", fontsize=8, color="#1565C0")
    a2.text(13.4, 67, "20 épocas sem melhora", fontsize=8, color="#1565C0")
    a1.legend(fontsize=8)
    a2.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)


def figura_matriz(destino: Path) -> None:
    rot = ["Bacteriana", "Fúngica", "Saudável"]
    m = [[103, 3, 0], [7, 66, 5], [0, 1, 139]]
    norm = [[v / sum(linha) for v in linha] for linha in m]
    fig, eixos = plt.subplots(1, 2, figsize=(10, 4.0))
    for ax, dados, titulo, fmt in (
        (eixos[0], m, "(a) Valores absolutos", "{:d}"),
        (eixos[1], norm, "(b) Proporção por classe real", "{:.3f}"),
    ):
        maximo = max(max(l) for l in dados)
        ax.imshow(dados, cmap="Greens", vmin=0, vmax=maximo)
        for i in range(3):
            for j in range(3):
                v = dados[i][j]
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=11,
                        color="white" if v > maximo * 0.55 else "black")
        ax.set_xticks(range(3), rot)
        ax.set_yticks(range(3), rot)
        ax.set_xlabel("Classe predita")
        ax.set_ylabel("Classe real")
        ax.set_title(titulo, fontsize=10)
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)


# ====================================================================== #
# Conteúdo
# ====================================================================== #

def ler_historico() -> list[dict]:
    with RESULTS_CSV.open(encoding="utf-8") as arq:
        leitor = csv.DictReader(arq)
        linhas = []
        for linha in leitor:
            linha = {k.strip(): v.strip() for k, v in linha.items()}
            linhas.append({
                "epoch": int(linha["epoch"]),
                "time": float(linha["time"]),
                "train/loss": float(linha["train/loss"]),
                "val/loss": float(linha["val/loss"]),
                "metrics/accuracy_top1": float(linha["metrics/accuracy_top1"]),
                "lr": float(linha["lr/pg0"]),
            })
    return linhas


def br(valor: float, casas: int) -> str:
    """Formata número no padrão brasileiro (vírgula decimal, ponto de milhar)."""
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


# Blocos: ("h", nivel, texto, rotulo) | ("p", texto) | ("al", [itens])
#         ("eq", omml, rotulo) | ("tab", rotulo, titulo, cab, linhas, larguras, alinh, fonte)
#         ("qua", ...) | ("fig", rotulo, titulo, caminho, largura_cm, fonte) | ("ref", texto)

# ---------------------------- OMML ------------------------------------- #

def mr(t: str, normal: bool = False) -> str:
    sty = '<m:rPr><m:sty m:val="p"/></m:rPr>' if normal else ""
    return f'<m:r>{sty}<m:t xml:space="preserve">{escape(t)}</m:t></m:r>'


def msub(b: str, s: str) -> str:
    return f"<m:sSub><m:e>{b}</m:e><m:sub>{s}</m:sub></m:sSub>"


def msup(b: str, s: str) -> str:
    return f"<m:sSup><m:e>{b}</m:e><m:sup>{s}</m:sup></m:sSup>"


def msubsup(b: str, s: str, p: str) -> str:
    return f"<m:sSubSup><m:e>{b}</m:e><m:sub>{s}</m:sub><m:sup>{p}</m:sup></m:sSubSup>"


def mfrac(n: str, d: str) -> str:
    return f"<m:f><m:num>{n}</m:num><m:den>{d}</m:den></m:f>"


def msoma(sub: str, sup: str, corpo: str) -> str:
    sup_xml = f"<m:sup>{sup}</m:sup>" if sup else "<m:sup/>"
    oculto = "" if sup else '<m:supHide m:val="1"/>'
    return (f'<m:nary><m:naryPr><m:chr m:val="∑"/>{oculto}</m:naryPr>'
            f"<m:sub>{sub}</m:sub>{sup_xml}<m:e>{corpo}</m:e></m:nary>")


def mpar(corpo: str) -> str:
    return f"<m:d><m:e>{corpo}</m:e></m:d>"


def construir_conteudo(hist: list[dict]) -> list[tuple]:
    melhor = max(hist, key=lambda r: (r["metrics/accuracy_top1"], -r["epoch"]))
    tempo_total = hist[-1]["time"]
    n_ep = len(hist)
    b: list[tuple] = []
    H = lambda n, t, r=None: b.append(("h", n, t, r))  # noqa: E731
    P = lambda t: b.append(("p", t))  # noqa: E731
    AL = lambda itens: b.append(("al", itens))  # noqa: E731

    # ================================================================== #
    H(1, "FUNDAMENTAÇÃO TEÓRICA DO MÓDULO DE INTELIGÊNCIA ARTIFICIAL", "cap_fund")
    # ================================================================== #
    P("Este capítulo apresenta os conceitos que fundamentam o módulo de inteligência "
      "artificial do sistema proposto, responsável por identificar, a partir de uma "
      "fotografia, se uma folha de alface (*Lactuca sativa*) está saudável ou acometida por "
      "doença bacteriana ou fúngica. São descritos o funcionamento das redes neurais "
      "convolucionais, o mecanismo pelo qual tais redes reconhecem padrões visuais de "
      "doença, a arquitetura adotada, o processo de treinamento e as métricas de avaliação "
      "empregadas.")

    H(2, "Visão computacional e aprendizado de máquina", "sec_visao")
    P("A visão computacional é a área da computação dedicada a extrair informação "
      "útil de imagens digitais. Uma imagem colorida é representada computacionalmente "
      "como uma matriz de *pixels*, em que cada *pixel* contém três valores de intensidade, "
      "correspondentes aos canais vermelho, verde e azul (padrão RGB), tipicamente no "
      "intervalo de 0 a 255. Uma imagem de 224 × 224 *pixels* corresponde, portanto, a "
      "150.528 valores numéricos.")
    P("Na abordagem clássica da visão computacional, o projetista define manualmente as "
      "características a serem extraídas — por exemplo, limiares de cor para segmentar "
      "manchas ou contagens de regiões escuras — e regras de decisão sobre elas. Na "
      "abordagem por aprendizado de máquina, e em particular por aprendizado profundo "
      "(*deep learning*), as características são aprendidas automaticamente a partir de "
      "exemplos rotulados (LECUN; BENGIO; HINTON, 2015). O {qua:abordagens} sintetiza as "
      "diferenças entre as duas abordagens.")
    b.append(("qua", "abordagens",
              "Comparação entre a abordagem clássica e a abordagem por aprendizado profundo",
              ["Aspecto", "Visão computacional clássica", "Aprendizado profundo (abordagem adotada)"],
              [["Definição das características", "Manual, pelo projetista (por exemplo, faixas de matiz no espaço HSV)", "Automática, aprendida durante o treinamento"],
               ["Uso da cor", "Regra explícita sobre valores de *pixel*", "Implícito, combinado com textura e forma"],
               ["Robustez a variações de iluminação", "Baixa: limiares fixos deixam de funcionar sob outra iluminação", "Maior, quando o treinamento inclui variações"],
               ["Interpretabilidade", "Alta: a regra é legível", "Baixa: requer técnicas de explicabilidade"],
               ["Volume de dados rotulados", "Pequeno", "Grande, mitigado por transferência de aprendizado"]],
              [2300, 3385, 3386], "lll", FONTE_PADRAO))

    H(2, "Redes neurais convolucionais", "sec_cnn")
    P("As redes neurais convolucionais (*Convolutional Neural Networks*, CNN) constituem a "
      "classe de modelos de aprendizado profundo predominante em tarefas de "
      "reconhecimento de imagens (GOODFELLOW; BENGIO; COURVILLE, 2016). São compostas por "
      "camadas sucessivas que transformam a imagem de entrada em representações "
      "progressivamente mais abstratas, até produzir a decisão final.")

    H(3, "Operação de convolução")
    P("O elemento fundamental da rede é a operação de convolução. Um filtro (*kernel*) é "
      "uma pequena matriz de pesos — no modelo adotado, predominantemente de 3 × 3 — que "
      "percorre a imagem. Em cada posição, os valores do filtro são multiplicados pelos "
      "valores da região sob ele e os produtos são somados, conforme a {eq:conv}:")
    b.append(("eq",
              msub(mr("S"), mr("i,j")) + mr("=") +
              msoma(mr("m"), "", msoma(mr("n"), "",
                    mr("I") + mpar(mr("i+m, j+n")) + mr("⋅") + mr("K") + mpar(mr("m, n")))) +
              mr("+") + mr("b"),
              "conv"))
    P("em que *I* representa a imagem (ou o mapa de características da camada anterior), "
      "*K* o filtro, *b* um termo de deslocamento (*bias*) e *S* o mapa de saída. O valor "
      "resultante em cada posição indica o grau de correspondência entre a região "
      "analisada e o padrão representado pelo filtro. Um filtro com pesos positivos de um "
      "lado e negativos do outro, por exemplo, responde intensamente a bordas — como a "
      "transição entre o tecido verde da folha e uma lesão escura.")
    P("Os pesos dos filtros não são definidos pelo projetista: são parâmetros ajustados "
      "durante o treinamento, de modo a minimizar o erro de classificação. Cada camada "
      "aplica dezenas ou centenas de filtros em paralelo, e cada filtro produz um mapa de "
      "características distinto, denominado canal.")

    H(3, "Normalização e função de ativação")
    P("Após cada convolução, o modelo adotado aplica a normalização em lote (*batch "
      "normalization*), que padroniza a escala dos valores entre camadas e estabiliza o "
      "treinamento, e a função de ativação SiLU (*Sigmoid Linear Unit*), definida pela "
      "{eq:silu}:")
    b.append(("eq",
              mr("f") + mpar(mr("x")) + mr("=") + mr("x") + mr("⋅") + mr("σ", True) +
              mpar(mr("x")) + mr("=") + mfrac(mr("x"), mr("1+") + msup(mr("e", True), mr("−x"))),
              "silu"))
    P("A função de ativação introduz não linearidade no modelo. Sem ela, a composição de "
      "todas as camadas equivaleria a uma única transformação linear, incapaz de "
      "representar padrões visuais complexos.")

    H(3, "Redução de resolução e hierarquia de características")
    P("Algumas convoluções são aplicadas com passo (*stride*) igual a 2, isto é, o filtro "
      "avança dois *pixels* por vez, o que reduz pela metade a altura e a largura do mapa "
      "de saída. Com a redução da resolução, cada posição do mapa passa a resumir uma "
      "região maior da imagem original, e as camadas seguintes passam a combinar padrões "
      "detectados pelas anteriores.")
    P("Disso resulta uma hierarquia de características: as primeiras camadas respondem a "
      "padrões elementares, como bordas e gradientes de cor; as intermediárias, a texturas "
      "e formas simples; e as finais, a composições complexas. Esse comportamento foi "
      "demonstrado experimentalmente por Zeiler e Fergus (2014), por meio da visualização "
      "dos padrões que ativam cada camada de uma rede treinada. O {qua:hierarquia} "
      "relaciona essa hierarquia ao domínio deste trabalho.")
    b.append(("qua", "hierarquia",
              "Hierarquia de características de uma rede convolucional aplicada a folhas de alface",
              ["Nível", "Resolução espacial no modelo adotado", "Padrões tipicamente detectados", "Exemplos no domínio"],
              [["Baixo", "112 e 56 *pixels*", "Bordas, gradientes de cor, contrastes locais", "Contorno da folha; transição do verde para o amarelo; borda de uma mancha"],
               ["Intermediário", "28 e 14 *pixels*", "Texturas e formas simples", "Superfície crespa; nervuras; manchas pequenas e angulosas; aspecto aveludado"],
               ["Alto", "7 *pixels*", "Composições e distribuição espacial", "Lesões delimitadas por nervuras; manchas concentradas nas bordas; folha uniformemente verde"],
               ["Decisão", "Vetor sem dimensão espacial", "Combinação ponderada das evidências", "Conjunto de evidências compatível com uma das classes"]],
              [1500, 2000, 2600, 2971], "llll",
              "Fonte: elaborado pelos autores (2026), com base em Zeiler e Fergus (2014)."))
    P("Ressalte-se que o {qua:hierarquia} descreve a tendência geral documentada na "
      "literatura. Os filtros aprendidos não possuem significado rotulado, de modo que não "
      "é possível afirmar com exatidão qual conceito visual cada um representa no modelo "
      "treinado.")

    H(2, "Mecanismo de identificação das doenças foliares", "sec_mecanismo")
    H(3, "Características visuais empregadas pela rede")
    P("O modelo desenvolvido não contém regras explícitas de diagnóstico. Em nenhuma "
      "etapa do processamento existe instrução do tipo “manchas marrons indicam doença "
      "bacteriana” ou “coloração pálida indica míldio”. A rede aprendeu, a partir de 1.566 "
      "imagens rotuladas, quais combinações de características visuais distinguem cada "
      "classe.")
    P("A cor participa do processo desde a primeira camada, uma vez que a entrada da rede "
      "são os três canais RGB, e os filtros iniciais combinam esses canais, respondendo a "
      "variações cromáticas como o amarelecimento e o escurecimento do tecido. Contudo, a "
      "decisão final resulta da combinação da cor com outras três famílias de "
      "características:")
    AL(["**textura**: o padrão de repetição da superfície, como o aspecto aveludado da "
        "esporulação ou o aspecto encharcado de lesões bacterianas;",
        "**forma**: a geometria das lesões, como contornos angulosos ou circulares;",
        "**distribuição espacial**: a localização das lesões na folha, como a "
        "concentração junto às bordas ou a delimitação pelas nervuras."])
    P("Essa combinação confere ao modelo vantagem sobre critérios exclusivamente "
      "cromáticos, que tenderiam a confundir, por exemplo, o amarelecimento causado por "
      "deficiência nutricional com os sintomas iniciais do míldio. Tal capacidade de "
      "distinção, entretanto, depende da presença de exemplos desses casos no conjunto de "
      "treinamento, condição não atendida pelo conjunto empregado, conforme discutido no "
      "{sec:cap_lim}.")

    H(3, "Sintomas visuais das classes")
    P("O {qua:sintomas} descreve os sintomas visuais característicos de cada classe "
      "considerada. Trata-se dos padrões presentes nas imagens de treinamento e, "
      "portanto, disponíveis para o aprendizado da rede. A classe denominada fúngica "
      "corresponde ao míldio, causado por *Bremia lactucae*[[nota:Taxonomicamente, *Bremia "
      "lactucae* é um oomiceto (reino Stramenopila), e não um fungo verdadeiro, embora seja "
      "tradicionalmente estudado pela micologia e controlado com fungicidas. Manteve-se a "
      "denominação “fúngica” por consistência com a especificação do sistema.]]; a classe "
      "bacteriana corresponde à mancha bacteriana, causada por *Xanthomonas campestris* pv. "
      "*vitians*.")
    b.append(("qua", "sintomas", "Características visuais das classes consideradas",
              ["Classe", "Cor", "Textura", "Forma e distribuição"],
              [["Saudável", "Verde uniforme, conforme a cultivar", "Superfície regular, lisa ou crespa conforme a cultivar", "Ausência de manchas; nervuras contínuas"],
               ["Fúngica (míldio, *Bremia lactucae*)", "Áreas verde-pálidas a amareladas na face superior, que escurecem com a progressão", "Crescimento esbranquiçado e aveludado, sobretudo na face inferior", "Lesões angulosas, delimitadas pelas nervuras"],
               ["Bacteriana (*Xanthomonas campestris* pv. *vitians*)", "Manchas inicialmente translúcidas, depois marrom-escuras a negras, por vezes com halo amarelado", "Aspecto encharcado; tecido necrosado", "Manchas pequenas e angulosas, que podem coalescer; frequentes junto às bordas"]],
              [1900, 2500, 2300, 2371], "llll",
              "Fonte: elaborado pelos autores (2026). [[preencher:referência fitopatológica que descreva os sintomas]]"))
    P("Observa-se que ambas as doenças produzem lesões angulosas. Essa semelhança visual "
      "é coerente com o padrão de erros observado nos resultados, em que a principal "
      "confusão ocorre entre as classes fúngica e bacteriana ({sec:sec_erros}).")

    H(2, "Arquitetura YOLO11 para classificação", "sec_yolo")
    P("A família de modelos YOLO (*You Only Look Once*) é amplamente empregada em visão "
      "computacional por equilibrar exatidão e velocidade de execução. Em sua décima "
      "primeira geração, denominada YOLO11, a biblioteca Ultralytics disponibiliza "
      "variantes para detecção de objetos, segmentação, estimativa de pose e "
      "classificação de imagens (JOCHER; QIU, 2024). Este trabalho emprega a variante de "
      "classificação no menor porte disponível, denominada YOLO11n-cls (*nano*), cujas "
      "características constam da {tab:modelo}.")
    b.append(("tab", "modelo", "Características do modelo YOLO11n-cls empregado",
              ["Característica", "Valor"],
              [["Parâmetros treináveis (três classes)", "1.534.947"],
               ["Camadas", "86 (47 após a fusão de convolução e normalização)"],
               ["Dimensão da entrada", "3 × 224 × 224"],
               ["Dimensão da saída", "3 probabilidades"],
               ["Tamanho do arquivo de pesos", "3,19 MB"]],
              [5500, 3571], "lr", FONTE_PADRAO))
    P("O modelo divide-se em duas partes. O extrator de características (*backbone*) "
      "transforma a imagem em um conjunto de mapas de características de baixa resolução "
      "e grande número de canais. A cabeça de classificação converte esses mapas em uma "
      "probabilidade para cada classe. A {fig:arquitetura} ilustra a sequência de blocos "
      "e a evolução das dimensões ao longo da rede, detalhada no {qua:camadas}.")
    b.append(("fig", "arquitetura", "Estrutura do modelo YOLO11n-cls", "arquitetura", 16.0,
              "Fonte: elaborado pelos autores (2026), com base em Jocher e Qiu (2024)."))
    b.append(("qua", "camadas", "Camadas do modelo YOLO11n-cls para entrada de 224 × 224 *pixels*",
              ["Ordem", "Bloco", "Função", "Saída (canais × altura × largura)"],
              [["—", "Entrada", "Imagem RGB normalizada", "3 × 224 × 224"],
               ["0", "Conv 3 × 3, passo 2", "Bordas e cores elementares; reduz a resolução", "16 × 112 × 112"],
               ["1", "Conv 3 × 3, passo 2", "Padrões locais simples", "32 × 56 × 56"],
               ["2", "C3k2", "Combinação de padrões; texturas iniciais", "64 × 56 × 56"],
               ["3", "Conv 3 × 3, passo 2", "Redução de resolução", "64 × 28 × 28"],
               ["4", "C3k2", "Texturas e formas pequenas", "128 × 28 × 28"],
               ["5", "Conv 3 × 3, passo 2", "Redução de resolução", "128 × 14 × 14"],
               ["6", "C3k2", "Formas maiores e agrupamentos", "128 × 14 × 14"],
               ["7", "Conv 3 × 3, passo 2", "Redução de resolução", "256 × 7 × 7"],
               ["8", "C3k2", "Padrões de alto nível", "256 × 7 × 7"],
               ["9", "C2PSA", "Atenção espacial entre regiões da folha", "256 × 7 × 7"],
               ["10", "Conv 1 × 1", "Expansão para 1.280 características", "1.280 × 7 × 7"],
               ["10", "Agrupamento médio global", "Média de cada canal sobre as 49 posições", "1.280"],
               ["10", "*Dropout* (somente no treinamento)", "Regularização", "1.280"],
               ["10", "Camada linear", "Uma pontuação por classe", "3"],
               ["10", "*Softmax*", "Conversão das pontuações em probabilidades", "3"]],
              [900, 2500, 3371, 2300], "clll",
              "Fonte: elaborado pelos autores (2026), com base em Jocher e Qiu (2024)."))
    P("Os blocos indicados como Conv compreendem convolução, normalização em lote e "
      "ativação SiLU. Após cinco convoluções com passo 2, a resolução de 224 *pixels* "
      "reduz-se a 7, de modo que cada posição do mapa final resume uma região de 32 × 32 "
      "*pixels* da imagem original.")
    P("O bloco C3k2 segue o princípio *Cross Stage Partial*: divide os canais em dois "
      "ramos, processa um deles por uma sequência de convoluções com conexões residuais e "
      "reúne os dois ao final, ampliando a capacidade de representação com baixo número "
      "de parâmetros. O bloco C2PSA (*Position-Sensitive Attention*) aplica um mecanismo "
      "de autoatenção, pelo qual cada posição do mapa de 7 × 7 pode ponderar as "
      "informações de todas as demais, o que permite relacionar lesões dispersas pela "
      "folha.")
    P("Na cabeça de classificação, o agrupamento médio global (*Global Average Pooling*) "
      "calcula a média de cada um dos 1.280 canais sobre todas as posições espaciais. O "
      "vetor resultante descreve quais padrões estão presentes na imagem, mas não onde "
      "estão. Por essa razão, o modelo de classificação não fornece a localização das "
      "lesões.")

    H(2, "Função *softmax* e grau de confiança", "sec_softmax")
    P("A camada linear final produz três valores reais, denominados *logits*, um para "
      "cada classe. A função *softmax*, expressa na {eq:softmax}, converte esses valores "
      "em probabilidades:")
    b.append(("eq",
              msub(mr("p"), mr("i")) + mr("=") +
              mfrac(msup(mr("e", True), msub(mr("z"), mr("i"))),
                    msoma(mr("j=1"), mr("3"), msup(mr("e", True), msub(mr("z"), mr("j"))))),
              "softmax"))
    P("em que *z*~i~ é o *logit* da classe *i* e *p*~i~ a probabilidade correspondente. Os "
      "valores resultantes situam-se entre 0 e 1 e somam exatamente 1. A classe de maior "
      "probabilidade constitui a resposta do modelo, e o valor dessa probabilidade é "
      "denominado confiança.")
    P("Uma consequência dessa formulação é que o modelo sempre atribui a imagem a uma das "
      "três classes conhecidas, inclusive quando a imagem não contém uma folha de alface. "
      "Essa propriedade motiva a etapa de verificação de domínio descrita na "
      "{sec:sec_dominio}.")

    H(2, "Treinamento supervisionado", "sec_treino_teoria")
    P("Treinar a rede consiste em ajustar seus parâmetros de modo a minimizar o erro "
      "sobre um conjunto de imagens cujas classes são conhecidas. O processo repete, a "
      "cada lote de imagens, quatro etapas:")
    AL(["propagação direta: o lote atravessa a rede, que produz as probabilidades de cada "
        "classe;",
        "cálculo da perda: as probabilidades são comparadas aos rótulos verdadeiros;",
        "retropropagação (*backpropagation*): calcula-se, pela regra da cadeia, o gradiente "
        "da perda em relação a cada parâmetro, isto é, a contribuição de cada parâmetro "
        "para o erro;",
        "atualização: o otimizador ajusta cada parâmetro na direção que reduz a perda."])
    P("Denomina-se época uma passagem completa por todas as imagens do conjunto de "
      "treinamento.")

    H(3, "Função de perda")
    P("Empregou-se a entropia cruzada (*cross-entropy*), que, para uma imagem cuja classe "
      "verdadeira é *y*, assume a forma da {eq:ce}:")
    b.append(("eq", mr("L") + mr("=") + mr("−") + mr("ln", True) +
              mpar(msub(mr("p"), mr("y"))), "ce"))
    P("Se o modelo atribui probabilidade 0,99 à classe correta, a perda é de 0,01; se "
      "atribui 0,10, a perda é de 2,30. A função penaliza, assim, de forma acentuada as "
      "predições incorretas feitas com alta confiança.")

    H(3, "Otimizador")
    P("Utilizou-se o otimizador AdamW (LOSHCHILOV; HUTTER, 2019), que adapta o tamanho do "
      "passo de cada parâmetro individualmente, com base em médias móveis do gradiente e "
      "de seu quadrado, e aplica o decaimento de pesos (*weight decay*) de forma "
      "desacoplada da atualização pelo gradiente. O decaimento de pesos favorece "
      "parâmetros de pequena magnitude, o que reduz a tendência ao sobreajuste.")

    H(3, "Taxa de aprendizado")
    P("A taxa de aprendizado determina a magnitude de cada passo de atualização. Adotou-se "
      "um período inicial de aquecimento (*warmup*) de três épocas, durante o qual a taxa "
      "cresce gradualmente até o valor nominal, seguido de decaimento cossenoidal "
      "(LOSHCHILOV; HUTTER, 2017), descrito pela {eq:cos}:")
    b.append(("eq",
              mr("η") + mpar(mr("t")) + mr("=") + msub(mr("η"), mr("0")) + mr("⋅") +
              mpar(mr("1−") + mpar(mr("1−") + mr("f")) + mr("⋅") +
                   mfrac(mr("1−") + mr("cos", True) + mpar(mfrac(mr("π") + mr("t"), mr("T"))),
                         mr("2"))),
              "cos"))
    P("em que *η*~0~ é a taxa inicial, *f* a fração final, *t* a época (contada a partir "
      "de zero) e *T* o número máximo de épocas. Com *η*~0~ = 0,001, *f* = 0,01 e *T* = "
      "100, a taxa decresce suavemente até 0,00001. O aquecimento evita que os gradientes "
      "elevados das primeiras iterações degradem o conhecimento previamente adquirido "
      "pelo modelo.")

    H(3, "Sobreajuste e regularização")
    P("O sobreajuste (*overfitting*) ocorre quando o modelo memoriza as particularidades "
      "das imagens de treinamento, em vez de aprender padrões generalizáveis, e se "
      "manifesta por desempenho elevado no treinamento e inferior em imagens novas. "
      "Foram adotados quatro mecanismos de regularização: o aumento de dados "
      "({sec:sec_aumento}); o *dropout*, que anula aleatoriamente uma fração das "
      "características durante o treinamento; o decaimento de pesos; e a parada "
      "antecipada (*early stopping*), que interrompe o treinamento quando o desempenho "
      "no subconjunto de validação deixa de melhorar por um número predefinido de épocas, "
      "denominado paciência, preservando os parâmetros da melhor época.")

    H(2, "Transferência de aprendizado", "sec_transfer")
    P("O treinamento não partiu de parâmetros aleatórios. Utilizaram-se como ponto de "
      "partida os parâmetros do modelo previamente treinado sobre o conjunto ImageNet, "
      "composto por aproximadamente 1,28 milhão de imagens de 1.000 categorias. Apenas a "
      "camada linear final, originalmente com 1.000 saídas, foi substituída por outra com "
      "três saídas; os demais parâmetros foram aproveitados e ajustados (*fine-tuning*).")
    P("A estratégia fundamenta-se na constatação de que as camadas iniciais de redes "
      "treinadas sobre grandes conjuntos aprendem detectores genéricos de bordas, cores e "
      "texturas, úteis a domínios visuais diversos (YOSINSKI et al., 2014). O "
      "aproveitamento desse conhecimento reduz substancialmente o volume de dados "
      "necessário, fator decisivo diante das 1.566 imagens de treinamento disponíveis. "
      "Mohanty, Hughes e Salathé (2016) demonstraram a eficácia da transferência de "
      "aprendizado especificamente no diagnóstico de doenças de plantas por imagem.")

    H(2, "Explicabilidade por Grad-CAM", "sec_gradcam")
    P("Para permitir a inspeção das decisões do modelo, adotou-se a técnica Grad-CAM "
      "(*Gradient-weighted Class Activation Mapping*), proposta por Selvaraju et al. "
      "(2017). A técnica calcula a importância de cada canal *k* da última camada "
      "convolucional para a classe *c*, conforme a {eq:alfa}:")
    b.append(("eq",
              msubsup(mr("α"), mr("k"), mr("c")) + mr("=") + mfrac(mr("1"), mr("Z")) +
              msoma(mr("i"), "", msoma(mr("j"), "",
                    mfrac(mr("∂") + msup(mr("y"), mr("c")),
                          mr("∂") + msubsup(mr("A"), mr("ij"), mr("k"))))),
              "alfa"))
    P("em que *y*^c^ é a pontuação da classe *c*, *A*^k^ o mapa de ativação do canal *k* "
      "e *Z* o número de posições espaciais. O mapa de calor resulta da combinação "
      "ponderada dos mapas de ativação, mantidos apenas os valores positivos, conforme a "
      "{eq:gradcam}:")
    b.append(("eq",
              msubsup(mr("L"), mr("Grad-CAM", True), mr("c")) + mr("=") + mr("ReLU", True) +
              mpar(msoma(mr("k"), "", msubsup(mr("α"), mr("k"), mr("c")) + msup(mr("A"), mr("k")))),
              "gradcam"))
    P("No modelo adotado, o mapa possui resolução de 7 × 7 e corresponde ao recorte "
      "central de 224 × 224 *pixels* analisado pela rede ({sec:sec_preproc}). Para a "
      "sobreposição, é ampliado até as dimensões dessa região da imagem original, e as "
      "faixas descartadas pelo recorte permanecem sem sobreposição; esticá-lo sobre a "
      "imagem inteira deslocaria as regiões destacadas em imagens não quadradas, como as "
      "de 1280 × 720 *pixels* da câmera do sistema. As regiões destacadas correspondem às que mais "
      "influenciaram a decisão. A técnica atende a duas finalidades: fornece indicação "
      "aproximada da localização da lesão, sem exigir anotação manual; e permite "
      "identificar o aprendizado por atalho (*shortcut learning*), situação em que o "
      "modelo acerta com base em elementos irrelevantes, como o fundo da imagem "
      "(GEIRHOS et al., 2020). Cumpre observar que o Grad-CAM indica onde o modelo "
      "concentrou sua atenção, mas não qual característica — cor, textura ou forma — "
      "determinou a decisão.")

    H(2, "Métricas de avaliação", "sec_metricas")
    P("O desempenho foi avaliado por métricas derivadas da matriz de confusão, tabela que "
      "cruza a classe real de cada imagem com a classe predita. Para cada classe, "
      "denominam-se verdadeiros positivos (VP) as imagens da classe corretamente "
      "identificadas; falsos positivos (FP) as imagens de outras classes atribuídas a "
      "ela; e falsos negativos (FN) as imagens da classe atribuídas a outras. As métricas "
      "são definidas pelas Equações {eqn:prec} a {eqn:f1}:")
    b.append(("eq", mr("Precisão", True) + mr("=") + mfrac(mr("VP", True), mr("VP+FP", True)), "prec"))
    b.append(("eq", mr("Revocação", True) + mr("=") + mfrac(mr("VP", True), mr("VP+FN", True)), "rev"))
    b.append(("eq", msub(mr("F", True), mr("1", True)) + mr("=") +
              mfrac(mr("2⋅", True) + mr("Precisão⋅Revocação", True),
                    mr("Precisão+Revocação", True)), "f1"))
    P("A precisão indica, entre as imagens atribuídas a uma classe, a proporção que de fato "
      "pertence a ela; precisão elevada corresponde a poucos alarmes falsos. A revocação "
      "(*recall*) indica, entre as imagens de uma classe, a proporção identificada pelo "
      "modelo; revocação elevada corresponde a poucas doenças não detectadas. O F1-*score* "
      "é a média harmônica das duas. A acurácia é a proporção de classificações corretas "
      "sobre o total, e a média macro é a média aritmética de uma métrica entre as três "
      "classes, sem ponderação pelo número de imagens, o que atribui igual importância à "
      "classe menos numerosa.")

    # ================================================================== #
    H(1, "DESENVOLVIMENTO DO MODELO DE CLASSIFICAÇÃO", "cap_metodo")
    # ================================================================== #
    P("Este capítulo descreve os materiais e os procedimentos empregados no "
      "desenvolvimento do modelo: a definição da tarefa, o tratamento do conjunto de "
      "dados, o pré-processamento, o aumento de dados, o treinamento na plataforma Google "
      "Colab, o procedimento de avaliação e a integração ao sistema. Descreve, ainda, a "
      "etapa de verificação de que a imagem contém uma folha de alface, cujos resultados "
      "constam da {sec:sec_res_dominio}, e a implantação do serviço em contêiner na "
      "nuvem.")

    H(2, "Definição da tarefa", "sec_tarefa")
    P("O projeto original previa a detecção de objetos, com delimitação das lesões por "
      "caixas delimitadoras. A análise do conjunto de dados obtido demonstrou, contudo, a "
      "inexistência de anotações de localização: as imagens são rotuladas apenas em nível "
      "de imagem inteira, por meio da organização em diretórios por classe. Adotou-se, "
      "por conseguinte, a tarefa de classificação de imagens, que atende ao requisito "
      "RF02 — identificar se a folha está saudável ou com infecção bacteriana ou fúngica — "
      "e aos casos de teste CT01 a CT03, formulados em termos de classe e confiança. A "
      "orientação espacial prevista para a interface de monitoramento foi recuperada pela "
      "técnica Grad-CAM ({sec:sec_gradcam}).")

    H(2, "Conjunto de dados", "sec_dados")
    H(3, "Origem e composição")
    P("Utilizou-se conjunto público de imagens de folhas de alface obtido na plataforma "
      "Kaggle [[preencher:referência completa do conjunto de dados — autor, título, ano e "
      "endereço eletrônico]], organizado em diretórios por classe e previamente dividido "
      "em subconjuntos de treinamento, validação e teste, conforme a {tab:original}.")
    b.append(("tab", "original", "Composição do conjunto de dados original",
              ["Subconjunto", "Bacteriana", "Fúngica", "Saudável", "Total", "Participação"],
              [["Treinamento", "732", "581", "1.049", "2.362", "84,0%"],
               ["Validação", "75", "75", "75", "225", "8,0%"],
               ["Teste", "75", "75", "76", "226", "8,0%"],
               ["**Total**", "**882**", "**731**", "**1.200**", "**2.813**", "**100,0%**"]],
              [2071, 1400, 1400, 1400, 1400, 1400], "lrrrrr",
              "Fonte: elaborado pelos autores (2026), a partir do conjunto de dados original."))
    P("As imagens apresentam resolução variável, entre 129 e 2.048 *pixels* em cada "
      "dimensão, com mediana de 474 × 355 *pixels*. Distribuem-se em 1.826 arquivos PNG e "
      "987 arquivos JPEG; 2.812 encontram-se no modo de cor RGB e uma no modo RGBA; "
      "apenas 354 são quadradas.")

    H(3, "Auditoria do conjunto de dados")
    P("Previamente ao treinamento, o conjunto foi submetido a auditoria sistemática, "
      "mediante o cálculo do resumo criptográfico MD5 de cada arquivo — que identifica "
      "arquivos idênticos byte a byte — e a análise dos padrões de nomenclatura. Os "
      "problemas identificados constam do {qua:problemas}.")
    b.append(("qua", "problemas", "Problemas identificados na auditoria do conjunto de dados",
              ["Código", "Problema", "Evidência"],
              [["P1", "Vazamento por duplicatas exatas entre subconjuntos", "104 grupos de arquivos idênticos presentes simultaneamente no treinamento e na validação ou no teste"],
               ["P2", "Vazamento por variantes da mesma folha", "115 grupos de origem distribuídos por mais de um subconjunto"],
               ["P3", "Baixa diversidade da classe bacteriana", "As 282 imagens bacterianas não sintéticas derivam de aproximadamente 53 folhas"],
               ["P4", "Predomínio de imagens sintéticas na avaliação", "800 imagens geradas artificialmente, correspondentes a dois terços dos subconjuntos de validação e teste"],
               ["P5", "Proporções divergentes da metodologia", "Divisão de 84/8/8, em vez de 70/15/15"],
               ["P6", "Desbalanceamento entre classes", "Participação de 44,4%, 31,0% e 24,6% no treinamento"]],
              [1000, 3300, 4771], "cll", FONTE_PADRAO))
    P("O problema P1 é o mais grave do ponto de vista metodológico. Dos 2.813 arquivos, "
      "apenas 2.216 possuem conteúdo distinto, o que revela 597 cópias redundantes. A "
      "avaliação de um modelo sobre imagens presentes em seu próprio conjunto de "
      "treinamento mede a capacidade de memorização, e não a de generalização, "
      "invalidando as conclusões sobre desempenho (KAUFMAN et al., 2012). O problema P2 "
      "decorre de variantes de uma mesma folha, geradas por transformações e gravadas "
      "como arquivos distintos, que foram distribuídas entre subconjuntos diferentes: "
      "ainda que não idênticas, representam a mesma unidade experimental.")

    H(3, "Procedimento de preparação")
    P("Desenvolveu-se rotina automatizada de preparação, executada obrigatoriamente antes "
      "de qualquer treinamento. O conjunto original foi tratado como fonte imutável, sem "
      "escrita em nenhuma etapa. O procedimento compreende:")
    AL(["inventário determinístico de todos os arquivos, com registro de classe, resumo "
        "criptográfico, dimensões, formato e modo de cor;",
        "unificação de grupos de origem que compartilham arquivos idênticos, por meio de "
        "componentes conexas, o que revelou sete casos de nomes distintos referentes à "
        "mesma folha;",
        "remoção das 597 duplicatas exatas, mantida uma única cópia de cada conteúdo, "
        "escolhida por critério determinístico;",
        "nova divisão nas proporções de 70%, 15% e 15%, estratificada por classe e "
        "disjunta por grupo de folha de origem (*StratifiedGroupKFold*; PEDREGOSA et al., "
        "2011), com semente aleatória fixa igual a 42;",
        "verificação de integridade, que interrompe o processo com erro caso algum "
        "conteúdo ou grupo de origem ocorra em mais de um subconjunto;",
        "conversão de todas as imagens para o modo RGB e gravação em formato JPEG com "
        "qualidade 95, sem redimensionamento."])

    H(3, "Conjunto de dados resultante", "sec_dados_resultante")
    P("O conjunto resultante contém 2.216 imagens, distribuídas conforme a {tab:limpo}. A "
      "verificação de integridade confirmou a ausência total de vazamento entre os "
      "subconjuntos.")
    b.append(("tab", "limpo", "Composição do conjunto de dados após a preparação",
              ["Subconjunto", "Bacteriana", "Fúngica", "Saudável", "Total", "Participação"],
              [["Treinamento", "510", "390", "666", "1.566", "70,7%"],
               ["Validação", "114", "75", "137", "326", "14,7%"],
               ["Teste", "106", "78", "140", "324", "14,6%"],
               ["**Total**", "**730**", "**543**", "**943**", "**2.216**", "**100,0%**"]],
              [2071, 1400, 1400, 1400, 1400, 1400], "lrrrrr", FONTE_PADRAO))
    P("A estratificação preservou a proporção entre classes nos três subconjuntos: a "
      "classe bacteriana representa 32,6%, 35,0% e 32,7%; a fúngica, 24,9%, 23,0% e "
      "24,1%; e a saudável, 42,5%, 42,0% e 43,2% dos subconjuntos de treinamento, "
      "validação e teste, respectivamente.")

    H(2, "Pré-processamento das imagens", "sec_preproc")
    P("Toda imagem submetida ao modelo, tanto na avaliação quanto em operação, passa "
      "pelas etapas representadas na {fig:fluxo}. Inicialmente, a imagem é validada: "
      "arquivos vazios, truncados, corrompidos ou em formato não suportado são rejeitados "
      "com erro explícito e jamais recebem classificação. Essa decisão decorre da análise "
      "dos modos de falha do sistema: a classificação silenciosa de uma imagem corrompida "
      "como folha saudável suprimiria o alerta sem qualquer indicação de anomalia.")
    b.append(("fig", "fluxo", "Etapas do processamento de uma imagem pelo módulo de inteligência artificial",
              "fluxo", 12.5, FONTE_PADRAO))
    P("Em seguida, a imagem é convertida para o padrão RGB; seu lado menor é "
      "redimensionado para 224 *pixels*, preservada a proporção; e recorta-se a região "
      "quadrada central de 224 × 224 *pixels*. Por fim, os valores dos *pixels* são "
      "divididos por 255, resultando em um tensor de dimensão 3 × 224 × 224 com valores no "
      "intervalo de 0 a 1. A resolução de 224 *pixels* é a nativa da tarefa de "
      "classificação e inferior à mediana das imagens do conjunto.")

    H(2, "Aumento de dados", "sec_aumento")
    P("O aumento de dados (*data augmentation*) consiste em aplicar transformações "
      "aleatórias às imagens de treinamento, de modo que o modelo observe, a cada época, "
      "versões diferentes de cada imagem. A técnica amplia a variabilidade de "
      "apresentação e reduz o sobreajuste. Neste trabalho, as transformações foram "
      "aplicadas exclusivamente em memória, no momento do treinamento, sem gravação das "
      "imagens transformadas — precisamente a prática que originou os problemas P1 e P2 no "
      "conjunto original. As transformações efetivamente aplicadas pela biblioteca na "
      "tarefa de classificação constam da {tab:aumento}.")
    b.append(("tab", "aumento", "Transformações de aumento de dados aplicadas no treinamento",
              ["Ordem", "Transformação", "Parâmetro", "Justificativa no domínio"],
              [["1", "Recorte aleatório redimensionado para 224 × 224", "Área de 60% a 100% da imagem; proporção de 3:4 a 4:3", "Variação da distância e do enquadramento da câmera"],
               ["2", "Espelhamento horizontal", "Probabilidade de 0,5", "Folhas não possuem orientação preferencial"],
               ["3", "Espelhamento vertical", "Probabilidade de 0,3", "A câmera pode assumir orientações diversas"],
               ["4", "RandAugment", "Duas operações sorteadas por imagem, magnitude 9", "Variações de iluminação, contraste, cor e posição"],
               ["5", "Apagamento aleatório", "Probabilidade de 0,2", "Oclusão parcial por folhas sobrepostas"]],
              [800, 2600, 2700, 2971], "clll", FONTE_PADRAO))
    P("O RandAugment (CUBUK et al., 2020) sorteia, para cada imagem, duas operações entre "
      "identidade, cisalhamento, translação, rotação, brilho, saturação de cor, contraste, "
      "nitidez, posterização, solarização, autocontraste e equalização, aplicadas com "
      "intensidade fixa. Na configuração padrão adotada, a rotação alcança até 9° e o "
      "fator de brilho varia em até 27% para mais ou para menos. A variação de brilho e "
      "de contraste contribui para mitigar o efeito das condições de iluminação da "
      "estufa, que alteram as características cromáticas das lesões.")
    P("Registre-se que, na tarefa de classificação da biblioteca empregada, os parâmetros "
      "específicos de rotação e translação não são utilizados — aplicam-se apenas às "
      "tarefas de detecção —, e a variação de matiz, saturação e brilho por parâmetros "
      "próprios é desativada quando o RandAugment está ativo, por sobrepor-se a ele. Nas "
      "etapas de validação, teste e operação, nenhuma transformação aleatória é "
      "aplicada.")

    H(2, "Ambiente de treinamento: Google Colab", "sec_colab")
    H(3, "Justificativa")
    P("O ambiente de desenvolvimento local, cujas características constam da "
      "{tab:ambiente}, não dispõe de unidade de processamento gráfico (GPU) compatível com "
      "a plataforma CUDA. Uma medição realizada nesse ambiente registrou duração de "
      "aproximadamente 3,8 minutos por época, o que projetava cerca de seis horas para "
      "100 épocas.")
    b.append(("tab", "ambiente", "Ambiente computacional local",
              ["Componente", "Especificação"],
              [["Sistema operacional", "Windows 11"],
               ["Processador", "AMD Ryzen 5 3500U"],
               ["GPU compatível com CUDA", "Não disponível"],
               ["Linguagem", "Python 3.13.5"],
               ["Aprendizado profundo", "PyTorch 2.13.0 (CPU)"],
               ["Visão computacional", "Ultralytics 8.4.95"],
               ["Métricas e divisão dos dados", "scikit-learn 1.6.1"],
               ["Execução do modelo exportado", "ONNX Runtime 1.28.0"]],
              [4200, 4871], "ll", FONTE_PADRAO))
    P("Optou-se, por isso, pelo treinamento na plataforma Google Colab, ambiente de "
      "cadernos interativos (*notebooks*) Jupyter hospedado pela Google, que disponibiliza "
      "GPU sob demanda [[preencher:modelo da GPU disponibilizada na execução]]. As redes "
      "convolucionais são compostas majoritariamente por multiplicações de matrizes, "
      "operação que a GPU executa em paralelo em milhares de núcleos de processamento. "
      "No Colab, o treinamento completo foi concluído em {tempo_min} minutos.")
    P("Adotou-se a seguinte divisão de tarefas: apenas o treinamento foi executado no "
      "Colab, enquanto a preparação do conjunto de dados, a avaliação, a exportação e a "
      "medição de latência foram executadas no ambiente local. Assim, as métricas "
      "oficiais foram medidas em processador central (CPU), configuração equivalente à do "
      "servidor em que o modelo opera.")

    H(3, "Procedimento de execução")
    P("A {fig:colab} representa o fluxo de trabalho entre o ambiente local e o Google "
      "Colab.")
    b.append(("fig", "colab", "Fluxo de trabalho entre o ambiente local e o Google Colab",
              "colab", 14.0, FONTE_PADRAO))
    P("No Colab, o caderno de treinamento foi organizado em células executadas na "
      "seguinte ordem:")
    AL(["verificação da GPU: consulta ao utilitário *nvidia-smi* e confirmação, via "
        "PyTorch, da disponibilidade da plataforma CUDA, com interrupção do processo caso "
        "nenhuma GPU seja detectada, evitando treinamento acidental em CPU;",
        "instalação da biblioteca Ultralytics, na versão 8.4.95 ou superior "
        "[[preencher:versões exatas do Ultralytics e do PyTorch no Colab]];",
        "envio do conjunto de dados preparado, compactado em arquivo de aproximadamente "
        "86 MB, por *upload* direto ou por meio do Google Drive;",
        "extração e conferência automática: verificação da existência dos subconjuntos "
        "de treinamento, validação e teste, da presença exata das três classes em cada um "
        "e da contagem total de 2.216 imagens (1.566, 326 e 324), com interrupção diante "
        "de qualquer divergência;",
        "treinamento do modelo com os hiperparâmetros da {tab:hiper};",
        "conferência da melhor acurácia no subconjunto de validação e da época "
        "correspondente;",
        "transferência dos parâmetros da melhor época e do histórico de treinamento para "
        "o ambiente local."])

    H(2, "Configuração do treinamento", "sec_config")
    P("Os hiperparâmetros adotados, registrados pela própria biblioteca ao término da "
      "execução, constam da {tab:hiper}. Em relação à configuração do ambiente local, "
      "alteraram-se apenas os parâmetros dependentes de *hardware*: dispositivo de "
      "execução, precisão mista automática e número de processos de carregamento de "
      "dados.")
    b.append(("tab", "hiper", "Hiperparâmetros do treinamento",
              ["Hiperparâmetro", "Valor", "Finalidade"],
              [["Modelo de partida", "YOLO11n-cls pré-treinado no ImageNet", "Transferência de aprendizado"],
               ["Épocas máximas", "100", "Limite superior de iterações"],
               ["Paciência", "20 épocas", "Parada antecipada"],
               ["Tamanho do lote", "32 imagens", "Imagens por passo de propagação"],
               ["Tamanho nominal do lote", "64 imagens", "Acúmulo de gradientes de dois lotes por atualização"],
               ["Resolução de entrada", "224 × 224 *pixels*", "Dimensão da imagem processada"],
               ["Otimizador", "AdamW (β~1~ = 0,937; β~2~ = 0,999)", "Atualização dos parâmetros"],
               ["Taxa de aprendizado inicial", "0,001", "Magnitude do passo"],
               ["Fração final da taxa", "0,01", "Taxa final de 0,00001"],
               ["Programação da taxa", "Cossenoidal, com 3 épocas de aquecimento", "Decaimento progressivo"],
               ["Decaimento de pesos", "0,0005", "Regularização"],
               ["*Dropout*", "0,1", "Regularização"],
               ["Semente aleatória", "42", "Reprodutibilidade"],
               ["Execução determinística", "Ativada", "Reprodutibilidade"],
               ["Dispositivo", "GPU", "Aceleração do treinamento"],
               ["Precisão mista automática", "Ativada", "Uso de meia precisão onde seguro, na GPU"],
               ["Processos de carregamento", "2", "Leitura paralela das imagens"]],
              [2900, 3300, 2871], "lll", FONTE_PADRAO))
    P("A técnica de suavização de rótulos (*label smoothing*), inicialmente prevista, não "
      "foi aplicada: o parâmetro correspondente foi removido da biblioteca Ultralytics a "
      "partir da versão 8.4, sendo aceito na chamada, mas descartado sem aviso. A "
      "configuração registrada pela biblioteca ao término do treinamento confirma a "
      "ausência do parâmetro.")
    P("Com 1.566 imagens e lotes de 32, cada época compreende 49 lotes. Os gradientes de "
      "dois lotes consecutivos são acumulados antes de cada atualização dos parâmetros, o "
      "que equivale a um lote efetivo de 64 imagens. Ao final de cada época, o modelo é "
      "avaliado sobre as 326 imagens de validação; os parâmetros da época de maior "
      "acurácia nesse subconjunto são preservados como modelo final.")

    H(2, "Procedimento de avaliação", "sec_avaliacao")
    P("A avaliação observou três princípios. O primeiro é a separação estrita entre ajuste "
      "e medição: o subconjunto de validação foi empregado para a seleção da melhor época, "
      "e o subconjunto de teste foi medido uma única vez, ao final do processo, sem que "
      "nenhuma decisão fosse tomada com base nele. A avaliação repetida sobre o teste o "
      "converteria em um segundo conjunto de validação, comprometendo a imparcialidade da "
      "medida final.")
    P("O segundo é a medição no caminho de execução real: a latência foi aferida sobre a "
      "mesma função invocada pelo servidor de aplicação, com uma imagem por chamada, sobre "
      "100 imagens do subconjunto de teste, descartadas as dez primeiras inferências, "
      "destinadas à estabilização de memória e de *caches*. O terceiro é a integridade do "
      "conjunto de medição: todas as métricas foram obtidas sobre o conjunto preparado, "
      "isento de vazamento.")
    P("Complementarmente, o comportamento do *software* foi verificado por 221 testes "
      "automatizados, que abrangem a correspondência dos rótulos de classe com os valores "
      "aceitos pelo banco de dados, a ausência de vazamento e a reprodutibilidade da "
      "divisão do conjunto, o formato da saída da inferência, o tratamento de imagens "
      "inválidas, a projeção do mapa Grad-CAM sobre a região analisada e, no verificador "
      "de domínio ({sec:sec_dominio}), a montagem e a divisão do seu conjunto de dados, a "
      "cascata e a resposta às imagens recusadas.")

    H(2, "Exportação e integração ao sistema", "sec_integracao")
    P("O modelo treinado foi convertido para o formato ONNX (*Open Neural Network "
      "Exchange*), que permite sua execução pelo ONNX Runtime sem dependência do "
      "PyTorch e com menor latência de inferência ({sec:sec_res_latencia}). No serviço "
      "implantado, contudo, o PyTorch é mantido e o classificador é executado no formato "
      "original, por exigência do Grad-CAM e do verificador de domínio; o formato ONNX "
      "permanece disponível por configuração ({sec:sec_imagem}). A equivalência numérica entre o modelo convertido e o original foi verificada, e "
      "não presumida, mediante comparação das saídas sobre 20 imagens do subconjunto de "
      "teste, com tolerância absoluta de 10^−3^.")
    P("O modelo é disponibilizado ao restante do sistema por meio de uma interface de "
      "programação de aplicações (API) HTTP, que recebe a imagem capturada pela câmera do "
      "sistema, uma Logitech C270, e retorna a classe, a confiança, as probabilidades das "
      "três classes, o tempo de inferência e a indicação de alerta. O alerta é emitido "
      "somente quando a classe predita corresponde a uma doença e a confiança atinge o "
      "limiar de 0,85, estabelecido pelos critérios de aceitação CT01 e CT02. Folhas "
      "classificadas como saudáveis e detecções de doença com confiança inferior ao limiar "
      "não geram notificação, o que reduz a incidência de alertas espúrios. Com o "
      "verificador de domínio ativo, a imagem sem alface recebe a resposta descrita na "
      "{sec:sec_dom_arquitetura}.")

    H(2, "Verificação de domínio: identificação da alface", "sec_dominio")
    P("Esta seção descreve a etapa de verificação de domínio, que antecede o classificador "
      "e decide se a imagem contém alface. Foram desenvolvidas duas versões do verificador, "
      "ambas treinadas e medidas segundo os procedimentos aqui descritos; os resultados "
      "constam da {sec:sec_res_dominio}. A implantação da segunda versão foi decidida após "
      "o conhecimento do resultado do teste, embora o critério fixado antes dele não tenha "
      "sido atingido, conforme registrado na {sec:sec_dom_decisao}.")

    H(3, "O problema do conjunto fechado de classes")
    P("O modelo desenvolvido é um classificador de conjunto fechado: conhece apenas três "
      "respostas possíveis e, em razão da função *softmax* ({sec:sec_softmax}), sempre "
      "atribui a imagem a uma delas. Em operação na estufa, entretanto, a câmera pode "
      "capturar imagens que não contêm folhas de alface: o solo, a bancada ou a "
      "estrutura, em caso de desalinhamento; a mão do agricultor, ferramentas ou insetos "
      "à frente da lente; plantas espontâneas; ou imagens escuras e desfocadas. Em todos "
      "esses casos, o modelo atual emitiria um diagnóstico, possivelmente um alerta falso, "
      "com consequente aplicação desnecessária de defensivo.")
    P("A rejeição de imagens com confiança reduzida não constitui solução suficiente. "
      "Hendrycks e Gimpel (2017) demonstraram que redes neurais frequentemente atribuem "
      "confiança elevada a imagens alheias ao domínio de treinamento, e Guo et al. (2017) "
      "documentaram a tendência das redes modernas à confiança excessiva. O próprio "
      "modelo deste trabalho registrou erros com confiança igual a 1,000 em imagens do "
      "domínio ({sec:sec_erros}).")

    H(3, "Arquitetura adotada", "sec_dom_arquitetura")
    P("A inferência foi organizada em duas etapas sucessivas (cascata), representadas na "
      "{fig:cascata}. Na primeira, um verificador de domínio — um segundo modelo "
      "YOLO11n-cls, treinado para a questão binária “a imagem contém uma folha de "
      "alface?” — decide se a imagem segue para diagnóstico: a imagem é aceita quando a "
      "probabilidade atribuída à classe alface atinge o limiar de aceitação "
      "({sec:sec_dom_treino}). Na segunda, o classificador de doenças desenvolvido neste "
      "trabalho, sem alteração, classifica as imagens aceitas. Imagens recusadas pelo "
      "verificador não recebem diagnóstico nem geram alerta.")
    b.append(("fig", "cascata", "Arquitetura de inferência em duas etapas",
              "cascata", 13.0, FONTE_PADRAO))
    P("Na interface HTTP ({sec:sec_integracao}), a imagem recusada pelo verificador "
      "recebe resposta bem-sucedida, com a classe “" + CLASSE_NAO_ALFACE + "” e o rótulo "
      "de exibição “" + EXIBICAO_NAO_ALFACE + "” no mesmo campo que, nas demais "
      "respostas, traz a classe da doença. Nesse caso, a confiança informada é o "
      "complemento da probabilidade de alface atribuída pelo verificador, não há "
      "probabilidades das classes de doença, e os indicadores de patógeno e de alerta são "
      "falsos. A resposta inclui ainda a probabilidade de alface, o limiar e o tempo de "
      "inferência do verificador. Imagens inválidas continuam rejeitadas com erro antes de "
      "qualquer modelo. Se o verificador estiver configurado e não puder ser carregado, o "
      "serviço recusa a inferência em vez de prosseguir sem ele, para que nenhuma imagem "
      "sem alface receba diagnóstico.")
    P("A separação em dois modelos preserva o classificador já validado e seus "
      "resultados e atribui a cada modelo uma única responsabilidade. O conjunto de "
      "classes de doença não se altera; a gravação das respostas “" + CLASSE_NAO_ALFACE +
      "” exige, contudo, que a coluna de classe do banco de dados aceite esse valor, ou "
      "que tais respostas não sejam gravadas ({sec:sec_lim}).")

    H(3, "Critérios visuais para a identificação da alface")
    P("O verificador emprega o mesmo mecanismo descrito na {sec:sec_mecanismo}: "
      "aprende, a partir de exemplos, as características que distinguem imagens de "
      "alface das demais, sem regras manuais de cor. O {qua:criterios_alface} apresenta as "
      "características visuais disponíveis para esse aprendizado.")
    b.append(("qua", "criterios_alface", "Características visuais que distinguem a alface de outros elementos da estufa",
              ["Característica", "Manifestação na alface", "Elementos contrastantes"],
              [["Cor", "Tons de verde, do claro ao escuro; roxo-avermelhado em cultivares roxas", "Solo e substrato (marrom); bancada e estrutura (cinza); pele"],
               ["Textura", "Superfície lisa, crespa ou bolhosa, com brilho característico", "Plástico, metal, terra"],
               ["Forma", "Folhas largas, de bordas onduladas ou recortadas, dispostas em roseta", "Folhas estreitas de gramíneas; folhas compostas de outras culturas"],
               ["Nervação", "Nervura central evidente, com ramificações", "Objetos sem nervação"],
               ["Contexto", "Folha ou planta ocupando a maior parte do enquadramento", "Estrutura da estufa ocupando o enquadramento"]],
              [1700, 3800, 3571], "lll", FONTE_PADRAO))
    P("Um critério exclusivamente cromático, como a exigência de predominância de verde, "
      "seria inadequado por quatro razões: rejeitaria folhas com míldio avançado, que "
      "apresentam extensas áreas amareladas e necrosadas — suprimindo o alerta justamente "
      "nos casos mais graves; rejeitaria cultivares roxas; aceitaria plantas espontâneas "
      "verdes; e seria sensível às variações de iluminação. A rede convolucional, ao "
      "combinar cor, textura e forma, pode tratar tais casos, desde que estejam "
      "representados no conjunto de treinamento.")

    H(3, "Alternativas consideradas")
    P("O {qua:alternativas} sintetiza as alternativas avaliadas para a verificação de "
      "domínio.")
    b.append(("qua", "alternativas", "Alternativas para a verificação de domínio",
              ["Alternativa", "Funcionamento", "Limitação", "Decisão"],
              [["Verificador binário em cascata", "Segundo modelo YOLO11n-cls anterior ao classificador", "Exige coleta de imagens negativas; duplica o custo de inferência", "Adotada"],
               ["Quarta classe no modelo atual", "Retreinamento com a classe “não alface”", "Altera o conjunto de classes do classificador e invalida os resultados medidos", "Descartada"],
               ["Limiar de confiança", "Rejeição de imagens com confiança reduzida", "Pouco confiável para imagens alheias ao domínio", "Complementar"],
               ["Distância no espaço de características", "Comparação do vetor de 1.280 características com a distribuição do treinamento (LEE et al., 2018)", "Requer calibração do limiar de distância", "Complementar futura"],
               ["Modelo visão-linguagem", "Consulta textual a modelo pré-treinado de grande porte", "Incompatível com os requisitos de latência e de custo em CPU", "Descartada"]],
              [2200, 2700, 2671, 1500], "llll", FONTE_PADRAO))

    H(3, "Critérios de aceitação", "sec_dom_criterios")
    P("A implantação do verificador foi condicionada ao atendimento simultâneo dos "
      "critérios da {tab:criterios_verif}, medidos no subconjunto de teste da versão "
      "correspondente. O primeiro critério, medido apenas sobre as folhas doentes do "
      "conjunto do Kaggle, protege a função do sistema: a recusa de uma folha doente "
      "suprime o alerta. O segundo limita os diagnósticos atribuídos a imagens sem alface, "
      "e o terceiro preserva o requisito RNF01 para as duas etapas somadas. Na segunda "
      "versão, acrescentaram-se dois critérios sobre as imagens da câmera do sistema, "
      "ausentes da avaliação da primeira: sem eles, o verificador poderia atingir as metas "
      "sem evidência de funcionar com a câmera em operação, onde a primeira versão falhou "
      "({sec:sec_dom_teste}).")
    b.append(("tab", "criterios_verif", "Critérios de aceitação do verificador de domínio",
              ["Critério", "Métrica", "Meta", "Mínimo no teste da versão 2"],
              [["(1) Não suprimir doenças", "Revocação da classe alface em folhas doentes do conjunto do Kaggle", "≥ 99%", "183 de 184"],
               ["(2) Rejeitar imagens alheias", "Revocação da classe não alface, todas as fontes", "≥ 95%", "337 de 354"],
               ["(3) Tempo de resposta", "Latência mediana das duas etapas", "< 100 ms", "—"],
               ["(4) Aceitar alface na câmera do sistema", "Revocação da classe alface nas imagens da C270", "≥ 95%", "72 de 75"],
               ["(5) Rejeitar imagens alheias na câmera do sistema", "Revocação da classe não alface nas imagens da C270", "≥ 95%", "46 de 48"]],
              [2600, 3271, 1100, 2100], "llcc",
              "Fonte: elaborado pelos autores (2026). Nota: os três primeiros critérios foram "
              "fixados em 18/10/2026, para a primeira versão, e mantidos para a segunda em "
              "25/10/2026; os dois últimos foram acrescentados em 25/10/2026, apenas para a "
              "segunda versão. Todos foram fixados antes da medição no subconjunto de teste da "
              "versão a que se aplicam."))
    P("As metas foram fixadas antes de qualquer medição no subconjunto de teste, e a "
      "comparação usa a razão exata de acertos, e não a taxa arredondada. O subconjunto de "
      "teste de cada versão foi medido uma única vez: o programa de avaliação recusa uma "
      "segunda medição da mesma versão, e nenhum ajuste foi feito com base nos resultados "
      "obtidos. As taxas são acompanhadas do intervalo de confiança de 95% (IC95) "
      "calculado pelo método de Wilson (WILSON, 1927), adequado a proporções próximas de 0 "
      "ou de 1 e a amostras pequenas. A latência das duas etapas é medida com o "
      "procedimento da {sec:sec_avaliacao}, sobre 100 imagens do subconjunto de validação, "
      "com pesos PyTorch em CPU no ambiente local.")

    H(3, "Conjunto de dados do verificador", "sec_dom_dados")
    P("A classe positiva (alface) reúne as 2.216 imagens do conjunto da {sec:sec_dados}, "
      "das três classes, e fotografias de alface capturadas pela câmera do sistema, uma "
      "Logitech C270. As imagens do Kaggle mantêm o subconjunto atribuído na "
      "{sec:sec_dados_resultante}, de modo que o efeito da cascata pôde ser medido sobre "
      "as mesmas 324 imagens de teste do classificador. Quarenta fotografias tiradas com "
      "telefone celular integram apenas o treinamento, por não provirem da câmera de "
      "operação.")
    P("A classe negativa (não alface) reúne até 500 imagens de cada um de quatro conjuntos "
      "públicos — PlantVillage (HUGHES; SALATHÉ, 2015), com folhas de outras culturas; "
      "PlantDoc (SINGH et al., 2020), com folhas de outras culturas, sadias e doentes, "
      "fotografadas em campo; DeepWeeds (OLSEN et al., 2019), com plantas espontâneas de "
      "pastagem; e COCO (LIN et al., 2014), com cenas gerais — e fotografias da C270 sem "
      "alface: a estrutura vazia, o piso, mãos, objetos e imagens escuras ou com a lente "
      "obstruída. Nenhum dos conjuntos públicos contém alface; do COCO, excluíram-se as "
      "imagens com alimentos, vasos de planta, mesas de refeição ou tigelas, em que poderia "
      "haver alface. Não se produziram imagens negativas escurecendo ou desfocando "
      "fotografias de alface, o que atribuiria a mesma folha às duas classes.")
    P("Cada fotografia da câmera do sistema foi rotulada por inspeção visual segundo uma "
      "regra única: é alface a imagem em que uma pessoa reconhece a folha de alface e "
      "consegue avaliar se há doença; as demais são não alface, e os casos ambíguos foram "
      "excluídos.")
    P("A divisão manteve os cuidados metodológicos do classificador. Imagens idênticas "
      "foram eliminadas pelo resumo MD5, e uma imagem presente nas duas classes "
      "interromperia a montagem. Os conjuntos públicos foram divididos na proporção "
      "70/15/15 por grupo de origem — a folha, no PlantVillage; a data de captura, no "
      "DeepWeeds; e a própria imagem, no PlantDoc e no COCO —, de forma estável à inclusão "
      "de novas fontes. As fotografias da C270 foram agrupadas em sessões de captura, "
      "sequências de fotos sem pausa superior a duas horas, e cada sessão foi alocada "
      "inteira a um único subconjunto, em ordem cronológica, as mais recentes no teste. As "
      "sessões iniciadas até 24/10/2026 foram destinadas apenas ao treinamento, para que a "
      "validação e o teste da segunda versão proviessem somente de sessões nunca "
      "empregadas anteriormente. O aumento de dados foi aplicado em memória, como no "
      "classificador. A {tab:dominio_dados} apresenta a composição do conjunto da segunda "
      "versão.")
    b.append(("tab", "dominio_dados", "Composição do conjunto de dados do verificador de domínio (versão 2)",
              ["Fonte", "Classe", "Treinamento", "Validação", "Teste", "Total"],
              [["Conjunto do Kaggle ({sec:sec_dados})", "alface", "1.566", "326", "324", "2.216"],
               ["Câmera do sistema (C270)", "alface", "350", "0", "75", "425"],
               ["Câmera do sistema (C270)", "não alface", "166", "0", "48", "214"],
               ["Telefone celular", "alface", "40", "0", "0", "40"],
               ["PlantVillage", "não alface", "363", "64", "73", "500"],
               ["PlantDoc", "não alface", "354", "62", "84", "500"],
               ["DeepWeeds", "não alface", "316", "102", "82", "500"],
               ["COCO", "não alface", "367", "66", "67", "500"],
               ["**Total**", "—", "**3.522**", "**620**", "**753**", "**4.895**"]],
              [2771, 1400, 1200, 1200, 1200, 1300], "llrrrr", FONTE_PADRAO))
    P("O conjunto da primeira versão, com 4.417 imagens (3.085 de treinamento, 622 de "
      "validação e 710 de teste), continha 161 fotografias da C270 (79, 2 e 80 em cada "
      "subconjunto). Na segunda versão, todas as sessões da C270 até 24/10/2026, inclusive "
      "as que integraram a validação e o teste da primeira, passaram ao treinamento, e o "
      "teste recebeu uma sessão nova, de 25/10/2026, com 75 imagens de alface e 48 sem "
      "alface. Em consequência, a validação da segunda versão não contém imagens da C270, "
      "e as 630 imagens de teste do Kaggle e dos conjuntos públicos são as mesmas da "
      "primeira versão.")
    P("Foi requisito essencial que a classe positiva incluísse folhas doentes em todos os "
      "estágios de progressão: um verificador que associasse a alface apenas à folha verde "
      "e saudável rejeitaria as folhas mais afetadas, suprimindo o alerta. Por essa razão, "
      "a métrica prioritária do verificador é a revocação da classe alface medida "
      "separadamente sobre as folhas doentes ({tab:criterios_verif}).")

    H(3, "Treinamento e escolha do limiar", "sec_dom_treino")
    P("As duas versões do verificador partiram do modelo YOLO11n-cls pré-treinado no "
      "ImageNet, com a camada de saída substituída por duas classes, e foram treinadas no "
      "Google Colab com os hiperparâmetros da {tab:hiper}, os mesmos do classificador. "
      "Para o envio ao Colab, o lado menor das imagens foi reduzido a 320 *pixels*, acima "
      "da resolução de entrada de 224 *pixels*. A {tab:dominio_treino} sintetiza as "
      "execuções.")
    b.append(("tab", "dominio_treino", "Treinamento das duas versões do verificador de domínio",
              ["Item", "Versão 1", "Versão 2"],
              [["Imagens de treinamento / validação / teste", "3.085 / 622 / 710", "3.522 / 620 / 753"],
               ["Imagens da C270 por subconjunto", "79 / 2 / 80", "516 / 0 / 123"],
               ["GPU", "Tesla T4", "Tesla T4"],
               ["Versão do Ultralytics", "8.4.181", "8.4.187"],
               ["Versão do PyTorch", "2.11.0 (CUDA 13.0)", "2.11.0 (CUDA 13.0)"],
               ["Épocas executadas", "26 de 100", "56 de 100"],
               ["Época de melhor desempenho", "6", "36"],
               ["Acurácia de validação na melhor época", "99,84% (621 de 622)", "99,84% (619 de 620)"],
               ["Tempo de treinamento", "7,8 min", "19,7 min"],
               ["Limiar de aceitação", "0,50", "0,50"]],
              [4071, 2500, 2500], "lcc", FONTE_PADRAO))
    P("A acurácia de validação, próxima de 100% nas duas versões, não constitui critério "
      "de aceitação: com um único erro em mais de 600 imagens, a escolha da melhor época é "
      "instável, e a validação da segunda versão não contém imagens da C270. O histórico "
      "por época e os gráficos da execução da segunda versão não foram preservados; o "
      "registro desse treinamento é o *log* da execução, que contém os argumentos efetivos "
      "e a acurácia de validação de cada época.")
    P("O limiar de aceitação foi escolhido no subconjunto de validação, favorecendo a "
      "aceitação: recusar uma folha doente suprime o alerta, enquanto aceitar uma imagem "
      "alheia custa, no máximo, um alerta indevido. Na primeira versão, 0,50 foi o menor "
      "limiar da grade avaliada com a maior recusa de imagens sem alface (293 de 294), "
      "mantendo aceitas as 189 folhas doentes; acima desse valor, a recusa não melhorava e "
      "passavam a ser recusadas folhas doentes. Na segunda versão, nenhuma imagem sem "
      "alface da validação recebeu probabilidade de alface superior a 0,327, e nenhuma "
      "folha doente, inferior a 0,735; o valor de 0,50 situa-se nesse intervalo e foi "
      "mantido. Nas duas versões, o limiar foi fixado antes da medição no teste.")

    H(2, "Implantação do serviço de inferência", "sec_implantacao")
    P("O serviço de inferência — a interface HTTP da {sec:sec_integracao}, que executa o "
      "verificador de domínio da {sec:sec_dominio} e o classificador de doenças — é "
      "distribuído como imagem de contêiner Docker e executado no serviço Azure Container "
      "Apps. A {fig:implantacao} representa a arquitetura de implantação.")
    b.append(("fig", "implantacao", "Arquitetura de implantação do serviço de inferência",
              "implantacao", 14.0, FONTE_PADRAO))
    P("A interface HTTP não acessa diretamente o armazenamento nem o banco de dados: "
      "recebe os bytes da imagem e devolve a classificação. A orquestração entre o "
      "armazenamento das imagens, o serviço de inferência e o banco de dados cabe a uma "
      "função de nuvem, ainda não implementada, e às fases seguintes do projeto.")

    H(3, "Justificativa do uso de contêiner")
    P("Um contêiner reúne a aplicação e todas as suas dependências numa imagem única, "
      "imutável e versionada, executada de forma isolada sobre o sistema operacional "
      "hospedeiro, sem o custo de uma máquina virtual completa (MERKEL, 2014). O serviço "
      "deste trabalho depende de um conjunto extenso de componentes — o interpretador "
      "Python, o PyTorch, a biblioteca Ultralytics, o servidor de aplicação e bibliotecas "
      "de sistema exigidas pelo OpenCV —, cujas versões precisam coincidir com as "
      "empregadas na avaliação. A adoção do contêiner justifica-se por quatro razões:")
    AL(["reprodutibilidade: o ambiente avaliado é o mesmo implantado. A ausência das "
        "bibliotecas de sistema exigidas pelo OpenCV, por exemplo, não impede a construção "
        "da imagem, mas impede o serviço de iniciar; executada localmente antes da "
        "publicação, a imagem revela esse tipo de falha antes que alcance a nuvem;",
        "portabilidade: a mesma imagem é executada no computador de desenvolvimento e na "
        "nuvem, sem reinstalação de dependências;",
        "requisito do serviço de hospedagem: o Azure Container Apps executa exclusivamente "
        "imagens de contêiner;",
        "separação de responsabilidades: a biblioteca de inferência e a interface HTTP "
        "permanecem sem dependência de serviços de nuvem. O contêiner apenas as empacota, "
        "de modo que a troca do provedor de nuvem não exige alteração do código avaliado "
        "neste trabalho."])

    H(3, "Composição da imagem", "sec_imagem")
    P("A imagem foi construída segundo as seguintes decisões:")
    AL(["imagem base python:3.13-slim, com a mesma versão do interpretador empregada no "
        "desenvolvimento e sem os componentes dispensáveis da distribuição completa;",
        "PyTorch obtido do repositório de pacotes exclusivo para CPU e instalado antes das "
        "demais dependências. O pacote padrão inclui as bibliotecas CUDA, destinadas a "
        "GPU, que aumentariam substancialmente o tamanho da imagem sem utilidade num "
        "servidor sem GPU;",
        "dependências instaladas em camada anterior à do código, de modo que uma alteração "
        "no código reaproveita a camada já construída, sem reinstalar o PyTorch;",
        "cópia seletiva da biblioteca, da interface HTTP e dos pesos, com exclusão do "
        "conjunto de dados e dos demais artefatos de desenvolvimento do contexto de "
        "construção;",
        "pesos incluídos: o classificador no formato PyTorch, configuração padrão por "
        "manter a explicabilidade por Grad-CAM ({sec:sec_gradcam}), que não opera com o "
        "modelo exportado; o classificador no formato ONNX, selecionável por configuração; "
        "e a segunda versão do verificador de domínio ({sec:sec_dominio});",
        "configuração por variáveis de ambiente — modelos, limiares, verificador de "
        "domínio, chave de acesso e origens permitidas —, alterável sem reconstrução da "
        "imagem;",
        "execução com usuário sem privilégios administrativos, de modo que um eventual "
        "comprometimento da aplicação não obtenha controle do contêiner."])
    P("Embora o formato ONNX dispense o PyTorch ({sec:sec_integracao}), a imagem o mantém, "
      "porque o Grad-CAM e o verificador de domínio empregam pesos no formato PyTorch. A "
      "imagem da segunda versão do serviço, com o verificador de domínio, mede 614 MB "
      "compactada e 2,63 GB em disco, praticamente o mesmo tamanho da primeira (615 MB e "
      "2,66 GB): o verificador acrescenta apenas os seus pesos, enquanto o volume da imagem "
      "decorre do PyTorch e das demais dependências.")

    H(3, "Hospedagem", "sec_hospedagem")
    P("O serviço é executado no Azure Container Apps, na região Brazil South, com 1 vCPU e "
      "2 GiB de memória por réplica e escala a zero: após cinco minutos sem requisições, a "
      "última réplica é removida, e uma nova é iniciada na requisição seguinte, até o "
      "máximo de duas réplicas simultâneas. A cobrança ocorre pelo tempo em que há réplica "
      "ativa, e não pelo número de requisições. Para o uso previsto — análises periódicas "
      "e consultas esporádicas —, o serviço permanece desligado na maior parte do tempo. O "
      "{qua:hospedagem} apresenta as alternativas consideradas.")
    b.append(("qua", "hospedagem", "Alternativas de hospedagem do serviço de inferência",
              ["Alternativa", "Característica", "Limitação", "Decisão"],
              [["Azure App Service, plano gratuito (F1)", "Hospedagem gerenciada sem custo", "Memória e disco insuficientes para o PyTorch e as demais dependências; no Linux, não executa contêiner próprio", "Descartada"],
               ["Azure App Service, plano básico (B1)", "Instância dedicada", "Custo fixo, com a instância sempre ligada, incompatível com o crédito da assinatura estudantil", "Descartada"],
               ["Inferência apenas com ONNX Runtime no plano gratuito", "Dispensa o PyTorch e cabe nos recursos do plano", "Exigiria reescrever o módulo de inferência, objeto deste trabalho, e eliminaria o Grad-CAM", "Descartada"],
               ["Azure Container Apps com réplica mínima permanente", "Sem inicialização a frio", "O contêiner nunca é desligado, e o custo deixa de ser compatível com a assinatura", "Descartada"],
               ["Azure Container Apps com escala a zero", "Contêiner desligado quando ocioso", "Inicialização a frio na primeira requisição após o período ocioso", "**Adotada**"]],
              [2300, 2100, 3471, 1200], "llll", FONTE_PADRAO))
    P("A imagem é publicada no GitHub Container Registry, gratuito para imagens públicas, "
      "e não no Azure Container Registry, que é pago; como os pesos já são distribuídos "
      "publicamente, a publicação da imagem não expõe informação adicional. Cada versão "
      "recebe uma etiqueta nova (v1, v2), pois a republicação de uma etiqueta existente "
      "pode levar o serviço a manter a versão anterior em cache. A chave de acesso da "
      "interface é armazenada como segredo do serviço, e não como variável em texto "
      "claro.")
    P("A escala a zero impõe uma restrição às demais partes do sistema: consultas "
      "automáticas em intervalos curtos, como verificações periódicas de estado a partir "
      "da interface web, impedem o desligamento do contêiner e eliminam a economia. Por "
      "essa razão, prevê-se que a interface web consulte o banco de dados, e não o serviço "
      "de inferência.")
    P("A latência e o tempo de inicialização a frio do serviço em nuvem constam da "
      "{sec:sec_res_latencia}; o custo mensal efetivo não foi medido. A latência reportada "
      "para o requisito RNF01, apresentada "
      "na {sec:sec_res_latencia} e na {sec:sec_dom_teste}, refere-se ao servidor de "
      "inferência; os tempos de transmissão, de inicialização a frio e de orquestração "
      "situam-se fora dessa fronteira de medição.")

    # ================================================================== #
    H(1, "RESULTADOS", "cap_res")
    # ================================================================== #
    P("Este capítulo apresenta os resultados do treinamento, da avaliação nos "
      "subconjuntos de validação e de teste, da medição de latência e da verificação dos "
      "casos de teste, dos requisitos e das hipóteses pertinentes ao módulo.")

    H(2, "Processo de treinamento", "sec_res_treino")
    P(f"O treinamento foi encerrado por parada antecipada após a {n_ep}ª época, tendo o "
      f"melhor desempenho no subconjunto de validação ocorrido na {melhor['epoch']}ª, "
      "cujos parâmetros foram preservados como modelo final. A {tab:sintese} sintetiza a "
      "execução.")
    menor_tr = min(hist, key=lambda r: r["train/loss"])
    menor_val = min(hist, key=lambda r: r["val/loss"])
    b.append(("tab", "sintese", "Síntese do processo de treinamento",
              ["Parâmetro", "Valor"],
              [["Épocas executadas", f"{n_ep} de 100"],
               ["Época de melhor desempenho", str(melhor["epoch"])],
               ["Acurácia de validação na melhor época", f"{br(100 * melhor['metrics/accuracy_top1'], 2)}% (318 de 326)"],
               ["Perda de validação na melhor época", br(melhor["val/loss"], 4)],
               ["Menor perda de treinamento", f"{br(menor_tr['train/loss'], 4)} (época {menor_tr['epoch']})"],
               ["Menor perda de validação", f"{br(menor_val['val/loss'], 4)} (época {menor_val['epoch']})"],
               ["Tempo total de treinamento", f"{br(tempo_total, 1)} s ({br(tempo_total / 60, 1)} min)"],
               ["Tempo médio por época", f"{br(tempo_total / n_ep, 1)} s"],
               ["Taxa de aprendizado na última época", br(hist[-1]["lr"], 6)]],
              [5200, 3871], "lr", FONTE_PADRAO))
    P("A {tab:epocas} apresenta o histórico completo, época a época, e a {fig:curvas} sua "
      "representação gráfica.")
    linhas_ep = []
    for r in hist:
        destaque = r["epoch"] == melhor["epoch"]
        cel = [str(r["epoch"]), br(r["train/loss"], 4), br(r["val/loss"], 4),
               f"{br(100 * r['metrics/accuracy_top1'], 2)}%", br(r["time"], 1)]
        linhas_ep.append([f"**{c}**" for c in cel] if destaque else cel)
    b.append(("tab", "epocas", "Histórico do treinamento por época",
              ["Época", "Perda de treinamento", "Perda de validação", "Acurácia de validação", "Tempo acumulado (s)"],
              linhas_ep, [1300, 1950, 1950, 1950, 1921], "crrrr",
              "Fonte: elaborado pelos autores (2026). Nota: em negrito, a época selecionada."))
    b.append(("fig", "curvas", "Curvas de perda e de acurácia ao longo do treinamento",
              "curvas", 16.0, FONTE_PADRAO))
    P("Distinguem-se quatro fases no processo:")
    AL(["adaptação inicial (épocas 1 e 2): a acurácia de validação elevou-se de 70,86% "
        "para 88,34% em duas épocas, efeito direto da transferência de aprendizado — o "
        "modelo já dispunha de detectores visuais genéricos e precisou apenas aprender a "
        "combiná-los para as três classes;",
        "instabilidade do aquecimento (épocas 3 a 5): a acurácia recuou a 81,60% na "
        "quarta época, coincidindo com o término do aquecimento, quando a taxa de "
        "aprendizado atinge seu valor máximo e os passos de atualização são maiores;",
        "convergência (épocas 6 a 13): a acurácia superou 90% na sexta época e atingiu o "
        "máximo de 97,55% na décima terceira;",
        "estabilização (épocas 14 a 33): a acurácia oscilou entre 93,56% e 97,55%, sem "
        "superar o máximo. A trigésima época igualou o valor de 97,55%, mas o critério de "
        "seleção exige melhora estrita. Transcorridas 20 épocas sem melhora, a parada "
        "antecipada encerrou o processo, dispensando as 67 épocas restantes."])
    P("Não se observaram indícios de sobreajuste: a perda de validação manteve-se na "
      "mesma ordem de grandeza da perda de treinamento ao longo de todo o processo, sem "
      "tendência sustentada de crescimento. O fato de a perda de validação ser, em geral, "
      "inferior à de treinamento explica-se pelo aumento de dados e pelo *dropout*, que "
      "atuam apenas no treinamento e tornam aquelas imagens mais difíceis.")

    H(2, "Desempenho no subconjunto de validação", "sec_res_val")
    P("A {tab:val} apresenta as métricas obtidas sobre as 326 imagens de validação. Por "
      "ter participado da seleção da melhor época, esse subconjunto fornece estimativa "
      "otimista do desempenho e não constitui o resultado oficial do trabalho.")
    b.append(("tab", "val", "Métricas por classe no subconjunto de validação",
              ["Classe", "Precisão", "Revocação", "F1-*score*", "Imagens"],
              [["Bacteriana", "0,9825", "0,9825", "0,9825", "114"],
               ["Fúngica", "0,9595", "0,9467", "0,9530", "75"],
               ["Saudável", "0,9783", "0,9854", "0,9818", "137"],
               ["**Média macro**", "—", "—", "**0,9724**", "**326**"]],
              [2471, 1650, 1650, 1650, 1650], "lrrrr", FONTE_PADRAO))
    P("A acurácia global na validação foi de 97,55%.")

    H(2, "Desempenho no subconjunto de teste", "sec_res_teste")
    P("O subconjunto de teste, avaliado uma única vez ao final do processo, fornece o "
      "resultado oficial do trabalho. O modelo alcançou **acurácia global de 95,06%**, com "
      "308 classificações corretas em 324 imagens, e F1-*score* macro de 0,9415. As "
      "métricas por classe constam da {tab:teste}.")
    b.append(("tab", "teste", "Métricas por classe no subconjunto de teste",
              ["Classe", "Precisão", "Revocação", "F1-*score*", "Imagens", "Confiança média nos acertos", "Acertos com confiança ≥ 0,85"],
              [["Bacteriana", "0,9364", "0,9717", "0,9537", "106", "0,9834", "98,1%"],
               ["Fúngica", "0,9429", "0,8462", "0,8919", "78", "0,9561", "89,4%"],
               ["Saudável", "0,9653", "0,9929", "0,9789", "140", "0,9884", "97,1%"],
               ["**Média macro**", "**0,9482**", "**0,9369**", "**0,9415**", "**324**", "—", "—"]],
              [1671, 1150, 1200, 1150, 1050, 1450, 1400], "lrrrrrr", FONTE_PADRAO))
    P("A acurácia no teste foi 2,49 pontos percentuais inferior à de validação. A "
      "diferença é esperada, uma vez que a validação participou da escolha da época, e "
      "sua magnitude indica preservação da capacidade de generalização; diferenças "
      "superiores a dez pontos percentuais caracterizariam sobreajuste.")

    H(2, "Matriz de confusão", "sec_res_matriz")
    P("A {fig:matriz} e a {tab:matriz} apresentam a matriz de confusão do subconjunto de "
      "teste.")
    b.append(("fig", "matriz", "Matriz de confusão no subconjunto de teste", "matriz", 16.0,
              FONTE_PADRAO))
    b.append(("tab", "matriz", "Matriz de confusão no subconjunto de teste",
              ["Classe real", "Predita bacteriana", "Predita fúngica", "Predita saudável", "Total"],
              [["Bacteriana", "**103**", "3", "0", "106"],
               ["Fúngica", "7", "**66**", "5", "78"],
               ["Saudável", "0", "1", "**139**", "140"],
               ["Total", "110", "70", "144", "324"]],
              [2271, 1750, 1700, 1750, 1600], "lrrrr", FONTE_PADRAO))
    P("A diagonal principal concentra os acertos. Os erros concentram-se nas folhas com "
      "míldio: 12 das 78 foram classificadas incorretamente, sete como bacterianas e cinco "
      "como saudáveis. Nenhuma folha bacteriana foi classificada como saudável, e nenhuma "
      "folha saudável foi classificada como bacteriana.")

    H(2, "Análise dos erros", "sec_erros")
    P("O {qua:gravidade} classifica os 16 erros observados no teste segundo sua "
      "consequência operacional para o agricultor.")
    b.append(("qua", "gravidade", "Erros de classificação no subconjunto de teste, por gravidade operacional",
              ["Tipo de erro", "Ocorrências", "Consequência"],
              [["Fúngica classificada como saudável", "5", "Doença não reportada; nenhum alerta emitido"],
               ["Fúngica classificada como bacteriana", "7", "Alerta emitido; possível indicação de tratamento inadequado"],
               ["Bacteriana classificada como fúngica", "3", "Alerta emitido; possível indicação de tratamento inadequado"],
               ["Saudável classificada como fúngica", "1", "Alerta indevido; possível aplicação desnecessária de defensivo"]],
              [3300, 1400, 4371], "lcl", FONTE_PADRAO))
    P("A revocação de 0,8462 da classe fúngica constitui o principal ponto de fragilidade "
      "do modelo. A gravidade dessas ocorrências é heterogênea: nos sete casos "
      "classificados como bacterianos, o alerta é emitido e o agricultor é notificado da "
      "presença de doença, divergindo apenas a indicação do tratamento; nos cinco casos "
      "classificados como saudáveis, nenhum alerta é gerado — situação que corresponde "
      "exatamente ao evento que o sistema se destina a evitar. A semelhança entre as "
      "lesões angulosas de ambas as doenças ({sec:sec_mecanismo}) e a proximidade visual "
      "entre o míldio em estágio inicial e a folha saudável são hipóteses compatíveis com "
      "esse padrão.")
    P("Em contrapartida, a classe saudável alcançou revocação de 0,9929, com uma única "
      "classificação incorreta em 140 imagens, o que indica baixa incidência de alarmes "
      "falsos. A {tab:piores} relaciona os dez erros de maior magnitude.")
    b.append(("tab", "piores", "Erros de maior magnitude no subconjunto de teste",
              ["Ordem", "Classe real", "Classe predita", "Confiança da predição"],
              [["1", "Fúngica", "Saudável", "1,000"],
               ["2", "Fúngica", "Bacteriana", "1,000"],
               ["3", "Fúngica", "Bacteriana", "1,000"],
               ["4", "Fúngica", "Bacteriana", "0,994"],
               ["5", "Fúngica", "Bacteriana", "0,990"],
               ["6", "Saudável", "Fúngica", "0,963"],
               ["7", "Bacteriana", "Fúngica", "0,936"],
               ["8", "Bacteriana", "Fúngica", "0,932"],
               ["9", "Fúngica", "Saudável", "0,894"],
               ["10", "Fúngica", "Bacteriana", "0,798"]],
              [1300, 2600, 2600, 2571], "cllr", FONTE_PADRAO))
    P("Os três erros de maior magnitude ocorreram com confiança de 1,000. Erros dessa "
      "natureza não são mitigáveis por ajuste do limiar de confiança e podem decorrer de "
      "rótulos incorretos no conjunto de dados de origem, hipótese plausível em conjuntos "
      "públicos de imagens agrícolas (BARBEDO, 2019). [[preencher:resultado da inspeção "
      "visual das três imagens — indicar se os rótulos originais estão corretos]]")

    H(2, "Latência de inferência", "sec_res_latencia")
    P("A {tab:latencia} apresenta a latência por imagem medida no ambiente local "
      "({tab:ambiente}), sem GPU.")
    b.append(("tab", "latencia", "Latência de inferência por imagem, em milissegundos",
              ["Configuração", "Mediana", "Média", "p95", "p99", "Mínimo", "Máximo"],
              [["PyTorch em CPU", "20,10", "23,23", "38,27", "81,34", "17,07", "91,20"],
               ["ONNX Runtime em CPU", "**12,73**", "16,15", "32,69", "78,98", "9,08", "85,37"]],
              [2371, 1100, 1100, 1100, 1100, 1150, 1150], "lrrrrrr",
              "Fonte: elaborado pelos autores (2026). Nota: p95 e p99 são os percentis 95 e 99."))
    P("A conversão para ONNX reduziu a latência mediana em 36,7%. Ambas as configurações "
      "atendem ao limite de 100 milissegundos estabelecido no requisito RNF01, inclusive "
      "no percentil 99. Medida de ponta a ponta por meio da API HTTP, em execução local, a "
      "latência foi de 53,3 milissegundos, dos quais 40,6 corresponderam à inferência e "
      "12,7 ao processamento do protocolo. A latência conjunta do verificador de domínio "
      "e do classificador, no ambiente local, é apresentada na {sec:sec_dom_teste}.")
    P("No serviço em nuvem ({sec:sec_implantacao}), com a segunda versão do verificador "
      "implantada, a latência foi medida em 100 imagens do subconjunto de validação, "
      "enviadas uma a uma, após dez requisições de aquecimento descartadas, com o critério "
      "da {sec:sec_dom_teste}: soma dos tempos de inferência das duas etapas, medidos pelo "
      "próprio servidor. A mediana das duas etapas foi de 22,62 milissegundos, com "
      "percentil 95 de 45,15 e máximo de 99,06 milissegundos; a do verificador "
      "isoladamente, de 11,27 milissegundos. Os valores são inferiores aos do ambiente "
      "local, que emprega processador distinto. A latência de ponta a ponta, medida a "
      "partir de um computador cliente em conexão doméstica, teve mediana de 104,85 "
      "milissegundos e percentil 95 de 505,87 milissegundos, variação atribuível à rede. A "
      "primeira requisição após o desligamento da réplica por inatividade levou 11,27 "
      "segundos, em uma única medição; esse tempo de inicialização a frio inclui a criação "
      "da réplica, a inicialização do contêiner e o carregamento dos dois modelos.")
    P("Os valores referem-se ao servidor de inferência, e não a dispositivo embarcado: "
      "na arquitetura adotada, a câmera Logitech C270 limita-se à captura da imagem. A "
      "comparação com latências de dispositivos de borda relatadas na literatura deve "
      "considerar essa distinção.")

    H(2, "Equivalência do modelo exportado", "sec_res_onnx")
    P("A comparação entre as saídas do modelo em formato ONNX e do modelo original, "
      "sobre 20 imagens do subconjunto de teste, resultou em divergência máxima de "
      "4,17 × 10^−7^ e média de 3,71 × 10^−8^, valores quatro ordens de grandeza inferiores "
      "à tolerância estabelecida de 10^−3^. Não houve divergência de classe predita em "
      "nenhuma das imagens. Os artefatos produzidos constam da {tab:artefatos}.")
    b.append(("tab", "artefatos", "Formatos do modelo produzidos",
              ["Formato", "Tamanho", "Finalidade"],
              [["PyTorch", "3,19 MB", "Treinamento, avaliação e explicabilidade"],
               ["ONNX", "6,17 MB", "Execução no servidor de inferência"],
               ["TorchScript", "6,33 MB", "Alternativa de implantação"]],
              [2300, 1700, 5071], "lrl", FONTE_PADRAO))

    H(2, "Desempenho do verificador de domínio", "sec_res_dominio")
    P("Esta seção apresenta os resultados das duas versões do verificador de domínio "
      "descrito na {sec:sec_dominio}: o desempenho nos subconjuntos de validação e de "
      "teste, o efeito da cascata sobre o diagnóstico, a análise dos erros e a decisão de "
      "implantação. Todas as medições foram realizadas no ambiente local da "
      "{tab:ambiente}, em CPU, com pesos PyTorch nas duas etapas e limiar de aceitação de "
      "0,50.")

    H(3, "Subconjunto de validação")
    P("A {tab:dominio_val} apresenta o desempenho no subconjunto de validação. Por ter "
      "servido à escolha da melhor época e do limiar, esse subconjunto fornece estimativa "
      "otimista e não constitui o resultado oficial.")
    b.append(("tab", "dominio_val", "Desempenho do verificador de domínio no subconjunto de validação",
              ["Critério", "Versão 1", "Versão 2", "Meta"],
              [["Alface aceita — folhas doentes", "189 de 189 (100,0%)", "189 de 189 (100,0%)", "≥ 99%"],
               ["Não alface recusada — todas as fontes", "293 de 294 (99,7%)", "294 de 294 (100,0%)", "≥ 95%"],
               ["Alface aceita — todas as imagens", "327 de 328 (99,7%)", "325 de 326 (99,7%)", "—"],
               ["Imagens da C270", "2 de 2 de alface aceitas; nenhuma sem alface", "Nenhuma", "—"],
               ["Latência mediana das duas etapas", "49,63 ms", "42,70 ms", "< 100 ms"]],
              [3300, 2300, 2300, 1171], "lccc",
              "Fonte: elaborado pelos autores (2026). Nota: na segunda versão, uma primeira "
              "medição de latência, com outros programas em execução no computador, registrou "
              "120,84 ms; a medição foi repetida com os demais programas encerrados, e a "
              "aceitação e a recusa foram idênticas nas duas execuções."))
    P("As duas versões atingiram, na validação, as metas da {tab:criterios_verif} "
      "aplicáveis a esse subconjunto. A validação da segunda versão não contém imagens da "
      "C270, de modo que seu comportamento no ambiente real só é evidenciado pelo teste.")

    H(3, "Subconjunto de teste", "sec_dom_teste")
    P("O subconjunto de teste de cada versão foi medido uma única vez, com o limiar e os "
      "critérios fixados antes da medição. A {tab:dominio_teste} apresenta os resultados e "
      "a situação de cada critério da {tab:criterios_verif}.")
    b.append(("tab", "dominio_teste", "Desempenho do verificador de domínio no subconjunto de teste",
              ["Critério", "Versão 1", "Situação", "Versão 2", "Situação"],
              [["(1) Alface aceita — folhas doentes", "181 de 184 (98,4%; IC95 95,3%–99,4%)", "Não atingida", "178 de 184 (96,7%; IC95 93,1%–98,5%)", "Não atingida"],
               ["(2) Não alface recusada — todas as fontes", "338 de 356 (94,9%; IC95 92,2%–96,8%)", "Não atingida", "340 de 354 (96,0%; IC95 93,5%–97,6%)", "Atingida"],
               ["(3) Latência mediana das duas etapas", "53,25 ms", "Atingida", "51,85 ms", "Atingida"],
               ["(4) Alface aceita — C270", "12 de 30 (40,0%; IC95 24,6%–57,7%)", "Sem meta", "75 de 75 (100,0%; IC95 95,1%–100,0%)", "Atingida"],
               ["(5) Não alface recusada — C270", "45 de 50 (90,0%; IC95 78,6%–95,7%)", "Sem meta", "42 de 48 (87,5%; IC95 75,3%–94,1%)", "Não atingida"]],
              [2471, 2100, 1100, 2300, 1100], "lcccc",
              "Fonte: elaborado pelos autores (2026). Nota: IC95 — intervalo de confiança de "
              "95% pelo método de Wilson. As imagens da C270 provêm de sessões de captura "
              "distintas em cada versão: de 10 a 11/10/2026 e de 17/10/2026, na primeira; de "
              "25/10/2026, na segunda. As 630 imagens restantes são as mesmas nas duas "
              "versões. Os critérios (4) e (5) não se aplicavam à primeira versão."))
    P("A primeira versão não atendeu aos critérios então vigentes: aceitou 181 das 184 "
      "folhas doentes, abaixo do mínimo de 183, e recusou 338 das 356 imagens sem alface, "
      "uma abaixo do mínimo de 339. O resultado mais grave para a operação foi a aceitação "
      "de apenas 12 das 30 imagens de alface da C270; nenhuma das 12 imagens de alface em "
      "vaso redondo preto foi aceita. Por isso, a primeira versão não foi implantada, e "
      "desenvolveu-se a segunda, com as sessões da C270 descritas na {sec:sec_dom_dados}.")
    P("A segunda versão aceitou todas as 75 imagens de alface da C270, recusou 340 das 354 "
      "imagens sem alface e manteve a latência mediana das duas etapas em 51,85 ms; a "
      "mediana do verificador isolado foi de 25,42 ms. Não atingiu, contudo, o critério "
      "(1), com 178 das 184 folhas doentes aceitas, contra o mínimo de 183, nem o critério "
      "(5), com 42 das 48 imagens sem alface da C270 recusadas, contra o mínimo de 46. "
      "**A segunda versão não atende ao critério de implantação.** Por grupo, foram "
      "aceitas todas as folhas bacterianas (106) e saudáveis (140) e 72 das 78 fúngicas; "
      "foram recusadas todas as imagens do COCO (67) e do PlantVillage (73), 81 das 82 do "
      "DeepWeeds e 77 das 84 do PlantDoc.")

    H(3, "Efeito da cascata sobre o diagnóstico", "sec_dom_cascata")
    P("A {tab:dominio_cascata} compara, nas 324 imagens de teste do classificador, o "
      "classificador isolado e o sistema completo, em que a imagem recusada pelo "
      "verificador não recebe diagnóstico e é contada como erro. O classificador não foi "
      "executado novamente: empregaram-se as predições da avaliação da "
      "{sec:sec_res_teste}.")
    b.append(("tab", "dominio_cascata", "Efeito do verificador de domínio sobre o diagnóstico no subconjunto de teste do classificador",
              ["Métrica", "Só classificador", "Cascata com a versão 1", "Cascata com a versão 2"],
              [["Revocação da classe bacteriana", "103 de 106 (97,2%)", "102 de 106 (96,2%)", "103 de 106 (97,2%)"],
               ["Revocação da classe fúngica", "66 de 78 (84,6%)", "64 de 78 (82,0%)", "63 de 78 (80,8%)"],
               ["Revocação da classe saudável", "139 de 140 (99,3%)", "138 de 140 (98,6%)", "139 de 140 (99,3%)"],
               ["Acurácia", "308 de 324 (95,1%)", "304 de 324 (93,8%)", "305 de 324 (94,1%)"],
               ["Alertas emitidos", "167", "164", "166"],
               ["Alertas verdadeiros suprimidos", "—", "3", "1"]],
              [2471, 2200, 2200, 2200], "lccc", FONTE_PADRAO))
    P("O classificador, isoladamente, não se altera, e seus resultados permanecem os das "
      "seções anteriores deste capítulo. No sistema completo com a segunda versão, a "
      "revocação da classe fúngica passa de 84,6% para 80,8%, e a acurácia, de 95,1% para "
      "94,1%. A diferença decorre exclusivamente das seis folhas fúngicas recusadas pelo "
      "verificador: o classificador acertava três delas, que ficam sem diagnóstico, e já "
      "errava as outras três. Apenas uma das seis gerava alerta, de modo que a segunda "
      "versão suprime um alerta verdadeiro, contra três da primeira.")

    H(3, "Análise dos erros", "sec_dom_erros")
    P("Na segunda versão, as seis folhas doentes recusadas, todas da classe fúngica, "
      "provêm de apenas três imagens de origem do conjunto do Kaggle: um detalhe ampliado "
      "de lesão, sem forma de folha; uma superfície em macrofotografia com estruturas do "
      "patógeno e um retângulo de ruído gravado na própria imagem pelo aumento de dados do "
      "conjunto de origem; e um fragmento de folha coberto de micélio sobre fundo branco. "
      "São imagens de microscópio ou de macrofotografia, sem a forma de folha ou de planta, "
      "e os erros são correlacionados: seis recusas, três imagens de origem.")
    P("As 14 imagens sem alface aceitas são todas de plantas: seis fotografias da C270 com "
      "outras plantas do quintal como assunto — babosa, violetas, samambaia, jiboia, "
      "lírio-da-paz e vasos com plantas junto ao muro —; sete do PlantDoc, com folhas e "
      "mudas de outras culturas, várias com lesões, como uma muda de pimentão em roseta e "
      "folhas crespas de tomateiro com virose; e uma do DeepWeeds. Nenhuma imagem de "
      "objeto, piso, céu ou lente obstruída foi aceita: as nove fotografias da C270 com a "
      "lente obstruída foram recusadas. Uma hipótese, não verificada, é que o modelo tenha "
      "associado a classe alface à presença de planta em vaso no ambiente doméstico, pois "
      "nenhuma imagem sem alface da C270 empregada no treinamento mostrava outra planta.")
    P("Na primeira versão, as falhas concentravam-se na C270: as 12 imagens de alface em "
      "vaso redondo preto foram recusadas, e quatro das cinco imagens sem alface aceitas "
      "tinham a lente obstruída ou eram quase pretas.")

    H(3, "Decisão de implantação", "sec_dom_decisao")
    P("O critério de implantação fixado antes do teste não foi atingido pela versão 2: a "
      "aceitação de folhas doentes (178 de 184, contra o mínimo de 183) e a recusa de "
      "imagens sem alface capturadas pela C270 (42 de 48, contra o mínimo de 46) ficaram "
      "abaixo das metas. Após o conhecimento desse resultado, optou-se por implantar a "
      "versão 2, com base na análise dos erros: as seis folhas recusadas provêm de três "
      "imagens de origem obtidas em microscópio ou em macrofotografia, sem a forma de "
      "folha ou de planta que a câmera do sistema captura, e só uma correspondia a alerta "
      "emitido pelo classificador; as catorze imagens sem alface aceitas são todas de "
      "outras plantas; e as falhas que inviabilizavam a versão 1, observadas em outras "
      "sessões de captura — a recusa de 18 das 30 imagens de alface da C270 e a aceitação "
      "de imagens com a lente obstruída —, não se repetiram. A decisão constitui exceção "
      "ao procedimento adotado, que fixava o critério antes da medição justamente para que "
      "o resultado não a influenciasse, e é registrada como tal: o critério permanece não "
      "atingido, e o subconjunto de teste da versão 2 não pode ser reutilizado para "
      "avaliar versões posteriores.")
    P("O classificador não se altera com essa decisão: isoladamente, mantém revocação "
      "fúngica de 84,6% e acurácia de 95,06%, e os casos de teste da {sec:sec_res_ct} "
      "referem-se a ele. No sistema completo, a revocação fúngica é de 80,8% "
      "({sec:sec_dom_cascata}). O resultado de 75 de 75 imagens de alface aceitas na C270 "
      "provém de uma única sessão de captura, com a câmera na mão e as mesmas plantas do "
      "treinamento, e deve ser lido com as limitações da {sec:sec_lim}. As melhorias "
      "previstas para uma próxima versão constam da {sec:sec_melhorias}.")

    H(2, "Verificação dos casos de teste e dos requisitos", "sec_res_ct")
    P("A {tab:ct} apresenta o resultado dos casos de teste pertinentes ao módulo.")
    b.append(("tab", "ct", "Resultado dos casos de teste do módulo de inteligência artificial",
              ["Caso", "Descrição", "Critério", "Resultado medido", "Situação"],
              [["CT01", "Detecção de míldio (*Bremia lactucae*)", "Confiança ≥ 0,85", "Confiança média de 0,9561 nos acertos; 89,4% dos acertos acima do limiar", "Aprovado"],
               ["CT02", "Detecção de mancha bacteriana (*Xanthomonas campestris*)", "Confiança ≥ 0,85", "Confiança média de 0,9834 nos acertos; 98,1% dos acertos acima do limiar", "Aprovado"],
               ["CT03", "Classificação de folha saudável", "Revocação ≥ 90%", "Revocação de 99,29%", "Aprovado"],
               ["CT04", "Tempo de inferência por imagem", "< 100 ms", "Mediana de 12,73 ms; percentil 99 de 78,98 ms", "Aprovado"]],
              [900, 2400, 1500, 3071, 1200], "lllll", FONTE_PADRAO))
    P("Ressalte-se que os critérios dos casos CT01 e CT02 avaliam a confiança nas "
      "classificações corretas e, portanto, não refletem as 12 folhas com míldio não "
      "identificadas como tal. A revocação de 0,8462 da classe fúngica deve ser "
      "considerada em conjunto com o resultado do CT01. Os casos de teste referem-se ao "
      "classificador; no sistema completo, com o verificador de domínio, a revocação da "
      "classe fúngica nas mesmas imagens é de 80,8% ({sec:sec_dom_cascata}). A {tab:rnf} "
      "apresenta a verificação dos requisitos não funcionais.")
    b.append(("tab", "rnf", "Verificação dos requisitos não funcionais do módulo",
              ["Requisito", "Descrição", "Meta", "Medido", "Situação"],
              [["RNF01", "Tempo de inferência", "< 100 ms", "12,73 ms", "Atendido"],
               ["RNF02", "Acurácia na classificação das três classes", "> 90%", "95,06%", "Atendido"],
               ["RNF07", "Documentação e versionamento", "—", "Repositório versionado; 221 testes automatizados", "Atendido"]],
              [1200, 3000, 1200, 2471, 1200], "llrrl", FONTE_PADRAO))
    P("Com o verificador de domínio, a latência mediana das duas etapas, com pesos PyTorch "
      "em CPU, foi de 51,85 milissegundos no ambiente local ({sec:sec_dom_teste}) e de "
      "22,62 milissegundos no serviço em nuvem ({sec:sec_res_latencia}), ambas inferiores "
      "ao limite do requisito RNF01.")

    H(2, "Verificação das hipóteses", "sec_res_hip")
    P("A hipótese H1 previa acurácia superior a 90% na distinção entre manchas foliares de "
      "origem bacteriana e fúngica. A acurácia obtida no subconjunto de teste, de 95,06%, "
      "supera a meta em 5,06 pontos percentuais, sobre conjunto isento de vazamento. A "
      "hipótese H1 é confirmada.")
    P("A hipótese H2 comporta dois componentes. O primeiro, tempo de inferência inferior a "
      "100 milissegundos, é confirmado, com mediana de 12,73 milissegundos para o "
      "classificador em ONNX e de 51,85 milissegundos para as duas etapas da cascata com "
      "pesos PyTorch. O segundo, "
      "contenção dos focos infecciosos antes que atinjam 5% da área da estufa, não pôde "
      "ser verificado neste módulo, por depender da operação do protótipo em ambiente de "
      "cultivo.")
    P("As hipóteses H3, H4 e H5, relativas à redução do consumo de defensivos, à "
      "usabilidade das interfaces e ao custo de implementação, não constituem objeto de "
      "verificação deste módulo. Registre-se, contudo, que a revocação de 0,9929 da classe "
      "saudável, com um único alerta indevido em 140 folhas sadias, é favorável à "
      "hipótese H3.")

    H(2, "Síntese dos resultados", "sec_res_sintese")
    P("O modelo YOLO11n-cls, treinado em {tempo_min} minutos no Google Colab, alcançou "
      "acurácia de 95,06% no subconjunto de teste e latência mediana de 12,73 "
      "milissegundos em CPU, atendendo aos requisitos RNF01 e RNF02 e confirmando a "
      "hipótese H1. O principal ponto de fragilidade é a revocação da classe fúngica, de "
      "0,8462.")
    P("O verificador de domínio, em sua segunda versão, aceitou todas as 75 imagens de "
      "alface da câmera do sistema e recusou 96,0% das imagens sem alface, mas não atingiu "
      "o critério de implantação fixado antes do teste quanto à aceitação de folhas "
      "doentes e à recusa de imagens sem alface da C270; sua implantação foi decidida após "
      "o resultado, como exceção registrada na {sec:sec_dom_decisao}. No sistema completo, "
      "a revocação da classe fúngica passa a 80,8%.")
    P("Destaca-se o tratamento conferido à integridade do conjunto de dados. A manutenção "
      "da divisão original, com 104 grupos de imagens idênticas e 115 grupos de folhas "
      "distribuídos entre subconjuntos distintos, produziria métricas superiores às aqui "
      "reportadas, porém desprovidas de validade, por medirem memorização em vez de "
      "generalização. A redivisão por grupo de folha de origem assegura que os resultados "
      "reflitam o desempenho do modelo diante de amostras inéditas.")

    # ================================================================== #
    H(1, "LIMITAÇÕES E MELHORIAS FUTURAS", "cap_lim")
    # ================================================================== #
    H(2, "Limitações do módulo", "sec_lim")
    P("Identificaram-se as seguintes limitações do módulo de inteligência artificial:")
    AL(["diversidade da classe bacteriana: as imagens não sintéticas dessa classe derivam "
        "de aproximadamente 53 folhas. O aumento de dados amplia a variabilidade de "
        "apresentação, mas não introduz novas amostras biológicas, de modo que a "
        "generalização para lesões em outras condições, cultivares ou estágios permanece "
        "não verificada;",
        "domínio das imagens: os subconjuntos de treinamento e de teste do classificador "
        "provêm do mesmo conjunto público, com condições de captura semelhantes. O "
        "desempenho do classificador sobre imagens capturadas pela câmera do sistema, uma "
        "Logitech C270, não foi medido, pois não há imagens dessa câmera rotuladas quanto à "
        "doença, e pode divergir do reportado; as imagens da C270 empregadas no verificador "
        "de domínio indicam apenas se a imagem contém alface;",
        "conjunto fechado de classes: o modelo reconhece exclusivamente as três classes "
        "treinadas. Viroses, deficiências nutricionais, queimaduras e danos mecânicos com "
        "aparência semelhante serão atribuídos a uma delas. A verificação de domínio "
        "implementada trata apenas de imagens que não contêm alface, e não de outras "
        "afecções da alface;",
        "recorte central: o pré-processamento descarta as faixas laterais de imagens "
        "retangulares, de modo que lesões situadas apenas nas extremidades podem não ser "
        "analisadas, o que recomenda o enquadramento centralizado da folha pela câmera;",
        "ausência de localização precisa: o mapa Grad-CAM, de resolução 7 × 7, fornece "
        "indicação aproximada, insuficiente para quantificar a área foliar afetada, "
        "informação necessária à verificação integral da hipótese H2;",
        "explicabilidade parcial: é possível identificar onde o modelo concentrou sua "
        "atenção, mas não qual característica visual determinou a decisão;",
        "revocação da classe fúngica abaixo de 0,90: técnicas como a sobreamostragem da "
        "classe minoritária ou a ponderação da função de perda, indicadas para esse "
        "cenário, não foram aplicadas;",
        "comparação entre variantes não realizada: a variante de maior porte do modelo, "
        "YOLO11s-cls, não foi treinada, de modo que não se avaliou o ganho de exatidão "
        "proporcionado por um modelo maior;",
        "critério de implantação do verificador não atingido: a segunda versão do "
        "verificador de domínio não atendeu aos critérios (1) e (5) da "
        "{tab:criterios_verif}, e sua implantação decorre de decisão tomada após o "
        "conhecimento do resultado ({sec:sec_dom_decisao});",
        "evidência do verificador no ambiente real: o teste da segunda versão com a C270 "
        "compreende uma única sessão de captura, com 75 imagens de alface e 48 sem alface, "
        "com a câmera na mão, fora do enquadramento de operação, e com as mesmas plantas do "
        "treinamento. Os intervalos de confiança são largos, e o resultado de 75 de 75 "
        "imagens de alface aceitas é provavelmente otimista para outras plantas e para a "
        "câmera fixa;",
        "calibração e reutilização do teste: o subconjunto de validação da segunda versão "
        "não contém imagens da C270, de modo que o limiar foi escolhido sem exemplos do "
        "ambiente real; e as 630 imagens de teste do Kaggle e dos conjuntos públicos são as "
        "mesmas nas duas versões, já examinadas na análise dos erros da primeira, de modo "
        "que apenas as imagens da C270 são inéditas para a segunda;",
        "imagens sem alface não cobertas: a jardineira com substrato e sem planta e o vaso "
        "vazio, provavelmente as imagens sem alface mais frequentes para uma câmera fixa, "
        "nunca foram fotografados, e o desempenho do verificador nesses casos não foi "
        "medido. O verificador aceitou plantas de outras espécies ({sec:sec_dom_erros}); "
        "em operação, outra planta diante da câmera pode receber diagnóstico, com "
        "frequência não medida;",
        "procedência dos dados do verificador: a deduplicação por resumo MD5 não detecta "
        "cópias aproximadas, como a mesma fotografia em outro tamanho, comuns em imagens "
        "obtidas da internet, como as do PlantDoc, e esse risco não foi verificado; a "
        "licença de uso do PlantVillage não está declarada no repositório de origem;",
        "rastreabilidade do treinamento da segunda versão: o histórico por época e os "
        "gráficos da execução não foram preservados, e o registro do treinamento é o *log* "
        "da execução;",
        "medições em nuvem restritas: a latência no serviço em nuvem foi medida em uma "
        "única sessão, e o tempo de inicialização a frio do contêiner, em uma única "
        "medição ({sec:sec_res_latencia}); a latência do fluxo completo, com o "
        "armazenamento das imagens e a função de orquestração, e o custo mensal efetivo não "
        "foram medidos;",
        "integração com o banco de dados: a coluna de classe do banco de dados aceita "
        "apenas as três classes de doença; enquanto não aceitar o valor “" +
        CLASSE_NAO_ALFACE + "”, as respostas das imagens recusadas pelo verificador não "
        "podem ser gravadas."])

    H(2, "Melhorias futuras do verificador de domínio", "sec_melhorias")
    P("A análise dos erros da segunda versão ({sec:sec_dom_erros}) indica as seguintes "
      "melhorias para uma próxima versão do verificador, em ordem de prioridade. Nenhuma "
      "foi implementada nem medida.")
    AL(["outras plantas do ambiente real como imagens sem alface: fotografar com a C270, no "
        "enquadramento de operação, plantas que não são alface, incluindo as espécies "
        "aceitas no teste, e incluí-las no treinamento, pois nenhuma imagem sem alface da "
        "C270 empregada no treinamento da segunda versão mostrava outra planta;",
        "hortaliças folhosas semelhantes à alface, como rúcula, agrião, couve, espinafre, "
        "repolho, chicória e escarola, além das plantas espontâneas da estufa, que "
        "constituem as imagens sem alface mais difíceis e mais prováveis em operação;",
        "mais folhas e mudas de outras culturas, sadias e com lesões, do tipo de imagem do "
        "PlantDoc aceito no teste;",
        "imagens de alface em detalhe e com doença avançada capturadas pela C270, para que "
        "o verificador não dependa da forma da folha inteira, acompanhadas da auditoria das "
        "imagens de microscópio ou de macrofotografia e das que trazem artefatos gravados "
        "no conjunto do Kaggle, para decidir se pertencem ao domínio de operação, sem "
        "alterar o teste já medido;",
        "imagens da C270 das duas classes no subconjunto de validação, para que o limiar "
        "seja escolhido com exemplos do ambiente real;",
        "sessões de captura com a C270 fixa no enquadramento de operação, em várias datas, "
        "para a validação e o teste da próxima versão, incluindo a jardineira com substrato "
        "e sem planta e o vaso vazio;",
        "registro, em operação, da probabilidade atribuída pelo verificador a cada captura, "
        "para que as imagens reais sejam rotuladas posteriormente e o desempenho com a "
        "câmera fixa seja medido;",
        "comparação no espaço de características ({qua:alternativas}), como verificação "
        "complementar para imagens muito diferentes das do treinamento."])
    P("Qualquer nova versão exige subconjuntos de validação e de teste inéditos, pois o "
      "subconjunto de teste da segunda versão já foi empregado na medição única.")

    # ================================================================== #
    H(1, "REFERÊNCIAS", "cap_ref", )
    # ================================================================== #
    refs = [
        "BARBEDO, J. G. A. Plant disease identification from individual lesions and spots using deep learning. **Biosystems Engineering**, v. 180, p. 96-107, 2019.",
        "CUBUK, E. D. et al. RandAugment: practical automated data augmentation with a reduced search space. In: IEEE/CVF CONFERENCE ON COMPUTER VISION AND PATTERN RECOGNITION WORKSHOPS, 2020, Seattle. **Proceedings** [...]. Piscataway: IEEE, 2020. p. 702-703.",
        "GEIRHOS, R. et al. Shortcut learning in deep neural networks. **Nature Machine Intelligence**, v. 2, p. 665-673, 2020.",
        "GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. **Deep learning**. Cambridge: MIT Press, 2016.",
        "GUO, C. et al. On calibration of modern neural networks. In: INTERNATIONAL CONFERENCE ON MACHINE LEARNING, 34., 2017, Sydney. **Proceedings** [...]. PMLR, 2017. p. 1321-1330.",
        "HENDRYCKS, D.; GIMPEL, K. A baseline for detecting misclassified and out-of-distribution examples in neural networks. In: INTERNATIONAL CONFERENCE ON LEARNING REPRESENTATIONS, 5., 2017, Toulon. **Proceedings** [...]. 2017.",
        "HUGHES, D. P.; SALATHÉ, M. An open access repository of images on plant health to enable the development of mobile disease diagnostics. **arXiv**, 1511.08060, 2015.",
        "JOCHER, G.; QIU, J. **Ultralytics YOLO11**. Versão 8.4.95. 2024. Disponível em: https://docs.ultralytics.com. Acesso em: [[preencher:data de acesso]].",
        "KAUFMAN, S. et al. Leakage in data mining: formulation, detection, and avoidance. **ACM Transactions on Knowledge Discovery from Data**, v. 6, n. 4, p. 1-21, 2012.",
        "LECUN, Y.; BENGIO, Y.; HINTON, G. Deep learning. **Nature**, v. 521, n. 7553, p. 436-444, 2015.",
        "LEE, K. et al. A simple unified framework for detecting out-of-distribution samples and adversarial attacks. In: CONFERENCE ON NEURAL INFORMATION PROCESSING SYSTEMS, 32., 2018, Montreal. **Advances in Neural Information Processing Systems**, v. 31, 2018.",
        "LIN, T.-Y. et al. Microsoft COCO: common objects in context. In: EUROPEAN CONFERENCE ON COMPUTER VISION, 13., 2014, Zurique. **Proceedings** [...]. Cham: Springer, 2014. p. 740-755.",
        "LOSHCHILOV, I.; HUTTER, F. SGDR: stochastic gradient descent with warm restarts. In: INTERNATIONAL CONFERENCE ON LEARNING REPRESENTATIONS, 5., 2017, Toulon. **Proceedings** [...]. 2017.",
        "LOSHCHILOV, I.; HUTTER, F. Decoupled weight decay regularization. In: INTERNATIONAL CONFERENCE ON LEARNING REPRESENTATIONS, 7., 2019, New Orleans. **Proceedings** [...]. 2019.",
        "MERKEL, D. Docker: lightweight Linux containers for consistent development and deployment. **Linux Journal**, v. 2014, n. 239, art. 2, 2014.",
        "MOHANTY, S. P.; HUGHES, D. P.; SALATHÉ, M. Using deep learning for image-based plant disease detection. **Frontiers in Plant Science**, v. 7, art. 1419, 2016.",
        "OLSEN, A. et al. DeepWeeds: a multiclass weed species image dataset for deep learning. **Scientific Reports**, v. 9, art. 2058, 2019.",
        "PEDREGOSA, F. et al. Scikit-learn: machine learning in Python. **Journal of Machine Learning Research**, v. 12, p. 2825-2830, 2011.",
        "SELVARAJU, R. R. et al. Grad-CAM: visual explanations from deep networks via gradient-based localization. In: IEEE INTERNATIONAL CONFERENCE ON COMPUTER VISION, 2017, Veneza. **Proceedings** [...]. Piscataway: IEEE, 2017. p. 618-626.",
        "SINGH, D. et al. PlantDoc: a dataset for visual plant disease detection. In: ACM INDIA JOINT INTERNATIONAL CONFERENCE ON DATA SCIENCE AND MANAGEMENT OF DATA, 7., 2020, Hyderabad. **Proceedings** [...]. New York: ACM, 2020. p. 249-253.",
        "WILSON, E. B. Probable inference, the law of succession, and statistical inference. **Journal of the American Statistical Association**, v. 22, n. 158, p. 209-212, 1927.",
        "YOSINSKI, J. et al. How transferable are features in deep neural networks? In: CONFERENCE ON NEURAL INFORMATION PROCESSING SYSTEMS, 28., 2014, Montreal. **Advances in Neural Information Processing Systems**, v. 27, p. 3320-3328, 2014.",
        "ZEILER, M. D.; FERGUS, R. Visualizing and understanding convolutional networks. In: EUROPEAN CONFERENCE ON COMPUTER VISION, 13., 2014, Zurique. **Proceedings** [...]. Cham: Springer, 2014. p. 818-833.",
        "[[preencher:referência do conjunto de dados obtido no Kaggle e referência fitopatológica]]",
    ]
    for r in refs:
        b.append(("ref", r))

    # Substitui marcador de tempo em todos os parágrafos.
    tempo_min = br(tempo_total / 60, 1)
    final = []
    for bloco in b:
        if bloco[0] == "p":
            final.append(("p", bloco[1].replace("{tempo_min}", tempo_min)))
        else:
            final.append(bloco)
    return final


# ====================================================================== #
# Geração do WordprocessingML
# ====================================================================== #

class Documento:
    def __init__(self, blocos: list[tuple], figuras: dict[str, Path]):
        self.blocos = blocos
        self.figuras = figuras
        self.notas: list[str] = []
        self.imagens: list[tuple[str, Path]] = []
        self.rotulos: dict[str, str] = {}
        self._numerar()

    # ------------------------------------------------------------- #
    def _numerar(self) -> None:
        cont = {"tab": 0, "qua": 0, "fig": 0, "eq": 0}
        secoes = [0, 0, 0, 0]
        self.num_secao: dict[int, str] = {}
        for i, bl in enumerate(self.blocos):
            tipo = bl[0]
            if tipo == "h":
                nivel, rotulo = bl[1], bl[3]
                if bl[2] == "REFERÊNCIAS":
                    self.num_secao[i] = ""
                    continue
                secoes[nivel - 1] += 1
                for k in range(nivel, 4):
                    secoes[k] = 0
                num = ".".join(str(s) for s in secoes[:nivel])
                self.num_secao[i] = num
                if rotulo:
                    self.rotulos[f"sec:{rotulo}"] = f"Seção {num}" if nivel > 1 else f"Capítulo {num}"
            elif tipo in cont:
                cont[tipo] += 1
                rotulo = bl[1] if tipo != "eq" else bl[2]
                nomes = {"tab": "Tabela", "qua": "Quadro", "fig": "Figura", "eq": "Equação"}
                self.rotulos[f"{tipo}:{rotulo}"] = (
                    f"{nomes[tipo]} {cont[tipo]}" if tipo != "eq" else f"Equação ({cont[tipo]})"
                )
                if tipo == "eq":
                    self.num_eq = getattr(self, "num_eq", {})
                    self.num_eq[i] = cont[tipo]
                else:
                    self.num_ilustracao = getattr(self, "num_ilustracao", {})
                    self.num_ilustracao[i] = cont[tipo]

    def _resolver(self, texto: str) -> str:
        def troca(m):
            prefixo, nome = m.group(1), m.group(2)
            # {eqn:x} devolve só o número, para "as Equações (9) a (11)".
            chave = f"{'eq' if prefixo == 'eqn' else prefixo}:{nome}"
            if chave not in self.rotulos:
                raise KeyError(f"Rótulo não definido: {chave}")
            valor = self.rotulos[chave]
            return valor.replace("Equação ", "") if prefixo == "eqn" else valor
        return re.sub(r"\{(tab|qua|fig|eqn|eq|sec):([a-z_0-9]+)\}", troca, texto)

    # ------------------------------------------------------------- #
    def _runs(self, texto: str, base_rpr: str = "") -> str:
        texto = self._resolver(texto)
        padrao = re.compile(
            r"(\[\[nota:.+?\]\]|\[\[preencher:.+?\]\]|\*\*.+?\*\*|\*.+?\*|~.+?~|\^.+?\^)",
            re.S,
        )
        partes = padrao.split(texto)
        xml = []
        for parte in partes:
            if not parte:
                continue
            if parte.startswith("[[nota:"):
                self.notas.append(parte[7:-2])
                nid = len(self.notas)
                xml.append(f'<w:r><w:rPr><w:rStyle w:val="RefNota"/></w:rPr>'
                           f'<w:footnoteReference w:id="{nid}"/></w:r>')
            elif parte.startswith("[[preencher:"):
                t = escape("[PREENCHER: " + parte[12:-2] + "]")
                xml.append(f'<w:r><w:rPr>{base_rpr}<w:highlight w:val="yellow"/></w:rPr>'
                           f'<w:t xml:space="preserve">{t}</w:t></w:r>')
            elif parte.startswith("**"):
                xml.append(self._runs_simples(parte[2:-2], base_rpr + "<w:b/>"))
            elif parte.startswith("*") and len(parte) > 2:
                xml.append(self._runs_simples(parte[1:-1], base_rpr + "<w:i/>"))
            elif parte.startswith("~") and len(parte) > 2:
                xml.append(self._runs_simples(parte[1:-1], base_rpr + '<w:vertAlign w:val="subscript"/>'))
            elif parte.startswith("^") and len(parte) > 2:
                xml.append(self._runs_simples(parte[1:-1], base_rpr + '<w:vertAlign w:val="superscript"/>'))
            else:
                xml.append(self._runs_simples(parte, base_rpr))
        return "".join(xml)

    def _runs_simples(self, texto: str, rpr: str) -> str:
        # Permite itálico dentro de negrito e vice-versa.
        if "*" in texto and not rpr.endswith("<w:i/>"):
            pedacos = re.split(r"(\*.+?\*)", texto)
            return "".join(
                self._run(p[1:-1], rpr + "<w:i/>") if p.startswith("*") and len(p) > 2
                else self._run(p, rpr)
                for p in pedacos if p
            )
        return self._run(texto, rpr)

    @staticmethod
    def _run(texto: str, rpr: str) -> str:
        rpr_xml = f"<w:rPr>{rpr}</w:rPr>" if rpr else ""
        return f'<w:r>{rpr_xml}<w:t xml:space="preserve">{escape(texto)}</w:t></w:r>'

    @staticmethod
    def _p(conteudo: str, estilo: str = "", ppr_extra: str = "") -> str:
        est = f'<w:pStyle w:val="{estilo}"/>' if estilo else ""
        return f"<w:p><w:pPr>{est}{ppr_extra}</w:pPr>{conteudo}</w:p>"

    # ------------------------------------------------------------- #
    def _tabela(self, i, bl, quadro: bool) -> str:
        _, rotulo, titulo, cab, linhas, larguras, alinh, fonte = bl
        num = self.num_ilustracao[i]
        nome = "Quadro" if quadro else "Tabela"
        xml = [self._p(self._runs(f"{nome} {num} – {titulo}"), "Legenda",
                       "<w:keepNext/>")]
        if quadro:
            bordas = ('<w:tblBorders>' + "".join(
                f'<w:{l} w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
                for l in ("top", "left", "bottom", "right", "insideH", "insideV")) +
                '</w:tblBorders>')
        else:
            bordas = ('<w:tblBorders>'
                      '<w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
                      '<w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
                      '</w:tblBorders>')
        grade = "".join(f'<w:gridCol w:w="{w}"/>' for w in larguras)
        xml.append(
            f'<w:tbl><w:tblPr><w:tblW w:w="{sum(larguras)}" w:type="dxa"/>'
            f'<w:jc w:val="center"/>{bordas}<w:tblLayout w:type="fixed"/>'
            '<w:tblCellMar><w:left w:w="85" w:type="dxa"/><w:right w:w="85" w:type="dxa"/>'
            '</w:tblCellMar></w:tblPr>'
            f'<w:tblGrid>{grade}</w:tblGrid>')
        mapa = {"l": "left", "r": "right", "c": "center"}
        todas = [cab] + linhas
        for li, linha in enumerate(todas):
            cabecalho = li == 0
            ultima = li == len(todas) - 1
            trpr = "<w:trPr><w:tblHeader/><w:cantSplit/></w:trPr>" if cabecalho else "<w:trPr><w:cantSplit/></w:trPr>"
            xml.append(f"<w:tr>{trpr}")
            for ci, celula in enumerate(linha):
                tcb = ""
                if cabecalho and not quadro:
                    tcb = ('<w:tcBorders><w:bottom w:val="single" w:sz="6" w:space="0" '
                           'w:color="000000"/></w:tcBorders>')
                if not quadro and ultima and linha[0].startswith("**Total") or (
                        not quadro and ultima and "Média" in linha[0]):
                    tcb = ('<w:tcBorders><w:top w:val="single" w:sz="4" w:space="0" '
                           'w:color="000000"/></w:tcBorders>')
                sombra = ('<w:shd w:val="clear" w:color="auto" w:fill="EDEDED"/>'
                          if cabecalho and quadro else "")
                jc = "center" if cabecalho else mapa[alinh[ci]]
                rpr = "<w:b/>" if cabecalho else ""
                conteudo = self._runs(celula, rpr)
                xml.append(
                    f'<w:tc><w:tcPr><w:tcW w:w="{larguras[ci]}" w:type="dxa"/>{tcb}{sombra}'
                    f'<w:vAlign w:val="center"/></w:tcPr>'
                    + self._p(conteudo, "TextoTabela", f'<w:jc w:val="{jc}"/>')
                    + "</w:tc>")
            xml.append("</w:tr>")
        xml.append("</w:tbl>")
        xml.append(self._p(self._runs(fonte), "Fonte"))
        return "".join(xml)

    def _figura(self, i, bl) -> str:
        _, rotulo, titulo, chave, largura_cm, fonte = bl
        num = self.num_ilustracao[i]
        caminho = self.figuras[chave]
        larg_px, alt_px = _dimensoes_png(caminho)
        cx = int(largura_cm * 360000)
        cy = int(cx * alt_px / larg_px)
        max_cy = int(21.0 * 360000)
        if cy > max_cy:
            cx = int(cx * max_cy / cy)
            cy = max_cy
        self.imagens.append((f"rIdImg{len(self.imagens) + 1}", caminho))
        rid = self.imagens[-1][0]
        pid = len(self.imagens)
        desenho = (
            f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
            f'<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{pid}" name="Figura {num}"/>'
            '<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:pic><pic:nvPicPr><pic:cNvPr id="{pid}" name="{caminho.name}"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>'
            '</a:graphicData></a:graphic></wp:inline></w:drawing></w:r>')
        return (self._p(self._runs(f"Figura {num} – {titulo}"), "Legenda", "<w:keepNext/>")
                + self._p(desenho, "Imagem", "<w:keepNext/>")
                + self._p(self._runs(fonte), "Fonte"))

    def _equacao(self, i, bl) -> str:
        num = self.num_eq[i]
        conteudo = (f'<w:r><w:tab/></w:r><m:oMath>{bl[1]}</m:oMath>'
                    f'<w:r><w:tab/><w:t>({num})</w:t></w:r>')
        return self._p(conteudo, "Equacao")

    def _titulo(self, i, bl) -> str:
        nivel, texto = bl[1], bl[2]
        num = self.num_secao[i]
        if nivel <= 2:
            texto = texto.upper().replace("*SOFTMAX*", "*SOFTMAX*")
        rotulo = f"{num} {texto}" if num else texto
        estilo = "TituloSemNumero" if not num else f"Heading{nivel}"
        return self._p(self._runs(rotulo), estilo)

    def corpo(self) -> str:
        partes = []
        for i, bl in enumerate(self.blocos):
            tipo = bl[0]
            if tipo == "h":
                partes.append(self._titulo(i, bl))
            elif tipo == "p":
                partes.append(self._p(self._runs(bl[1]), "Corpo"))
            elif tipo == "al":
                letras = "abcdefghijklmnopqrstuvwxyz"
                for k, item in enumerate(bl[1]):
                    partes.append(self._p(self._runs(f"{letras[k]}) {item}"), "Alinea"))
            elif tipo == "eq":
                partes.append(self._equacao(i, bl))
            elif tipo == "tab":
                partes.append(self._tabela(i, bl, quadro=False))
            elif tipo == "qua":
                partes.append(self._tabela(i, bl, quadro=True))
            elif tipo == "fig":
                partes.append(self._figura(i, bl))
            elif tipo == "ref":
                partes.append(self._p(self._runs(bl[1]), "Referencia"))
        return "".join(partes)


def _dimensoes_png(caminho: Path) -> tuple[int, int]:
    with caminho.open("rb") as arq:
        cabecalho = arq.read(24)
    largura, altura = struct.unpack(">II", cabecalho[16:24])
    return largura, altura


NS = (
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
    'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
    'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
    'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"'
)


def _ppr(keep_next=False, quebra=False, tabs="", antes=0, depois=0, linha=360,
         ind="", jc="", outline=None) -> str:
    """Monta um ``w:pPr`` respeitando a ordem de elementos exigida pelo esquema OOXML."""
    xml = ""
    if keep_next:
        xml += "<w:keepNext/>"
    if quebra:
        xml += "<w:pageBreakBefore/>"
    xml += tabs
    xml += (f'<w:spacing w:before="{antes}" w:after="{depois}" '
            f'w:line="{linha}" w:lineRule="auto"/>')
    xml += ind
    if jc:
        xml += f'<w:jc w:val="{jc}"/>'
    if outline is not None:
        xml += f'<w:outlineLvl w:val="{outline}"/>'
    return xml


def _estilos() -> str:
    f = f'<w:rFonts w:ascii="{FONTE}" w:hAnsi="{FONTE}" w:cs="{FONTE}" w:eastAsia="{FONTE}"/>'

    def estilo_p(sid, nome, ppr, rpr, embutido=False):
        custom = "" if embutido else ' w:customStyle="1"'
        return (f'<w:style w:type="paragraph"{custom} w:styleId="{sid}">'
                f'<w:name w:val="{nome}"/><w:basedOn w:val="Normal"/><w:qFormat/>'
                f'<w:pPr>{ppr}</w:pPr><w:rPr>{rpr}</w:rPr></w:style>')

    pequeno = '<w:sz w:val="20"/><w:szCs w:val="20"/>'
    sem_recuo = '<w:ind w:firstLine="0"/>'
    tabs_eq = (f'<w:tabs><w:tab w:val="center" w:pos="{LARGURA_TEXTO // 2}"/>'
               f'<w:tab w:val="right" w:pos="{LARGURA_TEXTO}"/></w:tabs>')
    return (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles {NS}>'
        f'<w:docDefaults><w:rPrDefault><w:rPr>{f}<w:sz w:val="24"/><w:szCs w:val="24"/>'
        '<w:lang w:val="pt-BR"/></w:rPr></w:rPrDefault>'
        '<w:pPrDefault><w:pPr><w:spacing w:after="0" w:line="360" w:lineRule="auto"/></w:pPr>'
        '</w:pPrDefault></w:docDefaults>'
        '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>'
        + estilo_p("Corpo", "Texto ABNT",
                   _ppr(ind='<w:ind w:firstLine="709"/>', jc="both"), "")
        + estilo_p("Alinea", "Alínea ABNT",
                   _ppr(ind='<w:ind w:left="1134" w:hanging="425"/>', jc="both"), "")
        + estilo_p("Heading1", "heading 1",
                   _ppr(keep_next=True, quebra=True, depois=360, ind=sem_recuo, jc="left", outline=0),
                   "<w:b/><w:caps/>", embutido=True)
        + estilo_p("TituloSemNumero", "Título sem indicativo numérico",
                   _ppr(keep_next=True, quebra=True, depois=360, ind=sem_recuo, jc="center", outline=0),
                   "<w:b/><w:caps/>")
        + estilo_p("Heading2", "heading 2",
                   _ppr(keep_next=True, antes=360, depois=360, ind=sem_recuo, jc="left", outline=1),
                   "<w:caps/>", embutido=True)
        + estilo_p("Heading3", "heading 3",
                   _ppr(keep_next=True, antes=360, depois=360, ind=sem_recuo, jc="left", outline=2),
                   "<w:b/>", embutido=True)
        + estilo_p("Heading4", "heading 4",
                   _ppr(keep_next=True, antes=360, depois=360, ind=sem_recuo, jc="left", outline=3),
                   "", embutido=True)
        + estilo_p("Legenda", "Legenda ABNT",
                   _ppr(keep_next=True, antes=240, depois=60, linha=240, jc="center"), pequeno)
        + estilo_p("Fonte", "Fonte ABNT",
                   _ppr(antes=60, depois=240, linha=240, jc="left"), pequeno)
        + estilo_p("TextoTabela", "Texto de tabela", _ppr(linha=240), pequeno)
        + estilo_p("Imagem", "Imagem", _ppr(keep_next=True, linha=240, jc="center"), "")
        + estilo_p("Equacao", "Equação", _ppr(tabs=tabs_eq, antes=120, depois=120), "")
        + estilo_p("Referencia", "Referência ABNT",
                   _ppr(depois=240, linha=240, jc="left"), "")
        + estilo_p("TextoNota", "Nota de rodapé", _ppr(linha=240, jc="both"), pequeno)
        + '<w:style w:type="character" w:customStyle="1" w:styleId="RefNota"><w:name w:val="Referência de nota"/>'
          '<w:rPr><w:vertAlign w:val="superscript"/></w:rPr></w:style>'
        + '</w:styles>'
    )


def _notas(notas: list[str], doc: Documento) -> str:
    sep = ('<w:footnote w:type="separator" w:id="-1"><w:p><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr>'
           '<w:r><w:separator/></w:r></w:p></w:footnote>'
           '<w:footnote w:type="continuationSeparator" w:id="0"><w:p><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr>'
           '<w:r><w:continuationSeparator/></w:r></w:p></w:footnote>')
    corpo = []
    for i, texto in enumerate(notas, start=1):
        runs = doc._runs_simples(texto, "")
        corpo.append(
            f'<w:footnote w:id="{i}"><w:p><w:pPr><w:pStyle w:val="TextoNota"/></w:pPr>'
            f'<w:r><w:rPr><w:rStyle w:val="RefNota"/></w:rPr><w:footnoteRef/></w:r>'
            f'<w:r><w:t xml:space="preserve"> </w:t></w:r>{runs}</w:p></w:footnote>')
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:footnotes {NS}>'
            + sep + "".join(corpo) + "</w:footnotes>")


def _cabecalho() -> str:
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:hdr {NS}>'
            '<w:p><w:pPr><w:jc w:val="right"/></w:pPr>'
            '<w:r><w:rPr><w:sz w:val="20"/></w:rPr><w:fldChar w:fldCharType="begin"/></w:r>'
            '<w:r><w:rPr><w:sz w:val="20"/></w:rPr><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
            '<w:r><w:rPr><w:sz w:val="20"/></w:rPr><w:fldChar w:fldCharType="separate"/></w:r>'
            '<w:r><w:rPr><w:sz w:val="20"/></w:rPr><w:t>1</w:t></w:r>'
            '<w:r><w:rPr><w:sz w:val="20"/></w:rPr><w:fldChar w:fldCharType="end"/></w:r>'
            '</w:p></w:hdr>')


def gravar_docx(doc: Documento, destino: Path) -> None:
    corpo = doc.corpo()  # gera também notas e imagens
    sect = ('<w:sectPr><w:headerReference w:type="default" r:id="rIdCab"/>'
            '<w:footnotePr><w:numFmt w:val="decimal"/></w:footnotePr>'
            '<w:pgSz w:w="11906" w:h="16838"/>'
            '<w:pgMar w:top="1701" w:right="1134" w:bottom="1134" w:left="1701" '
            'w:header="1134" w:footer="709" w:gutter="0"/></w:sectPr>')
    documento = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                 f'<w:document {NS}><w:body>{corpo}{sect}</w:body></w:document>')
    rels_img = "".join(
        f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
        f'Target="media/{caminho.name}"/>' for rid, caminho in doc.imagens)
    rels_doc = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rIdEst" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        '<Relationship Id="rIdConf" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>'
        '<Relationship Id="rIdNotas" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footnotes" Target="footnotes.xml"/>'
        '<Relationship Id="rIdCab" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/header" Target="header1.xml"/>'
        + rels_img + '</Relationships>')
    configuracoes = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:settings {NS}>'
        '<w:defaultTabStop w:val="709"/>'
        '<w:footnotePr><w:footnote w:id="-1"/><w:footnote w:id="0"/></w:footnotePr>'
        '<w:compat><w:compatSetting w:name="compatibilityMode" w:uri="http://schemas.microsoft.com/office/word" w:val="15"/></w:compat>'
        '<m:mathPr><m:mathFont m:val="Cambria Math"/><m:dispDef/></m:mathPr>'
        '</w:settings>')
    tipos = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="png" ContentType="image/png"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
        '<Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>'
        '<Override PartName="/word/footnotes.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml"/>'
        '<Override PartName="/word/header1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"/>'
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '</Types>')
    rels_raiz = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
        '</Relationships>')
    agora = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    core = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        '<dc:title>Módulo de inteligência artificial — fundamentação, desenvolvimento e resultados</dc:title>'
        '<dc:language>pt-BR</dc:language>'
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{agora}</dcterms:created>'
        '</cp:coreProperties>')
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", tipos)
        z.writestr("_rels/.rels", rels_raiz)
        z.writestr("docProps/core.xml", core)
        z.writestr("word/document.xml", documento)
        z.writestr("word/styles.xml", _estilos())
        z.writestr("word/settings.xml", configuracoes)
        z.writestr("word/footnotes.xml", _notas(doc.notas, doc))
        z.writestr("word/header1.xml", _cabecalho())
        z.writestr("word/_rels/document.xml.rels", rels_doc)
        for _, caminho in doc.imagens:
            z.write(caminho, f"word/media/{caminho.name}")


def main() -> None:
    SAIDA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    hist = ler_historico()

    figuras = {
        "fluxo": FIG_DIR / "figura_fluxo_inferencia.png",
        "arquitetura": FIG_DIR / "figura_arquitetura_yolo11n_cls.png",
        "colab": FIG_DIR / "figura_fluxo_colab.png",
        "cascata": FIG_DIR / "figura_verificacao_alface.png",
        "implantacao": FIG_DIR / "figura_implantacao.png",
        "curvas": FIG_DIR / "figura_curvas_treinamento.png",
        "matriz": FIG_DIR / "figura_matriz_confusao.png",
    }
    figura_fluxo_inferencia(figuras["fluxo"])
    figura_arquitetura(figuras["arquitetura"])
    figura_fluxo_colab(figuras["colab"])
    figura_cascata(figuras["cascata"])
    figura_implantacao(figuras["implantacao"])
    figura_curvas(hist, figuras["curvas"])
    figura_matriz(figuras["matriz"])

    doc = Documento(construir_conteudo(hist), figuras)
    gravar_docx(doc, SAIDA_DOCX)
    print(f"Documento gerado: {SAIDA_DOCX.relative_to(RAIZ)}")
    print(f"Figuras em:       {FIG_DIR.relative_to(RAIZ)}")
    print(f"Épocas lidas:     {len(hist)}")


if __name__ == "__main__":
    main()
