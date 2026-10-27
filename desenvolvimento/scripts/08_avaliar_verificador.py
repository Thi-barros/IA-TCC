"""Etapa 8 — Avaliação do verificador de domínio e da cascata.

Dois modos:

* ``--split val`` — revocação por grupo (folhas sadias, bacterianas,
  fúngicas, câmera do projeto, cada fonte de "não alface") e **curva de
  limiar**, base para escolher o limiar. Pode rodar quantas vezes for preciso.
* ``--split test`` — as mesmas métricas no limiar escolhido e a **cascata de
  ponta a ponta** sobre as predições oficiais do classificador no teste
  (``reports/predicoes.csv``), sem executá-lo de novo. Roda **uma única vez
  por versão**: recusa-se a continuar se o JSON de teste da versão já existir.

``--rotulo`` identifica a versão do verificador. Sem ele valem os nomes da v1
(``modelos/yolo11n_cls_dominio.pt``, ``reports/verificador_test.json``...);
com ``--rotulo v2``, pesos e saídas levam a versão no nome
(``modelos/yolo11n_cls_dominio_v2.pt``, ``reports/verificador_v2_test.json``...),
sem sobrescrever os arquivos da v1.

A latência das duas etapas é medida sempre em imagens do ``val``.

Uso:
    python desenvolvimento/scripts/08_avaliar_verificador.py --split val --rotulo v2
    python desenvolvimento/scripts/08_avaliar_verificador.py --split test --rotulo v2 --limiar 0.35

Referência: docs/VERIFICADOR_DOMINIO.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path, PureWindowsPath

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd

from ia_alface import avaliacao_dominio as ad
from ia_alface.cascata import AnalisadorCascata, ImagemForaDoDominio
from ia_alface.config import CAMINHOS, ConfigAvaliacaoDominio
from ia_alface.dados_dominio import FONTE_KAGGLE
from ia_alface.inferencia import DetectorDoencasAlface
from ia_alface.verificacao import LIMIAR_ALFACE_PADRAO, VerificadorDominio

logger = logging.getLogger("avaliar_verificador")

#: Rótulos dos agregados no relatório.
_NOMES_AGREGADOS = {
    "alface_folhas_doentes": "Alface aceita — folhas doentes (Kaggle)",
    "alface_total": "Alface aceita — todas",
    "alface_camera_projeto": "Alface aceita — câmera do projeto (C270)",
    "nao_alface_total": "Não alface recusada — todas",
    "nao_alface_camera_projeto": "Não alface recusada — câmera do projeto (C270)",
}


def _texto(medida: dict) -> str:
    """Formata uma proporção como 'acertos/total (taxa; IC95 a–b)'."""
    if not medida["total"]:
        return "não medido (0 imagens)"
    return (
        f"{medida['acertos']}/{medida['total']} ({medida['taxa']:.1%}; "
        f"IC95 {medida['ic95_inf']:.1%}–{medida['ic95_sup']:.1%})"
    )


def _situacao(medida: dict, meta: float) -> str:
    """Compara com a meta pela razão exata — a taxa arredondada poderia passar raspando."""
    if not medida["total"]:
        return "não medido"
    return "atingida" if medida["acertos"] / medida["total"] >= meta else "**não atingida**"


def _registros(tabela: pd.DataFrame) -> list[dict]:
    """Converte em registros trocando NaN por None (JSON válido e '—' no relatório)."""
    return tabela.astype(object).where(tabela.notna(), None).to_dict("records")


def _carregar_inventario(caminho: Path) -> pd.DataFrame:
    """Lê o inventário gerado pelo script 06."""
    if not caminho.is_file():
        raise FileNotFoundError(
            f"{caminho} não encontrado. Execute "
            "'python desenvolvimento/scripts/06_preparar_dataset_dominio.py' primeiro."
        )
    inventario = pd.read_csv(caminho, encoding="utf-8")
    if "caminho_destino" not in inventario.columns:
        raise ValueError(
            "O inventário não tem 'caminho_destino': o script 06 rodou com "
            "--somente-auditoria. Rode-o sem essa opção."
        )
    for coluna in ("camera", "dia", "estrato"):
        inventario[coluna] = inventario[coluna].fillna("").astype(str)
    return inventario


def _verificar(verificador: VerificadorDominio, linhas: pd.DataFrame) -> pd.DataFrame:
    """Roda o verificador em cada imagem do split."""
    probabilidades: list[float] = []
    for indice, caminho in enumerate(linhas["caminho_destino"], start=1):
        probabilidades.append(verificador.verificar(caminho).probabilidade_alface)
        if indice % 250 == 0:
            logger.info("  %d/%d imagens verificadas", indice, len(linhas))
    resultado = linhas.copy()
    resultado["prob_alface"] = probabilidades
    resultado["grupo"] = [
        ad.grupo_de_avaliacao(fonte, estrato, camera)
        for fonte, estrato, camera in zip(
            resultado["fonte"], resultado["estrato"], resultado["camera"]
        )
    ]
    return resultado


def _estatisticas(tempos: list[float]) -> dict:
    vetor = np.asarray(tempos, dtype=float)
    return {
        "n": len(vetor),
        "mediana": round(float(np.median(vetor)), 2),
        "media": round(float(np.mean(vetor)), 2),
        "p95": round(float(np.percentile(vetor, 95)), 2),
        "p99": round(float(np.percentile(vetor, 99)), 2),
        "minimo": round(float(vetor.min()), 2),
        "maximo": round(float(vetor.max()), 2),
    }


def _medir_latencia(
    verificador: VerificadorDominio,
    caminho_classificador: Path,
    imagens: list[str],
    config: ConfigAvaliacaoDominio,
    dispositivo: str | None,
) -> dict:
    """Mede o tempo de inferência das duas etapas juntas, imagem a imagem.

    Mesmo critério de ``05_benchmark_latencia.py``: soma dos tempos de
    inferência, sem a decodificação do arquivo, com aquecimento descartado.
    """
    detector = DetectorDoencasAlface(caminho_classificador, dispositivo=dispositivo)
    analisador = AnalisadorCascata(detector, verificador)

    def uma(caminho: str) -> tuple[float, float]:
        try:
            resultado = analisador.analisar(caminho)
            return resultado.tempo_total_ms, resultado.verificacao.tempo_inferencia_ms
        except ImagemForaDoDominio as recusa:
            tempo = recusa.verificacao.tempo_inferencia_ms
            return tempo, tempo

    for caminho in imagens[: config.aquecimento_latencia]:
        uma(caminho)
    medidas = [uma(caminho) for caminho in imagens]
    return {
        "dispositivo": detector.dispositivo,
        "aquecimento": min(config.aquecimento_latencia, len(imagens)),
        "duas_etapas": _estatisticas([total for total, _ in medidas]),
        "so_verificador": _estatisticas([verificacao for _, verificacao in medidas]),
    }


def _carregar_predicoes_oficiais(caminho: Path) -> pd.DataFrame:
    """Lê as predições oficiais do classificador e confere que são do teste."""
    if not caminho.is_file():
        raise FileNotFoundError(f"Predições oficiais do classificador não encontradas: {caminho}")
    predicoes = pd.read_csv(caminho, encoding="utf-8")
    splits = {PureWindowsPath(c).parent.parent.name for c in predicoes["caminho"]}
    if splits != {"test"}:
        raise ValueError(
            f"{caminho} não contém só o split de teste (encontrado: {sorted(splits)})."
        )
    return predicoes


def _escrever_relatorio(resultado: dict, caminho_md: Path) -> None:
    """Monta o relatório Markdown da avaliação."""
    config = resultado["config_avaliacao"]
    agregados = resultado["agregados"]
    linhas: list[str] = []
    ad_ = linhas.append

    ad_(f"# Avaliação do verificador de domínio — split `{resultado['split']}`")
    ad_("")
    ad_("*Gerado automaticamente por `desenvolvimento/scripts/08_avaliar_verificador.py` em "
        f"{resultado['gerado_em']}.*")
    ad_("")
    ad_("| Parâmetro | Valor |")
    ad_("| --- | --- |")
    ad_(f"| Versão do verificador | {resultado['versao']} |")
    ad_(f"| Pesos do verificador | `{resultado['pesos']}` |")
    ad_(f"| Limiar de aceitação | {resultado['limiar']:.2f} |")
    ad_(f"| Imagens avaliadas | {resultado['imagens']} |")
    ad_(f"| Dispositivo | {resultado['dispositivo']} |")
    ad_("")
    if resultado["split"] == "test":
        ad_("> Medição **única** do split de teste. O limiar foi escolhido antes, no `val`. "
            "Nenhum ajuste pode ser feito com base nestes números.")
    else:
        ad_("> Split de **validação**: use a curva da Seção 3 para escolher o limiar. "
            "Estes números não são o resultado oficial.")
    ad_("")

    ad_("## 1. Metas")
    ad_("")
    ad_("| Critério | Medido | Meta | Situação |")
    ad_("| --- | --- | ---: | --- |")
    doentes = agregados["alface_folhas_doentes"]
    ad_(f"| Não suprimir doenças: alface aceita em folhas doentes | {_texto(doentes)} | "
        f"≥ {config['meta_alface_doentes']:.0%} | {_situacao(doentes, config['meta_alface_doentes'])} |")
    nao_alface = agregados["nao_alface_total"]
    ad_(f"| Rejeitar imagens alheias: não alface recusada | {_texto(nao_alface)} | "
        f"≥ {config['meta_nao_alface']:.0%} | {_situacao(nao_alface, config['meta_nao_alface'])} |")
    latencia = resultado.get("latencia")
    if latencia:
        mediana = latencia["duas_etapas"]["mediana"]
        situacao = "atingida" if mediana < config["meta_latencia_ms"] else "**não atingida**"
        ad_(f"| Tempo de resposta: mediana das duas etapas ({latencia['dispositivo']}) | "
            f"{mediana:.2f} ms | < {config['meta_latencia_ms']:.0f} ms | {situacao} |")
    else:
        ad_(f"| Tempo de resposta | não medido | < {config['meta_latencia_ms']:.0f} ms | não medido |")
    ad_("")
    ad_("Evidência do ambiente real (sem meta própria; amostra pequena — leia o IC95):")
    ad_("")
    for chave in ("alface_camera_projeto", "nao_alface_camera_projeto"):
        ad_(f"- {_NOMES_AGREGADOS[chave]}: {_texto(agregados[chave])}")
    ad_("")

    ad_("## 2. Por grupo")
    ad_("")
    ad_("| Classe | Grupo | Acertos / total | Taxa | IC95 |")
    ad_("| --- | --- | ---: | ---: | --- |")
    for linha in resultado["por_grupo"]:
        taxa = "—" if linha["taxa"] is None else f"{linha['taxa']:.1%}"
        intervalo = ("—" if linha["ic95_inf"] is None
                     else f"{linha['ic95_inf']:.1%}–{linha['ic95_sup']:.1%}")
        ad_(f"| {linha['classe']} | {linha['grupo']} | {linha['acertos']}/{linha['total']} | "
            f"{taxa} | {intervalo} |")
    ad_("")

    if resultado.get("curva"):
        ad_("## 3. Curva de limiar (somente validação)")
        ad_("")
        ad_("Recusar uma folha doente suprime um alerta; aceitar uma imagem alheia custa "
            "no máximo um alerta indevido. Prefira o maior limiar que mantém a primeira "
            "coluna na meta.")
        ad_("")
        colunas = list(_NOMES_AGREGADOS)
        ad_("| Limiar | " + " | ".join(_NOMES_AGREGADOS[c] for c in colunas) + " |")
        ad_("| ---: | " + " | ".join("---:" for _ in colunas) + " |")
        for ponto in resultado["curva"]:
            valores = ["—" if ponto[c] is None else f"{ponto[c]:.1%}" for c in colunas]
            ad_(f"| {ponto['limiar']:.2f} | " + " | ".join(valores) + " |")
        ad_("")

    cascata = resultado.get("cascata")
    if cascata:
        ad_("## 4. Cascata de ponta a ponta (teste oficial do classificador)")
        ad_("")
        ad_(f"{cascata['imagens']} imagens; diagnóstico só para as aceitas pelo verificador. "
            "O classificador não foi executado de novo: usam-se as predições oficiais.")
        ad_("")
        ad_("| Classe | Imagens | Recusadas pelo verificador | Revocação só classificador | Revocação com a cascata |")
        ad_("| --- | ---: | ---: | --- | --- |")
        for classe, dados in cascata["por_classe"].items():
            ad_(f"| {classe} | {dados['total']} | {dados['recusadas_pelo_verificador']} | "
                f"{_texto(dados['revocacao_classificador'])} | {_texto(dados['revocacao_cascata'])} |")
        ad_("")
        ad_(f"- Alertas do classificador: {cascata['alertas_classificador']}; com a cascata: "
            f"{cascata['alertas_cascata']}.")
        ad_(f"- **Alertas verdadeiros suprimidos pelo verificador: "
            f"{cascata['alertas_verdadeiros_suprimidos']}.**")
        ad_(f"- Acurácia: só classificador {_texto(cascata['acuracia_classificador'])}; "
            f"cascata {_texto(cascata['acuracia_cascata'])}.")
        ad_("")

    ad_("## 5. Latência")
    ad_("")
    if latencia:
        ad_(f"Medida em {latencia['duas_etapas']['n']} imagens do `val` ({latencia['dispositivo']}), "
            f"{latencia['aquecimento']} de aquecimento descartadas; soma dos tempos de "
            "inferência, sem decodificação — mesmo critério do `05_benchmark_latencia.py`.")
        ad_("")
        ad_("| Etapas | Mediana | Média | p95 | p99 | Mín. | Máx. |")
        ad_("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for nome, chave in (("Verificador + classificador", "duas_etapas"),
                            ("Só verificador", "so_verificador")):
            e = latencia[chave]
            ad_(f"| {nome} | {e['mediana']} | {e['media']} | {e['p95']} | {e['p99']} | "
                f"{e['minimo']} | {e['maximo']} |")
    else:
        ad_("Não medida nesta execução.")
    ad_("")

    caminho_md.write_text("\n".join(linhas), encoding="utf-8")
    logger.info("Relatório escrito em %s", caminho_md)


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada do script."""
    padrao = ConfigAvaliacaoDominio()
    parser = argparse.ArgumentParser(description="Avalia o verificador de domínio e a cascata.")
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--rotulo", default=None,
                        help="Versão do verificador (por exemplo, v2). Omitido: nomes da v1.")
    parser.add_argument("--pesos", type=Path, default=None,
                        help="Pesos do verificador (padrão: derivado do rótulo, "
                             "modelos/yolo11n_cls_dominio[_<rotulo>].pt).")
    parser.add_argument("--limiar", type=float, default=LIMIAR_ALFACE_PADRAO,
                        help="Limiar de aceitação. No teste, use o escolhido no val.")
    parser.add_argument("--pesos-classificador", type=Path,
                        default=CAMINHOS.modelos / "yolo11n_cls_alface.pt",
                        help="Pesos do classificador, para a latência das duas etapas.")
    parser.add_argument("--limiar-confianca", type=float, default=0.85,
                        help="Limiar de alerta do classificador (CT01/CT02).")
    parser.add_argument("--inventario", type=Path,
                        default=CAMINHOS.relatorios / "dataset_dominio_inventario.csv")
    parser.add_argument("--predicoes-classificador", type=Path,
                        default=CAMINHOS.relatorios / "predicoes.csv",
                        help="Predições oficiais do classificador no teste.")
    parser.add_argument("--dispositivo", default="cpu",
                        help="Dispositivo (padrão: cpu, o do serviço implantado).")
    parser.add_argument("--sem-latencia", action="store_true")
    args = parser.parse_args(argv)
    try:
        nomes = ad.nomes_da_avaliacao(args.split, args.rotulo)
    except ValueError as erro:
        parser.error(str(erro))
    pesos = args.pesos or CAMINHOS.modelos / nomes["pesos"]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    CAMINHOS.garantir_diretorios()
    config = padrao

    caminho_json = CAMINHOS.relatorios / nomes["json"]
    if args.split == "test" and caminho_json.exists():
        logger.error(
            "%s já existe: o split de teste foi medido. Medi-lo de novo o transforma "
            "em um segundo conjunto de validação (CLAUDE.md, regra 3). Se a medição "
            "anterior foi inválida por erro técnico, registre o motivo em "
            "docs/VERIFICADOR_DOMINIO.md e apague o arquivo conscientemente.",
            caminho_json,
        )
        return 1

    try:
        inventario = _carregar_inventario(args.inventario)
        verificador = VerificadorDominio(pesos, dispositivo=args.dispositivo,
                                         limiar_alface=args.limiar)
        predicoes_oficiais = (_carregar_predicoes_oficiais(args.predicoes_classificador)
                              if args.split == "test" else None)
    except (FileNotFoundError, ValueError) as erro:
        logger.error("%s", erro)
        return 1

    linhas = inventario[inventario["split"] == args.split].reset_index(drop=True)
    logger.info("Verificando %d imagens do split '%s' com %s e limiar %.2f.",
                len(linhas), args.split, pesos.name, args.limiar)
    previsoes = _verificar(verificador, linhas)
    previsoes.to_csv(CAMINHOS.relatorios / nomes["predicoes"], index=False, encoding="utf-8")

    resultado: dict = {
        "split": args.split,
        "versao": args.rotulo or "v1",
        "gerado_em": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "pesos": pesos.name,
        "limiar": args.limiar,
        "imagens": len(previsoes),
        "dispositivo": verificador.dispositivo,
        "config_avaliacao": config.para_dicionario(),
        "agregados": ad.agregados(previsoes, args.limiar),
        "por_grupo": _registros(ad.metricas_por_grupo(previsoes, args.limiar)),
    }

    if args.split == "val":
        resultado["curva"] = _registros(ad.curva_limiar(previsoes, ad.limiares_padrao()))
    else:
        kaggle = previsoes[previsoes["fonte"] == FONTE_KAGGLE]
        verificacoes = pd.DataFrame({
            "arquivo": [PureWindowsPath(c).name for c in kaggle["caminho"]],
            "verdadeiro": kaggle["estrato"].to_list(),
            "prob_alface": kaggle["prob_alface"].to_list(),
        })
        try:
            tabela, resumo = ad.cascata_ponta_a_ponta(
                predicoes_oficiais, verificacoes, args.limiar, args.limiar_confianca
            )
        except ValueError as erro:
            logger.error("%s", erro)
            return 1
        tabela.to_csv(CAMINHOS.relatorios / nomes["cascata"], index=False, encoding="utf-8")
        resultado["cascata"] = resumo

    if not args.sem_latencia:
        val = inventario[(inventario["split"] == "val") & (inventario["fonte"] == FONTE_KAGGLE)]
        imagens = sorted(val["caminho_destino"])[: config.amostras_latencia]
        logger.info("Medindo a latência das duas etapas em %d imagens do val.", len(imagens))
        resultado["latencia"] = _medir_latencia(
            verificador, args.pesos_classificador, imagens, config, args.dispositivo
        )

    caminho_json.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Métricas escritas em %s", caminho_json)
    _escrever_relatorio(resultado, CAMINHOS.relatorios / nomes["md"])

    for chave, nome in _NOMES_AGREGADOS.items():
        logger.info("%s: %s", nome, _texto(resultado["agregados"][chave]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
