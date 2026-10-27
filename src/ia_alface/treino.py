"""Treinamento do modelo YOLO11-cls para classificação de doenças em alface.

O aumento de dados é aplicado **em tempo de treino** pelo Ultralytics e nunca
gravado em disco: gravar variantes em disco foi exatamente a causa do vazamento
identificado nos problemas P1 e P2 da auditoria do dataset (spec.md, Seção 6).

Referência: spec.md, Seção 6.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

import pandas as pd

from .config import CAMINHOS, Caminhos, ConfigTreino, descrever_ambiente

logger = logging.getLogger(__name__)

__all__ = [
    "validar_dataset",
    "treinar",
    "treinar_verificador",
    "gerar_comparativo_variantes",
]


def validar_dataset(raiz: Path, script_preparo: str = "01_preparar_dataset.py") -> None:
    """Confere que um dataset de classificação está no formato esperado pelo Ultralytics.

    Args:
        raiz: Diretório raiz do dataset de classificação.
        script_preparo: Script que gera o dataset, citado na mensagem de erro.

    Raises:
        FileNotFoundError: Se o diretório ou os splits obrigatórios não existirem.
        RuntimeError: Se algum split estiver vazio.
    """
    raiz = Path(raiz)
    if not raiz.is_dir():
        raise FileNotFoundError(
            f"Dataset de treino não encontrado em {raiz}. "
            f"Execute 'python desenvolvimento/scripts/{script_preparo}' primeiro."
        )

    for split in ("train", "val"):
        pasta = raiz / split
        if not pasta.is_dir():
            raise FileNotFoundError(
                f"O Ultralytics exige o subdiretório '{split}' em {raiz}. "
                "Note que o diretório de validação deve chamar-se 'val', não 'valid'."
            )
        arquivos = list(pasta.rglob("*.jpg")) + list(pasta.rglob("*.png"))
        if not arquivos:
            raise RuntimeError(f"O split '{split}' em {pasta} está vazio.")
        logger.info("Split '%s': %d imagens.", split, len(arquivos))


def _melhor_acuracia_validacao(diretorio_execucao: Path) -> float | None:
    """Extrai a melhor acurácia top-1 de validação do ``results.csv``.

    Para a tarefa de classificação, o Ultralytics usa a acurácia top-1 como
    métrica de aptidão, portanto o máximo desta coluna corresponde aos pesos
    salvos em ``best.pt``.

    Args:
        diretorio_execucao: Pasta da execução do Ultralytics.

    Returns:
        A melhor acurácia, ou ``None`` se o CSV não existir ou não tiver a coluna.
    """
    caminho = Path(diretorio_execucao) / "results.csv"
    if not caminho.exists():
        return None
    historico = pd.read_csv(caminho)
    historico.columns = [c.strip() for c in historico.columns]
    if "metrics/accuracy_top1" not in historico.columns:
        return None
    return round(float(historico["metrics/accuracy_top1"].max()), 4)


def _contar_parametros(modelo) -> int:
    """Conta os parâmetros treináveis e não treináveis do modelo."""
    return int(sum(p.numel() for p in modelo.model.parameters()))


def _latencia_rapida(
    caminho_pesos: Path,
    raiz_dataset: Path,
    limite_imagens: int = 30,
) -> dict | None:
    """Mede uma latência indicativa em CPU para o quadro comparativo.

    A medição definitiva, com aquecimento e percentis, é feita por
    ``desenvolvimento/scripts/05_benchmark_latencia.py``.

    Args:
        caminho_pesos: Pesos treinados.
        raiz_dataset: Raiz do ``desenvolvimento/dataset_clean/``.
        limite_imagens: Número máximo de imagens a usar.

    Returns:
        Estatísticas de latência, ou ``None`` se a medição falhar.
    """
    try:
        from .avaliacao import listar_imagens, medir_latencia
        from .inferencia import DetectorDoencasAlface

        itens = listar_imagens(raiz_dataset / "val")
        imagens = [caminho for caminho, _ in itens][:limite_imagens]
        detector = DetectorDoencasAlface(caminho_pesos, dispositivo="cpu")
        return medir_latencia(detector, imagens, aquecimento=5)
    except Exception as erro:  # noqa: BLE001 - medição é acessória
        logger.warning("Não foi possível medir a latência indicativa: %s", erro)
        return None


def treinar(
    config: ConfigTreino,
    caminhos: Caminhos = CAMINHOS,
    medir_latencia_indicativa: bool = True,
) -> dict:
    """Treina o modelo e registra os artefatos para o TCC.

    Args:
        config: Hiperparâmetros de treino.
        caminhos: Caminhos do projeto.
        medir_latencia_indicativa: Se ``True``, mede a latência em CPU ao final,
            para preencher o quadro comparativo de variantes.

    Returns:
        Dicionário com o resumo da execução: caminho dos pesos, diretório da
        execução, melhor acurácia de validação, número de parâmetros, duração e
        latência indicativa.

    Raises:
        FileNotFoundError: Se ``desenvolvimento/dataset_clean/`` não existir.
        RuntimeError: Se o Ultralytics não produzir o arquivo ``best.pt``.
    """
    from ultralytics import YOLO

    caminhos.garantir_diretorios()
    validar_dataset(caminhos.dataset_limpo)

    dispositivo = config.resolver_dispositivo()
    argumentos = config.para_argumentos_ultralytics(
        dados=caminhos.dataset_limpo, projeto=caminhos.runs / "classify"
    )

    logger.info(
        "Treinando %s por %d épocas em %s (imgsz=%d, batch=%d).",
        config.modelo_base,
        config.epocas,
        dispositivo,
        config.imgsz,
        config.batch,
    )
    if dispositivo == "cpu":
        logger.info(
            "Execução em CPU: espere alguns minutos por época. Para acelerar, use "
            "desenvolvimento/notebooks/treino_colab.ipynb com GPU."
        )

    modelo = YOLO(config.modelo_base)
    inicio = time.perf_counter()
    modelo.train(**argumentos)
    duracao_s = time.perf_counter() - inicio

    diretorio_execucao = Path(modelo.trainer.save_dir)
    melhores_pesos = diretorio_execucao / "weights" / "best.pt"
    if not melhores_pesos.exists():
        raise RuntimeError(
            f"O treinamento terminou, mas {melhores_pesos} não foi criado. "
            "Verifique o log do Ultralytics acima."
        )

    destino_pesos = caminhos.modelos / config.nome_arquivo_pesos
    shutil.copy2(melhores_pesos, destino_pesos)
    logger.info("Pesos finais copiados para %s", destino_pesos)

    acuracia_val = _melhor_acuracia_validacao(diretorio_execucao)
    n_parametros = _contar_parametros(modelo)

    latencia = None
    if medir_latencia_indicativa:
        latencia = _latencia_rapida(destino_pesos, caminhos.dataset_limpo)

    resumo = {
        "modelo_base": config.modelo_base,
        "nome_execucao": config.nome_execucao,
        "pesos": str(destino_pesos),
        "diretorio_execucao": str(diretorio_execucao),
        "melhor_acuracia_val": acuracia_val,
        "parametros": n_parametros,
        "parametros_milhoes": round(n_parametros / 1e6, 2),
        "tamanho_arquivo_mb": round(destino_pesos.stat().st_size / 1e6, 2),
        "duracao_segundos": round(duracao_s, 1),
        "duracao_formatada": f"{duracao_s / 60:.1f} min",
        "dispositivo": dispositivo,
        "latencia_cpu_ms": latencia,
        "config": config.para_dicionario(),
        "argumentos_ultralytics": argumentos,
        "ambiente": descrever_ambiente(),
    }

    caminho_config = caminhos.relatorios / "treino_config.json"
    historico = _carregar_json_lista(caminho_config)
    historico = [r for r in historico if r.get("nome_execucao") != config.nome_execucao]
    historico.append(resumo)
    caminho_config.write_text(
        json.dumps(historico, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Configuração e resumo do treino salvos em %s", caminho_config)

    gerar_comparativo_variantes(caminhos)
    return resumo


def treinar_verificador(config: ConfigTreino, caminhos: Caminhos = CAMINHOS) -> dict:
    """Treina o verificador de domínio (alface / não alface) sobre ``dataset_dominio/``.

    Separado de :func:`treinar` para não misturar o verificador nos relatórios
    do classificador: o resumo vai para ``reports/treino_verificador.json`` e o
    comparativo de variantes não é tocado. Também não mede latência aqui — a
    latência que importa é a das duas etapas juntas, medida na avaliação.

    Args:
        config: Hiperparâmetros. ``nome_execucao`` define o arquivo de pesos
            (por exemplo, ``yolo11n_cls_dominio`` → ``modelos/yolo11n_cls_dominio.pt``).
        caminhos: Caminhos do projeto.

    Returns:
        Resumo da execução.

    Raises:
        FileNotFoundError: Se ``dataset_dominio/`` não existir.
        RuntimeError: Se o Ultralytics não produzir ``best.pt``.
    """
    from ultralytics import YOLO

    caminhos.garantir_diretorios()
    validar_dataset(caminhos.dataset_dominio, "06_preparar_dataset_dominio.py")

    dispositivo = config.resolver_dispositivo()
    argumentos = config.para_argumentos_ultralytics(
        dados=caminhos.dataset_dominio, projeto=caminhos.runs / "classify"
    )
    logger.info(
        "Treinando o verificador %s por até %d épocas em %s.",
        config.nome_execucao,
        config.epocas,
        dispositivo,
    )

    modelo = YOLO(config.modelo_base)
    inicio = time.perf_counter()
    modelo.train(**argumentos)
    duracao_s = time.perf_counter() - inicio

    diretorio_execucao = Path(modelo.trainer.save_dir)
    melhores_pesos = diretorio_execucao / "weights" / "best.pt"
    if not melhores_pesos.exists():
        raise RuntimeError(
            f"O treinamento terminou, mas {melhores_pesos} não foi criado. "
            "Verifique o log do Ultralytics acima."
        )
    destino_pesos = caminhos.modelos / config.nome_arquivo_pesos
    shutil.copy2(melhores_pesos, destino_pesos)
    logger.info("Pesos do verificador copiados para %s", destino_pesos)

    n_parametros = _contar_parametros(modelo)
    resumo = {
        "modelo_base": config.modelo_base,
        "nome_execucao": config.nome_execucao,
        "pesos": str(destino_pesos),
        "diretorio_execucao": str(diretorio_execucao),
        "melhor_acuracia_val": _melhor_acuracia_validacao(diretorio_execucao),
        "parametros": n_parametros,
        "duracao_segundos": round(duracao_s, 1),
        "duracao_formatada": f"{duracao_s / 60:.1f} min",
        "dispositivo": dispositivo,
        "config": config.para_dicionario(),
        "argumentos_ultralytics": argumentos,
        "ambiente": descrever_ambiente(),
    }

    caminho_resumo = caminhos.relatorios / "treino_verificador.json"
    historico = [
        r for r in _carregar_json_lista(caminho_resumo)
        if r.get("nome_execucao") != config.nome_execucao
    ]
    historico.append(resumo)
    caminho_resumo.write_text(
        json.dumps(historico, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Resumo do treino do verificador salvo em %s", caminho_resumo)
    return resumo


def _carregar_json_lista(caminho: Path) -> list[dict]:
    """Lê um JSON que contém uma lista, tolerando arquivo ausente ou inválido."""
    if not caminho.exists():
        return []
    try:
        conteudo = json.loads(caminho.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("%s está corrompido e será reescrito.", caminho)
        return []
    return conteudo if isinstance(conteudo, list) else [conteudo]


def gerar_comparativo_variantes(caminhos: Caminhos = CAMINHOS) -> Path | None:
    """Monta ``desenvolvimento/reports/comparativo_variantes.md`` a partir dos treinos realizados.

    Preenche com dados próprios a Tabela 3 do TCC ("Características das
    variantes YOLOv11"), que hoje traz apenas valores de referência da
    literatura.

    Args:
        caminhos: Caminhos do projeto.

    Returns:
        O caminho do relatório, ou ``None`` se nenhum treino foi registrado.
    """
    from datetime import datetime

    registros = _carregar_json_lista(caminhos.relatorios / "treino_config.json")
    if not registros:
        logger.info("Nenhum treino registrado; comparativo não gerado.")
        return None

    registros = sorted(registros, key=lambda r: r.get("parametros", 0))

    linhas: list[str] = []
    ad = linhas.append
    ad("# Comparativo de variantes YOLO11-cls treinadas neste trabalho")
    ad("")
    ad(f"*Gerado por `src/ia_alface/treino.py` em "
       f"{datetime.now().strftime('%d/%m/%Y %H:%M')}.*")
    ad("")
    ad("Substitui, com medições próprias, os valores de referência da Tabela 3 "
       "do TCC. A latência é indicativa (30 imagens do split de validação, 5 de "
       "aquecimento, CPU); a medição definitiva está em "
       "`desenvolvimento/reports/benchmark_latencia.md`.")
    ad("")
    ad("| Variante | Parâmetros (M) | Acurácia top-1 (val) | "
       "Latência mediana CPU (ms) | Arquivo (MB) | Épocas | Duração |")
    ad("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for registro in registros:
        latencia = registro.get("latencia_cpu_ms") or {}
        acuracia = registro.get("melhor_acuracia_val")
        ad(
            f"| `{registro.get('modelo_base', '—')}` "
            f"| {registro.get('parametros_milhoes', '—')} "
            f"| {'não medido' if acuracia is None else f'{acuracia:.2%}'} "
            f"| {latencia.get('mediana', 'não medido')} "
            f"| {registro.get('tamanho_arquivo_mb', '—')} "
            f"| {registro.get('config', {}).get('epocas', '—')} "
            f"| {registro.get('duracao_formatada', '—')} |"
        )
    ad("")
    ad("## Ambiente das medições")
    ad("")
    ambiente = registros[-1].get("ambiente", {})
    for chave, valor in ambiente.items():
        ad(f"- **{chave}**: {valor}")
    ad("")

    caminho = caminhos.relatorios / "comparativo_variantes.md"
    caminho.write_text("\n".join(linhas), encoding="utf-8")
    logger.info("Comparativo de variantes escrito em %s", caminho)
    return caminho
