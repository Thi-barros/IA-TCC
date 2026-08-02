"""Inventário, deduplicação e re-divisão do dataset de folhas de alface.

Este módulo resolve os problemas P1–P6 documentados na Seção 4.2 do spec.md:

* **P1** — 104 grupos de imagens byte-idênticas cruzam a fronteira dos splits
  originais, o que inflaria artificialmente as métricas de teste.
* **P2** — variantes aumentadas da mesma folha (``ba12_0``, ``ba12_2``, ...)
  aparecem em splits distintos.
* **P5** — a divisão original é 84/8/8, não os 70/15/15 exigidos pelo TCC.

A estratégia é: inventariar tudo com hash MD5, deduplicar, agrupar por folha de
origem e re-dividir de forma **estratificada por classe** e **disjunta por
grupo**, de modo que nenhuma folha (nem suas variantes) apareça em mais de um
split.

O diretório ``desenvolvimento/dataset/`` é tratado como fonte imutável: nada é escrito nele.

Referência: spec.md, Seção 4.3.
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from PIL import Image

from .classes import CANONICO_PARA_INDICE, ROTULOS_CANONICOS, canonico_de_pasta

logger = logging.getLogger(__name__)

__all__ = [
    "SPLITS_SAIDA",
    "PROPORCOES_PADRAO",
    "normalizar_grupo_origem",
    "hash_arquivo",
    "inventariar",
    "deduplicar",
    "grupos_duplicados_entre_splits",
    "dividir_por_grupos",
    "verificar_integridade",
    "escrever_dataset_limpo",
]

#: Nomes dos splits de saída. O Ultralytics espera ``val`` (não ``valid``)
#: para a tarefa de classificação — ver spec.md, Seção 4.3, item 6.
SPLITS_SAIDA: tuple[str, str, str] = ("train", "val", "test")

#: Proporções exigidas pela Seção 6.1.1 do TCC.
PROPORCOES_PADRAO: tuple[float, float, float] = (0.70, 0.15, 0.15)

#: Extensões de imagem aceitas.
EXTENSOES_IMAGEM: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

# Sufixos que o Windows acrescenta a cópias de arquivo: " - Copy", " - Copy (2)",
# e as variantes em português " - Copia", " - Cópia (2)".
_PADRAO_COPIA = re.compile(
    r"\s*[-–]\s*(?:copy|copia|cópia)\s*(?:\(\d+\))?\s*$",
    flags=re.IGNORECASE,
)

# Sufixo numérico de variante de augmentation: "ba12_3" -> "ba12".
_PADRAO_SUFIXO_NUMERICO = re.compile(r"_\d+$")

_PREFIXO_AUMENTADO = "augmented_"


def normalizar_grupo_origem(
    nome_arquivo: str,
    classe_canonica: str,
    indice_unico: int | None = None,
) -> str:
    """Deriva a chave de agrupamento (folha de origem) de um nome de arquivo.

    O dataset nomeia variantes aumentadas de uma mesma folha como
    ``<origem>_<n>`` (por exemplo ``ba12_0``, ``ba12_2``, ``ba12_3``). Agrupar
    por ``<origem>`` impede que variantes da mesma folha caiam em splits
    diferentes — o vazamento de grupo descrito no problema P2 do spec.

    Os arquivos ``augmented_*`` não têm origem recuperável, então cada um forma
    um **grupo unitário**. Para garantir que sejam realmente unitários, o
    chamador fornece um ``indice_unico`` sequencial: usar o número do próprio
    nome do arquivo faria ``train/.../augmented_0.jpg`` e
    ``test/.../augmented_0.jpg`` — que são imagens distintas — colapsarem no
    mesmo grupo.

    Args:
        nome_arquivo: Nome do arquivo, com ou sem diretório, com extensão.
        classe_canonica: Rótulo canônico da classe do arquivo.
        indice_unico: Índice sequencial, obrigatório apenas para arquivos
            ``augmented_*``.

    Returns:
        A chave de agrupamento, sem o nome da classe.

    Raises:
        ValueError: Se o arquivo for ``augmented_*`` e ``indice_unico`` for
            ``None``, ou se o nome resultante ficar vazio.

    Examples:
        >>> normalizar_grupo_origem("ba12_3.png", "bacteriano")
        'ba12'
        >>> normalizar_grupo_origem("ba1_4 - Copy.png", "bacteriano")
        'ba1'
        >>> normalizar_grupo_origem("h100_2.png", "saudavel")
        'h100'
        >>> normalizar_grupo_origem("augmented_0.jpg", "bacteriano", 7)
        'augmented_bacteriano_7'
    """
    stem = Path(nome_arquivo).stem

    if stem.lower().startswith(_PREFIXO_AUMENTADO):
        if indice_unico is None:
            raise ValueError(
                f"O arquivo {nome_arquivo!r} é do tipo 'augmented_*' e exige um "
                "indice_unico para formar um grupo unitário."
            )
        return f"{_PREFIXO_AUMENTADO}{classe_canonica}_{indice_unico}"

    # Remove sufixos de cópia repetidamente: "ba1_4 - Copy - Copy" -> "ba1_4".
    anterior = None
    while anterior != stem:
        anterior = stem
        stem = _PADRAO_COPIA.sub("", stem)

    # Remove o sufixo de variante: "ba12_3" -> "ba12".
    stem = _PADRAO_SUFIXO_NUMERICO.sub("", stem)
    stem = stem.strip()

    if not stem:
        raise ValueError(
            f"Não foi possível derivar um grupo de origem de {nome_arquivo!r}."
        )
    return stem


def hash_arquivo(caminho: Path, tamanho_bloco: int = 1 << 20) -> str:
    """Calcula o hash MD5 do conteúdo de um arquivo.

    O MD5 é usado apenas para detectar duplicatas exatas, não para fins de
    segurança.

    Args:
        caminho: Caminho do arquivo.
        tamanho_bloco: Tamanho do bloco de leitura, em bytes.

    Returns:
        O digest hexadecimal.
    """
    digest = hashlib.md5()
    with caminho.open("rb") as fluxo:
        for bloco in iter(lambda: fluxo.read(tamanho_bloco), b""):
            digest.update(bloco)
    return digest.hexdigest()


def inventariar(raiz_dataset: Path) -> pd.DataFrame:
    """Cataloga todas as imagens do dataset original.

    A varredura é determinística (diretórios e arquivos em ordem alfabética),
    de modo que os índices sequenciais atribuídos aos arquivos ``augmented_*``
    sejam reprodutíveis entre execuções.

    Args:
        raiz_dataset: Diretório ``desenvolvimento/dataset/``, contendo ``train/``, ``valid/`` e
            ``test/``, cada um com subdiretórios por classe.

    Returns:
        DataFrame com uma linha por imagem e as colunas: ``caminho``,
        ``arquivo``, ``split_original``, ``pasta_classe``, ``classe``,
        ``indice_classe``, ``grupo_origem``, ``grupo_id``, ``hash_md5``,
        ``largura``, ``altura``, ``formato``, ``modo``, ``bytes``.

    Raises:
        FileNotFoundError: Se ``raiz_dataset`` não existir.
        RuntimeError: Se nenhuma imagem legível for encontrada.
    """
    raiz_dataset = Path(raiz_dataset)
    if not raiz_dataset.is_dir():
        raise FileNotFoundError(
            f"Dataset não encontrado em {raiz_dataset}. "
            "Verifique se o diretório 'desenvolvimento/dataset/' está na raiz do projeto."
        )

    registros: list[dict] = []
    ilegiveis: list[tuple[str, str]] = []
    contador_aumentado: Counter[str] = Counter()

    splits = sorted(p for p in raiz_dataset.iterdir() if p.is_dir())
    for dir_split in splits:
        for dir_classe in sorted(p for p in dir_split.iterdir() if p.is_dir()):
            try:
                classe = canonico_de_pasta(dir_classe.name)
            except KeyError:
                logger.warning(
                    "Ignorando diretório de classe desconhecido: %s", dir_classe
                )
                continue

            arquivos = sorted(
                p
                for p in dir_classe.iterdir()
                if p.is_file() and p.suffix.lower() in EXTENSOES_IMAGEM
            )
            for arquivo in arquivos:
                try:
                    with Image.open(arquivo) as imagem:
                        largura, altura = imagem.size
                        formato = imagem.format or arquivo.suffix.lstrip(".").upper()
                        modo = imagem.mode
                except Exception as erro:  # noqa: BLE001 - queremos catalogar a falha
                    ilegiveis.append((str(arquivo), str(erro)))
                    logger.warning("Imagem ilegível, ignorada: %s (%s)", arquivo, erro)
                    continue

                indice_unico = None
                if arquivo.stem.lower().startswith(_PREFIXO_AUMENTADO):
                    indice_unico = contador_aumentado[classe]
                    contador_aumentado[classe] += 1

                grupo_origem = normalizar_grupo_origem(
                    arquivo.name, classe, indice_unico
                )

                registros.append(
                    {
                        "caminho": str(arquivo),
                        "arquivo": arquivo.name,
                        "split_original": dir_split.name,
                        "pasta_classe": dir_classe.name,
                        "classe": classe,
                        "indice_classe": CANONICO_PARA_INDICE[classe],
                        "grupo_origem": grupo_origem,
                        "grupo_id": f"{classe}/{grupo_origem}",
                        "hash_md5": hash_arquivo(arquivo),
                        "largura": largura,
                        "altura": altura,
                        "formato": formato,
                        "modo": modo,
                        "bytes": arquivo.stat().st_size,
                    }
                )

    if not registros:
        raise RuntimeError(
            f"Nenhuma imagem legível encontrada em {raiz_dataset}. "
            f"Extensões aceitas: {sorted(EXTENSOES_IMAGEM)}."
        )

    inventario = pd.DataFrame(registros)
    inventario.attrs["ilegiveis"] = ilegiveis
    logger.info(
        "Inventário concluído: %d imagens catalogadas, %d ilegíveis.",
        len(inventario),
        len(ilegiveis),
    )
    return inventario


def grupos_duplicados_entre_splits(inventario: pd.DataFrame) -> pd.DataFrame:
    """Identifica duplicatas exatas que cruzam a fronteira dos splits originais.

    Este é o problema P1 do spec: imagens byte-idênticas presentes ao mesmo
    tempo em ``train`` e em ``valid``/``test`` tornam as métricas de teste
    inválidas.

    Args:
        inventario: Saída de :func:`inventariar`.

    Returns:
        DataFrame com uma linha por grupo de duplicatas que aparece em mais de
        um split, com as colunas ``hash_md5``, ``classe``, ``n_arquivos``,
        ``splits`` e ``arquivos``.
    """
    linhas: list[dict] = []
    for hash_md5, grupo in inventario.groupby("hash_md5", sort=True):
        if len(grupo) < 2:
            continue
        splits = sorted(set(grupo["split_original"]))
        if len(splits) < 2:
            continue
        arquivos = sorted(
            f"{r.split_original}/{r.pasta_classe}/{r.arquivo}"
            for r in grupo.itertuples()
        )
        linhas.append(
            {
                "hash_md5": hash_md5,
                "classe": ", ".join(sorted(set(grupo["classe"]))),
                "n_arquivos": len(grupo),
                "splits": ", ".join(splits),
                "arquivos": " | ".join(arquivos),
            }
        )
    return pd.DataFrame(
        linhas,
        columns=["hash_md5", "classe", "n_arquivos", "splits", "arquivos"],
    )


def deduplicar(inventario: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove duplicatas exatas, mantendo uma cópia por hash MD5.

    A cópia mantida é a de menor caminho em ordem lexicográfica, o que torna a
    escolha determinística entre execuções.

    Args:
        inventario: Saída de :func:`inventariar`.

    Returns:
        Uma tupla ``(mantidos, descartados)`` com os dois subconjuntos.
    """
    ordenado = inventario.sort_values("caminho", kind="stable")
    mascara_manter = ~ordenado.duplicated(subset="hash_md5", keep="first")

    mantidos = ordenado[mascara_manter].reset_index(drop=True)
    descartados = ordenado[~mascara_manter].reset_index(drop=True)

    logger.info(
        "Deduplicação: %d imagens mantidas, %d descartadas (duplicatas exatas).",
        len(mantidos),
        len(descartados),
    )
    return mantidos, descartados


def _unificar_grupos_por_hash(inventario: pd.DataFrame) -> pd.DataFrame:
    """Funde grupos de origem que compartilham imagens idênticas.

    O dataset contém casos como ``ba15_1.png`` e ``ba6_0.png`` sendo o mesmo
    arquivo: dois nomes de origem diferentes para a mesma folha. Sem esta fusão,
    a deduplicação manteria apenas uma cópia e os grupos ``ba15`` e ``ba6``
    seriam tratados como folhas distintas, permitindo que outras variantes de
    cada um caíssem em splits diferentes — vazamento residual.

    A fusão é feita por componentes conexas: hashes e grupos são nós de um grafo,
    e cada arquivo é uma aresta entre o seu hash e o seu grupo.

    Args:
        inventario: Saída de :func:`inventariar` (**antes** da deduplicação).

    Returns:
        Cópia do inventário com a coluna ``grupo_id`` reescrita para o
        identificador canônico da componente conexa.
    """
    pai: dict[str, str] = {}

    def encontrar(no: str) -> str:
        pai.setdefault(no, no)
        raiz = no
        while pai[raiz] != raiz:
            raiz = pai[raiz]
        # Compressão de caminho.
        while pai[no] != raiz:
            pai[no], no = raiz, pai[no]
        return raiz

    def unir(a: str, b: str) -> None:
        raiz_a, raiz_b = encontrar(a), encontrar(b)
        if raiz_a == raiz_b:
            return
        # Mantém a menor raiz em ordem lexicográfica, para determinismo.
        if raiz_b < raiz_a:
            raiz_a, raiz_b = raiz_b, raiz_a
        pai[raiz_b] = raiz_a

    for registro in inventario.itertuples():
        unir(f"g::{registro.grupo_id}", f"h::{registro.hash_md5}")

    # Para cada componente, escolhe como nome canônico o menor grupo_id contido.
    grupos_por_componente: dict[str, set[str]] = defaultdict(set)
    for registro in inventario.itertuples():
        componente = encontrar(f"g::{registro.grupo_id}")
        grupos_por_componente[componente].add(registro.grupo_id)

    nome_canonico = {
        componente: min(grupos) for componente, grupos in grupos_por_componente.items()
    }

    resultado = inventario.copy()
    resultado["grupo_id_original"] = resultado["grupo_id"]
    resultado["grupo_id"] = [
        nome_canonico[encontrar(f"g::{g}")] for g in resultado["grupo_id"]
    ]

    n_fundidos = sum(1 for grupos in grupos_por_componente.values() if len(grupos) > 1)
    if n_fundidos:
        logger.info(
            "Fusão de grupos por identidade de conteúdo: %d componentes reuniram "
            "mais de um nome de origem.",
            n_fundidos,
        )
    return resultado


def dividir_por_grupos(
    inventario: pd.DataFrame,
    proporcoes: Sequence[float] = PROPORCOES_PADRAO,
    seed: int = 42,
    n_splits_kfold: int = 20,
) -> pd.DataFrame:
    """Re-divide o dataset em 70/15/15, estratificado e disjunto por grupo.

    Usa ``StratifiedGroupKFold`` do scikit-learn com ``n_splits_kfold`` dobras
    disjuntas por grupo, combinando dobras para atingir as proporções pedidas
    (com 20 dobras: 14 para treino, 3 para validação, 3 para teste). Quando
    alguma classe tem menos grupos do que dobras, cai para um alocador guloso
    por classe, que garante as mesmas propriedades.

    Args:
        inventario: Inventário já deduplicado, com as colunas ``classe`` e
            ``grupo_id``.
        proporcoes: Frações de ``(train, val, test)``. Devem somar 1,0.
        seed: Semente para reprodutibilidade.
        n_splits_kfold: Número de dobras do ``StratifiedGroupKFold``.

    Returns:
        Cópia do inventário com a coluna ``split`` preenchida com ``train``,
        ``val`` ou ``test``.

    Raises:
        ValueError: Se as proporções não somarem 1,0.
    """
    if abs(sum(proporcoes) - 1.0) > 1e-6:
        raise ValueError(f"As proporções devem somar 1,0; recebido {sum(proporcoes)}.")

    grupos_por_classe = inventario.groupby("classe")["grupo_id"].nunique()
    minimo_grupos = int(grupos_por_classe.min())

    if minimo_grupos >= n_splits_kfold:
        resultado = _dividir_stratified_group_kfold(
            inventario, proporcoes, seed, n_splits_kfold
        )
    else:
        logger.warning(
            "A classe com menos grupos possui %d grupos, abaixo das %d dobras "
            "exigidas pelo StratifiedGroupKFold. Usando o alocador guloso.",
            minimo_grupos,
            n_splits_kfold,
        )
        resultado = _dividir_guloso(inventario, proporcoes, seed)

    contagem = resultado.groupby(["split", "classe"]).size()
    logger.info("Divisão concluída:\n%s", contagem.to_string())
    return resultado


def _dividir_stratified_group_kfold(
    inventario: pd.DataFrame,
    proporcoes: Sequence[float],
    seed: int,
    n_splits: int,
) -> pd.DataFrame:
    """Divide combinando dobras de ``StratifiedGroupKFold``."""
    from sklearn.model_selection import StratifiedGroupKFold

    dobrador = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    indices_dobra = pd.Series(-1, index=inventario.index, dtype=int)
    for numero_dobra, (_, indices_teste) in enumerate(
        dobrador.split(
            X=inventario.index.to_numpy(),
            y=inventario["indice_classe"].to_numpy(),
            groups=inventario["grupo_id"].to_numpy(),
        )
    ):
        indices_dobra.iloc[indices_teste] = numero_dobra

    n_test = max(1, round(proporcoes[2] * n_splits))
    n_val = max(1, round(proporcoes[1] * n_splits))

    def rotular(dobra: int) -> str:
        if dobra < n_test:
            return "test"
        if dobra < n_test + n_val:
            return "val"
        return "train"

    resultado = inventario.copy()
    resultado["split"] = [rotular(d) for d in indices_dobra]
    return resultado


def _dividir_guloso(
    inventario: pd.DataFrame,
    proporcoes: Sequence[float],
    seed: int,
) -> pd.DataFrame:
    """Aloca grupos gulosamente, por classe, respeitando as proporções.

    Para cada classe, embaralha os grupos de forma determinística e atribui cada
    grupo ao split com o maior déficit em relação à sua meta de imagens. Garante
    disjunção por grupo por construção e estratificação exata por classe.
    """
    atribuicao: dict[str, str] = {}

    for classe, bloco in inventario.groupby("classe", sort=True):
        tamanhos = bloco.groupby("grupo_id").size().sort_index()
        grupos = list(tamanhos.items())
        random.Random(f"{seed}:{classe}").shuffle(grupos)
        # Grupos maiores primeiro reduzem o desvio das proporções finais.
        grupos.sort(key=lambda item: (-item[1], item[0]))

        total = int(tamanhos.sum())
        metas = dict(zip(SPLITS_SAIDA, (p * total for p in proporcoes)))
        atuais = {split: 0 for split in SPLITS_SAIDA}

        for grupo_id, tamanho in grupos:
            escolhido = max(
                SPLITS_SAIDA, key=lambda s: (metas[s] - atuais[s], s == "train", s)
            )
            atribuicao[grupo_id] = escolhido
            atuais[escolhido] += tamanho

    resultado = inventario.copy()
    resultado["split"] = resultado["grupo_id"].map(atribuicao)
    return resultado


def verificar_integridade(inventario: pd.DataFrame) -> list[str]:
    """Verifica que a divisão não contém vazamento entre splits.

    Esta é a verificação obrigatória do item 8 da Seção 4.3 do spec: nenhum
    ``hash_md5`` e nenhum ``grupo_id`` pode aparecer em mais de um split.

    Args:
        inventario: Inventário já dividido, com a coluna ``split``.

    Returns:
        Lista de mensagens de problema, em português. Lista vazia significa
        integridade confirmada.
    """
    problemas: list[str] = []

    if "split" not in inventario.columns:
        return ["O inventário não possui a coluna 'split'."]

    faltando = inventario["split"].isna().sum()
    if faltando:
        problemas.append(f"{faltando} imagens ficaram sem split atribuído.")

    for coluna, descricao in (
        ("hash_md5", "hashes MD5 idênticos"),
        ("grupo_id", "grupos de origem"),
    ):
        por_chave = inventario.groupby(coluna)["split"].nunique()
        vazando = por_chave[por_chave > 1]
        if len(vazando):
            exemplos = ", ".join(map(str, vazando.index[:5]))
            problemas.append(
                f"{len(vazando)} {descricao} aparecem em mais de um split "
                f"(exemplos: {exemplos})."
            )

    for split in SPLITS_SAIDA:
        bloco = inventario[inventario["split"] == split]
        if bloco.empty:
            problemas.append(f"O split '{split}' ficou vazio.")
            continue
        classes_ausentes = set(ROTULOS_CANONICOS) - set(bloco["classe"])
        if classes_ausentes:
            problemas.append(
                f"O split '{split}' não contém as classes: "
                f"{sorted(classes_ausentes)}."
            )

    return problemas


def _sanitizar_nome(texto: str) -> str:
    """Reduz um nome de arquivo a caracteres seguros para qualquer sistema."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", texto).strip("_") or "imagem"


def escrever_dataset_limpo(
    inventario: pd.DataFrame,
    destino: Path,
    qualidade_jpeg: int = 95,
) -> pd.DataFrame:
    """Materializa o dataset re-dividido no layout esperado pelo Ultralytics.

    As imagens são convertidas para RGB (tratando o único arquivo RGBA do
    dataset) e reescritas como JPEG de qualidade 95. **Não há redimensionamento**
    — a proporção original é preservada e o ajuste de tamanho fica a cargo do
    parâmetro ``imgsz`` do Ultralytics em tempo de treino.

    Args:
        inventario: Inventário dividido, com a coluna ``split``.
        destino: Diretório de saída (``desenvolvimento/dataset_clean/``). Se já existir, é
            removido antes da escrita, para evitar mistura entre execuções.
        qualidade_jpeg: Qualidade de compressão JPEG.

    Returns:
        Cópia do inventário com a coluna ``caminho_destino`` preenchida.
    """
    import shutil

    destino = Path(destino)
    if destino.exists():
        logger.info("Removendo %s de uma execução anterior.", destino)
        shutil.rmtree(destino)

    for split in SPLITS_SAIDA:
        for classe in ROTULOS_CANONICOS:
            (destino / split / classe).mkdir(parents=True, exist_ok=True)

    caminhos_destino: list[str] = []
    for registro in inventario.itertuples():
        pasta = destino / registro.split / registro.classe
        nome = f"{_sanitizar_nome(Path(registro.arquivo).stem)}__{registro.hash_md5[:8]}.jpg"
        caminho_saida = pasta / nome

        with Image.open(registro.caminho) as imagem:
            if imagem.mode != "RGB":
                imagem = imagem.convert("RGB")
            imagem.save(caminho_saida, format="JPEG", quality=qualidade_jpeg)

        caminhos_destino.append(str(caminho_saida))

    resultado = inventario.copy()
    resultado["caminho_destino"] = caminhos_destino

    total = len(resultado)
    logger.info("Dataset limpo escrito em %s (%d imagens).", destino, total)
    return resultado


def resumo_por_split(inventario: pd.DataFrame, coluna_split: str = "split") -> pd.DataFrame:
    """Monta a tabela de contagens por split e classe.

    Args:
        inventario: Inventário com uma coluna de split.
        coluna_split: Nome da coluna de split a usar.

    Returns:
        DataFrame com splits nas linhas, classes nas colunas, e as colunas
        adicionais ``Total`` e ``%``.
    """
    tabela = (
        inventario.pivot_table(
            index=coluna_split,
            columns="classe",
            values="caminho",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(columns=list(ROTULOS_CANONICOS), fill_value=0)
    )
    ordem = [s for s in SPLITS_SAIDA if s in tabela.index]
    ordem += [s for s in tabela.index if s not in ordem]
    tabela = tabela.reindex(ordem)

    tabela["Total"] = tabela.sum(axis=1)
    total_geral = int(tabela["Total"].sum())
    tabela["%"] = (tabela["Total"] / total_geral * 100).round(1)
    return tabela


def contar_grupos_por_classe(inventario: pd.DataFrame) -> pd.DataFrame:
    """Conta imagens e grupos de origem distintos por classe.

    Evidencia o problema P3 do spec: a classe ``bacteriano`` deriva de poucas
    dezenas de folhas originais.

    Args:
        inventario: Inventário com as colunas ``classe`` e ``grupo_id``.

    Returns:
        DataFrame indexado por classe, com ``imagens``, ``grupos`` e
        ``imagens_por_grupo``.
    """
    tabela = inventario.groupby("classe").agg(
        imagens=("caminho", "count"),
        grupos=("grupo_id", "nunique"),
    )
    tabela["imagens_por_grupo"] = (tabela["imagens"] / tabela["grupos"]).round(2)
    return tabela.reindex(list(ROTULOS_CANONICOS))


def estatisticas_resolucao(inventario: pd.DataFrame) -> dict[str, object]:
    """Resume as resoluções presentes no dataset.

    Args:
        inventario: Inventário com as colunas ``largura`` e ``altura``.

    Returns:
        Dicionário com mínimos, medianas, máximos, contagem de imagens
        quadradas e os cinco tamanhos mais frequentes.
    """
    larguras = inventario["largura"]
    alturas = inventario["altura"]
    tamanhos = Counter(zip(larguras, alturas))
    return {
        "largura_min": int(larguras.min()),
        "largura_mediana": int(larguras.median()),
        "largura_max": int(larguras.max()),
        "altura_min": int(alturas.min()),
        "altura_mediana": int(alturas.median()),
        "altura_max": int(alturas.max()),
        "quadradas": int((larguras == alturas).sum()),
        "mais_frequentes": tamanhos.most_common(5),
        "modos_de_cor": dict(Counter(inventario["modo"])),
        "formatos": dict(Counter(inventario["formato"])),
    }


def preparar(
    raiz_dataset: Path,
    destino: Path,
    proporcoes: Sequence[float] = PROPORCOES_PADRAO,
    seed: int = 42,
    escrever: bool = True,
) -> dict[str, object]:
    """Executa o pipeline completo de preparação do dataset.

    Etapas, na ordem da Seção 4.3 do spec: inventariar, fundir grupos por
    identidade de conteúdo, deduplicar por MD5, re-dividir 70/15/15 com
    disjunção por grupo, verificar integridade e materializar em disco.

    Args:
        raiz_dataset: Diretório ``desenvolvimento/dataset/`` (fonte imutável).
        destino: Diretório ``desenvolvimento/dataset_clean/`` a ser criado.
        proporcoes: Frações de ``(train, val, test)``.
        seed: Semente para reprodutibilidade.
        escrever: Se ``False``, executa tudo menos a escrita em disco (útil em
            testes e verificações rápidas).

    Returns:
        Dicionário com todos os artefatos intermediários, para uso na geração
        do relatório de auditoria.
    """
    inventario_bruto = inventariar(raiz_dataset)
    duplicatas_cruzadas = grupos_duplicados_entre_splits(inventario_bruto)
    inventario_unificado = _unificar_grupos_por_hash(inventario_bruto)
    mantidos, descartados = deduplicar(inventario_unificado)
    dividido = dividir_por_grupos(mantidos, proporcoes=proporcoes, seed=seed)
    problemas = verificar_integridade(dividido)

    if escrever and not problemas:
        dividido = escrever_dataset_limpo(dividido, destino)

    return {
        "inventario_bruto": inventario_bruto,
        "duplicatas_cruzadas": duplicatas_cruzadas,
        "descartados": descartados,
        "dividido": dividido,
        "problemas": problemas,
        "proporcoes": tuple(proporcoes),
        "seed": seed,
        "raiz_dataset": Path(raiz_dataset),
        "destino": Path(destino),
    }
