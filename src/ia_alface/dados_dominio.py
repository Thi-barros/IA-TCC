"""Montagem do dataset do verificador de domínio (alface / não alface).

Lê as positivas de ``desenvolvimento/dataset_clean/`` e as demais fontes de
``desenvolvimento/dataset_dominio_bruto/``, sorteia, deduplica, divide por
grupo e materializa ``desenvolvimento/dataset_dominio/`` no layout do
Ultralytics (``train|val|test / alface|nao_alface``).

**Nenhuma pasta de origem é escrita.** As imagens são copiadas byte a byte; só
as que não são JPEG RGB são reconvertidas, sem redimensionar. Não há aumento de
dados em disco (regra 6 do ``CLAUDE.md``).

Regras de divisão (``docs/VERIFICADOR_DOMINIO.md``, Seção 1.2):

* positivas do Kaggle mantêm o split de ``dataset_clean/``, para que a cascata
  possa ser medida nas mesmas imagens de teste do classificador;
* fontes públicas: split pelo hash do grupo — estável quando novas fontes
  entram; PlantVillage agrupado por folha, DeepWeeds por data de captura;
* estufa, câmera do projeto: por sessão de captura — fotos em sequência, até
  uma pausa de mais de ``pausa_sessao_horas`` —, em ordem cronológica — as
  sessões mais recentes no teste —, aplicada **separadamente** às sessões que
  têm "não alface" e às que só têm alface, para que as fotos de "não alface"
  também cheguem ao treino. Sessões iniciadas até ``data_corte_treino`` vão
  inteiras para o treino e ficam fora dessa divisão;
* estufa, celular: somente treino.

Referência: docs/VERIFICADOR_DOMINIO.md.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import shutil
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd
from PIL import Image, ImageOps

from .classes import CLASSE_ALFACE, CLASSE_NAO_ALFACE, ROTULOS_CANONICOS, ROTULOS_DOMINIO
from .config import ConfigDatasetDominio
from .dados import EXTENSOES_IMAGEM, SPLITS_SAIDA, hash_arquivo

logger = logging.getLogger(__name__)

__all__ = [
    "FONTE_KAGGLE",
    "FONTE_PLANTVILLAGE",
    "FONTE_PLANTDOC",
    "FONTE_DEEPWEEDS",
    "FONTE_COCO",
    "FONTE_ESTUFA",
    "CAMERA_PROJETO",
    "CAMERA_CELULAR",
    "listar_kaggle",
    "listar_plantvillage",
    "listar_plantdoc",
    "listar_deepweeds",
    "listar_coco",
    "listar_estufa",
    "sessoes_de_captura",
    "amostrar_estratificado",
    "split_por_hash",
    "dividir_sessoes",
    "dividir_sessoes_estratificado",
    "conflitos_de_classe",
    "deduplicar_dominio",
    "atribuir_splits",
    "verificar_integridade_dominio",
    "escrever_dataset_dominio",
    "preparar_dominio",
]

FONTE_KAGGLE = "kaggle"
FONTE_PLANTVILLAGE = "plantvillage"
FONTE_PLANTDOC = "plantdoc"
FONTE_DEEPWEEDS = "deepweeds"
FONTE_COCO = "coco"
FONTE_ESTUFA = "estufa"

#: Câmera de operação do projeto (Logitech C270) e câmera auxiliar.
CAMERA_PROJETO = "projeto"
CAMERA_CELULAR = "celular"
CAMERAS: tuple[str, ...] = (CAMERA_PROJETO, CAMERA_CELULAR)

#: Subpasta de sessão da estufa ignorada: casos ambíguos, fora do dataset.
PASTA_DUVIDOSAS = "duvidosas"

#: Categorias do COCO em que pode haver alface (salada, horta, vaso).
SUPERCATEGORIAS_COCO_EXCLUIDAS: frozenset[str] = frozenset({"food"})
CATEGORIAS_COCO_EXCLUIDAS: frozenset[str] = frozenset({"potted plant", "dining table", "bowl"})

#: Ordem de preferência ao manter uma entre cópias idênticas: o Kaggle tem
#: split fixo e a estufa é a fonte mais valiosa.
_PRIORIDADE_FONTE: dict[str, int] = {
    FONTE_KAGGLE: 0,
    FONTE_ESTUFA: 1,
    FONTE_PLANTVILLAGE: 2,
    FONTE_PLANTDOC: 3,
    FONTE_DEEPWEEDS: 4,
    FONTE_COCO: 5,
}

_PADRAO_PASTA = re.compile(r"^(\d{8})_([a-z]+)$")
#: Data e hora de captura no nome do arquivo: "20260926T204727-425793" (script
#: de captura) ou "WIN_20261017_16_01_40_Pro" (app Câmera do Windows).
_PADRAO_CAPTURA = re.compile(r"^(?:WIN_)?(\d{8})[T_](\d{2})_?(\d{2})_?(\d{2})")
_PADRAO_DEEPWEEDS = re.compile(r"^(\d{8})-\d{6}-\d+$")


# --------------------------------------------------------------------------- #
# Auxiliares
# --------------------------------------------------------------------------- #


def _registro(
    arquivo: Path,
    fonte: str,
    classe: str,
    estrato: str,
    grupo: str,
    *,
    camera: str = "",
    dia: str = "",
    sessao: str = "",
    split_fixo: str | None = None,
) -> dict:
    """Monta uma linha do inventário."""
    return {
        "caminho": str(arquivo),
        "arquivo": arquivo.name,
        "fonte": fonte,
        "classe": classe,
        "estrato": estrato,
        "grupo": grupo,
        "camera": camera,
        "dia": dia,
        "sessao": sessao,
        "split_fixo": split_fixo,
    }


def _imagens(pasta: Path, recursivo: bool = False) -> list[Path]:
    """Lista os arquivos de imagem de uma pasta, em ordem determinística."""
    iterador = pasta.rglob("*") if recursivo else pasta.iterdir()
    return sorted(
        p for p in iterador if p.is_file() and p.suffix.lower() in EXTENSOES_IMAGEM
    )


def _subpastas(pasta: Path) -> list[Path]:
    """Lista as subpastas imediatas, em ordem alfabética."""
    return sorted(p for p in pasta.iterdir() if p.is_dir())


def _sanitizar_nome(texto: str) -> str:
    """Reduz um nome de arquivo a caracteres seguros para qualquer sistema."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", texto).strip("_") or "imagem"


def _relacionados(a: Path, b: Path) -> bool:
    """Indica se um caminho é igual a outro ou está contido nele."""
    a, b = Path(a).resolve(), Path(b).resolve()
    return a == b or a in b.parents or b in a.parents


# --------------------------------------------------------------------------- #
# Leitura das fontes
# --------------------------------------------------------------------------- #


def listar_kaggle(raiz_limpo: Path) -> tuple[list[dict], dict[str, int]]:
    """Lista as positivas do dataset do classificador, com o split de origem.

    Args:
        raiz_limpo: Diretório ``desenvolvimento/dataset_clean/``.

    Returns:
        Registros (um por imagem) e estatísticas da fonte.

    Raises:
        FileNotFoundError: Se o dataset limpo ou algum split/classe faltar —
            ``01_preparar_dataset.py`` precisa ter rodado antes.
    """
    raiz_limpo = Path(raiz_limpo)
    registros: list[dict] = []
    for split in SPLITS_SAIDA:
        for doenca in ROTULOS_CANONICOS:
            pasta = raiz_limpo / split / doenca
            if not pasta.is_dir():
                raise FileNotFoundError(
                    f"Pasta {pasta} não encontrada. Execute "
                    "'python desenvolvimento/scripts/01_preparar_dataset.py' primeiro."
                )
            for arquivo in _imagens(pasta):
                registros.append(
                    _registro(
                        arquivo,
                        FONTE_KAGGLE,
                        CLASSE_ALFACE,
                        estrato=doenca,
                        grupo=f"{FONTE_KAGGLE}/{split}/{doenca}/{arquivo.name}",
                        split_fixo=split,
                    )
                )
    if not registros:
        raise FileNotFoundError(f"Nenhuma imagem em {raiz_limpo}.")
    return registros, {"imagens": len(registros)}


def listar_plantvillage(raiz: Path) -> tuple[list[dict], dict[str, int]]:
    """Lista uma imagem por folha do PlantVillage.

    O PlantVillage tem várias fotos da mesma folha. O ``leaf-map.json`` dos
    autores associa o nome original de cada foto a ``<classe>:::<nº da folha>``.
    Fotos sem mapeamento ou associadas a mais de uma folha são descartadas: sem
    o grupo, não há como garantir que a mesma folha não cruze splits.

    Args:
        raiz: Pasta do clone do PlantVillage.

    Returns:
        Candidatos (um por folha, estratificados por cultura) e estatísticas.

    Raises:
        FileNotFoundError: Se ``leaf-map.json`` ou ``raw/color/`` faltarem.
    """
    raiz = Path(raiz)
    caminho_mapa = raiz / "leaf-map.json"
    pasta_cor = raiz / "raw" / "color"
    if not caminho_mapa.is_file() or not pasta_cor.is_dir():
        raise FileNotFoundError(
            f"PlantVillage incompleto em {raiz}: esperados 'leaf-map.json' e 'raw/color/'."
        )
    mapa: dict[str, list[str]] = json.loads(caminho_mapa.read_text(encoding="utf-8"))

    por_folha: dict[str, list[Path]] = defaultdict(list)
    cultura_da_folha: dict[str, str] = {}
    total = sem_mapa = ambiguas = 0
    for pasta_classe in _subpastas(pasta_cor):
        cultura = pasta_classe.name.split("___")[0]
        for arquivo in _imagens(pasta_classe):
            total += 1
            # "<uuid>___<nome original>.JPG" -> chave "<nome original>" em minúsculas
            _, separador, original = arquivo.stem.partition("___")
            chave = (original if separador else arquivo.stem).lower().strip()
            folhas = set(mapa.get(chave, []))
            if not folhas:
                sem_mapa += 1
                continue
            if len(folhas) > 1:
                ambiguas += 1
                continue
            folha = folhas.pop()
            por_folha[folha].append(arquivo)
            cultura_da_folha[folha] = cultura

    candidatos = [
        _registro(
            arquivos[0],
            FONTE_PLANTVILLAGE,
            CLASSE_NAO_ALFACE,
            estrato=cultura_da_folha[folha],
            grupo=f"{FONTE_PLANTVILLAGE}/{folha}",
        )
        for folha, arquivos in sorted(por_folha.items())
    ]
    estatisticas = {
        "imagens": total,
        "sem_mapa": sem_mapa,
        "ambiguas": ambiguas,
        "folhas": len(por_folha),
    }
    return candidatos, estatisticas


def listar_plantdoc(raiz: Path) -> tuple[list[dict], dict[str, int]]:
    """Lista as imagens do PlantDoc (``train/`` e ``test/``), estratificadas por classe.

    Args:
        raiz: Pasta extraída por ``desenvolvimento/scripts/extrair_plantdoc.py``.

    Returns:
        Candidatos (cada imagem é um grupo) e estatísticas.

    Raises:
        FileNotFoundError: Se nenhuma imagem for encontrada.
    """
    raiz = Path(raiz)
    candidatos: list[dict] = []
    for split_original in ("train", "test"):
        pasta = raiz / split_original
        if not pasta.is_dir():
            continue
        for pasta_classe in _subpastas(pasta):
            for arquivo in _imagens(pasta_classe):
                candidatos.append(
                    _registro(
                        arquivo,
                        FONTE_PLANTDOC,
                        CLASSE_NAO_ALFACE,
                        estrato=pasta_classe.name.strip().lower(),
                        grupo=f"{FONTE_PLANTDOC}/{split_original}/{pasta_classe.name}/{arquivo.name}",
                    )
                )
    if not candidatos:
        raise FileNotFoundError(f"Nenhuma imagem em {raiz}/train ou {raiz}/test.")
    return candidatos, {"imagens": len(candidatos)}


def listar_deepweeds(raiz: Path) -> tuple[list[dict], dict[str, int]]:
    """Lista as imagens do DeepWeeds, agrupadas pela data de captura.

    Os nomes seguem ``AAAAMMDD-HHMMSS-N``. Fotos do mesmo dia podem ser quase
    idênticas; agrupar por data impede que cruzem splits. O sorteio percorre as
    datas em rodízio, o que maximiza a variedade de dias.

    Args:
        raiz: Pasta com as imagens extraídas.

    Returns:
        Candidatos e estatísticas.

    Raises:
        FileNotFoundError: Se nenhuma imagem for encontrada.
    """
    candidatos: list[dict] = []
    for arquivo in _imagens(Path(raiz), recursivo=True):
        correspondencia = _PADRAO_DEEPWEEDS.match(arquivo.stem)
        dia = correspondencia.group(1) if correspondencia else "sem_data"
        candidatos.append(
            _registro(
                arquivo,
                FONTE_DEEPWEEDS,
                CLASSE_NAO_ALFACE,
                estrato=dia,
                grupo=f"{FONTE_DEEPWEEDS}/{dia}",
                dia=dia,
            )
        )
    if not candidatos:
        raise FileNotFoundError(f"Nenhuma imagem em {raiz}.")
    dias = {c["dia"] for c in candidatos}
    return candidatos, {"imagens": len(candidatos), "dias": len(dias)}


def listar_coco(raiz: Path) -> tuple[list[dict], dict[str, int]]:
    """Lista as imagens do COCO val2017 em que não pode haver alface.

    Exclui as imagens com qualquer objeto de comida ou de vaso, mesa e tigela,
    e as imagens sem anotação — sem anotação não há como garantir a ausência de
    alface.

    Args:
        raiz: Pasta com ``val2017/`` e ``annotations/instances_val2017.json``.

    Returns:
        Candidatos e estatísticas do filtro.

    Raises:
        FileNotFoundError: Se o arquivo de anotações faltar.
    """
    raiz = Path(raiz)
    caminho_anotacoes = raiz / "annotations" / "instances_val2017.json"
    if not caminho_anotacoes.is_file():
        raise FileNotFoundError(f"Anotações do COCO não encontradas: {caminho_anotacoes}")
    anotacoes = json.loads(caminho_anotacoes.read_text(encoding="utf-8"))

    excluidas = {
        c["id"]
        for c in anotacoes["categories"]
        if c["supercategory"] in SUPERCATEGORIAS_COCO_EXCLUIDAS
        or c["name"] in CATEGORIAS_COCO_EXCLUIDAS
    }
    anotadas: set[int] = set()
    com_excluida: set[int] = set()
    for anotacao in anotacoes["annotations"]:
        anotadas.add(anotacao["image_id"])
        if anotacao["category_id"] in excluidas:
            com_excluida.add(anotacao["image_id"])

    candidatos: list[dict] = []
    ausentes = 0
    imagens = sorted(anotacoes["images"], key=lambda im: im["id"])
    for imagem in imagens:
        if imagem["id"] in com_excluida or imagem["id"] not in anotadas:
            continue
        arquivo = raiz / "val2017" / imagem["file_name"]
        if not arquivo.is_file():
            ausentes += 1
            continue
        candidatos.append(
            _registro(
                arquivo,
                FONTE_COCO,
                CLASSE_NAO_ALFACE,
                estrato=FONTE_COCO,
                grupo=f"{FONTE_COCO}/{imagem['id']}",
            )
        )

    estatisticas = {
        "imagens": len(imagens),
        "com_categoria_excluida": len(com_excluida),
        "sem_anotacao": len({im["id"] for im in imagens} - anotadas),
        "arquivo_ausente": ausentes,
        "utilizaveis": len(candidatos),
    }
    return candidatos, estatisticas


def _momento_de_captura(stem: str, data_pasta: str) -> tuple[str, datetime | None]:
    """Lê o dia ``AAAAMMDD`` e o momento de captura do nome de um arquivo.

    Sem data no nome, vale a data da pasta, sem hora.
    """
    captura = _PADRAO_CAPTURA.match(stem)
    if not captura:
        return data_pasta, None
    dia = captura.group(1)
    try:
        return dia, datetime.strptime("".join(captura.groups()), "%Y%m%d%H%M%S")
    except ValueError:
        logger.warning("Data ou hora inválida no nome '%s'; o dia fica numa única sessão.", stem)
        return dia, None


def sessoes_de_captura(momentos: Sequence[datetime], pausa_horas: float) -> list[str]:
    """Agrupa momentos de captura em sessões.

    Fotos tiradas em sequência formam uma sessão até que uma pausa **maior**
    que ``pausa_horas`` separe duas fotos consecutivas. A sessão pode
    atravessar a meia-noite: fotos de 23:59 e 00:01 são a mesma cena e não
    podem cair em splits diferentes. A sessão é identificada por
    ``AAAAMMDD_HHMM`` da primeira foto, o que preserva a ordem cronológica na
    ordenação textual.

    Args:
        momentos: Data e hora de cada foto, em qualquer ordem.
        pausa_horas: Pausa acima da qual começa uma nova sessão.

    Returns:
        Identificador da sessão de cada momento, na ordem recebida.
    """
    pausa = timedelta(hours=pausa_horas)
    sessoes = [""] * len(momentos)
    inicio: datetime | None = None
    anterior: datetime | None = None
    for indice in sorted(range(len(momentos)), key=momentos.__getitem__):
        momento = momentos[indice]
        if anterior is None or momento - anterior > pausa:
            inicio = momento
        sessoes[indice] = inicio.strftime("%Y%m%d_%H%M")
        anterior = momento
    return sessoes


def _atribuir_sessoes(
    registros: list[dict], momentos: list[datetime | None], pausa_horas: float
) -> None:
    """Preenche ``sessao`` e ``grupo`` das fotos da estufa, alterando os registros.

    Câmera do projeto: sessões de :func:`sessoes_de_captura` sobre todas as
    fotos com hora no nome. Se alguma foto de um dia não tiver hora, esse dia
    inteiro vira uma sessão à parte — separar as fotos sem hora poderia pôr a
    mesma cena em dois splits. Celular: uma sessão por pasta (só vai para o
    treino).
    """
    indices_por_dia: dict[str, list[int]] = defaultdict(list)
    for indice, registro in enumerate(registros):
        if registro["camera"] == CAMERA_PROJETO:
            indices_por_dia[registro["dia"]].append(indice)
        else:
            registro["sessao"] = registro["dia"]

    com_hora: list[int] = []
    for dia, indices in indices_por_dia.items():
        if all(momentos[i] is not None for i in indices):
            com_hora.extend(indices)
            continue
        if any(momentos[i] is not None for i in indices):
            logger.warning(
                "Dia %s tem fotos sem hora no nome: o dia inteiro fica numa única sessão.", dia
            )
        for indice in indices:
            registros[indice]["sessao"] = dia

    sessoes = sessoes_de_captura([momentos[i] for i in com_hora], pausa_horas)
    for indice, sessao in zip(com_hora, sessoes):
        registros[indice]["sessao"] = sessao

    for registro in registros:
        registro["grupo"] = f"{FONTE_ESTUFA}/{registro['camera']}/{registro['sessao']}"


def listar_estufa(
    raiz: Path, pausa_sessao_horas: float = ConfigDatasetDominio.pausa_sessao_horas
) -> tuple[list[dict], dict[str, int]]:
    """Lista as fotos da estufa, organizadas em ``<AAAAMMDD>_<camera>/<classe>/``.

    Para a câmera do projeto, a data e a hora vêm do nome do arquivo quando ele
    começa com ``AAAAMMDDTHHMMSS`` ou ``WIN_AAAAMMDD_HH_MM_SS``; caso contrário,
    a data vem da pasta. As fotos são agrupadas em sessões de captura, que são
    os grupos da divisão. Fotos de celular usam a data da pasta (o WhatsApp
    apaga a data de captura).

    Args:
        raiz: Pasta ``dataset_dominio_bruto/estufa/``. Pode não existir.
        pausa_sessao_horas: Pausa entre fotos do mesmo dia que inicia uma nova
            sessão (ver :func:`sessoes_de_captura`).

    Returns:
        Registros e estatísticas.

    Raises:
        ValueError: Se uma pasta de data/câmera ou de classe fugir do padrão —
            um erro de digitação colocaria fotos na classe errada sem aviso.
    """
    raiz = Path(raiz)
    if not raiz.is_dir():
        return [], {"imagens": 0, "pastas": 0, "sessoes_projeto": 0}

    registros: list[dict] = []
    momentos: list[datetime | None] = []
    pastas = _subpastas(raiz)
    for pasta in pastas:
        correspondencia = _PADRAO_PASTA.match(pasta.name)
        if not correspondencia or correspondencia.group(2) not in CAMERAS:
            raise ValueError(
                f"Pasta fora do padrão <AAAAMMDD>_<{'|'.join(CAMERAS)}>: {pasta}"
            )
        data_pasta, camera = correspondencia.groups()

        for pasta_classe in _subpastas(pasta):
            if pasta_classe.name == PASTA_DUVIDOSAS:
                logger.info("Ignorando casos ambíguos em %s", pasta_classe)
                continue
            if pasta_classe.name not in ROTULOS_DOMINIO:
                raise ValueError(
                    f"Pasta de classe desconhecida: {pasta_classe}. Esperado "
                    f"{list(ROTULOS_DOMINIO)} ou '{PASTA_DUVIDOSAS}'."
                )
            for arquivo in _imagens(pasta_classe):
                dia, momento = data_pasta, None
                if camera == CAMERA_PROJETO:
                    dia, momento = _momento_de_captura(arquivo.stem, data_pasta)
                registros.append(
                    _registro(
                        arquivo,
                        FONTE_ESTUFA,
                        pasta_classe.name,
                        estrato=camera,
                        grupo="",  # preenchido por _atribuir_sessoes
                        camera=camera,
                        dia=dia,
                    )
                )
                momentos.append(momento)

    _atribuir_sessoes(registros, momentos, pausa_sessao_horas)
    sessoes_projeto = {r["sessao"] for r in registros if r["camera"] == CAMERA_PROJETO}
    estatisticas = {
        "imagens": len(registros),
        "pastas": len(pastas),
        "sessoes_projeto": len(sessoes_projeto),
    }
    return registros, estatisticas


# --------------------------------------------------------------------------- #
# Sorteio e divisão
# --------------------------------------------------------------------------- #


def amostrar_estratificado(candidatos: Sequence[dict], n: int, semente: str) -> list[dict]:
    """Sorteia ``n`` candidatos percorrendo os estratos em rodízio.

    O rodízio distribui o sorteio por igual entre culturas, classes ou datas,
    e um estrato pequeno simplesmente se esgota antes dos demais.

    Args:
        candidatos: Registros com a chave ``estrato``.
        n: Quantidade desejada. Se for maior que o total, todos são devolvidos.
        semente: Semente textual; o mesmo valor sempre produz o mesmo sorteio.

    Returns:
        Os registros sorteados.
    """
    if n >= len(candidatos):
        return list(candidatos)

    rng = random.Random(semente)
    por_estrato: dict[str, list[dict]] = defaultdict(list)
    for candidato in candidatos:
        por_estrato[candidato["estrato"]].append(candidato)

    filas: list[list[dict]] = []
    for estrato in sorted(por_estrato):
        fila = sorted(por_estrato[estrato], key=lambda r: r["caminho"])
        rng.shuffle(fila)
        filas.append(fila)
    rng.shuffle(filas)

    escolhidos: list[dict] = []
    while len(escolhidos) < n:
        for fila in filas:
            if fila and len(escolhidos) < n:
                escolhidos.append(fila.pop())
        filas = [fila for fila in filas if fila]
    return escolhidos


def split_por_hash(grupo: str, seed: int, proporcoes: Sequence[float]) -> str:
    """Atribui um split a um grupo a partir do hash do seu identificador.

    O resultado depende só do grupo e da semente: incluir uma fonte nova não
    muda o split de nenhum grupo já existente.

    Args:
        grupo: Identificador do grupo.
        seed: Semente.
        proporcoes: Frações ``(train, val, test)``, somando 1.

    Returns:
        ``"train"``, ``"val"`` ou ``"test"``.

    Raises:
        ValueError: Se as proporções não somarem 1.
    """
    if abs(sum(proporcoes) - 1.0) > 1e-6:
        raise ValueError(f"As proporções devem somar 1; recebido {list(proporcoes)}.")
    valor = int(hashlib.sha1(f"{seed}:{grupo}".encode("utf-8")).hexdigest()[:8], 16) / 2**32
    limite = 0.0
    for split, fracao in zip(SPLITS_SAIDA, proporcoes):
        limite += fracao
        if valor < limite:
            return split
    return SPLITS_SAIDA[-1]


def dividir_sessoes(
    sessoes: Iterable[str], fracao_teste: float = 0.25, fracao_val: float = 0.25
) -> dict[str, str]:
    """Divide sessões de captura em ordem cronológica: as mais recentes no teste.

    Com poucas sessões, um sorteio poderia não deixar nenhuma no teste; a ordem
    cronológica também simula o uso real, em que o modelo vê dias futuros.

    Args:
        sessoes: Identificadores ``AAAAMMDD`` ou ``AAAAMMDD_HHMM``, cuja ordem
            textual é a cronológica (repetições são ignoradas).
        fracao_teste: Fração das sessões para teste (mínimo de uma).
        fracao_val: Fração das sessões para validação (só a partir de três).

    Returns:
        Mapa sessão -> split.
    """
    ordenadas = sorted(set(sessoes))
    n = len(ordenadas)
    if n == 0:
        return {}
    n_teste = min(n, max(1, round(n * fracao_teste)))
    n_val = min(n - n_teste, round(n * fracao_val)) if n >= 3 else 0
    n_treino = n - n_teste - n_val

    mapa: dict[str, str] = {}
    for posicao, sessao in enumerate(ordenadas):
        if posicao < n_treino:
            mapa[sessao] = "train"
        elif posicao < n_treino + n_val:
            mapa[sessao] = "val"
        else:
            mapa[sessao] = "test"
    return mapa


def dividir_sessoes_estratificado(
    sessoes: Iterable[str],
    sessoes_com_nao_alface: Iterable[str],
    fracao_teste: float = 0.25,
    fracao_val: float = 0.25,
    data_corte: str | None = None,
) -> dict[str, str]:
    """Aplica :func:`dividir_sessoes` separadamente a dois tipos de sessão.

    As primeiras fotos da câmera do projeto eram só de alface; as de "não
    alface" vieram depois. Numa única ordem cronológica, toda sessão de "não
    alface" cairia no teste e nunca no treino. Separando as sessões com "não
    alface" das que só têm alface, cada tipo é dividido na própria ordem
    cronológica. Cada sessão continua inteira num único split.

    Args:
        sessoes: Todas as sessões de captura da câmera do projeto.
        sessoes_com_nao_alface: Sessões com pelo menos uma foto de "não alface".
        fracao_teste: Fração das sessões de cada tipo para teste (mínimo de uma).
        fracao_val: Fração das sessões de cada tipo para validação (a partir de três).
        data_corte: Data ``AAAAMMDD`` (inclusive). Sessões iniciadas até ela
            vão inteiras para o treino, e só as posteriores são divididas.
            Uma sessão que começa no dia do corte e atravessa a meia-noite
            também vai para o treino. ``None`` desliga o corte.

    Returns:
        Mapa sessão -> split.

    Raises:
        ValueError: Se ``data_corte`` não for uma data ``AAAAMMDD`` válida.
    """
    todas = set(sessoes)
    mapa: dict[str, str] = {}
    if data_corte is not None:
        try:
            datetime.strptime(data_corte, "%Y%m%d")
        except ValueError as erro:
            raise ValueError(
                f"data_corte deve ser uma data AAAAMMDD; recebido {data_corte!r}."
            ) from erro
        ate_corte = {sessao for sessao in todas if sessao[:8] <= data_corte}
        mapa.update(dict.fromkeys(ate_corte, "train"))
        todas -= ate_corte

    com_nao_alface = todas & set(sessoes_com_nao_alface)
    mapa.update(dividir_sessoes(todas - com_nao_alface, fracao_teste, fracao_val))
    mapa.update(dividir_sessoes(com_nao_alface, fracao_teste, fracao_val))
    return mapa


def conflitos_de_classe(inventario: pd.DataFrame) -> list[str]:
    """Detecta imagens idênticas rotuladas nas duas classes.

    Precisa rodar **antes** da deduplicação, que manteria uma das cópias e
    esconderia o conflito.

    Args:
        inventario: Inventário com ``hash_md5`` e ``classe``.

    Returns:
        Mensagens de problema; vazia se não houver conflito.
    """
    por_hash = inventario.groupby("hash_md5")["classe"].nunique()
    conflitos = por_hash[por_hash > 1]
    if conflitos.empty:
        return []
    caminhos = inventario[inventario["hash_md5"].isin(conflitos.index)]["caminho"]
    return [
        f"{len(conflitos)} imagens idênticas aparecem como '{CLASSE_ALFACE}' e "
        f"'{CLASSE_NAO_ALFACE}' (exemplos: {', '.join(caminhos.head(4))})."
    ]


def deduplicar_dominio(inventario: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove cópias byte-idênticas, mantendo a da fonte de maior prioridade.

    Args:
        inventario: Inventário com ``hash_md5``, ``fonte`` e ``caminho``.

    Returns:
        ``(mantidos, descartados)``.
    """
    ordenado = inventario.assign(
        _prioridade=inventario["fonte"].map(_PRIORIDADE_FONTE)
    ).sort_values(["_prioridade", "caminho"], kind="mergesort")
    repetido = ordenado.duplicated("hash_md5", keep="first")
    mantidos = ordenado[~repetido].drop(columns="_prioridade").reset_index(drop=True)
    descartados = ordenado[repetido].drop(columns="_prioridade").reset_index(drop=True)
    return mantidos, descartados


def atribuir_splits(
    inventario: pd.DataFrame, config: ConfigDatasetDominio
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Atribui o split de cada imagem conforme a regra da sua fonte.

    Args:
        inventario: Inventário deduplicado.
        config: Parâmetros de divisão.

    Returns:
        Inventário com a coluna ``split`` e o mapa sessão -> split da câmera do
        projeto.
    """
    eh_projeto = (inventario["fonte"] == FONTE_ESTUFA) & (
        inventario["camera"] == CAMERA_PROJETO
    )
    projeto = inventario[eh_projeto]
    mapa_sessoes = dividir_sessoes_estratificado(
        projeto["sessao"],
        projeto.loc[projeto["classe"] == CLASSE_NAO_ALFACE, "sessao"],
        fracao_teste=config.fracao_sessoes_teste,
        fracao_val=config.fracao_sessoes_val,
        data_corte=config.data_corte_treino,
    )

    splits: list[str] = []
    for registro in inventario.to_dict("records"):
        if registro["fonte"] == FONTE_KAGGLE:
            splits.append(registro["split_fixo"])
        elif registro["fonte"] == FONTE_ESTUFA and registro["camera"] == CAMERA_PROJETO:
            splits.append(mapa_sessoes[registro["sessao"]])
        elif registro["fonte"] == FONTE_ESTUFA:
            splits.append("train")
        else:
            splits.append(split_por_hash(registro["grupo"], config.seed, config.proporcoes))

    resultado = inventario.copy()
    resultado["split"] = splits
    return resultado, mapa_sessoes


def verificar_integridade_dominio(inventario: pd.DataFrame) -> list[str]:
    """Verifica que a divisão não tem vazamento e respeita as regras por fonte.

    Args:
        inventario: Inventário dividido, com a coluna ``split``.

    Returns:
        Mensagens de problema, em português. Lista vazia = aprovado.
    """
    if "split" not in inventario.columns:
        return ["O inventário não possui a coluna 'split'."]

    problemas: list[str] = []
    faltando = int(inventario["split"].isna().sum())
    if faltando:
        problemas.append(f"{faltando} imagens ficaram sem split atribuído.")

    for coluna, descricao in (
        ("hash_md5", "hashes MD5 idênticos"),
        ("grupo", "grupos (folha, data ou sessão)"),
    ):
        por_chave = inventario.groupby(coluna)["split"].nunique()
        vazando = por_chave[por_chave > 1]
        if len(vazando):
            exemplos = ", ".join(map(str, vazando.index[:5]))
            problemas.append(
                f"{len(vazando)} {descricao} aparecem em mais de um split (exemplos: {exemplos})."
            )

    for split in SPLITS_SAIDA:
        bloco = inventario[inventario["split"] == split]
        ausentes = set(ROTULOS_DOMINIO) - set(bloco["classe"])
        if ausentes:
            problemas.append(f"O split '{split}' não contém as classes: {sorted(ausentes)}.")

    kaggle = inventario[inventario["fonte"] == FONTE_KAGGLE]
    fora = kaggle[kaggle["split"] != kaggle["split_fixo"]]
    if len(fora):
        problemas.append(
            f"{len(fora)} imagens do Kaggle mudaram de split em relação a dataset_clean/."
        )

    celular_no_teste = inventario[
        (inventario["camera"] == CAMERA_CELULAR) & (inventario["split"] == "test")
    ]
    if len(celular_no_teste):
        problemas.append(
            f"{len(celular_no_teste)} fotos de celular caíram no teste; só a câmera "
            "do projeto serve como evidência."
        )
    return problemas


# --------------------------------------------------------------------------- #
# Escrita e pipeline
# --------------------------------------------------------------------------- #


def _inspecionar(registros: list[dict]) -> tuple[list[dict], list[dict]]:
    """Abre cada imagem por completo, calcula o MD5 e separa as ilegíveis."""
    validos: list[dict] = []
    ilegiveis: list[dict] = []
    for registro in registros:
        caminho = Path(registro["caminho"])
        try:
            with Image.open(caminho) as imagem:
                imagem.load()
                largura, altura = imagem.size
                formato, modo = imagem.format, imagem.mode
        except Exception as erro:  # noqa: BLE001 - queremos catalogar a falha
            ilegiveis.append({**registro, "erro": str(erro)})
            logger.warning("Imagem ilegível, ignorada: %s (%s)", caminho, erro)
            continue
        if largura < 2 or altura < 2:
            ilegiveis.append({**registro, "erro": f"dimensões degeneradas {largura}×{altura}"})
            continue
        validos.append(
            {
                **registro,
                "hash_md5": hash_arquivo(caminho),
                "largura": largura,
                "altura": altura,
                "formato": formato,
                "modo": modo,
            }
        )
    return validos, ilegiveis


def escrever_dataset_dominio(
    inventario: pd.DataFrame, destino: Path, qualidade_jpeg: int = 95
) -> pd.DataFrame:
    """Materializa o dataset no layout do Ultralytics.

    JPEGs em RGB são copiados byte a byte. Os demais (PNG, RGBA, tons de
    cinza, CMYK) são reconvertidos para JPEG RGB, respeitando a orientação EXIF
    e **sem redimensionar**.

    Args:
        inventario: Inventário dividido.
        destino: Diretório de saída. Se existir, é removido antes da escrita.
        qualidade_jpeg: Qualidade das imagens reconvertidas.

    Returns:
        Cópia do inventário com a coluna ``caminho_destino``.

    Raises:
        RuntimeError: Se dois arquivos de saída colidirem no nome.
    """
    destino = Path(destino)
    if destino.exists():
        logger.info("Removendo %s de uma execução anterior.", destino)
        shutil.rmtree(destino)
    for split in SPLITS_SAIDA:
        for classe in ROTULOS_DOMINIO:
            (destino / split / classe).mkdir(parents=True, exist_ok=True)

    caminhos_destino: list[str] = []
    for registro in inventario.itertuples():
        stem = _sanitizar_nome(Path(registro.arquivo).stem)[:40]
        nome = f"{registro.fonte}__{stem}__{registro.hash_md5[:8]}.jpg"
        saida = destino / registro.split / registro.classe / nome
        if saida.exists():
            raise RuntimeError(f"Nome de saída repetido: {saida}")

        if registro.formato == "JPEG" and registro.modo == "RGB":
            shutil.copyfile(registro.caminho, saida)
        else:
            with Image.open(registro.caminho) as aberta:
                imagem = ImageOps.exif_transpose(aberta)
                if imagem.mode != "RGB":
                    imagem = imagem.convert("RGB")
                imagem.save(saida, format="JPEG", quality=qualidade_jpeg)
        caminhos_destino.append(str(saida))

    resultado = inventario.copy()
    resultado["caminho_destino"] = caminhos_destino
    logger.info("Dataset do verificador escrito em %s (%d imagens).", destino, len(resultado))
    return resultado


#: Fontes negativas públicas, na ordem em que são lidas.
_LEITORES_PUBLICOS: dict[str, Callable[[Path], tuple[list[dict], dict[str, int]]]] = {
    FONTE_PLANTVILLAGE: listar_plantvillage,
    FONTE_PLANTDOC: listar_plantdoc,
    FONTE_DEEPWEEDS: listar_deepweeds,
    FONTE_COCO: listar_coco,
}


def preparar_dominio(
    raiz_bruto: Path,
    raiz_limpo: Path,
    destino: Path,
    config: ConfigDatasetDominio | None = None,
    escrever: bool = True,
) -> dict[str, object]:
    """Executa o pipeline completo de montagem do dataset do verificador.

    Etapas: listar as fontes, sortear as públicas, abrir e hashear cada
    imagem, detectar conflito de classe, deduplicar, dividir, verificar
    integridade e — só se aprovado — escrever em disco.

    Args:
        raiz_bruto: ``desenvolvimento/dataset_dominio_bruto/`` (somente leitura).
        raiz_limpo: ``desenvolvimento/dataset_clean/`` (somente leitura).
        destino: ``desenvolvimento/dataset_dominio/``.
        config: Parâmetros; o padrão é :class:`ConfigDatasetDominio`.
        escrever: Se ``False``, faz tudo menos a escrita.

    Returns:
        Dicionário com os artefatos, para o relatório de auditoria.

    Raises:
        ValueError: Se o destino coincidir com uma origem ou estiver contido
            nela (a escrita apagaria o destino antes de gravar).
        FileNotFoundError: Se ``dataset_clean/`` não existir.
    """
    config = config or ConfigDatasetDominio()
    raiz_bruto, raiz_limpo, destino = Path(raiz_bruto), Path(raiz_limpo), Path(destino)
    for origem in (raiz_bruto, raiz_limpo):
        if _relacionados(destino, origem):
            raise ValueError(
                f"O destino {destino} não pode coincidir com a origem {origem} nem "
                "estar contido nela: as origens nunca são escritas."
            )

    avisos: list[str] = []
    estatisticas: dict[str, dict[str, int]] = {}
    contagens: dict[str, dict[str, int]] = {}

    kaggle, estatisticas[FONTE_KAGGLE] = listar_kaggle(raiz_limpo)
    contagens[FONTE_KAGGLE] = {"disponiveis": len(kaggle), "sorteadas": len(kaggle)}
    registros = list(kaggle)

    for fonte, leitor in _LEITORES_PUBLICOS.items():
        pasta = raiz_bruto / CLASSE_NAO_ALFACE / fonte
        if not pasta.is_dir():
            avisos.append(f"Fonte ausente, ignorada: {pasta}")
            contagens[fonte] = {"disponiveis": 0, "sorteadas": 0}
            continue
        candidatos, estatisticas[fonte] = leitor(pasta)
        sorteados = amostrar_estratificado(candidatos, config.por_fonte, f"{config.seed}:{fonte}")
        contagens[fonte] = {"disponiveis": len(candidatos), "sorteadas": len(sorteados)}
        registros.extend(sorteados)

    estufa, estatisticas[FONTE_ESTUFA] = listar_estufa(
        raiz_bruto / FONTE_ESTUFA, config.pausa_sessao_horas
    )
    contagens[FONTE_ESTUFA] = {"disponiveis": len(estufa), "sorteadas": len(estufa)}
    if not any(r["classe"] == CLASSE_NAO_ALFACE for r in estufa):
        avisos.append("Nenhuma foto de 'nao_alface' da estufa: o teste real fica incompleto.")
    registros.extend(estufa)

    validos, ilegiveis = _inspecionar(registros)
    inventario = pd.DataFrame(validos)
    problemas = conflitos_de_classe(inventario)
    mantidos, descartados = deduplicar_dominio(inventario)
    dividido, mapa_sessoes = atribuir_splits(mantidos, config)
    corte = config.data_corte_treino
    if corte is not None and mapa_sessoes and all(s[:8] <= corte for s in mapa_sessoes):
        avisos.append(
            f"Nenhuma sessão da câmera do projeto depois da data de corte {corte}: "
            "validação e teste ficam sem fotos da C270."
        )
    problemas += verificar_integridade_dominio(dividido)

    if escrever and not problemas:
        dividido = escrever_dataset_dominio(dividido, destino, config.qualidade_jpeg)

    return {
        "inventario": inventario,
        "ilegiveis": pd.DataFrame(ilegiveis),
        "descartados": descartados,
        "dividido": dividido,
        "mapa_sessoes": mapa_sessoes,
        "estatisticas": estatisticas,
        "contagens": contagens,
        "avisos": avisos,
        "problemas": problemas,
        "config": config,
        "raiz_bruto": raiz_bruto,
        "raiz_limpo": raiz_limpo,
        "destino": destino,
    }
