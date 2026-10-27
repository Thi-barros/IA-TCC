"""Testes da montagem do dataset do verificador de domínio.

Cada fonte é reproduzida em miniatura numa pasta temporária, com a mesma
estrutura da real; a suíte não depende de ``dataset_dominio_bruto/`` existir.

Referência: docs/VERIFICADOR_DOMINIO.md.
"""

from __future__ import annotations

import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ia_alface import dados_dominio as dd
from ia_alface.classes import CLASSE_ALFACE, CLASSE_NAO_ALFACE, ROTULOS_CANONICOS, ROTULOS_DOMINIO
from ia_alface.config import ConfigDatasetDominio
from ia_alface.dados import SPLITS_SAIDA

# --------------------------------------------------------------------------- #
# Fontes em miniatura
# --------------------------------------------------------------------------- #


def _imagem(caminho: Path, semente: int, canais: int = 3, formato: str = "JPEG") -> Path:
    """Grava uma imagem 12×12 de ruído; sementes diferentes geram arquivos diferentes."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    forma = (12, 12, canais) if canais > 1 else (12, 12)
    pixels = np.random.default_rng(semente).integers(0, 255, forma, dtype=np.uint8)
    Image.fromarray(pixels).save(caminho, format=formato)
    return caminho


#: Fotos só com alface da câmera do projeto na miniatura; as duas últimas
#: atravessam a meia-noite, como 10/10 23:59 → 11/10 00:01.
FOTOS_SO_ALFACE = [
    "20260926T120000-0.jpg",
    "20260928T120000-1.jpg",
    "20260929T120000-2.jpg",
    "20261010T235900-3.jpg",
    "20261011T000100-4.jpg",
]

#: Dia misto da miniatura, como 17/10: tarde e noite com as duas classes,
#: separadas por uma pausa de 2 h 28 min.
FOTOS_DIA_MISTO = [
    (CLASSE_NAO_ALFACE, "WIN_20261017_16_01_40_Pro.jpg"),
    (CLASSE_ALFACE, "WIN_20261017_16_11_56_Pro.jpg"),
    (CLASSE_NAO_ALFACE, "WIN_20261017_18_40_40_Pro.jpg"),
    (CLASSE_ALFACE, "WIN_20261017_19_00_22_Pro.jpg"),
]


@pytest.fixture
def fontes(tmp_path: Path) -> dict[str, Path]:
    """Monta dataset_clean, as quatro fontes públicas e a estufa em miniatura."""
    semente = itertools.count(1)

    limpo = tmp_path / "dataset_clean"
    for split in SPLITS_SAIDA:
        for doenca in ROTULOS_CANONICOS:
            for i in range(2):
                _imagem(limpo / split / doenca / f"k{i}.jpg", next(semente))

    bruto = tmp_path / "bruto"
    nao = bruto / CLASSE_NAO_ALFACE

    # PlantVillage: folha Peach 1 com duas fotos, uma foto sem mapa, uma ambígua.
    pv = nao / "plantvillage"
    nomes_pv = {
        "Peach___healthy": ["Rutg._HL 1", "Rutg._HL 2", "Rutg._HL 3", "Rutg._HL 4", "Rutg._HL 5"],
        "Tomato___healthy": ["Com.G_HL 6", "Com.G_HL 7"],
    }
    for classe_pv, nomes in nomes_pv.items():
        for indice, nome in enumerate(nomes):
            _imagem(pv / "raw" / "color" / classe_pv / f"uuid{indice}___{nome}.JPG", next(semente))
    mapa = {
        "rutg._hl 1": ["Peach___healthy:::1.0"],
        "rutg._hl 2": ["Peach___healthy:::1.0"],
        "rutg._hl 3": ["Peach___healthy:::2.0"],
        "rutg._hl 5": ["Peach___healthy:::3.0", "Peach___healthy:::4.0"],
        "com.g_hl 6": ["Tomato___healthy:::1.0"],
        "com.g_hl 7": ["Tomato___healthy:::2.0"],
    }
    (pv / "leaf-map.json").write_text(json.dumps(mapa), encoding="utf-8")

    # PlantDoc: um PNG com transparência e um arquivo corrompido.
    plantdoc = nao / "plantdoc"
    _imagem(plantdoc / "train" / "Apple leaf" / "a1.jpg", next(semente))
    _imagem(plantdoc / "train" / "Apple leaf" / "a2.jpg", next(semente))
    _imagem(plantdoc / "test" / "Corn leaf" / "c1.png", next(semente), canais=4, formato="PNG")
    (plantdoc / "train" / "Apple leaf" / "corrompida.jpg").write_bytes(b"isto nao eh uma imagem")
    (plantdoc / "README.md").write_text("não é imagem", encoding="utf-8")

    # DeepWeeds: 30 datas, uma delas com duas fotos.
    deepweeds = nao / "deepweeds"
    for i in range(30):
        _imagem(deepweeds / f"2017{i // 28 + 1:02d}{i % 28 + 1:02d}-120000-0.jpg", next(semente))
    _imagem(deepweeds / "20170101-130000-1.jpg", next(semente))

    # COCO: 20 utilizáveis, 1 com brócolis, 1 com mesa, 1 sem anotação.
    coco = nao / "coco"
    imagens, anotacoes = [], []
    for i in range(1, 24):
        nome = f"{i:012d}.jpg"
        _imagem(coco / "val2017" / nome, next(semente))
        imagens.append({"id": i, "file_name": nome})
    for i in range(1, 21):
        anotacoes.append({"image_id": i, "category_id": 1 if i % 2 else 4})
    anotacoes += [
        {"image_id": 21, "category_id": 1},
        {"image_id": 21, "category_id": 2},
        {"image_id": 22, "category_id": 3},
    ]
    categorias = [
        {"id": 1, "name": "person", "supercategory": "person"},
        {"id": 2, "name": "broccoli", "supercategory": "food"},
        {"id": 3, "name": "dining table", "supercategory": "furniture"},
        {"id": 4, "name": "car", "supercategory": "vehicle"},
    ]
    (coco / "annotations").mkdir(parents=True)
    (coco / "annotations" / "instances_val2017.json").write_text(
        json.dumps({"images": imagens, "annotations": anotacoes, "categories": categorias}),
        encoding="utf-8",
    )

    # Estufa: 4 sessões só com alface (uma atravessa a meia-noite), cópia
    # "(1)", um dia com tarde e noite mistas, celular e duvidosas.
    estufa = bruto / "estufa"
    projeto = estufa / "20260930_projeto" / CLASSE_ALFACE
    for nome in FOTOS_SO_ALFACE:
        _imagem(projeto / nome, next(semente))
    original = projeto / "20260926T120000-0.jpg"
    (projeto / "20260926T120000-0 (1).jpg").write_bytes(original.read_bytes())
    for classe, nome in FOTOS_DIA_MISTO:
        _imagem(estufa / "20261017_projeto" / classe / nome, next(semente))
    celular = estufa / "20261015_celular"
    _imagem(celular / CLASSE_ALFACE / "WhatsApp 1.jpeg", next(semente))
    _imagem(celular / CLASSE_ALFACE / "WhatsApp 2.jpeg", next(semente))
    _imagem(celular / "duvidosas" / "talvez.jpg", next(semente))

    return {"bruto": bruto, "limpo": limpo, "destino": tmp_path / "dataset_dominio"}


@pytest.fixture
def artefatos(fontes) -> dict:
    """Executa o pipeline completo sobre as fontes em miniatura, sem data de corte.

    As sessões da miniatura são todas anteriores à data de corte padrão; sem
    desligá-la, todas iriam para o treino e a divisão não seria exercitada.
    """
    return dd.preparar_dominio(
        fontes["bruto"],
        fontes["limpo"],
        fontes["destino"],
        ConfigDatasetDominio(data_corte_treino=None),
    )


# --------------------------------------------------------------------------- #
# Pipeline completo
# --------------------------------------------------------------------------- #


def test_pipeline_aprovado_escreve_layout_do_ultralytics(fontes, artefatos):
    """Sem vazamento, o dataset é escrito em split/classe, uma imagem por linha."""
    assert artefatos["problemas"] == []
    for split in SPLITS_SAIDA:
        for classe in ROTULOS_DOMINIO:
            assert (fontes["destino"] / split / classe).is_dir()
    escritos = list(fontes["destino"].rglob("*.jpg"))
    assert len(escritos) == len(artefatos["dividido"])


def _filtrar(df: pd.DataFrame, **igualdades) -> pd.DataFrame:
    """Linhas em que cada coluna informada é igual ao valor dado."""
    mascara = pd.Series(True, index=df.index)
    for coluna, valor in igualdades.items():
        mascara &= df[coluna] == valor
    return df[mascara]


def test_kaggle_mantem_o_split_original(artefatos):
    """As positivas do classificador ficam no mesmo split de dataset_clean/."""
    kaggle = _filtrar(artefatos["dividido"], fonte=dd.FONTE_KAGGLE)
    assert len(kaggle) == len(SPLITS_SAIDA) * len(ROTULOS_CANONICOS) * 2
    assert (kaggle["split"] == kaggle["split_fixo"]).all()
    assert (kaggle["classe"] == CLASSE_ALFACE).all()


def test_fotos_de_celular_ficam_so_no_treino(artefatos):
    """O celular não é a câmera de operação: nunca serve de evidência no teste."""
    celular = _filtrar(artefatos["dividido"], camera=dd.CAMERA_CELULAR)
    assert len(celular) == 2
    assert set(celular["split"]) == {"train"}
    assert set(celular["sessao"]) == {"20261015"}


def test_sessoes_da_camera_do_projeto_em_ordem_cronologica(artefatos):
    """Sessões só com alface e sessões com "não alface" são divididas separadamente.

    Só alface (4 sessões, com 10/10 e 11/10 juntas): 2 treino, 1 validação,
    1 teste. Com "não alface" (tarde e noite de 17/10): a tarde vai para o
    treino e a noite para o teste.
    """
    assert artefatos["mapa_sessoes"] == {
        "20260926_1200": "train",
        "20260928_1200": "train",
        "20260929_1200": "val",
        "20261010_2359": "test",
        "20261017_1601": "train",
        "20261017_1840": "test",
    }
    for split in ("train", "test"):
        projeto = _filtrar(artefatos["dividido"], camera=dd.CAMERA_PROJETO, split=split)
        assert set(projeto["classe"]) == set(ROTULOS_DOMINIO)


def test_sessao_que_atravessa_a_meia_noite_fica_inteira(artefatos):
    """As fotos de 10/10 23:59 e 11/10 00:01 são uma sessão, num único split."""
    sessao = _filtrar(artefatos["dividido"], sessao="20261010_2359")
    assert set(sessao["dia"]) == {"20261010", "20261011"}
    assert set(sessao["split"]) == {"test"}


def test_copia_identica_e_removida(artefatos):
    """O arquivo '(1)' do Windows é byte-idêntico e sai na deduplicação."""
    descartados = artefatos["descartados"]
    assert len(descartados) == 1
    assert descartados.iloc[0]["fonte"] == dd.FONTE_ESTUFA
    assert artefatos["dividido"]["hash_md5"].is_unique


def test_imagem_corrompida_vai_para_ilegiveis(artefatos):
    """Arquivo que não abre é catalogado e fica fora do dataset."""
    ilegiveis = artefatos["ilegiveis"]
    assert list(ilegiveis["arquivo"]) == ["corrompida.jpg"]
    assert "corrompida.jpg" not in set(artefatos["dividido"]["arquivo"])


def test_duvidosas_ficam_fora(artefatos):
    """A pasta 'duvidosas' não entra em nenhuma classe."""
    assert "talvez.jpg" not in set(artefatos["dividido"]["arquivo"])


def test_png_com_transparencia_vira_jpeg_rgb(artefatos):
    """Imagens que não são JPEG RGB são reconvertidas, sem redimensionar."""
    linha = _filtrar(artefatos["dividido"], arquivo="c1.png").iloc[0]
    with Image.open(linha["caminho_destino"]) as convertida:
        assert convertida.format == "JPEG"
        assert convertida.mode == "RGB"
        assert convertida.size == (12, 12)


def test_jpeg_rgb_e_copiado_byte_a_byte(artefatos):
    """Sem reconversão desnecessária: o conteúdo de um JPEG RGB não muda."""
    linha = _filtrar(artefatos["dividido"], fonte=dd.FONTE_KAGGLE).iloc[0]
    assert Path(linha["caminho_destino"]).read_bytes() == Path(linha["caminho"]).read_bytes()


def test_origens_nunca_sao_alteradas(fontes):
    """dataset_dominio_bruto/ e dataset_clean/ são somente leitura."""

    def retrato(raiz: Path) -> dict[str, tuple[int, int]]:
        return {
            str(p): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in raiz.rglob("*")
            if p.is_file()
        }

    antes = {**retrato(fontes["bruto"]), **retrato(fontes["limpo"])}
    dd.preparar_dominio(fontes["bruto"], fontes["limpo"], fontes["destino"])
    depois = {**retrato(fontes["bruto"]), **retrato(fontes["limpo"])}
    assert antes == depois


def test_destino_dentro_da_origem_e_recusado(fontes):
    """A escrita apaga o destino: ele nunca pode ficar dentro de uma origem."""
    with pytest.raises(ValueError, match="origem"):
        dd.preparar_dominio(fontes["bruto"], fontes["limpo"], fontes["bruto"] / "saida")


def test_sem_dataset_clean_falha(fontes, tmp_path):
    """O 01_preparar_dataset.py precisa ter rodado antes."""
    with pytest.raises(FileNotFoundError, match="01_preparar_dataset"):
        dd.preparar_dominio(fontes["bruto"], tmp_path / "nao_existe", fontes["destino"])


def test_somente_auditoria_nao_escreve(fontes):
    """Com escrever=False nada é gravado no destino."""
    resultado = dd.preparar_dominio(
        fontes["bruto"], fontes["limpo"], fontes["destino"], escrever=False
    )
    assert resultado["problemas"] == []
    assert not fontes["destino"].exists()


# --------------------------------------------------------------------------- #
# Leitura das fontes
# --------------------------------------------------------------------------- #


def test_plantvillage_uma_imagem_por_folha(fontes):
    """Fotos da mesma folha viram um candidato; sem mapa ou ambíguas saem."""
    candidatos, estatisticas = dd.listar_plantvillage(
        fontes["bruto"] / CLASSE_NAO_ALFACE / "plantvillage"
    )
    assert estatisticas == {"imagens": 7, "sem_mapa": 1, "ambiguas": 1, "folhas": 4}
    nomes = {c["arquivo"] for c in candidatos}
    assert "uuid0___Rutg._HL 1.JPG" in nomes
    assert "uuid1___Rutg._HL 2.JPG" not in nomes  # mesma folha da anterior
    assert {c["estrato"] for c in candidatos} == {"Peach", "Tomato"}
    assert len({c["grupo"] for c in candidatos}) == len(candidatos)


def test_coco_exclui_comida_mesa_e_sem_anotacao(fontes):
    """Imagens onde pode haver alface, ou sem anotação, ficam de fora."""
    candidatos, estatisticas = dd.listar_coco(fontes["bruto"] / CLASSE_NAO_ALFACE / "coco")
    assert estatisticas["utilizaveis"] == 20
    assert estatisticas["com_categoria_excluida"] == 2
    assert estatisticas["sem_anotacao"] == 1
    excluidas = {f"{i:012d}.jpg" for i in (21, 22, 23)}
    assert not excluidas & {c["arquivo"] for c in candidatos}


def test_deepweeds_agrupa_pela_data(fontes):
    """Fotos do mesmo dia formam um único grupo."""
    candidatos, estatisticas = dd.listar_deepweeds(
        fontes["bruto"] / CLASSE_NAO_ALFACE / "deepweeds"
    )
    assert estatisticas == {"imagens": 31, "dias": 30}
    mesmo_dia = [c for c in candidatos if c["arquivo"].startswith("20170101")]
    assert len({c["grupo"] for c in mesmo_dia}) == 1


@pytest.mark.parametrize("nome", ["2026-10-16_projeto", "20261016_webcam", "20261016"])
def test_pasta_de_sessao_fora_do_padrao_falha(tmp_path, nome):
    """Um erro de digitação na sessão nunca pode passar despercebido."""
    _imagem(tmp_path / nome / CLASSE_ALFACE / "x.jpg", 1)
    with pytest.raises(ValueError, match="padrão"):
        dd.listar_estufa(tmp_path)


def test_pasta_de_classe_desconhecida_falha(tmp_path):
    """'alfaces' no lugar de 'alface' poria fotos na classe errada."""
    _imagem(tmp_path / "20261016_projeto" / "alfaces" / "x.jpg", 1)
    with pytest.raises(ValueError, match="classe desconhecida"):
        dd.listar_estufa(tmp_path)


@pytest.mark.parametrize(
    ("arquivo", "dia_esperado", "sessao_esperada"),
    [
        ("20260926T204727-425793.jpg", "20260926", "20260926_2047"),
        ("20260928T223645-363109 (1).jpg", "20260928", "20260928_2236"),
        ("WIN_20261017_16_01_40_Pro.jpg", "20261017", "20261017_1601"),
        ("WIN_20261017_16_03_49_Pro (2).jpg", "20261017", "20261017_1603"),
        ("foto_sem_data.jpg", "20261020", "20261020"),  # cai na data da pasta
    ],
)
def test_dia_e_sessao_vem_do_nome_do_arquivo(tmp_path, arquivo, dia_esperado, sessao_esperada):
    """Os dois padrões de nome da câmera do projeto são reconhecidos."""
    _imagem(tmp_path / "20261020_projeto" / CLASSE_NAO_ALFACE / arquivo, 1)
    registros, _ = dd.listar_estufa(tmp_path)
    assert [r["dia"] for r in registros] == [dia_esperado]
    assert [r["sessao"] for r in registros] == [sessao_esperada]
    assert [r["grupo"] for r in registros] == [f"estufa/projeto/{sessao_esperada}"]


def test_dia_com_foto_sem_hora_fica_numa_unica_sessao(tmp_path):
    """Sem a hora de todas as fotos, não há como separar o dia com segurança."""
    pasta = tmp_path / "20261017_projeto"
    _imagem(pasta / CLASSE_NAO_ALFACE / "WIN_20261017_16_01_40_Pro.jpg", 1)
    _imagem(pasta / CLASSE_ALFACE / "WIN_20261017_19_00_22_Pro.jpg", 2)
    _imagem(pasta / CLASSE_ALFACE / "foto_sem_hora.jpg", 3)
    registros, estatisticas = dd.listar_estufa(tmp_path)
    assert {r["sessao"] for r in registros} == {"20261017"}
    assert estatisticas == {"imagens": 3, "pastas": 1, "sessoes_projeto": 1}


def test_estufa_ausente_devolve_vazio(tmp_path):
    """Sem fotos da estufa, o pipeline segue com as demais fontes."""
    assert dd.listar_estufa(tmp_path / "nao_existe") == (
        [],
        {"imagens": 0, "pastas": 0, "sessoes_projeto": 0},
    )


def _momentos(*textos: str) -> list[datetime]:
    """Converte ``AAAA-MM-DD HH:MM:SS`` em datetime."""
    return [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in textos]


def test_sessoes_de_captura_cenario_real():
    """A pausa de 2 h 28 min de 17/10 separa tarde e noite; a de 1 h 20 min de 26/09 não."""
    momentos = _momentos(
        "2026-09-26 19:26:57", "2026-09-26 20:47:27", "2026-09-26 20:53:01",
        "2026-10-17 16:01:40", "2026-10-17 16:11:56",
        "2026-10-17 18:40:40", "2026-10-17 19:00:22",
    )
    assert dd.sessoes_de_captura(momentos, 2.0) == [
        "20260926_1926", "20260926_1926", "20260926_1926",
        "20261017_1601", "20261017_1601",
        "20261017_1840", "20261017_1840",
    ]


def test_sessao_atravessa_a_meia_noite():
    """10/10 23:59 e 11/10 00:01 são a mesma cena; uma pausa longa ainda separa."""
    momentos = _momentos("2026-10-10 23:59:23", "2026-10-11 00:01:22")
    assert dd.sessoes_de_captura(momentos, 2.0) == ["20261010_2359", "20261010_2359"]

    momentos = _momentos("2026-10-10 22:00:00", "2026-10-11 00:30:00")
    assert dd.sessoes_de_captura(momentos, 2.0) == ["20261010_2200", "20261011_0030"]


def test_pausa_igual_ao_limite_nao_separa():
    """Só uma pausa maior que o limite inicia uma nova sessão."""
    momentos = _momentos("2026-10-17 16:00:00", "2026-10-17 18:00:00", "2026-10-17 20:00:01")
    assert dd.sessoes_de_captura(momentos, 2.0) == [
        "20261017_1600", "20261017_1600", "20261017_2000",
    ]


def test_sessoes_de_captura_seguem_a_ordem_recebida():
    """A entrada fora de ordem é ordenada por dentro; a saída acompanha a entrada."""
    momentos = _momentos("2026-10-17 18:40:40", "2026-10-17 16:01:40")
    assert dd.sessoes_de_captura(momentos, 2.0) == ["20261017_1840", "20261017_1601"]
    assert dd.sessoes_de_captura([], 2.0) == []


# --------------------------------------------------------------------------- #
# Sorteio e divisão
# --------------------------------------------------------------------------- #


def _candidatos(tamanhos: dict[str, int]) -> list[dict]:
    """Candidatos sintéticos com os estratos e tamanhos informados."""
    return [
        {"caminho": f"{estrato}/{i}.jpg", "estrato": estrato}
        for estrato, n in tamanhos.items()
        for i in range(n)
    ]


def test_amostragem_reprodutivel_e_estratificada():
    """Mesma semente, mesmo sorteio; estratos pequenos se esgotam primeiro."""
    candidatos = _candidatos({"a": 10, "b": 10, "c": 2})

    primeiro = dd.amostrar_estratificado(candidatos, 9, "42:teste")
    segundo = dd.amostrar_estratificado(candidatos, 9, "42:teste")
    outro = dd.amostrar_estratificado(candidatos, 9, "7:teste")

    assert [c["caminho"] for c in primeiro] == [c["caminho"] for c in segundo]
    assert {c["caminho"] for c in primeiro} != {c["caminho"] for c in outro}
    por_estrato = pd.Series([c["estrato"] for c in primeiro]).value_counts()
    assert por_estrato["c"] == 2
    assert sorted([por_estrato["a"], por_estrato["b"]]) == [3, 4]


def test_amostragem_maior_que_o_disponivel_devolve_tudo():
    """Pedir mais do que existe não é erro."""
    candidatos = _candidatos({"a": 3})
    assert dd.amostrar_estratificado(candidatos, 10, "42") == candidatos


def test_split_por_hash_estavel_e_proporcional():
    """O split depende só do grupo; a distribuição segue as proporções."""
    proporcoes = (0.70, 0.15, 0.15)
    assert dd.split_por_hash("coco/1", 42, proporcoes) == dd.split_por_hash("coco/1", 42, proporcoes)

    splits = pd.Series([dd.split_por_hash(f"g/{i}", 42, proporcoes) for i in range(5000)])
    frequencias = splits.value_counts(normalize=True)
    for split, esperado in zip(SPLITS_SAIDA, proporcoes):
        assert frequencias[split] == pytest.approx(esperado, abs=0.03)


def test_split_por_hash_rejeita_proporcoes_invalidas():
    """Proporções que não somam 1 são erro de configuração."""
    with pytest.raises(ValueError, match="somar 1"):
        dd.split_por_hash("g", 42, (0.5, 0.2, 0.2))


@pytest.mark.parametrize(
    ("sessoes", "esperado"),
    [
        ([], {}),
        (["d1"], {"d1": "test"}),
        (["d2", "d1"], {"d1": "train", "d2": "test"}),
        (["d1", "d1", "d2"], {"d1": "train", "d2": "test"}),
        (
            ["d1", "d2", "d3", "d4", "d5"],
            {"d1": "train", "d2": "train", "d3": "train", "d4": "val", "d5": "test"},
        ),
    ],
)
def test_dividir_sessoes(sessoes, esperado):
    """As sessões mais recentes vão para o teste; validação só a partir de 3."""
    assert dd.dividir_sessoes(sessoes) == esperado


def test_dividir_sessoes_estratificado_cenario_real():
    """Cenário da estufa: 4 sessões só com alface; tarde e noite de 17/10 mistas.

    A noite de 10/10 vai até 00:49 de 11/10 e forma uma única sessão. A tarde
    de 17/10 vai para o treino e a noite para o teste — o modelo aprende "não
    alface" da câmera do projeto e ela também é medida.
    """
    so_alface = ["20260926_1926", "20260928_2154", "20260929_2112", "20261010_2354"]
    com_nao_alface = ["20261017_1601", "20261017_1840"]

    mapa = dd.dividir_sessoes_estratificado(so_alface + com_nao_alface, com_nao_alface)

    assert mapa == {
        "20260926_1926": "train",
        "20260928_2154": "train",
        "20260929_2112": "val",
        "20261010_2354": "test",
        "20261017_1601": "train",
        "20261017_1840": "test",
    }


def test_dividir_sessoes_estratificado_com_uma_sessao_de_nao_alface():
    """Com uma só sessão de "não alface", ela vai para o teste."""
    mapa = dd.dividir_sessoes_estratificado(["d1", "d2", "n1"], ["n1"])
    assert mapa == {"d1": "train", "d2": "test", "n1": "test"}


def test_dividir_sessoes_estratificado_independe_da_data_do_outro_tipo():
    """A ordem é cronológica dentro de cada tipo, não entre os tipos."""
    mapa = dd.dividir_sessoes_estratificado(
        ["20260901", "20260910", "20261020"], ["20260901", "20261020"]
    )
    # 20260901 é anterior à única sessão só de alface, mas vai para o treino do
    # seu próprio tipo; a sessão só de alface, sozinha, vai para o teste.
    assert mapa == {"20260901": "train", "20261020": "test", "20260910": "test"}


def test_data_corte_manda_sessoes_antigas_para_o_treino():
    """Até o corte, tudo no treino; só as sessões novas entram na divisão."""
    antigas = ["20260926_1926", "20261010_2354", "20261017_1601", "20261017_1840",
               "20261021_2350"]  # começa no dia do corte e atravessa a meia-noite
    novas = ["20261025_1500", "20261025_2200", "20261026_1500", "20261026_2200"]

    mapa = dd.dividir_sessoes_estratificado(
        antigas + novas, antigas[2:] + novas, data_corte="20261021"
    )

    assert {sessao: mapa[sessao] for sessao in antigas} == dict.fromkeys(antigas, "train")
    # 4 sessões novas com "não alface": 2 treino, 1 validação, 1 teste.
    assert {sessao: mapa[sessao] for sessao in novas} == {
        "20261025_1500": "train",
        "20261025_2200": "train",
        "20261026_1500": "val",
        "20261026_2200": "test",
    }


def test_data_corte_sem_sessoes_posteriores_deixa_tudo_no_treino():
    """Sem sessão depois do corte, nenhuma foto da câmera vai para validação ou teste."""
    mapa = dd.dividir_sessoes_estratificado(["20261010_2354", "20261017_1840"], ["20261017_1840"],
                                            data_corte="20261021")
    assert mapa == {"20261010_2354": "train", "20261017_1840": "train"}


@pytest.mark.parametrize("data_corte", ["2026-10-21", "20261321"])
def test_data_corte_invalida_falha(data_corte):
    """Um erro de digitação no corte não pode passar despercebido."""
    with pytest.raises(ValueError, match="AAAAMMDD"):
        dd.dividir_sessoes_estratificado(["20261017_1840"], [], data_corte=data_corte)


def test_pipeline_com_data_corte_padrao(fontes):
    """Com o corte padrão, as sessões da miniatura (até 17/10) vão todas para o treino."""
    resultado = dd.preparar_dominio(
        fontes["bruto"], fontes["limpo"], fontes["destino"], escrever=False
    )
    assert resultado["problemas"] == []
    assert set(resultado["mapa_sessoes"].values()) == {"train"}
    projeto = _filtrar(resultado["dividido"], camera=dd.CAMERA_PROJETO)
    assert set(projeto["split"]) == {"train"}
    assert any("data de corte" in aviso for aviso in resultado["avisos"])


def test_tarde_e_noite_do_mesmo_dia_ficam_em_splits_diferentes(artefatos):
    """Cada sessão de 17/10, com as duas classes, fica inteira num único split."""
    dia_misto = _filtrar(artefatos["dividido"], camera=dd.CAMERA_PROJETO, dia="20261017")
    for _, sessao in dia_misto.groupby("sessao"):
        assert set(sessao["classe"]) == set(ROTULOS_DOMINIO)
        assert sessao["split"].nunique() == 1
    assert dia_misto.groupby("sessao")["split"].first().to_dict() == {
        "20261017_1601": "train",
        "20261017_1840": "test",
    }


# --------------------------------------------------------------------------- #
# Verificações de integridade
# --------------------------------------------------------------------------- #


def _linha(**campos) -> dict:
    """Linha de inventário com valores padrão válidos."""
    padrao = {
        "caminho": "x.jpg",
        "fonte": dd.FONTE_COCO,
        "classe": CLASSE_NAO_ALFACE,
        "grupo": "coco/1",
        "camera": "",
        "split_fixo": None,
        "hash_md5": "h1",
        "split": "train",
    }
    return {**padrao, **campos}


def _base_valida() -> list[dict]:
    """Uma imagem de cada classe em cada split, sem repetições."""
    linhas = []
    for i, split in enumerate(SPLITS_SAIDA):
        for j, classe in enumerate(ROTULOS_DOMINIO):
            linhas.append(
                _linha(split=split, classe=classe, hash_md5=f"b{i}{j}", grupo=f"base/{i}{j}")
            )
    return linhas


def test_integridade_aprova_base_valida():
    """Sanidade do caso de referência usado nos testes abaixo."""
    assert dd.verificar_integridade_dominio(pd.DataFrame(_base_valida())) == []


def test_integridade_detecta_hash_em_dois_splits():
    """A mesma imagem em treino e teste é vazamento."""
    linhas = _base_valida() + [
        _linha(hash_md5="dup", grupo="g1", split="train"),
        _linha(hash_md5="dup", grupo="g2", split="test"),
    ]
    problemas = dd.verificar_integridade_dominio(pd.DataFrame(linhas))
    assert any("hashes MD5" in p for p in problemas)


def test_integridade_detecta_grupo_em_dois_splits():
    """Fotos da mesma folha, data ou sessão não podem se separar."""
    linhas = _base_valida() + [
        _linha(hash_md5="x1", grupo="deepweeds/20170101", split="train"),
        _linha(hash_md5="x2", grupo="deepweeds/20170101", split="val"),
    ]
    problemas = dd.verificar_integridade_dominio(pd.DataFrame(linhas))
    assert any("grupos" in p for p in problemas)


def test_integridade_detecta_kaggle_fora_do_split_e_celular_no_teste():
    """As regras por fonte também são verificadas."""
    linhas = _base_valida() + [
        _linha(fonte=dd.FONTE_KAGGLE, classe=CLASSE_ALFACE, hash_md5="k", grupo="k",
               split="train", split_fixo="test"),
        _linha(fonte=dd.FONTE_ESTUFA, classe=CLASSE_ALFACE, hash_md5="c", grupo="c",
               camera=dd.CAMERA_CELULAR, split="test"),
    ]
    problemas = dd.verificar_integridade_dominio(pd.DataFrame(linhas))
    assert any("Kaggle" in p for p in problemas)
    assert any("celular" in p for p in problemas)


def test_conflito_de_classe_detectado_antes_da_deduplicacao():
    """A mesma imagem rotulada como alface e como não alface é erro de rótulo."""
    linhas = [
        _linha(hash_md5="mesma", classe=CLASSE_ALFACE),
        _linha(hash_md5="mesma", classe=CLASSE_NAO_ALFACE),
    ]
    assert len(dd.conflitos_de_classe(pd.DataFrame(linhas))) == 1
    assert dd.conflitos_de_classe(pd.DataFrame(_base_valida())) == []
