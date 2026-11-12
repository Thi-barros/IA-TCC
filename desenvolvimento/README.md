# Desenvolvimento 

Esta pasta agrupa tudo que pertence ao ciclo de pesquisa do módulo de IA:
preparar o dataset, treinar, avaliar, exportar e documentar. Nada aqui é
necessário para **servir** o modelo.

O critério é simples: o servidor de inferência precisa apenas de `api/`,
`src/ia_alface/`, `modelos/` e `requirements.txt`. Se um arquivo não é lido em
tempo de execução pela API, ele está aqui.

---

## Conteúdo

| Caminho | O que é | Versionado no Git |
| --- | --- | :---: |
| `scripts/` | Pipeline 01–05 e utilitários | sim |
| `tests/` | 101 testes automatizados | sim |
| `reports/` | Métricas, figuras e relatórios do TCC | sim |
| `requirements-dev.txt` | Dependências extras de desenvolvimento | sim |
| `notebooks/` | Treinamento em GPU no Google Colab | **não** |
| `tcc/` | Texto da monografia e material de apoio | **não** |
| `dataset/` | Conjunto original do Kaggle (~966 MB) | **não** |
| `dataset_clean/` | Conjunto preparado (~217 MB) | **não** |
| `dataset_clean.zip` | Pacote para upload no Colab | **não** |
| `runs/` | Execuções do Ultralytics | **não** |
| `pesos_base/` | Pesos pré-treinados baixados | **não** |

`dataset_clean/`, `runs/`, `dataset_clean.zip` e `pesos_base/` são
**regeneráveis**: saem dos scripts ou são baixados automaticamente pelo
Ultralytics. `dataset/` vem do Kaggle (ver `../docs/README.md`). Pode apagá-los
para liberar espaço — cerca de 1,2 GB — e recriá-los quando precisar.

`notebooks/` e `tcc/`, por outro lado, **não são regeneráveis e não estão no
Git**. Mantenha um backup próprio deles:

- `notebooks/treino_colab.ipynb` — necessário para treinar variantes em GPU
- `tcc/` — o texto da monografia e o material de apoio

---

## Instalação

A partir da **raiz do projeto**:

```bash
pip install -r desenvolvimento/requirements-dev.txt
```

Isso instala as dependências de produção mais `pandas`, `scikit-learn`, `onnx` e
`pytest`, usadas apenas por `dados.py`, `treino.py`, `avaliacao.py` e
`exportacao.py`.

---

## Fluxo de trabalho

Todos os comandos são executados a partir da **raiz do projeto**, não de dentro
desta pasta.

```bash
python desenvolvimento/scripts/01_preparar_dataset.py
```

```bash
python desenvolvimento/scripts/02_treinar.py --modelo yolo11n-cls.pt
```

```bash
python desenvolvimento/scripts/03_avaliar.py --pesos modelos/yolo11n_cls_alface.pt
```

```bash
python desenvolvimento/scripts/04_exportar.py
```

```bash
python desenvolvimento/scripts/05_benchmark_latencia.py
```

```bash
python -m pytest -q
```

### Utilitários

```bash
python desenvolvimento/scripts/testar_imagem.py foto.jpg --heatmap
```

Classifica imagens carregando o modelo diretamente, sem passar pela API.

```bash
python desenvolvimento/scripts/preparar_zip_colab.py
```

Empacota `dataset_clean/` para upload no Colab, reduzindo o menor lado das
imagens para 320 px (de ~220 MB para ~86 MB).

---

## Onde os artefatos são gravados

Os caminhos são definidos em `src/ia_alface/config.py` e resolvidos a partir da
raiz do projeto:

| Artefato | Destino |
| --- | --- |
| Dataset preparado | `desenvolvimento/dataset_clean/` |
| Relatórios e figuras | `desenvolvimento/reports/` |
| Execuções de treino | `desenvolvimento/runs/` |
| Pesos finais | `modelos/` (raiz — é produção) |

Os pesos são a única saída que atravessa a fronteira: o treinamento acontece
aqui, mas o modelo resultante é artefato de produção.

---

## O que preservar

Se for limpar a pasta, tenha em mente:

- **`modelos/yolo11n_cls_alface.pt`** (na raiz) não está no Git e não é
  reproduzível bit a bit. Mantenha um backup fora do projeto.
- **`reports/`** contém as métricas que sustentam as Seções 7.1 e 7.2 do TCC.
  Está versionado, mas confirme antes de apagar.
- **`tcc/`** guarda o texto da monografia.

O resto se refaz com os comandos acima.
