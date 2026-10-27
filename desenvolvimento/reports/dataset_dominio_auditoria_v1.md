# Auditoria do dataset do verificador de domínio

*Gerado automaticamente por `desenvolvimento/scripts/06_preparar_dataset_dominio.py` em 17/10/2026 19:56.*

| Parâmetro | Valor |
| --- | --- |
| Positivas | `dataset_clean/` (split mantido) e fotos da estufa |
| Fontes brutas | `dataset_dominio_bruto/` (somente leitura) |
| Dataset gerado | `dataset_dominio/` |
| Máximo por fonte pública | 500 |
| Proporções (fontes públicas) | 70% / 15% / 15% |
| Sessões da câmera do projeto | fotos em sequência até uma pausa de mais de 2 h; 25% mais recentes no teste, 25% seguintes na validação — separadamente para as sessões com "não alface" e as só com alface |
| Semente | 42 |
| Python | 3.13.5 |

Decisões, licenças e procedência: `docs/VERIFICADOR_DOMINIO.md`.

## 1. Fontes

| Fonte | Classe | Disponíveis | Sorteadas | Ilegíveis | Duplicatas removidas | No dataset |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| kaggle | alface | 2216 | 2216 | 0 | 0 | 2216 |
| estufa | ambas | 202 | 202 | 0 | 1 | 201 |
| plantvillage | nao_alface | 7078 | 500 | 0 | 0 | 500 |
| plantdoc | nao_alface | 2578 | 500 | 0 | 0 | 500 |
| deepweeds | nao_alface | 17509 | 500 | 0 | 0 | 500 |
| coco | nao_alface | 3859 | 500 | 0 | 0 | 500 |

## 2. Detalhes por fonte

- **kaggle**: imagens = 2216
- **estufa**: imagens = 202, pastas = 2, sessoes_projeto = 6
- **plantvillage**: imagens = 54305, sem_mapa = 13194, ambiguas = 4370, folhas = 7078
- **plantdoc**: imagens = 2578
- **deepweeds**: imagens = 17509, dias = 60
- **coco**: imagens = 5000, com_categoria_excluida = 1093, sem_anotacao = 48, arquivo_ausente = 0, utilizaveis = 3859

PlantVillage: uma imagem por folha; fotos sem mapeamento no `leaf-map.json` ou associadas a mais de uma folha são descartadas. COCO: excluídas as imagens com comida, vaso de planta, mesa de refeição ou tigela, e as sem anotação.

## 3. Distribuição final

| split | alface | nao_alface | Total |
| --- | --- | --- | --- |
| train | 1628 | 1457 | 3085 |
| val | 328 | 294 | 622 |
| test | 354 | 356 | 710 |

| split | kaggle | estufa | plantvillage | plantdoc | deepweeds | coco | Total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1566 | 119 | 363 | 354 | 316 | 367 | 3085 |
| val | 326 | 2 | 64 | 62 | 102 | 66 | 622 |
| test | 324 | 80 | 73 | 84 | 82 | 67 | 710 |

## 4. Fotos da estufa

| camera | dia | sessao | classe | split | imagens |
| --- | --- | --- | --- | --- | --- |
| celular | 20261015 | 20261015 | alface | train | 40 |
| projeto | 20260926 | 20260926_1926 | alface | train | 3 |
| projeto | 20260928 | 20260928_2154 | alface | train | 2 |
| projeto | 20260929 | 20260929_2112 | alface | val | 2 |
| projeto | 20261010 | 20261010_2354 | alface | test | 3 |
| projeto | 20261011 | 20261010_2354 | alface | test | 7 |
| projeto | 20261017 | 20261017_1601 | alface | train | 17 |
| projeto | 20261017 | 20261017_1601 | nao_alface | train | 57 |
| projeto | 20261017 | 20261017_1840 | alface | test | 20 |
| projeto | 20261017 | 20261017_1840 | nao_alface | test | 50 |

Fotos da câmera do projeto (C270) por split:

| split | alface | nao_alface | Total |
| --- | --- | --- | --- |
| train | 22 | 57 | 79 |
| val | 2 | 0 | 2 |
| test | 30 | 50 | 80 |

Câmera do projeto (C270): sessões de captura — fotos em sequência até uma pausa de mais de 2 h, podendo atravessar a meia-noite, identificadas por `AAAAMMDD_HHMM` da primeira foto — em ordem cronológica, as mais recentes no teste, aplicada separadamente às sessões com "não alface" e às só com alface; cada sessão fica inteira num único split. Celular: somente treino — não é a câmera de operação.

## 5. Verificação de integridade

**APROVADO** — zero hashes MD5 e zero grupos compartilhados entre `train`, `val` e `test`; nenhuma imagem nas duas classes; Kaggle no split original; nenhuma foto de celular no teste; as duas classes presentes em todos os splits.

## 6. Avisos

- Nenhum aviso.

## 7. Limitações desta montagem

- A deduplicação é por MD5: cópias **aproximadas** (mesma foto em outro tamanho, comum em imagens da web como as do PlantDoc) não são detectadas.
- As fontes públicas negativas têm fundos e enquadramentos próprios; o desempenho no ambiente real só é evidenciado pelas fotos da câmera do projeto.
