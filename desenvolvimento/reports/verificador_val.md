# Avaliação do verificador de domínio — split `val`

*Gerado automaticamente por `desenvolvimento/scripts/08_avaliar_verificador.py` em 18/10/2026 17:15.*

| Parâmetro | Valor |
| --- | --- |
| Pesos do verificador | `yolo11n_cls_dominio.pt` |
| Limiar de aceitação | 0.50 |
| Imagens avaliadas | 622 |
| Dispositivo | cpu |

> Split de **validação**: use a curva da Seção 3 para escolher o limiar. Estes números não são o resultado oficial.

## 1. Metas

| Critério | Medido | Meta | Situação |
| --- | --- | ---: | --- |
| Não suprimir doenças: alface aceita em folhas doentes | 189/189 (100.0%; IC95 98.0%–100.0%) | ≥ 99% | atingida |
| Rejeitar imagens alheias: não alface recusada | 293/294 (99.7%; IC95 98.1%–99.9%) | ≥ 95% | atingida |
| Tempo de resposta: mediana das duas etapas (cpu) | 49.63 ms | < 100 ms | atingida |

Evidência do ambiente real (sem meta própria; amostra pequena — leia o IC95):

- Alface aceita — câmera do projeto (C270): 2/2 (100.0%; IC95 34.2%–100.0%)
- Não alface recusada — câmera do projeto (C270): não medido (0 imagens)

## 2. Por grupo

| Classe | Grupo | Acertos / total | Taxa | IC95 |
| --- | --- | ---: | ---: | --- |
| alface | estufa/projeto | 2/2 | 100.0% | 34.2%–100.0% |
| alface | kaggle/bacteriano | 114/114 | 100.0% | 96.7%–100.0% |
| alface | kaggle/fungico | 75/75 | 100.0% | 95.1%–100.0% |
| alface | kaggle/saudavel | 136/137 | 99.3% | 96.0%–99.9% |
| nao_alface | coco | 66/66 | 100.0% | 94.5%–100.0% |
| nao_alface | deepweeds | 102/102 | 100.0% | 96.4%–100.0% |
| nao_alface | plantdoc | 61/62 | 98.4% | 91.4%–99.7% |
| nao_alface | plantvillage | 64/64 | 100.0% | 94.3%–100.0% |

## 3. Curva de limiar (somente validação)

Recusar uma folha doente suprime um alerta; aceitar uma imagem alheia custa no máximo um alerta indevido. Prefira o maior limiar que mantém a primeira coluna na meta.

| Limiar | Alface aceita — folhas doentes (Kaggle) | Alface aceita — todas | Alface aceita — câmera do projeto (C270) | Não alface recusada — todas | Não alface recusada — câmera do projeto (C270) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 100.0% | 100.0% | 100.0% | 93.2% | — |
| 0.10 | 100.0% | 100.0% | 100.0% | 95.6% | — |
| 0.15 | 100.0% | 100.0% | 100.0% | 96.6% | — |
| 0.20 | 100.0% | 100.0% | 100.0% | 98.3% | — |
| 0.25 | 100.0% | 100.0% | 100.0% | 98.6% | — |
| 0.30 | 100.0% | 100.0% | 100.0% | 99.0% | — |
| 0.35 | 100.0% | 100.0% | 100.0% | 99.0% | — |
| 0.40 | 100.0% | 100.0% | 100.0% | 99.3% | — |
| 0.45 | 100.0% | 99.7% | 100.0% | 99.3% | — |
| 0.50 | 100.0% | 99.7% | 100.0% | 99.7% | — |
| 0.55 | 99.5% | 98.8% | 100.0% | 99.7% | — |
| 0.60 | 99.5% | 98.2% | 100.0% | 99.7% | — |
| 0.65 | 98.4% | 97.6% | 100.0% | 99.7% | — |
| 0.70 | 97.9% | 97.0% | 100.0% | 99.7% | — |
| 0.75 | 97.4% | 95.7% | 100.0% | 99.7% | — |
| 0.80 | 94.7% | 94.2% | 100.0% | 99.7% | — |
| 0.85 | 93.1% | 91.2% | 100.0% | 99.7% | — |
| 0.90 | 89.4% | 87.8% | 100.0% | 99.7% | — |
| 0.95 | 78.8% | 79.0% | 100.0% | 100.0% | — |

## 5. Latência

Medida em 100 imagens do `val` (cpu), 10 de aquecimento descartadas; soma dos tempos de inferência, sem decodificação — mesmo critério do `05_benchmark_latencia.py`.

| Etapas | Mediana | Média | p95 | p99 | Mín. | Máx. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Verificador + classificador | 49.63 | 62.66 | 139.72 | 181.07 | 41.97 | 191.06 |
| Só verificador | 24.8 | 31.11 | 72.49 | 80.95 | 20.88 | 105.9 |
