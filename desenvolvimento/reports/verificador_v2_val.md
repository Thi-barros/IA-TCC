# Avaliação do verificador de domínio — split `val`

*Gerado automaticamente por `desenvolvimento/scripts/08_avaliar_verificador.py` em 25/10/2026 10:30.*

| Parâmetro | Valor |
| --- | --- |
| Versão do verificador | v2 |
| Pesos do verificador | `yolo11n_cls_dominio_v2.pt` |
| Limiar de aceitação | 0.50 |
| Imagens avaliadas | 620 |
| Dispositivo | cpu |

> Split de **validação**: use a curva da Seção 3 para escolher o limiar. Estes números não são o resultado oficial.

## 1. Metas

| Critério | Medido | Meta | Situação |
| --- | --- | ---: | --- |
| Não suprimir doenças: alface aceita em folhas doentes | 189/189 (100.0%; IC95 98.0%–100.0%) | ≥ 99% | atingida |
| Rejeitar imagens alheias: não alface recusada | 294/294 (100.0%; IC95 98.7%–100.0%) | ≥ 95% | atingida |
| Tempo de resposta: mediana das duas etapas (cpu) | 42.70 ms | < 100 ms | atingida |

Evidência do ambiente real (sem meta própria; amostra pequena — leia o IC95):

- Alface aceita — câmera do projeto (C270): não medido (0 imagens)
- Não alface recusada — câmera do projeto (C270): não medido (0 imagens)

## 2. Por grupo

| Classe | Grupo | Acertos / total | Taxa | IC95 |
| --- | --- | ---: | ---: | --- |
| alface | kaggle/bacteriano | 114/114 | 100.0% | 96.7%–100.0% |
| alface | kaggle/fungico | 75/75 | 100.0% | 95.1%–100.0% |
| alface | kaggle/saudavel | 136/137 | 99.3% | 96.0%–99.9% |
| nao_alface | coco | 66/66 | 100.0% | 94.5%–100.0% |
| nao_alface | deepweeds | 102/102 | 100.0% | 96.4%–100.0% |
| nao_alface | plantdoc | 62/62 | 100.0% | 94.2%–100.0% |
| nao_alface | plantvillage | 64/64 | 100.0% | 94.3%–100.0% |

## 3. Curva de limiar (somente validação)

Recusar uma folha doente suprime um alerta; aceitar uma imagem alheia custa no máximo um alerta indevido. Prefira o maior limiar que mantém a primeira coluna na meta.

| Limiar | Alface aceita — folhas doentes (Kaggle) | Alface aceita — todas | Alface aceita — câmera do projeto (C270) | Não alface recusada — todas | Não alface recusada — câmera do projeto (C270) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 100.0% | 100.0% | — | 98.0% | — |
| 0.10 | 100.0% | 100.0% | — | 98.3% | — |
| 0.15 | 100.0% | 99.7% | — | 98.6% | — |
| 0.20 | 100.0% | 99.7% | — | 99.0% | — |
| 0.25 | 100.0% | 99.7% | — | 99.3% | — |
| 0.30 | 100.0% | 99.7% | — | 99.3% | — |
| 0.35 | 100.0% | 99.7% | — | 100.0% | — |
| 0.40 | 100.0% | 99.7% | — | 100.0% | — |
| 0.45 | 100.0% | 99.7% | — | 100.0% | — |
| 0.50 | 100.0% | 99.7% | — | 100.0% | — |
| 0.55 | 100.0% | 99.7% | — | 100.0% | — |
| 0.60 | 100.0% | 99.7% | — | 100.0% | — |
| 0.65 | 100.0% | 99.7% | — | 100.0% | — |
| 0.70 | 100.0% | 99.7% | — | 100.0% | — |
| 0.75 | 99.5% | 99.1% | — | 100.0% | — |
| 0.80 | 99.5% | 99.1% | — | 100.0% | — |
| 0.85 | 98.9% | 98.8% | — | 100.0% | — |
| 0.90 | 97.4% | 97.5% | — | 100.0% | — |
| 0.95 | 96.8% | 96.9% | — | 100.0% | — |

## 5. Latência

Medida em 100 imagens do `val` (cpu), 10 de aquecimento descartadas; soma dos tempos de inferência, sem decodificação — mesmo critério do `05_benchmark_latencia.py`.

| Etapas | Mediana | Média | p95 | p99 | Mín. | Máx. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Verificador + classificador | 42.7 | 53.45 | 117.19 | 137.7 | 35.05 | 180.12 |
| Só verificador | 21.95 | 26.68 | 58.08 | 66.62 | 17.75 | 116.41 |
