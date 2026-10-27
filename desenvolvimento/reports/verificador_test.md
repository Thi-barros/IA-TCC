# Avaliação do verificador de domínio — split `test`

*Gerado automaticamente por `desenvolvimento/scripts/08_avaliar_verificador.py` em 18/10/2026 17:33.*

| Parâmetro | Valor |
| --- | --- |
| Pesos do verificador | `yolo11n_cls_dominio.pt` |
| Limiar de aceitação | 0.50 |
| Imagens avaliadas | 710 |
| Dispositivo | cpu |

> Medição **única** do split de teste. O limiar foi escolhido antes, no `val`. Nenhum ajuste pode ser feito com base nestes números.

## 1. Metas

| Critério | Medido | Meta | Situação |
| --- | --- | ---: | --- |
| Não suprimir doenças: alface aceita em folhas doentes | 181/184 (98.4%; IC95 95.3%–99.4%) | ≥ 99% | **não atingida** |
| Rejeitar imagens alheias: não alface recusada | 338/356 (94.9%; IC95 92.2%–96.8%) | ≥ 95% | **não atingida** |
| Tempo de resposta: mediana das duas etapas (cpu) | 53.25 ms | < 100 ms | atingida |

Evidência do ambiente real (sem meta própria; amostra pequena — leia o IC95):

- Alface aceita — câmera do projeto (C270): 12/30 (40.0%; IC95 24.6%–57.7%)
- Não alface recusada — câmera do projeto (C270): 45/50 (90.0%; IC95 78.6%–95.7%)

## 2. Por grupo

| Classe | Grupo | Acertos / total | Taxa | IC95 |
| --- | --- | ---: | ---: | --- |
| alface | estufa/projeto | 12/30 | 40.0% | 24.6%–57.7% |
| alface | kaggle/bacteriano | 105/106 | 99.1% | 94.8%–99.8% |
| alface | kaggle/fungico | 76/78 | 97.4% | 91.1%–99.3% |
| alface | kaggle/saudavel | 139/140 | 99.3% | 96.1%–99.9% |
| nao_alface | coco | 67/67 | 100.0% | 94.6%–100.0% |
| nao_alface | deepweeds | 81/82 | 98.8% | 93.4%–99.8% |
| nao_alface | estufa/projeto | 45/50 | 90.0% | 78.6%–95.7% |
| nao_alface | plantdoc | 72/84 | 85.7% | 76.7%–91.6% |
| nao_alface | plantvillage | 73/73 | 100.0% | 95.0%–100.0% |

## 4. Cascata de ponta a ponta (teste oficial do classificador)

324 imagens; diagnóstico só para as aceitas pelo verificador. O classificador não foi executado de novo: usam-se as predições oficiais.

| Classe | Imagens | Recusadas pelo verificador | Revocação só classificador | Revocação com a cascata |
| --- | ---: | ---: | --- | --- |
| bacteriano | 106 | 1 | 103/106 (97.2%; IC95 92.0%–99.0%) | 102/106 (96.2%; IC95 90.7%–98.5%) |
| fungico | 78 | 2 | 66/78 (84.6%; IC95 75.0%–91.0%) | 64/78 (82.0%; IC95 72.1%–89.0%) |
| saudavel | 140 | 1 | 139/140 (99.3%; IC95 96.1%–99.9%) | 138/140 (98.6%; IC95 94.9%–99.6%) |

- Alertas do classificador: 167; com a cascata: 164.
- **Alertas verdadeiros suprimidos pelo verificador: 3.**
- Acurácia: só classificador 308/324 (95.1%; IC95 92.1%–96.9%); cascata 304/324 (93.8%; IC95 90.7%–96.0%).

## 5. Latência

Medida em 100 imagens do `val` (cpu), 10 de aquecimento descartadas; soma dos tempos de inferência, sem decodificação — mesmo critério do `05_benchmark_latencia.py`.

| Etapas | Mediana | Média | p95 | p99 | Mín. | Máx. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Verificador + classificador | 53.25 | 101.96 | 324.16 | 385.86 | 40.34 | 468.99 |
| Só verificador | 26.19 | 52.76 | 171.27 | 204.0 | 20.73 | 230.82 |
