# Avaliação do verificador de domínio — split `test`

*Gerado automaticamente por `desenvolvimento/scripts/08_avaliar_verificador.py` em 25/10/2026 13:00.*

| Parâmetro | Valor |
| --- | --- |
| Versão do verificador | v2 |
| Pesos do verificador | `yolo11n_cls_dominio_v2.pt` |
| Limiar de aceitação | 0.50 |
| Imagens avaliadas | 753 |
| Dispositivo | cpu |

> Medição **única** do split de teste. O limiar foi escolhido antes, no `val`. Nenhum ajuste pode ser feito com base nestes números.

## 1. Metas

| Critério | Medido | Meta | Situação |
| --- | --- | ---: | --- |
| Não suprimir doenças: alface aceita em folhas doentes | 178/184 (96.7%; IC95 93.1%–98.5%) | ≥ 99% | **não atingida** |
| Rejeitar imagens alheias: não alface recusada | 340/354 (96.0%; IC95 93.5%–97.6%) | ≥ 95% | atingida |
| Tempo de resposta: mediana das duas etapas (cpu) | 51.85 ms | < 100 ms | atingida |

Evidência do ambiente real (sem meta própria; amostra pequena — leia o IC95):

- Alface aceita — câmera do projeto (C270): 75/75 (100.0%; IC95 95.1%–100.0%)
- Não alface recusada — câmera do projeto (C270): 42/48 (87.5%; IC95 75.3%–94.1%)

## 2. Por grupo

| Classe | Grupo | Acertos / total | Taxa | IC95 |
| --- | --- | ---: | ---: | --- |
| alface | estufa/projeto | 75/75 | 100.0% | 95.1%–100.0% |
| alface | kaggle/bacteriano | 106/106 | 100.0% | 96.5%–100.0% |
| alface | kaggle/fungico | 72/78 | 92.3% | 84.2%–96.4% |
| alface | kaggle/saudavel | 140/140 | 100.0% | 97.3%–100.0% |
| nao_alface | coco | 67/67 | 100.0% | 94.6%–100.0% |
| nao_alface | deepweeds | 81/82 | 98.8% | 93.4%–99.8% |
| nao_alface | estufa/projeto | 42/48 | 87.5% | 75.3%–94.1% |
| nao_alface | plantdoc | 77/84 | 91.7% | 83.8%–95.9% |
| nao_alface | plantvillage | 73/73 | 100.0% | 95.0%–100.0% |

## 4. Cascata de ponta a ponta (teste oficial do classificador)

324 imagens; diagnóstico só para as aceitas pelo verificador. O classificador não foi executado de novo: usam-se as predições oficiais.

| Classe | Imagens | Recusadas pelo verificador | Revocação só classificador | Revocação com a cascata |
| --- | ---: | ---: | --- | --- |
| bacteriano | 106 | 0 | 103/106 (97.2%; IC95 92.0%–99.0%) | 103/106 (97.2%; IC95 92.0%–99.0%) |
| fungico | 78 | 6 | 66/78 (84.6%; IC95 75.0%–91.0%) | 63/78 (80.8%; IC95 70.7%–88.0%) |
| saudavel | 140 | 0 | 139/140 (99.3%; IC95 96.1%–99.9%) | 139/140 (99.3%; IC95 96.1%–99.9%) |

- Alertas do classificador: 167; com a cascata: 166.
- **Alertas verdadeiros suprimidos pelo verificador: 1.**
- Acurácia: só classificador 308/324 (95.1%; IC95 92.1%–96.9%); cascata 305/324 (94.1%; IC95 91.0%–96.2%).

## 5. Latência

Medida em 100 imagens do `val` (cpu), 10 de aquecimento descartadas; soma dos tempos de inferência, sem decodificação — mesmo critério do `05_benchmark_latencia.py`.

| Etapas | Mediana | Média | p95 | p99 | Mín. | Máx. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Verificador + classificador | 51.85 | 61.91 | 122.2 | 153.38 | 41.99 | 157.98 |
| Só verificador | 25.42 | 30.9 | 61.04 | 73.6 | 20.99 | 85.21 |
