# Ambiente de execução

Registro do ambiente em que o módulo de IA foi desenvolvido e verificado
(requisito RNF07 — rastreabilidade e reprodutibilidade).

## Versões verificadas

| Componente | Versão |
| --- | --- |
| Python | 3.13.5 |
| PyTorch | 2.13.0+cpu |
| Ultralytics | 8.4.95 |
| NumPy | 2.1.3 |
| pandas | 2.2.3 |
| scikit-learn | 1.6.1 |
| matplotlib | 3.10.0 |
| Pillow | 11.1.0 |
| ONNX Runtime | 1.28.0 |

## Hardware e sistema operacional

| Item | Valor |
| --- | --- |
| Sistema operacional | Windows-11-10.0.26200-SP0 |
| Processador | AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx (AMD64 Family 23 Model 24 Stepping 1) |
| CUDA disponível | **Não** — a instalação do PyTorch é `+cpu` |
| GPU usada no treino | Nenhuma |

Para regenerar esta tabela a qualquer momento:

```bash
python -c "import sys; sys.path.insert(0,'src'); from ia_alface.config import descrever_ambiente; print(descrever_ambiente())"
```

## Consequências práticas da ausência de GPU

O código detecta o dispositivo em tempo de execução (`torch.cuda.is_available()`)
e funciona nos dois cenários, mas o desempenho de **treinamento** difere muito.

Medição real nesta máquina, `yolo11n-cls` com `imgsz=224` e `batch=16` sobre 1.566
imagens de treino e 326 de validação:

| Métrica | Valor medido |
| --- | --- |
| Duração de 1 época (treino + validação) | ~3,8 min |
| Projeção para 100 épocas | **~6 horas** |

O *early stopping* (`patience=20`) pode encerrar antes, mas a ordem de grandeza é
de horas. Duas opções:

1. **Treinar localmente em CPU.** Funciona; deixe rodando. Use
   `--epocas 100 --batch 32` conforme a configuração padrão do spec.
2. **Treinar no Google Colab com GPU.** Use o notebook
   `desenvolvimento/notebooks/treino_colab.ipynb` e traga o arquivo `best.pt` de
   volta para `modelos/`. A avaliação, a exportação e o benchmark podem então
   rodar localmente em minutos.

   > O notebook não é versionado neste repositório — ver
   > [`../desenvolvimento/README.md`](../desenvolvimento/README.md).

## Ajustes obrigatórios no Windows

Configurados por padrão em `src/ia_alface/config.py`; não altere sem motivo:

| Parâmetro | Valor | Razão |
| --- | --- | --- |
| `workers` | `0` | O *multiprocessing* do DataLoader falha no Windows dentro do fluxo do Ultralytics |
| `amp` | `False` em CPU | *Mixed precision* só faz sentido em GPU |
| `deterministic` | `True` | Reprodutibilidade entre execuções |
| `seed` | `42` | Reprodutibilidade |

## Desempenho de inferência medido

Latência de `DetectorDoencasAlface.predizer` (uma imagem por chamada, o caminho
real de produção), 40 imagens do split de teste com 10 de aquecimento, neste
hardware:

| Configuração | Mediana | p95 |
| --- | ---: | ---: |
| PyTorch / CPU | ~35 ms | ~48 ms |
| ONNX Runtime / CPU | ~17 ms | ~32 ms |

Ambos atendem ao requisito RNF01 (< 100 ms). Os números atualizados após cada
execução ficam em [`../desenvolvimento/reports/benchmark_latencia.md`](../desenvolvimento/reports/benchmark_latencia.md).

> Estes valores referem-se ao **servidor de inferência**. A Tabela 16 do TCC
> compara latências em dispositivos de borda (ESP32); são contextos distintos.
> Na arquitetura proposta, o ESP32-CAM apenas captura e transmite a imagem.
