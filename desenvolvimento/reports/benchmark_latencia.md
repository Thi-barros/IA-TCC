# Benchmark de latência de inferência

*Gerado por `desenvolvimento/scripts/05_benchmark_latencia.py` em 02/08/2026 18:14.*

| Parâmetro | Valor |
| --- | --- |
| Imagens medidas | 100 (split `test` de `dataset_clean`) |
| Inferências de aquecimento | 10 (descartadas) |
| Caminho medido | `DetectorDoencasAlface.predizer` — 1 imagem por chamada |
| CPU | AMD64 Family 23 Model 24 Stepping 1, AuthenticAMD |
| Sistema operacional | Windows-11-10.0.26200-SP0 |
| Python / torch / ultralytics | 3.13.5 / 2.13.0+cpu / 8.4.95 |
| CUDA disponível | False |

## Resultados

| Configuração | Mediana | Média | p95 | p99 | Mínimo | Máximo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch / CPU | 20.10 ms | 23.23 ms | 38.27 ms | 81.34 ms | 17.07 ms | 91.20 ms |
| ONNX / CPU | 12.73 ms | 16.15 ms | 32.69 ms | 78.98 ms | 9.08 ms | 85.37 ms |

## Conclusão sobre o requisito RNF01

RNF01 e o caso de teste CT04 exigem inferência em menos de **100 ms** por imagem no servidor.

| Configuração | Mediana medida | Meta | Status |
| --- | ---: | ---: | --- |
| PyTorch / CPU | 20.10 ms | < 100 ms | **APROVADO** |
| ONNX / CPU | 12.73 ms | < 100 ms | **APROVADO** |

A configuração mais rápida medida é **ONNX / CPU**, com mediana de **12.73 ms** no hardware descrito acima.

## Observação metodológica

Estes números foram medidos no servidor de inferência (CPU/GPU do host que executou o script), **não** em dispositivo embarcado. A Tabela 16 do TCC reúne latências de literatura em ESP32 e em modelos de borda quantizados; os dois contextos não devem ser comparados diretamente. Na arquitetura proposta (Seção 6.2.6.3 do TCC), o ESP32-CAM apenas captura e transmite a imagem via HTTP; a inferência ocorre no servidor.
