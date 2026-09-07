# Imagem de produção da API de inferência.
#
# Empacota o mínimo para servir o classificador: a biblioteca `src/ia_alface/`,
# a camada HTTP `api/` e os pesos treinados. O ciclo de pesquisa
# (`desenvolvimento/`, com o dataset de ~966 MB) fica de fora pelo .dockerignore.
#
# Construir e testar localmente, a partir da raiz do projeto:
#     docker build -t ia-alface:local .
#     docker run --rm -p 8000:8000 ia-alface:local
#     curl http://localhost:8000/health
#
# Publicar no GitHub Container Registry:
#     docker build -t ghcr.io/<usuario>/ia-alface:v1 .
#     docker push ghcr.io/<usuario>/ia-alface:v1
#
# Implantação em Azure Container Apps: ver docs/IMPLANTACAO.md.

FROM python:3.13-slim

# Sem PYTHONUNBUFFERED o stdout do uvicorn fica retido no buffer e os logs só
# aparecem no painel do Container Apps aos blocos, atrapalhando o diagnóstico.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# O Ultralytics depende do opencv-python, que exige bibliotecas de vídeo do
# sistema mesmo em execução sem interface gráfica. Sem elas o import falha em
# tempo de execução com `libGL.so.1: cannot open shared object file` — erro que
# só aparece quando o container sobe, não durante a build.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch a partir do índice CPU-only. O pacote equivalente no PyPI embute as
# bibliotecas CUDA — mais de 2 GB que este serviço jamais usaria, já que roda em
# CPU. Instalado ANTES do requirements.txt para que a resolução seguinte o
# encontre satisfeito e não o troque pela variante com CUDA.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.13.0" torchvision

# Dependências em camada própria, antes do código: enquanto o requirements.txt
# não mudar, alterações em `api/` ou `src/` reaproveitam esta camada e a build
# leva segundos em vez de reinstalar o PyTorch inteiro.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Apenas o que a API importa em tempo de execução.
COPY src/ ./src/
COPY api/ ./api/
COPY modelos/yolo11n_cls_alface.pt   ./modelos/
COPY modelos/yolo11n_cls_alface.onnx ./modelos/

# Padrão: pesos .pt, que mantêm o Grad-CAM de `/analyze?heatmap=true` operante
# (o mapa de calor não funciona com modelo exportado). Para trocar pelo ONNX —
# mediana de latência menor, sem mapa de calor — basta definir a variável na
# implantação, sem reconstruir a imagem:
#     IA_ALFACE_MODELO=/app/modelos/yolo11n_cls_alface.onnx
ENV IA_ALFACE_MODELO=/app/modelos/yolo11n_cls_alface.pt

# O Ultralytics grava um settings.json próprio ao carregar. Sem esta variável
# ele tenta a home do usuário sem privilégios, não consegue, cai para /tmp e
# emite um aviso a cada inicialização — ruído que atrapalha o diagnóstico no
# painel do Container Apps. Apontar para dentro de /app torna o destino
# determinístico; o diretório é criado logo abaixo, já com a dona correta.
ENV YOLO_CONFIG_DIR=/app/.ultralytics

# ATENÇÃO: o padrão de IA_ALFACE_CORS_ORIGENS é ["*"], adequado a
# desenvolvimento. Defina as origens do dashboard na implantação.

# Processo sem privilégios: um comprometimento da aplicação não herda root.
RUN useradd --create-home --uid 10001 alface \
    && mkdir -p /app/.ultralytics \
    && chown -R alface:alface /app
USER alface

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
