"""Configuração central: caminhos do projeto e hiperparâmetros.

Todos os caminhos são resolvidos em relação à raiz do projeto — nenhum caminho
absoluto é codificado no fonte (spec.md, Seção 5).

Referência: spec.md, Seções 3, 5, 6 e 10.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "RAIZ_PROJETO",
    "Caminhos",
    "ConfigTreino",
    "CAMINHOS",
    "detectar_dispositivo",
    "descrever_ambiente",
]

# src/ia_alface/config.py -> src/ia_alface -> src -> raiz do projeto
RAIZ_PROJETO: Path = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Caminhos:
    """Caminhos canônicos do projeto, todos derivados da raiz.

    A raiz contém apenas o que vai para produção (``api/``, ``src/``,
    ``modelos/``, ``docs/``). Tudo que pertence ao ciclo de pesquisa — dataset,
    relatórios, execuções de treino — fica sob ``desenvolvimento/`` e não é
    implantado. Ver ``desenvolvimento/README.md``.
    """

    raiz: Path = RAIZ_PROJETO

    # ---------------------------------------------------------------- #
    # Produção — presentes no ambiente implantado
    # ---------------------------------------------------------------- #

    @property
    def modelos(self) -> Path:
        """Pesos treinados e modelos exportados. Necessário em produção."""
        return self.raiz / "modelos"

    @property
    def docs(self) -> Path:
        """Documentação do projeto."""
        return self.raiz / "docs"

    # ---------------------------------------------------------------- #
    # Desenvolvimento — ausentes no ambiente implantado
    # ---------------------------------------------------------------- #

    @property
    def desenvolvimento(self) -> Path:
        """Raiz dos artefatos que não vão para produção."""
        return self.raiz / "desenvolvimento"

    @property
    def dataset_original(self) -> Path:
        """Dataset de origem. **Imutável** — nunca escrever aqui."""
        return self.desenvolvimento / "dataset"

    @property
    def dataset_limpo(self) -> Path:
        """Dataset normalizado e re-dividido, gerado por ``01_preparar_dataset.py``."""
        return self.desenvolvimento / "dataset_clean"

    @property
    def relatorios(self) -> Path:
        """Saídas destinadas ao texto do TCC."""
        return self.desenvolvimento / "reports"

    @property
    def runs(self) -> Path:
        """Diretório de execuções do Ultralytics."""
        return self.desenvolvimento / "runs"

    def garantir_diretorios(self) -> None:
        """Cria os diretórios de saída, se ainda não existirem."""
        for caminho in (self.modelos, self.relatorios, self.docs):
            caminho.mkdir(parents=True, exist_ok=True)


CAMINHOS = Caminhos()


def detectar_dispositivo(preferencia: str | None = None) -> str:
    """Seleciona o dispositivo de execução do PyTorch.

    Nunca codifica o dispositivo: usa CUDA quando disponível e CPU caso
    contrário (spec.md, Seção 3).

    Args:
        preferencia: Se informado (``"cpu"`` ou ``"cuda"``), tenta respeitar.
            Um pedido de ``"cuda"`` sem GPU disponível cai para CPU com aviso.

    Returns:
        ``"cuda"`` ou ``"cpu"``.
    """
    import torch

    cuda_disponivel = torch.cuda.is_available()

    if preferencia is not None:
        preferencia = preferencia.strip().lower()
        if preferencia.startswith("cuda") and not cuda_disponivel:
            logger.warning(
                "Dispositivo 'cuda' solicitado, mas CUDA não está disponível. "
                "Usando CPU."
            )
            return "cpu"
        return preferencia

    return "cuda" if cuda_disponivel else "cpu"


def descrever_ambiente() -> dict[str, str]:
    """Coleta a identificação do ambiente para os relatórios.

    Returns:
        Dicionário com versões de bibliotecas e identificação de hardware/SO.
    """
    import numpy
    import torch
    import ultralytics

    descricao = {
        "python": platform.python_version(),
        "plataforma": platform.platform(),
        "processador": platform.processor() or "desconhecido",
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "numpy": numpy.__version__,
        "cuda_disponivel": str(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():  # pragma: no cover - depende de hardware
        descricao["gpu"] = torch.cuda.get_device_name(0)
    return descricao


@dataclass
class ConfigTreino:
    """Hiperparâmetros de treinamento do YOLO11-cls.

    Os valores padrão reproduzem exatamente a configuração base da Seção 6 do
    spec.md. Todos podem ser sobrescritos por argumentos de linha de comando.
    """

    modelo_base: str = "yolo11n-cls.pt"
    epocas: int = 100
    imgsz: int = 224
    batch: int = 32
    patience: int = 20
    optimizer: str = "AdamW"
    lr0: float = 0.001
    cos_lr: bool = True
    dropout: float = 0.1
    seed: int = 42
    deterministic: bool = True
    workers: int = 0  # obrigatório no Windows (spec.md, Seção 3)
    dispositivo: str | None = None  # None => autodetecta
    nome_execucao: str | None = None  # None => derivado do modelo_base

    # Data augmentation em tempo de treino (Seção 6.1.1 do TCC).
    # Nunca gravar imagens aumentadas em disco: foi a causa raiz do vazamento
    # identificado no dataset original (spec.md, problema P4).
    fliplr: float = 0.5
    flipud: float = 0.3
    degrees: float = 20.0
    scale: float = 0.4
    translate: float = 0.1
    hsv_h: float = 0.015
    hsv_s: float = 0.5
    hsv_v: float = 0.4  # variação de brilho — mitiga a limitação de iluminação
    erasing: float = 0.2

    def __post_init__(self) -> None:
        if self.nome_execucao is None:
            base = Path(self.modelo_base).stem.replace("-", "_")
            self.nome_execucao = f"{base}_alface"

    @property
    def nome_arquivo_pesos(self) -> str:
        """Nome do arquivo de pesos finais em ``modelos/``."""
        return f"{self.nome_execucao}.pt"

    def resolver_dispositivo(self) -> str:
        """Retorna o dispositivo efetivo para esta configuração."""
        return detectar_dispositivo(self.dispositivo)

    def para_argumentos_ultralytics(self, dados: Path, projeto: Path) -> dict:
        """Monta o dicionário de argumentos de ``YOLO.train``.

        Args:
            dados: Diretório raiz do dataset de classificação (com ``train/`` e ``val/``).
            projeto: Diretório onde o Ultralytics gravará a execução.

        Returns:
            Argumentos nomeados prontos para ``model.train(**args)``.
        """
        dispositivo = self.resolver_dispositivo()
        return {
            "data": str(dados),
            "epochs": self.epocas,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "patience": self.patience,
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "cos_lr": self.cos_lr,
            "dropout": self.dropout,
            # `label_smoothing` NÃO é passado: a partir do Ultralytics 8.4.x ele
            # consta de `removed_keys` em `ultralytics/cfg/__init__.py`, sendo
            # aceito sem erro mas silenciosamente ignorado. Passá-lo daria a falsa
            # impressão de que a regularização está ativa. Ver docs/DIVERGENCIAS.md.
            "seed": self.seed,
            "deterministic": self.deterministic,
            "device": dispositivo,
            # AMP só faz sentido em GPU; em CPU deve permanecer desligado.
            "amp": dispositivo.startswith("cuda"),
            "workers": self.workers,
            "project": str(projeto),
            "name": self.nome_execucao,
            "exist_ok": True,
            "plots": True,
            "fliplr": self.fliplr,
            "flipud": self.flipud,
            "degrees": self.degrees,
            "scale": self.scale,
            "translate": self.translate,
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            "erasing": self.erasing,
        }

    def para_dicionario(self) -> dict:
        """Serializa a configuração para ``desenvolvimento/reports/treino_config.json``."""
        return asdict(self)
