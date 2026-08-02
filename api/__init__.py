"""Serviço de inferência HTTP do módulo de IA.

Camada de exposição da biblioteca ``ia_alface`` por REST. Mantida separada de
``src/ia_alface/`` porque a biblioteca é deliberadamente pura: não conhece HTTP,
MQTT nem banco de dados.

Execução: ``uvicorn api.main:app --reload``
"""

from __future__ import annotations

__version__ = "1.0.0"
