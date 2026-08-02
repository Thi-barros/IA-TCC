"""Configuração compartilhada do pytest.

Coloca a raiz do projeto e ``src/`` no ``sys.path``, para que ``import ia_alface``
e ``import api`` funcionem sem exigir instalação do pacote — independentemente do
diretório a partir do qual o pytest é invocado.
"""

from __future__ import annotations

import sys
from pathlib import Path

# desenvolvimento/tests/conftest.py -> tests -> desenvolvimento -> raiz
RAIZ = Path(__file__).resolve().parents[2]

for caminho in (RAIZ, RAIZ / "src"):
    if str(caminho) not in sys.path:
        sys.path.insert(0, str(caminho))
