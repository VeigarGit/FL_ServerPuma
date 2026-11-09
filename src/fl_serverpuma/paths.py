# Responsabilidade: Define SOURCE_ROOT, usado para localizar datasets, resultados e recursos
# operacionais a partir do diretório src.
#

"""Base directory for the project's datasets, results and runtime resources."""

from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
