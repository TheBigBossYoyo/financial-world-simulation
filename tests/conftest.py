import sys
from pathlib import Path

# make `config`, `modules` and `evaluate_forecasts` importable from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
