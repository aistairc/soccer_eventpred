import os
from pathlib import Path

PROJECT_DIR = Path(__file__).parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_DIR / "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_DIR / "outputs"))
