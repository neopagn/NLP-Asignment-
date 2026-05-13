from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"


ENTITY_LABELS = ["PARTY", "MONEY", "DATE", "RATE", "PENALTY", "LAW"]
INTENT_LABELS = ["Obligation", "Prohibition", "Right", "Termination Condition"]
