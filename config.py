from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "MNIST" / "raw"
PROC_DIR = DATA_DIR / "processed"
DOCS_DIR = ROOT / "docs"

N_TRAIN = 50_000          # from the 60k training images
N_VAL = 10_000
GIVENS_LEVELS = (25, 30, 35)   # needed by Experiment 2
PUZZLES_PER_LEVEL = 100
BLANK_THRESHOLD = 0.05    # tile is blank if max pixel < this
MAX_GEN_ATTEMPTS = 200

N_SEED_GROUPS = 5
PUZZLES_PER_GROUP = PUZZLES_PER_LEVEL // N_SEED_GROUPS   # 20
SUDOKU_DIR = PROC_DIR / "sudoku"
SAMPLES_DIR = PROC_DIR / "samples"