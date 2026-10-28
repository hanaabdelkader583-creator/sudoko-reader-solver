"""Reproducible data pipeline.  Run from the repo root:  python make_data.py"""
import time
import numpy as np
import config
from src.data.mnist_loader import load_mnist_split
from src.data.sudoku_gen import make_puzzle
import matplotlib
matplotlib.use("Agg")                      # no display needed
import matplotlib.pyplot as plt
from src.data.mosaic import load_raw_test, sample_cell_indices, render_mosaic
import json


def build_mnist_manifest():
    """Save which original MNIST indices fall in train/val/test (audit trail)."""
    d = load_mnist_split(config.SEED)
    config.PROC_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.PROC_DIR / "mnist_split.npz",
        train_orig_idx=d["train_orig_idx"],
        val_orig_idx=d["val_orig_idx"],
        test_orig_idx=d["test_orig_idx"],
    )
    counts = {k: len(d[f"{k}_labels"]) for k in ("train", "val", "test")}
    print("MNIST (digit 0 removed):", counts)
    return counts


def puzzle_seed(givens, i):
    """Readable, collision-free per-puzzle seed derived from config.SEED."""
    return config.SEED * 1_000_000 + givens * 1_000 + i


def build_puzzles():
    config.SUDOKU_DIR.mkdir(parents=True, exist_ok=True)
    n = config.PUZZLES_PER_LEVEL
    for g in config.GIVENS_LEVELS:
        path = config.SUDOKU_DIR / f"puzzles_g{g}.npz"
        if path.exists():                       # resumable: skip finished levels
            print(f"g={g}: exists, skipping")
            continue
        P, S, seeds = [], [], []
        t0 = time.time()
        for i in range(n):
            seed = puzzle_seed(g, i)
            p, s = make_puzzle(g, seed)
            P.append(p); S.append(s); seeds.append(seed)
            if (i + 1) % 10 == 0:
                print(f"g={g}: {i+1}/{n}  ({time.time()-t0:.0f}s)")
        np.savez_compressed(
            path,
            puzzles=np.stack(P).astype(np.int8),
            solutions=np.stack(S).astype(np.int8),
            puzzle_seeds=np.array(seeds, dtype=np.int64),
            seed_group=(np.arange(n) // config.PUZZLES_PER_GROUP).astype(np.int8),
        )
        print(f"g={g}: saved {path.name}")

def attach_mosaic_indices():
    """Add `cell_img_idx` (N, 81) to each puzzle file. Deterministic, safe to re-run."""
    _, test_labels = load_raw_test()
    for g in config.GIVENS_LEVELS:
        path = config.SUDOKU_DIR / f"puzzles_g{g}.npz"
        d = dict(np.load(path))
        d["cell_img_idx"] = np.stack([
            sample_cell_indices(p, test_labels, np.random.default_rng([int(s), 1]))
            for p, s in zip(d["puzzles"], d["puzzle_seeds"])
        ]).astype(np.int32)
        np.savez_compressed(path, **d)
        print(f"g={g}: cell_img_idx attached")


def render_samples():
    """One PNG per givens level (puzzle 0) so we can eyeball the mosaics."""
    test_images, _ = load_raw_test()
    config.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for g in config.GIVENS_LEVELS:
        d = np.load(config.SUDOKU_DIR / f"puzzles_g{g}.npz")
        m = render_mosaic(d["cell_img_idx"][0], test_images)
        plt.imsave(config.SAMPLES_DIR / f"mosaic_g{g}_p0.png", m, cmap="gray", vmin=0, vmax=1)
    print("samples written to", config.SAMPLES_DIR)

def write_manifest(counts):
    manifest = {
        "seed": config.SEED,
        "mnist_counts_digit0_removed": counts,
        "givens_levels": list(config.GIVENS_LEVELS),
        "puzzles_per_level": config.PUZZLES_PER_LEVEL,
        "seed_groups": config.N_SEED_GROUPS,
        "puzzle_seed_rule": "SEED*1_000_000 + givens*1_000 + i",
        "blank_threshold": config.BLANK_THRESHOLD,
        "files": sorted(p.name for p in config.SUDOKU_DIR.glob("*.npz")),
    }
    (config.PROC_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("manifest.json written")


def write_dictionary(counts):
    config.DOCS_DIR.mkdir(exist_ok=True)
    text = f"""# Data dictionary

Everything is reproducible with `python make_data.py` (global seed {config.SEED}).

## MNIST (real)
Source: http://yann.lecun.com/exdb/mnist/ (raw IDX files in `data/MNIST/raw/`).
Pixels are uint8 on disk and scaled to 0-1 float32 when used.
The 60k training images are split with a seeded permutation: {config.N_TRAIN} train / {config.N_VAL} validation.
The 10k test set is untouched. Digit 0 is removed AFTER splitting, so labels are 1-9.

| Split | Images after removing 0 |
|---|---|
| train | {counts['train']} |
| val | {counts['val']} |
| test | {counts['test']} |

`data/processed/mnist_split.npz`: `train_orig_idx`, `val_orig_idx`, `test_orig_idx` (REAL; indices into the raw MNIST arrays).

## Sudoku puzzles (GENERATED)
Files: `data/processed/sudoku/puzzles_g{{25,30,35}}.npz`, {config.PUZZLES_PER_LEVEL} puzzles each.
Generation: a full grid is built by randomised backtracking, then cells are removed in random order
while the solution stays unique (checked by an independent solution counter), until the target number
of givens is reached. Puzzles are unique but not necessarily minimal.
Puzzle seed = SEED*1_000_000 + givens*1_000 + i.

| Array | Shape / dtype | Real or generated | Meaning |
|---|---|---|---|
| `puzzles` | (100, 81) int8 | generated | 0 = blank, 1-9 = given |
| `solutions` | (100, 81) int8 | generated | the unique solution |
| `puzzle_seeds` | (100,) int64 | generated | seed used for each puzzle |
| `seed_group` | (100,) int8 | generated | 0-4; 20 puzzles per group, used for mean/std over 5 seeds |
| `cell_img_idx` | (100, 81) int32 | real images, generated assignment | index into the RAW 10k MNIST test set for each given cell; -1 for blank cells |

## Mosaics (GENERATED from real images)
Built on demand by `src/data/mosaic.py`: each given cell gets a distinct real test image whose true label
equals the cell digit (sampling seed `[puzzle_seed, 1]`); blank cells are all-zero 28x28 tiles.
Mosaic size is 252x252, float32 in 0-1. Blank cells are detected by a brightness rule
(max pixel < {config.BLANK_THRESHOLD}), never by the NN.
Test images are used for mosaics because training and test writers are different people.

## Not yet added
Injected misreads (hard, low-confidence digits) are added in Phase 3, once NN probabilities exist.
"""
    (config.DOCS_DIR / "data_dictionary.md").write_text(text, encoding="utf-8")
    print("data_dictionary.md written")


if __name__ == "__main__":
    counts =build_mnist_manifest()
    build_puzzles()
    attach_mosaic_indices()
    render_samples()
    write_manifest(counts)
    write_dictionary(counts)