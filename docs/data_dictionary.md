# Data dictionary

Everything is reproducible with `python make_data.py` (global seed 42).

## MNIST (real)
Source: http://yann.lecun.com/exdb/mnist/ (raw IDX files in `data/MNIST/raw/`).
Pixels are uint8 on disk and scaled to 0-1 float32 when used.
The 60k training images are split with a seeded permutation: 50000 train / 10000 validation.
The 10k test set is untouched. Digit 0 is removed AFTER splitting, so labels are 1-9.

| Split | Images after removing 0 |
|---|---|
| train | 45073 |
| val | 9004 |
| test | 9020 |

`data/processed/mnist_split.npz`: `train_orig_idx`, `val_orig_idx`, `test_orig_idx` (REAL; indices into the raw MNIST arrays).

## Sudoku puzzles (GENERATED)
Files: `data/processed/sudoku/puzzles_g{25,30,35}.npz`, 100 puzzles each.
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
(max pixel < 0.05), never by the NN.
Test images are used for mosaics because training and test writers are different people.

## Neural Network & Vision Pipeline (CNN & Baseline)
* **MNIST Digits (Classes 1-9):** Real handwritten digit images (28x28 pixels) and labels sourced from the public MNIST database. Pixel intensities are scaled to [0, 1].
* **Blank Cells (Class 0):** Generated synthetically. These are 28x28 near-black image crops injected with simulated sensor noise, specks, and leftover grid-line fragments to train the model to ignore empty Sudoku cells. 
* **Trained Weights (`models/sudoku_digit_cnn.pt`):** Saved CNN checkpoint containing model state dictionaries, architectural metadata, and embedded normalization buffers (`MNIST_MEAN` and `MNIST_STD`).
* **Injected Misreads:** Low-confidence digits intentionally injected into test puzzles to allow the Search and CSP integration pipeline to evaluate correction handling.

## Not yet added
Injected misreads (hard, low-confidence digits) are added in Phase 3, once NN probabilities exist.