"""
src/data/misreads.py
====================
Find MNIST test images that the NN genuinely misreads (using its real
probabilities) and plant them in puzzle mosaics, so the correction search
has real errors to repair.
"""
from __future__ import annotations
import numpy as np
from src.data import mnist_loader as ml
from src.data.mosaic import render_mosaic, split_tiles

CHUNK = 81  # predict_cells takes exactly 81 tiles


def predict_all(test_images, predict=None, temperature=1.0):
    """Run the NN over every raw test image (81 at a time). Returns probs (N, 9)."""
    if predict is None:
        from src.nn.interface import predict_cells as predict
    n = len(test_images)
    probs = np.zeros((n, 9), dtype=np.float64)
    for s in range(0, n, CHUNK):
        batch = np.stack([ml.to_float(im) for im in test_images[s:s + CHUNK]])
        m = len(batch)
        if m < CHUNK:
            batch = np.concatenate([batch, np.zeros((CHUNK - m, 28, 28), dtype=np.float32)])
        p, _ = predict(batch, temperature=temperature)
        probs[s:s + m] = np.asarray(p)[:m]
    return probs


class MisreadPool:
    """Per digit 1-9: 'hard' images (NN top guess is wrong) and 'clean' images (NN reads them right).
    Images the NN treats as blank (all-zero probs) are in neither group."""

    def __init__(self, probs, labels):
        self.probs = np.asarray(probs, dtype=np.float64)
        self.labels = np.asarray(labels)
        self.pred = self.probs.argmax(axis=1) + 1
        readable = self.probs.sum(axis=1) > 0
        self.hard_by_digit = {d: np.flatnonzero(readable & (self.labels == d) & (self.pred != d))
                              for d in range(1, 10)}
        self.clean_by_digit = {d: np.flatnonzero(readable & (self.labels == d) & (self.pred == d))
                               for d in range(1, 10)}

    @classmethod
    def from_model(cls, test_images, test_labels, predict=None, temperature=1.0):
        return cls(predict_all(test_images, predict, temperature), test_labels)

    def hard(self, d, min_true_prob=0.0):
        """Hard images of digit d where the NN still gives the true digit >= min_true_prob."""
        idx = self.hard_by_digit[d]
        return idx[self.probs[idx, d - 1] >= min_true_prob]


def sample_clean_indices(puzzle, pool, rng):
    """Like mosaic.sample_cell_indices, but only from images the NN reads correctly."""
    puzzle = np.asarray(puzzle).reshape(81)
    idx = np.full(81, -1, dtype=np.int32)
    for d in range(1, 10):
        cells = np.flatnonzero(puzzle == d)
        if len(cells):
            idx[cells] = rng.choice(pool.clean_by_digit[d], size=len(cells), replace=False)
    return idx


def inject_misreads(puzzle, cell_img_idx, n, pool, rng, min_true_prob=0.0):
    """Swap n given cells' images for images the NN misreads.
    Returns (new_idx, {(row, col): (true_digit, nn_guess)})."""
    puzzle = np.asarray(puzzle).reshape(81)
    idx = np.array(cell_img_idx, copy=True)
    used = {int(k) for k in idx if k >= 0}
    injected = {}
    for cell in rng.permutation(np.flatnonzero(puzzle > 0)):
        if len(injected) == n:
            break
        d = int(puzzle[cell])
        cand = np.setdiff1d(pool.hard(d, min_true_prob), list(used))
        if len(cand) == 0:
            continue
        k = int(rng.choice(cand))
        used.discard(int(idx[cell]))
        used.add(k)
        idx[cell] = k
        injected[(int(cell) // 9, int(cell) % 9)] = (d, int(pool.pred[k]))
    if len(injected) < n:
        raise ValueError(f"could only inject {len(injected)} of {n} misreads")
    return idx, injected


def make_misread_instance(puzzle, n_misreads, seed, pool, test_images, min_true_prob=0.0):
    """puzzle (81 or 9x9, 0 = blank) -> (tiles (81,28,28), injected, cell_img_idx).
    Exactly n_misreads cells are read wrong by the NN; every other given is read right."""
    rng = np.random.default_rng(seed)
    idx = sample_clean_indices(puzzle, pool, rng)
    idx, injected = inject_misreads(puzzle, idx, n_misreads, pool, rng, min_true_prob)
    tiles = split_tiles(render_mosaic(idx, test_images))
    return tiles, injected, idx
