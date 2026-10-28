"""Turn per-cell MNIST indices into a 252x252 mosaic, and back into 81 tiles."""
import numpy as np
import config
from src.data import mnist_loader as ml


def load_raw_test():
    """Raw 10k test set (uint8 images, labels 0-9). Indices into this are what we store."""
    ml.ensure_mnist()
    return (ml.read_images(ml.FILES["test_images"]),
            ml.read_labels(ml.FILES["test_labels"]))


def sample_cell_indices(puzzle, test_labels, rng):
    """For each given cell, pick a distinct test image whose label equals the digit.
    Returns int32 array of length 81, -1 for blank cells."""
    idx = np.full(81, -1, dtype=np.int32)
    for d in range(1, 10):                       # fixed order -> deterministic
        cells = np.flatnonzero(puzzle == d)
        if len(cells):
            pool = np.flatnonzero(test_labels == d)
            idx[cells] = rng.choice(pool, size=len(cells), replace=False)
    return idx


def render_mosaic(cell_img_idx, test_images):
    """81 indices -> one (252, 252) float32 image in 0-1. Blank cells stay all zero."""
    tiles = np.zeros((81, 28, 28), dtype=np.float32)
    for c, k in enumerate(cell_img_idx):
        if k >= 0:
            tiles[c] = ml.to_float(test_images[k])
    return tiles.reshape(9, 9, 28, 28).transpose(0, 2, 1, 3).reshape(252, 252)


def split_tiles(mosaic):
    """Inverse of render_mosaic: (252, 252) -> (81, 28, 28)."""
    return mosaic.reshape(9, 28, 9, 28).transpose(0, 2, 1, 3).reshape(81, 28, 28)


def blank_mask(tiles):
    """The simple brightness rule from the brief: blank if max pixel < threshold."""
    return tiles.max(axis=(1, 2)) < config.BLANK_THRESHOLD