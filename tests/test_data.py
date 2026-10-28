import numpy as np
import config
from src.data.sudoku_gen import count_solutions
from src.data.mosaic import load_raw_test, render_mosaic, split_tiles, blank_mask


def _load(g):
    return np.load(config.SUDOKU_DIR / f"puzzles_g{g}.npz")


def test_puzzles_have_exact_givens_and_agree_with_solution():
    for g in config.GIVENS_LEVELS:
        d = _load(g)
        P, S = d["puzzles"], d["solutions"]
        assert ((P != 0).sum(1) == g).all()
        assert ((P == 0) | (P == S)).all()


def test_puzzles_are_unique():
    for g in config.GIVENS_LEVELS:
        assert all(count_solutions(p, 2) == 1 for p in _load(g)["puzzles"])


def test_mosaic_labels_match_cells_and_blank_rule_works():
    test_images, test_labels = load_raw_test()
    for g in config.GIVENS_LEVELS:
        d = _load(g)
        P, I = d["puzzles"], d["cell_img_idx"]
        given = P != 0
        assert (test_labels[I[given]] == P[given]).all()
        tiles = split_tiles(render_mosaic(I[0], test_images))
        assert (blank_mask(tiles) == (P[0] == 0)).all()