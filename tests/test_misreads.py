"""Tests for src/data/misreads.py (synthetic images + fake NN: no MNIST or weights needed)."""
import numpy as np
import pytest

from src.data import misreads as mr
from src.data import mnist_loader as ml
from tests.test_search import SOL

N = 150
LABELS = (np.arange(N) % 9) + 1
IMAGES = np.stack([np.full((28, 28), 100 + k, dtype=np.uint8) for k in range(N)])
# every 5th image is "hard": the fake NN reads it as the next digit up
PRED = np.where(np.arange(N) % 5 == 0, LABELS % 9 + 1, LABELS)


@pytest.fixture(autouse=True)
def plain_to_float(monkeypatch):
    monkeypatch.setattr(ml, "to_float", lambda im: np.asarray(im, dtype=np.float32) / 255.0)


def fake_predict(tiles, **kw):
    ids = np.rint(tiles[:, 0, 0] * 255).astype(int) - 100
    blank = tiles.max(axis=(1, 2)) < 0.3
    probs = np.full((len(tiles), 9), 0.01)
    for i, k in enumerate(ids):
        if not blank[i]:
            probs[i, PRED[k] - 1] = 0.9
    probs[blank] = 0.0
    return probs, blank


def read_grid(tiles):
    probs, blank = fake_predict(tiles)
    return np.where(blank, 0, probs.argmax(axis=1) + 1).reshape(9, 9)


def make_puzzle():
    p = np.asarray(SOL).copy().reshape(81)
    p[np.random.default_rng(1).permutation(81)[:20]] = 0
    return p


def make_pool():
    return mr.MisreadPool.from_model(IMAGES, LABELS, predict=fake_predict)


def test_pool_holds_only_images_the_nn_misreads():
    pool = make_pool()
    for d in range(1, 10):
        assert len(pool.hard_by_digit[d]) > 0
        assert all(LABELS[k] == d and PRED[k] != d for k in pool.hard_by_digit[d])
        assert all(LABELS[k] == d and PRED[k] == d for k in pool.clean_by_digit[d])


def test_instance_plants_exactly_n_misreads():
    puzzle = make_puzzle()
    tiles, injected, _ = mr.make_misread_instance(puzzle, 4, seed=0, pool=make_pool(),
                                                  test_images=IMAGES)
    assert tiles.shape == (81, 28, 28) and len(injected) == 4
    read, truth = read_grid(tiles), puzzle.reshape(9, 9)
    wrong = {(int(r), int(c)) for r, c in zip(*np.nonzero(read != truth))}
    assert wrong == set(injected)
    for (r, c), (true_d, guess) in injected.items():
        assert truth[r, c] == true_d and read[r, c] == guess and guess != true_d


def test_same_seed_gives_same_instance():
    puzzle, pool = make_puzzle(), make_pool()
    a = mr.make_misread_instance(puzzle, 3, 7, pool, IMAGES)
    b = mr.make_misread_instance(puzzle, 3, 7, pool, IMAGES)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]


def test_too_many_misreads_raises():
    with pytest.raises(ValueError):
        mr.make_misread_instance(make_puzzle(), 70, 0, make_pool(), IMAGES)
