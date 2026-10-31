import numpy as np
import config
from src.csp.model import (PEERS, UNITS, find_conflicts, is_solved,
                           initial_domains)


def test_structure():
    assert len(UNITS) == 27 and all(len(u) == 9 for u in UNITS)
    assert all(len(p) == 20 for p in PEERS)
    assert all(i not in PEERS[i] for i in range(81))


def test_solutions_from_dataset_are_solved():
    d = np.load(config.SUDOKU_DIR / "puzzles_g30.npz")
    assert all(is_solved(s) for s in d["solutions"])


def test_puzzles_are_not_solved_but_conflict_free():
    d = np.load(config.SUDOKU_DIR / "puzzles_g30.npz")
    for p in d["puzzles"]:
        assert not is_solved(p) and find_conflicts(p) == []


def test_conflict_detection_and_domains():
    g = [0] * 81
    g[0] = g[1] = 5                       # same row, same digit
    assert find_conflicts(g) == [(0, 1)]
    dom = initial_domains(g)
    assert dom[0] == {5} and dom[2] == set(range(1, 10))