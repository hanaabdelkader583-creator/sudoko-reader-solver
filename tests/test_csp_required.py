"""Required test cases for the CSP: 5 normal, 3 edge, 2 impossible."""
import numpy as np
import pytest
import config
from src.csp.solver import solve
from src.csp.model import is_solved


def _puzzle(givens, k):
    d = np.load(config.SUDOKU_DIR / f"puzzles_g{givens}.npz")
    return d["puzzles"][k], [int(v) for v in d["solutions"][k]]


# ---------- 5 normal ----------
def test_normal_1_bt_35_givens():
    p, s = _puzzle(35, 0)
    assert solve(p, use_fc=False, use_mrv=False)[0] == s


def test_normal_2_bt_fc_30_givens():
    p, s = _puzzle(30, 1)
    assert solve(p, use_fc=True, use_mrv=False)[0] == s


def test_normal_3_bt_fc_ac3_30_givens():
    p, s = _puzzle(30, 2)
    assert solve(p, use_fc=True, use_mrv=False, use_ac3=True)[0] == s


def test_normal_4_fc_mrv_25_givens():
    p, s = _puzzle(25, 3)
    assert solve(p, use_fc=True, use_mrv=True)[0] == s


def test_normal_5_maintained_ac3_25_givens():
    p, s = _puzzle(25, 4)
    assert solve(p, use_ac3=True, maintain_ac3=True)[0] == s


# ---------- 3 edge ----------
def test_edge_1_already_solved_grid():
    _, s = _puzzle(30, 5)
    sol, st = solve(s)
    assert sol == s and st.assignments == 0


def test_edge_2_exactly_one_blank():
    _, s = _puzzle(30, 6)
    g = list(s)
    g[40] = 0
    sol, st = solve(g)
    assert sol == s and st.assignments == 1


def test_edge_3_empty_grid_gets_a_valid_solution():
    sol, _ = solve([0] * 81)
    assert sol is not None and is_solved(sol)


# ---------- 2 impossible ----------
def test_impossible_1_clashing_givens():
    g = [0] * 81
    g[0] = g[1] = 5
    sol, st = solve(g)
    assert sol is None and st.assignments == 0


def test_impossible_2_empty_domain_without_clash():
    g = [0] * 81
    g[:8] = [1, 2, 3, 4, 5, 6, 7, 8]
    g[17] = 9
    for kw in ({"use_fc": False, "use_mrv": False}, {"use_ac3": True}):
        assert solve(g, **kw)[0] is None