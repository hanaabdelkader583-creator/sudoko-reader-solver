import numpy as np
import config
from src.csp.backtracking import solve_bt
from src.csp.model import is_solved


def test_solves_35_given_puzzles_correctly():
    d = np.load(config.SUDOKU_DIR / "puzzles_g35.npz")
    for p, s in zip(d["puzzles"][:5], d["solutions"][:5]):
        sol, st = solve_bt(p)
        assert sol is not None and is_solved(sol)
        assert sol == [int(v) for v in s]           # unique puzzle -> same answer
        assert st.assignments > 0


def test_already_solved_grid():
    d = np.load(config.SUDOKU_DIR / "puzzles_g30.npz")
    sol, st = solve_bt(d["solutions"][0])
    assert sol is not None and st.assignments == 0 and st.backtracks == 0


def test_clashing_givens_give_no_solution():
    g = [0] * 81
    g[0] = g[1] = 5
    sol, st = solve_bt(g)
    assert sol is None and st.assignments == 0


def test_unsolvable_but_conflict_free():
    # Row 0 holds 1-8, and a 9 sits in column 8 below, so cell 8 has no value.
    g = [0] * 81
    g[:8] = [1, 2, 3, 4, 5, 6, 7, 8]
    g[17] = 9
    sol, st = solve_bt(g)
    assert sol is None