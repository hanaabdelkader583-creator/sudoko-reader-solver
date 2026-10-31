import itertools
import numpy as np
import config
from src.csp.solver import solve
from src.csp.model import is_solved


def _puzzles(g, n):
    d = np.load(config.SUDOKU_DIR / f"puzzles_g{g}.npz")
    return list(zip(d["puzzles"][:n], d["solutions"][:n]))


def test_all_switch_combinations_solve_correctly():
    for fc, mrv in itertools.product([False, True], repeat=2):
        if mrv and not fc:
            continue                       # MRV needs domains, so it requires FC
        for p, s in _puzzles(35, 3):
            sol, _ = solve(p, use_fc=fc, use_mrv=mrv)
            assert sol == [int(v) for v in s]


def test_fc_and_mrv_reduce_work_on_hard_puzzles():
    for p, _ in _puzzles(25, 3):
        _, bt = solve(p, use_fc=False, use_mrv=False, max_assignments=200_000)
        _, fc = solve(p, use_fc=True, use_mrv=False)
        _, both = solve(p, use_fc=True, use_mrv=True)
        assert both.assignments <= fc.assignments
        assert fc.assignments <= bt.assignments


def test_impossible_cases():
    g = [0] * 81
    g[0] = g[1] = 5
    assert solve(g)[0] is None             # clash
    h = [0] * 81
    h[:8] = [1, 2, 3, 4, 5, 6, 7, 8]
    h[17] = 9
    sol, st = solve(h)
    assert sol is None                     # no clash, but cell 8 has an empty domain